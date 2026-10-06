# The scGD0 iteration on the GPU against the host, on the pb window of `pb_fixture.jl`: first on
# one shared setup, then end to end.
@testset "scGD0 CPU vs GPU" begin
    (; model, nk, window, ωs, ωs_dense, occ, η_init, η_min) = pb_scgd0_fixture()
    gpu = EP.gpu_backend()
    # Three iterations with the μ update, never converged (tol = 0). `run_scgd0` passes the same.
    loop_kwargs = (; η_min, η_init, maxiter = 3, tol = 0.0, Σ_init = nothing, fix_μ = false,
        ω_acoustic = EP.omega_acoustic, verbosity = 0)
    setup_host = EPSpectral._setup_scgd0(model, nk, window, occ; ωs, dω_dense = step(ωs_dense),
        backend = EP.CPUBackend(), symmetry = model.symmetry, verbosity = 0,
        electron_degen_cutoff = EP.electron_degen_cutoff)
    function run_loop(setup, backend; gpu_tile = nothing)
        @test_logs (:warn, r"not converged") match_mode = :any EPSpectral._loop_scgd0(setup,
            occ; loop_kwargs..., backend, gpu_tile)
    end
    # The loop on the host setup is what `run_scgd0` on the host returns.
    res_cpu = run_loop(setup_host, EP.CPUBackend())

    @testset "loop on one shared setup" begin
        # The only place a device loop sees host-extracted data, which `run_scgd0` never does.
        # Covers g2 resident on the device and streamed from host RAM, and run-to-run
        # reproducibility of each. The setup's arrays go to the device; g2 stays on the host,
        # where the loop expects it.
        on_device(name, value) = name in (:g2, :el_i, :multiplets) || !(value isa Array) ?
            value : EP.to_device(gpu, value)
        names = keys(setup_host)
        setup_device = NamedTuple{names}(map(on_device, names, values(setup_host)))
        # Resident, then streamed in chunks of 37 of the 120 inner states.
        for gpu_tile in (nothing, 37)
            res_gpu = run_loop(setup_device, gpu; gpu_tile)
            @test norm(res_gpu.Σ - res_cpu.Σ) <= 1e-10 * norm(res_cpu.Σ)
            @test res_gpu.μ ≈ res_cpu.μ rtol = 1e-10
            res_gpu_again = run_loop(setup_device, gpu; gpu_tile)
            @test res_gpu_again.Σ == res_gpu.Σ
            @test res_gpu_again.μ == res_gpu.μ
        end
    end

    @testset "end to end" begin
        # Each run extracts g2 and diagonalizes on its own backend, so the electron gauge inside a
        # multiplet differs between backends and only multiplet-averaged quantities agree.
        function run_gpu()
            @test_logs (:warn, r"not converged") match_mode = :any run_scgd0(model, nk, window;
                occ, ωs, η_init, η_min, maxiter = 3, tol = 0.0, verbosity = 0, backend = gpu)
        end
        res_gpu = run_gpu()
        el = setup_host.el_i
        @test el.iks == res_gpu.el_i.iks && el.ibands == res_gpu.el_i.ibands
        @test any(el.iks[i] == el.iks[j] && abs(el.es[i] - el.es[j]) < EP.electron_degen_cutoff
                  for i in 1:el.n, j in 1:el.n if i < j)
        # The multiplets of this model are split by up to 2.4e-7 eV, and the residual scales with
        # the split (measured 2.0e-7).
        @test norm(res_gpu.Σ - res_cpu.Σ) <= 1e-6 * norm(res_cpu.Σ)
        @test res_gpu.μ ≈ res_cpu.μ rtol = 1e-8
        # Spectral DOS D_T(ωd) = Σ_i w_i A_i(ωd) dω from each run's Σ; it carries the same
        # split-sized residual (measured 9.3e-8).
        P = EPSpectral.interp_matrix(ωs, ωs_dense)
        function spectral_dos(Σ, iT)
            A = .-imag.(inv.(ωs_dense .- transpose(el.es) .- P * Σ[:, :, iT])) ./ π
            A * state_weights(el) .* step(ωs_dense)
        end
        for iT in eachindex(occ.Tlist)
            D_cpu, D_gpu = spectral_dos(res_cpu.Σ, iT), spectral_dos(res_gpu.Σ, iT)
            @test norm(D_gpu - D_cpu) <= 1e-6 * norm(D_cpu)
        end
        # Two GPU runs, each with its own extraction and eigensolve.
        res_gpu_again = run_gpu()
        @test norm(res_gpu_again.Σ - res_gpu.Σ) <= 1e-12 * norm(res_gpu.Σ)
        @test res_gpu_again.μ ≈ res_gpu.μ rtol = 1e-12
    end
end
