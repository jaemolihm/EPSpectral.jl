# The scGD0 iteration on the GPU against the host, on the pb window of `test_scgd0_pb.jl` (a
# degenerate pair at X): first on one shared setup, then end to end.
@testset "scGD0 CPU vs GPU" begin
    folder = Pkg.Artifacts.ensure_artifact_installed("pb", joinpath(@__DIR__, "Artifacts.toml"))
    model = EP.load_model_from_epw_new(folder, "temp", "pb"; epmat_outer_momentum = "el")
    eV, K = unit_to_aru(:eV), unit_to_aru(:K)
    e_F = 11.83 * eV
    nk = (6, 6, 6)
    window = (e_F - 1.0eV, e_F + 1.0eV)
    ωs = e_F .+ vcat(-1.2:0.05:-0.25, -0.24:0.02:0.24, 0.25:0.05:1.2) .* eV
    ωs_dense = range(extrema(ωs)...; step = minimum(diff(ωs)))
    occ = ElectronOccupationParams(; Tlist = [300.0, 600.0] .* K, nlist = 0.0, nelec = 4,
        volume = model.volume, spin_degeneracy = 2, type = :Metal)
    η_min, η_init = [0.0, 0.03eV], 0.05eV
    gpu = EP.gpu_backend()

    @testset "loop on one shared setup" begin
        # The only place a device loop sees host-extracted data, which `run_scgd0` never does.
        # Covers g2 resident on the device and streamed from host RAM, and run-to-run
        # reproducibility of each.
        setup_host = EPSpectral._setup_scgd0(model, nk, window, occ; ωs,
            dω_dense = step(ωs_dense), backend = EP.CPUBackend(), symmetry = model.symmetry,
            verbosity = 0, electron_degen_cutoff = EP.electron_degen_cutoff)
        # The same setup with its arrays on the device; g2 stays on the host, where the loop
        # expects it.
        on_device(name, value) = name in (:g2, :el_i, :multiplets) || !(value isa Array) ?
            value : EP.to_device(gpu, value)
        names = keys(setup_host)
        setup_device = NamedTuple{names}(map(on_device, names, values(setup_host)))
        # Three iterations with the μ update, never converged (tol = 0).
        loop_kwargs = (; η_min, η_init, maxiter = 3, tol = 0.0, Σ_init = nothing, fix_μ = false,
            ω_acoustic = EP.omega_acoustic, verbosity = 0)
        function run_loop(setup, backend; gpu_tile = nothing)
            @test_logs (:warn, r"not converged") match_mode = :any EPSpectral._loop_scgd0(setup,
                occ; loop_kwargs..., backend, gpu_tile)
        end

        res_cpu = run_loop(setup_host, EP.CPUBackend())
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
        # multiplet differs between the two and only multiplet-averaged quantities agree.
        function run(backend)
            @test_logs (:warn, r"not converged") match_mode = :any run_scgd0(model, nk, window;
                occ, ωs, η_init, η_min, maxiter = 3, tol = 0.0, verbosity = 0, backend)
        end
        res_cpu = run(EP.CPUBackend())
        res_gpu = run(gpu)
        el = res_cpu.el_i
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
        function spectral_dos(res, iT)
            A = .-imag.(inv.(ωs_dense .- transpose(res.el_i.es) .- P * res.Σ[:, :, iT])) ./ π
            A * state_weights(res.el_i) .* step(ωs_dense)
        end
        for iT in eachindex(occ.Tlist)
            D_cpu, D_gpu = spectral_dos(res_cpu, iT), spectral_dos(res_gpu, iT)
            @test norm(D_gpu - D_cpu) <= 1e-6 * norm(D_cpu)
        end
    end
end
