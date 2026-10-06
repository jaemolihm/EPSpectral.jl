# The scGD0 iteration on the GPU against the host, on one shared setup (one g2 extraction): the
# only place a device loop sees host-extracted data, which `run_scgd0` never does. Covers g2
# resident on the device and streamed from host RAM, and run-to-run reproducibility of each.
@testset "scGD0 loop, CPU vs GPU" begin
    folder = Pkg.Artifacts.ensure_artifact_installed("pb", joinpath(@__DIR__, "Artifacts.toml"))
    model = EP.load_model_from_epw_new(folder, "temp", "pb"; epmat_outer_momentum = "el")
    eV, K = unit_to_aru(:eV), unit_to_aru(:K)
    e_F = 11.83 * eV
    ωs = e_F .+ vcat(-1.2:0.05:-0.25, -0.24:0.02:0.24, 0.25:0.05:1.2) .* eV
    occ = ElectronOccupationParams(; Tlist = [300.0, 600.0] .* K, nlist = 0.0, nelec = 4,
        volume = model.volume, spin_degeneracy = 2, type = :Metal)
    η_min, η_init = [0.0, 0.03eV], 0.05eV
    setup_host = EPSpectral._setup_scgd0(model, (6, 6, 6), (e_F - 1.0eV, e_F + 1.0eV), occ; ωs,
        dω_dense = minimum(diff(ωs)), backend = EP.CPUBackend(), symmetry = model.symmetry,
        verbosity = 0, electron_degen_cutoff = EP.electron_degen_cutoff)
    gpu = EP.gpu_backend()
    # The same setup with its arrays on the device; g2 stays on the host, where the loop expects it.
    on_device(name, value) =
        name in (:g2, :el_i, :multiplets) || !(value isa Array) ? value : EP.to_device(gpu, value)
    names = keys(setup_host)
    setup_device = NamedTuple{names}(map(on_device, names, values(setup_host)))
    # Three iterations with the μ update, never converged (tol = 0).
    loop_kwargs = (; η_min, η_init, maxiter = 3, tol = 0.0, Σ_init = nothing, fix_μ = false,
        ω_acoustic = EP.omega_acoustic, verbosity = 0)
    function run_loop(setup, backend; gpu_tile = nothing)
        @test_logs (:warn, r"not converged") match_mode = :any EPSpectral._loop_scgd0(setup, occ;
            loop_kwargs..., backend, gpu_tile)
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
