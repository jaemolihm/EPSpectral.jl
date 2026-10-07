# `run_scgd0` end to end on the 1D Holstein `Model` of ElectronPhonon
# (filter -> unfold -> G2Calculator -> phonon-table gather -> f_to_i -> kernel -> KK -> μ), against
# the closed-form one-shot and the local scalar iteration of `holstein_1d_reference.jl`.
@testset "scGD0 1D Holstein" begin
    t, λ, ω0, nk, T = 1.0, 0.5, 0.2, 64, 0.1
    # holstein_model stores epmat = g √(2ω0 M), so the extracted g2 = |g|² = 2·dimension·λ·ω0·|t|
    # for every pair.
    g2 = 0.2
    model = EP.holstein_model(; t, λ, ω₀ = ω0, dimension = 1, verbose = false)
    ωs = collect(-4.0:0.02:4.0)
    ωs_dense = range(extrema(ωs)...; step = minimum(diff(ωs)))
    η_init, tol = 0.05, 1e-6   # model units (t = 1), not the meV-scale defaults
    # `type = :Metal`: from nlist = 0 and an integer nelec the constructor would infer
    # :Semiconductor.
    occ_for(nlist; μlist = nothing, Tlist = [T]) = ElectronOccupationParams(; Tlist, nlist, μlist,
        nelec = 1, volume = model.volume, spin_degeneracy = 2, type = :Metal)
    run(occ; kwargs...) = run_scgd0(model, (nk, 1, 1), (-Inf, Inf); occ, ωs, η_init, tol,
        verbosity = 0, kwargs...)

    for (filling, nlist, μ_fixed) in (("half", 0.0, 0.0), ("doped", -0.3, 0.3))
        @testset "one-shot, $filling filling" begin
            # μ pinned: the closed form needs the μ that iteration 1 used.
            res = run(occ_for(nlist; μlist = μ_fixed); maxiter = 1, fix_μ = true)
            # k -> -k: 33 irreducible of 64.
            @test size(res.Σ) == (length(ωs), nk ÷ 2 + 1, 1)
            ImΣ = holstein_1d_oneshot_imsigma(ωs; nk, t, ω0, g2, T, μ = μ_fixed, η = η_init)
            Σ_ref = complex.(kramers_kronig(ωs, ImΣ), ImΣ)
            @test maximum(norm(res.Σ[:, i, 1] - Σ_ref) for i in axes(res.Σ, 2)) <=
                1e-12 * norm(Σ_ref)
            @test res.μ == [μ_fixed]
            @test res.converged == [false]
        end

        @testset "self-consistent, $filling filling" begin
            n_target = 1 + nlist
            res = run(occ_for(nlist))
            niter = size(res.history.err, 2)
            @test res.converged == [true]
            @test maximum(abs, res.Σ .- res.Σ[:, 1:1, :]) < 1e-12
            μ0 = holstein_1d_bare_μ(; nk, t, T, n_target, spin_degeneracy = 2)
            Σ_ref, μ_ref = holstein_1d_local_scgd0(ωs, ωs_dense; nk, t, ω0, g2, T, η_init,
                μ_start = μ0, niter, fix_μ = false, n_target, spin_degeneracy = 2)
            @test norm(res.Σ[:, 1, 1] - Σ_ref) <= 1e-10 * norm(Σ_ref)
            @test abs(res.μ[1] - μ_ref) < 1e-9
            nlist == 0 || @test abs(res.μ[1] - μ0) > 1e-6

            # Chunking the inner states (7 does not divide 64) only reorders the f sum.
            res_tiled = run(occ_for(nlist); gpu_tile = 7)
            @test norm(res_tiled.Σ - res.Σ) <= 1e-12 * norm(res.Σ)
            @test abs(res_tiled.μ[1] - res.μ[1]) < 1e-12   # μ ~ 1e-4 at half filling: absolute

            # Save and load.
            res_loaded = load_scgd0(save_scgd0(joinpath(mktempdir(), "scgd0.jld2"), res))
            for name in fieldnames(SCGD0Result)
                name === :el_i && continue
                @test isequal(getfield(res_loaded, name), getfield(res, name))
            end
            for name in fieldnames(typeof(res.el_i))
                name === :kpts && continue
                @test isequal(getfield(res_loaded.el_i, name), getfield(res.el_i, name))
            end
            for name in fieldnames(typeof(res.el_i.kpts))
                @test isequal(getfield(res_loaded.el_i.kpts, name), getfield(res.el_i.kpts, name))
            end

            # Restart from the converged result: one iteration, nothing moves beyond tol.
            res_restart = run(occ_for(nlist; μlist = res.μ); Σ_init = res.Σ)
            @test size(res_restart.history.err, 2) == 1
            @test res_restart.converged == [true]
            @test maximum(abs, res_restart.Σ - res.Σ) < tol
            @test abs(res_restart.μ[1] - res.μ[1]) < tol

            # Two temperatures in one run: each is the single-temperature run, and each freezes at
            # its own iteration.
            res_T2 = run(occ_for(nlist; Tlist = [0.2]))
            res_both = run(occ_for(nlist; Tlist = [T, 0.2]))
            for (iT, r) in enumerate((res, res_T2))
                @test norm(res_both.Σ[:, :, iT] - r.Σ[:, :, 1]) <= 1e-12 * norm(r.Σ)
                @test abs(res_both.μ[iT] - r.μ[1]) < 1e-12
            end
            @test res_both.history.iter_frozen == [size(r.history.err, 2) for r in (res, res_T2)]
        end
    end

    @testset "argument checks" begin
        @test_throws ArgumentError run(occ_for(0.0); fix_μ = true)
        @test_throws ArgumentError run(ElectronOccupationParams(; Tlist = [T], nlist = 0.0,
            nelec = 1, volume = model.volume, spin_degeneracy = 2, type = :Metal, occ_type = :MV))
        @test_throws ArgumentError run(occ_for(0.0); Σ_init = zeros(ComplexF64, length(ωs), 1, 1))
        @test_throws ArgumentError run(occ_for(0.0); η_min = [0.0, 0.0])
    end
end
