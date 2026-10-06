# `_loop_scgd0` on hand-built 1D Holstein input (see `holstein_1d_loop_setup`), against the
# references of `holstein_1d_reference.jl`. `test_scgd0_holstein.jl` repeats the main checks through
# `run_scgd0`, which builds the same input from the `Model`.
@testset "scGD0 loop, 1D Holstein by hand" begin
    nk, t, ω0, g2, T = 64, 1.0, 0.2, 0.2, 0.1
    ωs = collect(-4.0:0.02:4.0)
    ωs_dense = range(extrema(ωs)...; step = minimum(diff(ωs)))
    η_init, tol = 0.05, 1e-6
    occ_for(nlist; μlist = nothing, Tlist = [T], type = :Metal) = ElectronOccupationParams(;
        Tlist, nlist, μlist, nelec = 1, volume = 1.0, spin_degeneracy = 2, type)
    setup(μ_start) = holstein_1d_loop_setup(; nk, t, ω0, g2, ωs, μ_start)

    for (filling, nlist, μ_fixed) in (("half", 0.0, 0.0), ("doped", -0.3, 0.3))
        @testset "one-shot, $filling filling" begin
            res = run_loop_scgd0(setup([μ_fixed]), occ_for(nlist); η_init, tol, maxiter = 1,
                fix_μ = true)
            ImΣ = holstein_1d_oneshot_imsigma(ωs; nk, t, ω0, g2, T, μ = μ_fixed, η = η_init)
            Σ_ref = complex.(kramers_kronig(ωs, ImΣ), ImΣ)
            @test maximum(norm(res.Σ[:, i, 1] - Σ_ref) for i in 1:nk÷2+1) <= 1e-12 * norm(Σ_ref)
            @test res.μ == [μ_fixed]
            @test res.converged == [false]
        end

        @testset "self-consistent, $filling filling" begin
            n_target = 1 + nlist
            μ0 = holstein_1d_bare_μ(; nk, t, T, n_target, spin_degeneracy = 2)
            res = run_loop_scgd0(setup([μ0]), occ_for(nlist); η_init, tol)
            niter = size(res.history.err, 2)
            @test res.converged == [true]
            @test res.history.iter_frozen == [niter]
            @test maximum(abs, res.Σ .- res.Σ[:, 1:1, :]) < 1e-12
            Σ_ref, μ_ref = holstein_1d_local_scgd0(ωs, ωs_dense; nk, t, ω0, g2, T, η_init,
                μ_start = μ0, niter, fix_μ = false, n_target, spin_degeneracy = 2)
            @test norm(res.Σ[:, 1, 1] - Σ_ref) <= 1e-10 * norm(Σ_ref)
            @test abs(res.μ[1] - μ_ref) < 1e-9
            nlist == 0 || @test abs(res.μ[1] - μ0) > 1e-6

            # Chunking the inner states (7 does not divide 64) only reorders the f sum.
            res_tiled = run_loop_scgd0(setup([μ0]), occ_for(nlist); η_init, tol, gpu_tile = 7)
            @test norm(res_tiled.Σ - res.Σ) <= 1e-12 * norm(res.Σ)
            @test abs(res_tiled.μ[1] - res.μ[1]) < 1e-12   # μ ~ 1e-4 at half filling: absolute

            # Restart from the converged Σ and μ: one iteration, nothing moves beyond tol.
            res_restart = run_loop_scgd0(setup(res.μ), occ_for(nlist; μlist = res.μ); η_init, tol,
                Σ_init = res.Σ)
            @test size(res_restart.history.err, 2) == 1
            @test res_restart.converged == [true]
            @test maximum(abs, res_restart.Σ - res.Σ) < tol
            @test abs(res_restart.μ[1] - res.μ[1]) < tol

            # Two temperatures in one run: each is the single-temperature run, and each freezes at
            # its own iteration.
            T2 = 0.2
            μ0_T2 = holstein_1d_bare_μ(; nk, t, T = T2, n_target, spin_degeneracy = 2)
            res_T2 = run_loop_scgd0(setup([μ0_T2]), occ_for(nlist; Tlist = [T2]); η_init, tol)
            res_both = run_loop_scgd0(setup([μ0, μ0_T2]), occ_for(nlist; Tlist = [T, T2]);
                η_init, tol)
            for (iT, r) in enumerate((res, res_T2))
                @test norm(res_both.Σ[:, :, iT] - r.Σ[:, :, 1]) <= 1e-12 * norm(r.Σ)
                @test abs(res_both.μ[iT] - r.μ[1]) < 1e-12
            end
            @test res_both.history.iter_frozen == [size(r.history.err, 2) for r in (res, res_T2)]
            @test res_both.converged == [true, true]
        end
    end

    @testset "maxiter without convergence" begin
        res = @test_logs (:warn, r"not converged") run_loop_scgd0(setup([0.0]), occ_for(0.0);
            η_init, tol, maxiter = 2)
        @test res.converged == [false]
        @test res.history.iter_frozen == [0]
        @test size(res.history.err, 2) == 2
    end

    @testset "no chemical potential in the bracket" begin
        # 2.5 electrons per cell do not fit in one spin-degenerate band.
        @test_throws ArgumentError run_loop_scgd0(setup([0.0]), occ_for(1.5); η_init, tol)
    end

    @testset "multiplet average" begin
        # A coupling that depends on the outer state makes Σ_i differ; declaring states 2 and 3 a
        # multiplet replaces both by their mean and leaves the others alone.
        s = setup([0.0])
        s = merge(s, (; g2 = reshape([0.2 * (1 + 0.1i) for i in 1:nk÷2+1, f in 1:nk], 1, :, nk)))
        res = run_loop_scgd0(s, occ_for(0.0); η_init, tol, maxiter = 1, fix_μ = true)
        res_avg = run_loop_scgd0(merge(s, (; multiplets = [[2, 3]])), occ_for(0.0); η_init, tol,
            maxiter = 1, fix_μ = true)
        @test norm(res.Σ[:, 2, 1] - res.Σ[:, 3, 1]) > 1e-3 * norm(res.Σ[:, 2, 1])
        @test res_avg.Σ[:, 2, 1] == res_avg.Σ[:, 3, 1]
        @test res_avg.Σ[:, 2, 1] ≈ (res.Σ[:, 2, 1] + res.Σ[:, 3, 1]) / 2 rtol = 1e-14
        @test res_avg.Σ[:, [1; 4:end], :] == res.Σ[:, [1; 4:end], :]
    end
end
