# `_loop_scgd0` on hand-built 1D Holstein input (see `holstein_1d_loop_setup`), for what a run of
# `run_scgd0` on the Holstein `Model` cannot reach: an unconverged maxiter, a μ with no root, and a
# multiplet average that changes Σ (the Holstein Σ is k-independent).
@testset "scGD0 loop, 1D Holstein by hand" begin
    nk, t, ω0, g2, T = 64, 1.0, 0.2, 0.2, 0.1
    ωs = collect(-4.0:0.02:4.0)
    η_init, tol = 0.05, 1e-6
    occ_for(nlist) = ElectronOccupationParams(; Tlist = [T], nlist, nelec = 1, volume = 1.0,
        spin_degeneracy = 2, type = :Metal)
    setup(μ_start) = holstein_1d_loop_setup(; nk, t, ω0, g2, ωs, μ_start)

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
        g2_varied = reshape([0.2 * (1 + 0.1i) for i in 1:nk÷2+1, f in 1:nk], 1, :, nk)
        setup_varied = merge(setup([0.0]), (; g2 = g2_varied))
        res = run_loop_scgd0(setup_varied, occ_for(0.0); η_init, tol, maxiter = 1, fix_μ = true)
        res_avg = run_loop_scgd0(merge(setup_varied, (; multiplets = [[2, 3]])), occ_for(0.0);
            η_init, tol, maxiter = 1, fix_μ = true)
        @test norm(res.Σ[:, 2, 1] - res.Σ[:, 3, 1]) > 1e-3 * norm(res.Σ[:, 2, 1])
        @test res_avg.Σ[:, 2, 1] == res_avg.Σ[:, 3, 1]
        @test res_avg.Σ[:, 2, 1] ≈ (res.Σ[:, 2, 1] + res.Σ[:, 3, 1]) / 2 rtol = 1e-14
        @test res_avg.Σ[:, [1; 4:end], :] == res.Σ[:, [1; 4:end], :]
    end
end
