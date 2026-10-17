# The ladder on the 1D Holstein chain (constant g2 and ω0, ωs step below ω0): the vertex correction
# of the current vanishes identically, and the density vertex measures the Ward identity.

@testset "ladder 1D Holstein" begin
    @testset "current vertex correction vanishes" begin
        case = holstein_ladder_case()
        (; ladder, res, model, nk, window, occ) = case
        # Σ_f J_f = 0 for J_{-k} = -J_k, so L(J0) = 0 and σ_ladder = σ_bubble.
        for iT in 1:2, Ω in (0.0, 0.2)
            Gm, Gp = ladder_propagators(case, iT, Ω)
            J0 = zeros(ComplexF64, length(case.ωs), 4, 3, 2, res.el_i.n)
            EPSpectral.attach_greens_to_bare_vertex!(J0, Gm, ladder.v, Gp)
            LJ = apply_ladder_case(case, J0, iT, Ω)
            @test maximum(abs, LJ) <= 1e-12 * maximum(abs, J0)
            # A rotation without the sign of the time-reversed partners (the inner state -k
            # reached with +1 instead of -1) does not cancel.
            wrong = (; ladder..., R = abs.(ladder.R))
            workspace = EPSpectral._ladder_workspace(wrong, EP.CPUBackend(), nothing, 0)
            LJ_wrong = EPSpectral.apply_ladder_kernel!(similar(J0), J0, wrong, workspace, Gm, Gp, res.Tlist[iT];
                ω_acoustic = EP.omega_acoustic)
            @test maximum(abs, LJ_wrong) > 1e-3 * maximum(abs, J0)
        end
        out = run_scgd0_ladder(model, nk, window, res; occ, Ωs = [0.0, 0.2], verbosity = 0)
        @test norm(out.σ_ladder - out.σ_bubble) <= 1e-12 * norm(out.σ_bubble)
        @test all(out.converged)
    end

    @testset "Ward identity of the density vertex" begin
        # No symmetry, so the x slot of the vertex is not rotated. V = 1 in the x slot is the
        # density vertex at q = 0; for Ω ≠ 0 the conserving ladder gives χ^K_ρρ = 0, so the ratio
        # of the dressed to the bare response measures the discretization error of the kernel.
        case = holstein_ladder_case(; symmetry = nothing)
        (; ladder, res) = case
        v_density = zeros(3, res.el_i.n)
        v_density[1, :] .= 1
        Ωs = [0.1, 0.3]
        out = EPSpectral._loop_ladder((; ladder..., v = v_density), res.Σ, res.μ, res.Tlist, Ωs;
            solver = :gmres, rtol = 1e-8, maxiter = 200, store_vertex = false,
            backend = EP.CPUBackend(), gpu_tile = nothing, ω_acoustic = EP.omega_acoustic,
            verbosity = 0)
        ratio = out.σ_ladder[1, 1, :, :] ./ out.σ_bubble[1, 1, :, :]
        @info "Ward residual |χ^K_ladder / χ^K_bubble| (rows Ω, columns T)" Ωs res.Tlist ratio
        @test all(out.converged)
        @test all(isfinite, ratio) && all(!iszero, out.σ_bubble[1, 1, :, :])
    end
end
