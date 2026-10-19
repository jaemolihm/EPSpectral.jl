# The ladder on the pb window (multiple modes, IBZ with the TR-odd unfolding): the two solvers
# save/load and the stored vertex, and the argument checks of `run_scgd0_ladder`.

@testitem "ladder pb" setup = [Common] begin
    eV = unit_to_aru(:eV)
    pb = pb_ladder_case()
    (; model, nk, window, occ, res) = pb
    Ωs = [0.0, 0.1 * eV]
    run(; kwargs...) = run_scgd0_ladder(model, nk, window, res; occ, Ωs, verbosity = 0, kwargs...)

    @testset "solvers agree" begin
        # Ω > 0 only: the fixed point diverges at the DC point of the 300 K entry of this
        # fixture (the spectral radius of L exceeds 1 there), where GMRES converges.
        rtol = 1e-6
        gmres = run(; Ωs = Ωs[2:2], solver = :gmres, rtol, maxiter = 60)
        fixed = run(; Ωs = Ωs[2:2], solver = :fixed_point, rtol, maxiter = 200)
        @test all(gmres.converged) && all(fixed.converged)
        @test norm(gmres.σ_ladder - fixed.σ_ladder) <= 10 * rtol * norm(gmres.σ_ladder)
        # The correction is not negligible, so the agreement is not a trivial one.
        @test norm(gmres.σ_ladder - gmres.σ_bubble) > 1e-2 * norm(gmres.σ_bubble)
        @info "pb ladder/bubble" ratio = gmres.σ_ladder[1, 1, :, :] ./ gmres.σ_bubble[1, 1, :, :] niter_gmres = gmres.niter niter_fixed = fixed.niter
    end

    @testset "save, load and the stored vertex" begin
        mktempdir() do dir
            for store_vertex in (false, true)
                out = run(; solver = :gmres, rtol = 1e-8, maxiter = 60, store_vertex)
                file = save_scgd0_ladder(joinpath(dir, "ladder_$store_vertex.jld2"), out)
                back = load_scgd0_ladder(file)
                for name in fieldnames(SCGD0LadderResult)
                    @test getfield(back, name) == getfield(out, name)
                end
                @test (out.J === nothing) == !store_vertex
            end
        end

        # One fixed-point step from J0 stores J0 + L(J0), formed here from the definitions.
        out = @test_logs (:warn, r"not converged") run(; solver = :fixed_point, maxiter = 1,
            store_vertex = true)
        @test size(out.J) == (length(res.ωs), 4, 3, 2, res.el_i.n, length(Ωs), 2)
        for iT in 1:2, (iΩ, Ω) in enumerate(Ωs)
            Gm, Gp = ladder_propagators(pb, iT, Ω)
            J0 = zeros(ComplexF64, length(res.ωs), 4, 3, 2, res.el_i.n)
            EPSpectral.attach_greens_to_bare_vertex!(J0, Gm, pb.ladder.v, Gp)
            J1 = J0 .+ apply_ladder_case(pb, J0, iT, Ω)
            @test norm(out.J[:, :, :, :, :, iΩ, iT] - J1) <= 1e-12 * norm(J1)
        end
        # A converged solve satisfies the equation: J = J0 + L(J) to the solver tolerance.
        rtol = 1e-8
        out = run(; solver = :gmres, rtol, maxiter = 60, store_vertex = true)
        for iT in 1:2, (iΩ, Ω) in enumerate(Ωs)
            Gm, Gp = ladder_propagators(pb, iT, Ω)
            J0 = zeros(ComplexF64, length(res.ωs), 4, 3, 2, res.el_i.n)
            EPSpectral.attach_greens_to_bare_vertex!(J0, Gm, pb.ladder.v, Gp)
            J = out.J[:, :, :, :, :, iΩ, iT]
            @test norm(J - J0 - apply_ladder_case(pb, J, iT, Ω)) <= 10 * rtol * norm(J0)
        end
    end

    @testset "argument checks" begin
        @test_throws ArgumentError run(; solver = :newton)
        @test_throws ArgumentError run(; maxiter = 0)
        @test_throws ArgumentError run(; gpu_tile = 0)
        @test_throws ArgumentError run_scgd0_ladder(model, nk, window, res; occ, Ωs = Float64[],
            verbosity = 0)
        occ_other = ElectronOccupationParams(; Tlist = occ.Tlist[1:1], nlist = 0.0, nelec = 4,
            volume = model.volume, spin_degeneracy = 2, type = :Metal)
        @test_throws ArgumentError run_scgd0_ladder(model, nk, window, res; occ = occ_other,
            verbosity = 0)
        # The shift-grid coupling build has no row for a negative phonon frequency.
        @test_throws ArgumentError run(; ω_acoustic = -1.0)
        # A different window gives different states than scgd0.el_i.
        @test_throws ArgumentError run_scgd0_ladder(model, nk, (window[1], window[2] - 0.2 * eV),
            res; occ, verbosity = 0)
    end
end
