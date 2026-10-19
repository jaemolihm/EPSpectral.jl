# The grid ωs_spectral of `run_scgd0`: its default, its validation, a graded grid against a fine
# uniform one on the 1D Holstein chain (μ and min_i ∫A_i), and the JLD2 round trip of both grids.
@testset "scGD0 spectral grid" begin
    t, λ, ω0, nk, T = 1.0, 0.5, 0.2, 64, 0.1
    model = EP.holstein_model(; t, λ, ω₀ = ω0, dimension = 1, verbose = false)
    ωs = FrequencyGrid(collect(-4.0:0.02:4.0))
    occ = ElectronOccupationParams(; Tlist = [T], nlist = -0.3, nelec = 1, volume = model.volume,
        spin_degeneracy = 2, type = :Metal)
    # One iteration: μ and min_i ∫A_i are read from the Σ of that iteration, so two runs on
    # different spectral grids have the same Σ and differ only by the quadrature.
    run(; kwargs...) = run_scgd0(model, (nk, 1, 1), (-Inf, Inf); occ, ωs, η_init = 0.05,
        maxiter = 1, verbosity = 0, kwargs...)

    @testset "default is the uniform grid of step ωs.dω_min" begin
        res = run()
        @test collect(res.ωs_spectral) ≈ collect(range(extrema(ωs)...; step = ωs.dω_min)) rtol =
            1e-14
        @test res.ωs_spectral.dω_unit == ωs.dω_min
    end

    @testset "validation" begin
        # A grid on which Σ is flat everywhere is no spectral grid for ωs.
        @test_throws ArgumentError run(ωs_spectral = FrequencyGrid(5.0:0.1:6.0))
        @test_throws ArgumentError run(ωs_spectral = FrequencyGrid(-7.0:0.1:-4.0))
        # Not a FrequencyGrid.
        @test_throws ArgumentError run(ωs_spectral = collect(-4.0:0.1:4.0))
        # A grid cutting off part of ωs would drop spectral weight from the μ solve.
        @test_throws ArgumentError run(ωs_spectral = FrequencyGrid(-1.0:0.01:6.0))
        # A grid covering ωs, extending past it, is accepted.
        @test run(ωs_spectral = FrequencyGrid(-5.0:0.01:5.0)) isa SCGD0Result
    end

    @testset "graded grid against a fine uniform grid" begin
        # Step 0.005 around μ = -0.91 (nlist = -0.3), 0.02 outside, on the span of ωs.
        graded = graded_frequency_grid([(-1.4, -0.4, 0.005), (-4.0, 4.0, 0.02)])
        fine = FrequencyGrid(range(first(graded), last(graded); length = 8001))
        same_count = FrequencyGrid(range(first(graded), last(graded); length = length(graded)))
        res_graded, res_fine = run(ωs_spectral = graded), run(ωs_spectral = fine)
        @test res_graded.Σ == res_fine.Σ
        @test res_graded.ωs_spectral == graded
        @test length(graded) < length(fine) / 10
        # The reference: min_i ∫A_i of the same Σ on a still finer uniform grid, by the trapezoid
        # rule written out here.
        el = res_fine.el_i
        Σ_itp(i) = extrapolate(interpolate((ωs,), res_fine.Σ[:, i, 1], Gridded(Linear())), Flat())
        function integral_A(i, ωa)
            Wω = [(ωa[min(r + 1, end)] - ωa[max(r - 1, 1)]) / 2 for r in eachindex(ωa)]
            sum(-imag(1 / (ω - el.es[i] - Σ_itp(i)(ω))) / π * Wω[r] for (r, ω) in enumerate(ωa))
        end
        ωa_ref = range(first(graded), last(graded); length = 40001)
        minA_ref = minimum(integral_A(i, ωa_ref) for i in 1:el.n)
        @test res_fine.history.min_spectral_weight[1, 1] ≈ minA_ref rtol = 1e-6
        # Measured: 7e-6 relative in ∫A and 2e-6 in μ, against 5e-4 and 4e-5 for a uniform grid
        # of the same number of nodes.
        err_minA(res) = abs(res.history.min_spectral_weight[1, 1] - minA_ref) / minA_ref
        res_same_count = run(ωs_spectral = same_count)
        @test err_minA(res_graded) < 1e-4
        @test abs(res_graded.μ[1] - res_fine.μ[1]) < 1e-5
        @test err_minA(res_same_count) > 10 * err_minA(res_graded)
        @test abs(res_same_count.μ[1] - res_fine.μ[1]) > 10 * abs(res_graded.μ[1] - res_fine.μ[1])
    end

    @testset "save and load" begin
        graded = graded_frequency_grid([(-1.4, -0.4, 0.005), (-4.0, 4.0, 0.02)])
        res = run(ωs_spectral = graded)
        file = save_scgd0(joinpath(mktempdir(), "scgd0.jld2"), res)
        # Each grid is a group keyed by the field names of FrequencyGrid.
        jldopen(file, "r") do f
            for grid in ("ωs", "ωs_spectral")
                @test sort(collect(keys(f[grid]))) == ["dω_unit", "offsets", "ω_origin"]
            end
        end
        back = load_scgd0(file)
        for name in (:ωs, :ωs_spectral), field in (:ω_origin, :dω_unit, :offsets, :dω_min)
            @test isequal(getfield(getfield(back, name), field), getfield(getfield(res, name), field))
        end
    end
end
