@testset "scGD0 linear maps" begin
    # Nonuniform output grid (the SrVO3 grid, in eV around E_F) and the default dense grid.
    ωs = collect(vcat(-0.35:0.02:-0.11, -0.099:0.001:0.100, 0.12:0.02:0.34))
    ωs_dense = range(extrema(ωs)...; step = minimum(diff(ωs)))

    @testset "kk_matrix" begin
        K = EPSpectral.kk_matrix(ωs)
        z = pseudorandom(length(ωs); seed = 1)
        y = kramers_kronig(ωs, z)
        @test norm(K * z - y) <= 1e-13 * norm(y)
    end

    @testset "interp_matrix" begin
        # Targets beyond both ends of ωs cover the flat extrapolation.
        xs = range(ωs[1] - 0.05, ωs[end] + 0.05; length = 1111)
        y = pseudorandom_complex(length(ωs); seed = 2)
        P = EPSpectral.interp_matrix(ωs, xs)
        @test eltype(P) == ComplexF64
        y_ref = extrapolate(interpolate((ωs,), y, Gridded(Linear())), Flat()).(xs)
        @test norm(P * y - y_ref) <= 1e-13 * norm(y_ref)
        @test size(EPSpectral.interp_matrix(ωs, ωs_dense)) == (length(ωs_dense), length(ωs))
    end

    @testset "_lerp_flat" begin
        # The dense-grid lookup of the Fan-Migdal kernel, against Interpolations' uniform-grid
        # linear interpolation with flat extrapolation, inside and outside the grid. The two round
        # the grid position (x - ωd0)/dωd differently (Interpolations in the range's
        # TwicePrecision arithmetic), which on this data with O(1) jumps between neighbouring
        # points gives 2.4e-14.
        Σin = pseudorandom_complex(length(ωs_dense), 3; seed = 4)
        ωd0, dωd = first(ωs_dense), step(ωs_dense)
        xs = vcat(range(ωs[1] - 0.05, ωs[end] + 0.05; length = 997), collect(ωs_dense))
        for col in axes(Σin, 2)
            itp = extrapolate(scale(interpolate(Σin[:, col], BSpline(Linear())), ωs_dense), Flat())
            err = maximum(abs(EPSpectral._lerp_flat(Σin, col, x, ωd0, dωd) - itp(x)) for x in xs)
            @test err <= 5e-14 * maximum(abs, Σin)
        end
    end
end
