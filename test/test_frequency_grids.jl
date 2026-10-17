@testset "Frequency grids" begin
    # Nonuniform output grid (the SrVO3 grid, in eV around E_F) and the default dense grid.
    ωs = collect(vcat(-0.35:0.02:-0.11, -0.099:0.001:0.100, 0.12:0.02:0.34))
    ωs_dense = range(extrema(ωs)...; step = minimum(diff(ωs)))

    @testset "kk_matrix" begin
        K = EPSpectral.kk_matrix(ωs)
        z = pseudorandom(length(ωs); seed = 1)
        # Export smoke test: `kramers_kronig` is defined as `K * z`.
        @test kramers_kronig(ωs, z) == K * z
        # Independent reference on a small nonuniform grid: the PV integral of the interpolant
        # (hats on the nodes, one end cell of decay on each side) as the subtracted integral
        # P∫ h/(y-x) = ∫ (h(y) - h(x))/(y-x) dy + h(x) log|(R-x)/(L-x)|, composite Simpson.
        xg = [-0.3, -0.25, -0.1, 0.0, 0.07, 0.2, 0.3]
        zg = pseudorandom(length(xg); seed = 7)
        xe = vcat(2xg[1] - xg[2], xg, 2xg[end] - xg[end-1])
        ze = vcat(0.0, zg, 0.0)
        h(y) = extrapolate(interpolate((xe,), ze, Gridded(Linear())), Flat())(y)
        function pv(x)
            tot = h(x) * log(abs((xe[end] - x) / (xe[1] - x)))
            for c in 1:length(xe)-1
                m = 4000
                ys = range(xe[c], xe[c+1]; length = m + 1)
                f(y) = isapprox(y, x; atol = 1e-14) ? 0.0 : (h(y) - h(x)) / (y - x)
                # Cell containing x: the integrand is constant, take the cell slope.
                if xe[c] <= x <= xe[c+1]
                    sl = (ze[c+1] - ze[c]) / (xe[c+1] - xe[c])
                    tot += sl * (xe[c+1] - xe[c])
                else
                    dy = step(ys)
                    tot += dy / 3 * (f(ys[1]) + f(ys[end]) +
                                     4 * sum(f, ys[2:2:end-1]) + 2 * sum(f, ys[3:2:end-2]))
                end
            end
            tot / π
        end
        y_ref = pv.(xg)
        @test norm(EPSpectral.kk_matrix(xg) * zg - y_ref) <= 1e-8 * norm(y_ref)
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

    @testset "_interp_linear_uniform" begin
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
            err = maximum(abs(EPSpectral._interp_linear_uniform(Σin, col, x, ωd0, dωd) - itp(x)) for x in xs)
            @test err <= 5e-14 * maximum(abs, Σin)
        end
    end
end
