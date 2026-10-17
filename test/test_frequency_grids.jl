@testset "Frequency grids" begin
    # Nonuniform output grid (the SrVO3 grid, in eV around E_F) and the default dense grid.
    ωs = collect(vcat(-0.35:0.02:-0.11, -0.099:0.001:0.100, 0.12:0.02:0.34))
    ωs_dense = range(extrema(ωs)...; step = minimum(diff(ωs)))

    @testset "kramers_kronig_matrix" begin
        K = EPSpectral.kramers_kronig_matrix(ωs)
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
        @test norm(EPSpectral.kramers_kronig_matrix(xg) * zg - y_ref) <= 1e-8 * norm(y_ref)
    end

    # Targets beyond both ends of ωs, exactly on its nodes, and on the dense grid.
    n_out = 997
    xs = vcat(range(ωs[1] - 0.05, ωs[end] + 0.05; length = n_out), ωs, collect(ωs_dense))
    itp(y) = extrapolate(interpolate((ωs,), y, Gridded(Linear())), Flat())

    @testset "linear_interpolation_table, apply_interpolation_table!" begin
        y = pseudorandom_complex(length(ωs), 3; seed = 2)
        j, t = EPSpectral.linear_interpolation_table(ωs, xs)
        @test j isa Vector{Int32} && t isa Vector{Float64}
        @test all(in(1:length(ωs) - 1), j) && all(x -> 0 <= x <= 1, t)
        Y = zeros(ComplexF64, length(xs), 3)
        @test EPSpectral.apply_interpolation_table!(Y, y, j, t) === Y
        Y_ref = stack(itp(y[:, i]).(xs) for i in 1:3)
        @test norm(Y - Y_ref) <= 1e-13 * norm(Y_ref)
        # Outside the grid the table returns the end values exactly.
        @test Y[1, :] == y[1, :] && Y[n_out, :] == y[end, :]
    end

    @testset "_interp_linear_grid" begin
        # The Fan-Migdal lookup against Interpolations' `Gridded(Linear())` with `Flat()`, on the
        # nonuniform grid, inside, on the nodes and outside, and at a NaN (the first value).
        Σin = pseudorandom_complex(length(ωs), 3; seed = 4)
        for col in axes(Σin, 2)
            f = itp(Σin[:, col])
            err = maximum(abs(EPSpectral._interp_linear_grid(Σin, col, x, ωs) - f(x)) for x in xs)
            @test err <= 1e-13 * maximum(abs, Σin)
            @test EPSpectral._interp_linear_grid(Σin, col, NaN, ωs) == Σin[1, col]
        end
    end
end
