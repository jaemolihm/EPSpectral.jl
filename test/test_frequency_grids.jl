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

    @testset "cell_search_table, _interp_linear_grid" begin
        # The cell search finds the cell `searchsortedlast` does, so the lookup is bitwise the one
        # with a bisection, on the nodes and their neighbouring floats, outside both ends, at the
        # last node and at a NaN (the first value).
        function lookup_bisection(Σin, col, x, ωs)
            n = length(ωs)
            !(x > ωs[1]) && return Σin[1, col]
            x >= ωs[n] && return Σin[n, col]
            j = searchsortedlast(ωs, x)
            t = (x - ωs[j]) / (ωs[j+1] - ωs[j])
            (1 - t) * Σin[j, col] + t * Σin[j+1, col]
        end
        function same_as_bisection(ωs, xs)
            Σin = pseudorandom_complex(length(ωs), 2; seed = 4)
            cell_table = EPSpectral.cell_search_table(EP.CPUBackend(), ωs)
            all(isequal(lookup_bisection(Σin, col, x, ωs),
                        EPSpectral._interp_linear_grid(Σin, col, x, ωs, cell_table))
                for col in axes(Σin, 2), x in vcat(xs, nextfloat.(ωs), prevfloat.(ωs), NaN))
        end
        @test same_as_bisection(ωs, xs)
        # A grid with a 500x finer region.
        ωg = vcat(-1.0:0.5:-0.5, -0.499:1e-3:-0.4, 0.1:0.5:1.0)
        @test same_as_bisection(ωg, range(-1.2, 1.2; length = 20001))
        # The table has one entry per bucket, so its size follows the finest cell.
        @test length(EPSpectral.cell_search_table(EP.CPUBackend(), ωg).jtab) <= (ωg[end] - ωg[1]) / (0.999e-3) + 2
        # A near-coincident pair of nodes would need a huge table: refused, not allocated.
        @test_throws ArgumentError EPSpectral.cell_search_table(EP.CPUBackend(), [0.0, 1.0, 1.0 + 1e-9, 2.0])
        # Against Interpolations on the nonuniform grid.
        Σin = pseudorandom_complex(length(ωs), 3; seed = 4)
        cell_table = EPSpectral.cell_search_table(EP.CPUBackend(), ωs)
        for col in axes(Σin, 2)
            f = itp(Σin[:, col])
            err = maximum(abs(EPSpectral._interp_linear_grid(Σin, col, x, ωs, cell_table) - f(x)) for x in xs)
            @test err <= 1e-13 * maximum(abs, Σin)
        end
    end

    @testset "graded_frequency_grid" begin
        # 1 meV cells within 0.1 eV and 20 meV cells to 0.34 eV (the SrVO3 grid), written as nested
        # windows about 0.5; the lattice is anchored at the lower end of the first window, 0.4.
        grid = graded_frequency_grid([(0.4, 0.6, 0.001), (0.15, 0.85, 0.02)])
        @test issorted(grid) && grid[1] ≈ 0.4 - 0.24 && grid[end] ≈ 0.4 + 0.02 * 22
        @test count(x -> 0.4 <= x <= 0.6 + 1e-12, grid) == 201
        @test minimum(diff(grid)) ≈ 0.001 && maximum(diff(grid)) ≈ 0.02
        # The grid is commensurate with the finest step by construction, so the ladder derives it.
        @test EPSpectral.ladder_shift_step(grid, nothing) ≈ 0.001
        # No symmetry is imposed: asymmetric windows give an asymmetric grid, on the lattice 0.1 + 0.01 ℤ
        # (the outer window has 3x and 9x steps, and its ends are not multiples of them).
        windows = [(0.1, 0.15, 0.01), (0.05, 0.4, 0.03), (-0.2, 1.0, 0.09)]
        grid3 = graded_frequency_grid(windows)
        @test all(p -> abs(p - round(p)) < 1e-8, (grid3 .- 0.1) ./ 0.01)
        @test issorted(grid3) && allunique(round.(grid3; digits = 9))
        @test all(w -> any(≈(w), grid3), (0.1, 0.15, 0.13))
        @test grid3[1] >= -0.2 - 1e-12 && grid3[end] <= 1.0 + 1e-12 && 1.0 - grid3[end] < 0.09
        @test EPSpectral.ladder_shift_step(grid3, nothing) ≈ 0.01
        # A single window is the uniform grid.
        @test graded_frequency_grid([(-1.0, 1.0, 0.25)]) ≈ -1.0:0.25:1.0
        # A non-integer ratio, windows that are not nested, a bad window, an empty list.
        @test_throws ArgumentError graded_frequency_grid([(-1.0, 1.0, 0.01), (-3.0, 3.0, 0.025)])
        @test_throws ArgumentError graded_frequency_grid([(-1.0, 1.0, 0.01), (-0.5, 3.0, 0.02)])
        @test_throws ArgumentError graded_frequency_grid([(-1.0, 1.0, 0.01), (-3.0, 0.5, 0.02)])
        @test_throws ArgumentError graded_frequency_grid([(1.0, -1.0, 0.01)])
        @test_throws ArgumentError graded_frequency_grid([(-1.0, 1.0, 0.0)])
        @test_throws ArgumentError graded_frequency_grid([])
    end
end
