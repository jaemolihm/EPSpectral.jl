@testset "Frequency grids" begin
    # Nonuniform output grid (the SrVO3 grid, in eV around E_F) and the default spectral grid.
    ωs = collect(vcat(-0.35:0.02:-0.11, -0.099:0.001:0.100, 0.12:0.02:0.34))
    grid = FrequencyGrid(ωs)
    ωs_spectral = range(extrema(ωs)...; step = grid.dω_min)

    @testset "FrequencyGrid" begin
        # The structure: nodes ω_origin + offsets * dω_unit, read-only, the smallest spacing stored.
        g = FrequencyGrid(0.5, 0.25, [-2, 0, 1, 5])
        @test g isa AbstractVector{Float64} && size(g) == (4,)
        @test g == [0.0, 0.5, 0.75, 1.75] && g.dω_min == 0.25
        @test collect(g) == [g[j] for j in 1:4] && g[2:3] == [0.5, 0.75] && g[end] == 1.75
        offsets = [0, 1, 3]
        g = FrequencyGrid(0.0, 0.5, offsets)
        offsets[2] = 7
        @test g.offsets == [0, 1, 3]   # a copy
        @test_throws ArgumentError FrequencyGrid(0.0, 0.5, [0])
        @test_throws ArgumentError FrequencyGrid(0.0, 0.5, [0, 2, 2])
        @test_throws ArgumentError FrequencyGrid(0.0, 0.5, [0, 3, 2])
        @test_throws ArgumentError FrequencyGrid(0.0, 0.0, [0, 1])
        @test_throws ArgumentError FrequencyGrid(0.0, -1.0, [0, 1])
        # Non-finite origin or step.
        @test_throws ArgumentError FrequencyGrid(NaN, 0.5, [0, 1])
        @test_throws ArgumentError FrequencyGrid(Inf, 0.5, [0, 1])
        @test_throws ArgumentError FrequencyGrid(0.0, Inf, [0, 1])

        # The conversion of a vector of nodes: the largest commensurate step minimum(diff) / k.
        # k = 1 (uniform, and graded with steps that are multiples of the minimum) ...
        g = FrequencyGrid(collect(0.0:0.02:1.0))
        @test g.dω_unit ≈ 0.02 rtol = 1e-14
        @test g.ω_origin == 0.0 && g.offsets == 0:50
        @test FrequencyGrid(ωs).dω_unit ≈ 0.001 rtol = 1e-12
        @test FrequencyGrid(ωs).dω_min ≈ 0.001 rtol = 1e-12
        @test collect(FrequencyGrid(ωs)) ≈ ωs rtol = 1e-14
        # ... and k = 2, 0.05 and 0.02 steps: the commensurate step is 0.01, the minimum step 0.02.
        g = FrequencyGrid(vcat(0.0:0.05:0.5, 0.52:0.02:0.8))
        @test g.dω_unit ≈ 0.01 && g.dω_min ≈ 0.02
        @test FrequencyGrid(vcat(-0.5:0.05:-0.1, -0.09:0.01:0.09, 0.1:0.05:0.5)).dω_unit ≈ 0.01
        # A fine step far from zero: the smallest gap carries the accumulated round-off of the
        # nodes; the step is taken from the span (0.1 meV nodes around 11.83 eV in Ry).
        eV = 1 / 13.605693122994
        center = 11.83 * eV
        for g in (FrequencyGrid(collect(center .+ (-2eV:1e-4eV:2eV))),
                graded_frequency_grid([(center - 0.1eV, center + 0.1eV, 1e-4eV),
                                       (center - 1eV, center + 1eV, 2e-3eV)]),
                FrequencyGrid(collect(graded_frequency_grid([(center - 0.1eV, center + 0.1eV, 1e-4eV),
                                       (center - 1eV, center + 1eV, 2e-3eV)]))))
            @test g.dω_unit ≈ 1e-4eV rtol = 1e-12
        end

        # The tolerance, in units of the step: a node 1e-9 steps off is accepted and lies on the
        # grid ω_origin + p dω_unit to that tolerance, 1e-7 steps off is refused, naming the node.
        nodes = collect(0.0:0.1:1.0)
        nodes_9 = copy(nodes); nodes_9[6] += 1e-9 * 0.1
        g = FrequencyGrid(nodes_9)
        @test g.offsets == 0:10 && abs(g[6] - nodes_9[6]) <= 2e-8 * g.dω_unit
        nodes_7 = copy(nodes); nodes_7[6] += 1e-7 * 0.1
        err = try; FrequencyGrid(nodes_7); catch e; e; end
        @test err isa ArgumentError && occursin("ωs[6] = $(nodes_7[6])", err.msg) &&
            occursin("not commensurate", err.msg)
        @test FrequencyGrid(nodes_7; atol = 1e-6).offsets == 0:10
        # No commensurate step with k <= 64.
        ωs_none = [0.0, 1.0, 1.0 + sqrt(2) / 100]
        err = try; FrequencyGrid(ωs_none); catch e; e; end
        @test err isa ArgumentError && occursin("not commensurate", err.msg) &&
            occursin(r"the node ωs\[\d\] = [\d.]+ is [\d.e-]+ steps off", err.msg) &&
            occursin("graded_frequency_grid", err.msg)
        # Duplicated, unsorted or too few nodes, a 1-ulp pair and a 1e-12 pair.
        for bad in ([0.0, 0.5, 0.5, 1.0], [0.0, 1.0, 0.5], [0.0, 0.5, nextfloat(0.5), 1.0],
                    [0.0, 1e-12, 0.5], [0.0, NaN, 1.0])
            err = try; FrequencyGrid(bad); catch e; e; end
            @test err isa ArgumentError && occursin("apart", err.msg)
        end
        @test_throws ArgumentError FrequencyGrid([0.0])
    end

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
    xs = vcat(range(ωs[1] - 0.05, ωs[end] + 0.05; length = n_out), ωs, collect(ωs_spectral))
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
            cell_table = EPSpectral.cell_search_table(EP.CPUBackend(), FrequencyGrid(ωs))
            all(isequal(lookup_bisection(Σin, col, x, ωs),
                        EPSpectral._interp_linear_grid(Σin, col, x, ωs, cell_table))
                for col in axes(Σin, 2), x in vcat(xs, nextfloat.(ωs), prevfloat.(ωs), NaN))
        end
        @test same_as_bisection(ωs, xs)
        # A grid with a 500x finer region.
        ωg = vcat(-1.0:0.5:-0.5, -0.499:1e-3:-0.4, 0.1:0.5:1.0)
        @test same_as_bisection(ωg, range(-1.2, 1.2; length = 20001))
        # The table has one entry per bucket, so its size follows the finest cell.
        @test length(EPSpectral.cell_search_table(EP.CPUBackend(), FrequencyGrid(ωg)).jtab) <=
            (ωg[end] - ωg[1]) / (0.999e-3) + 2
        # A near-coincident pair of nodes would need a huge table: refused, not allocated.
        @test_throws ArgumentError EPSpectral.cell_search_table(EP.CPUBackend(),
            FrequencyGrid(0.0, 1e-9, [0, 10^9, 10^9 + 1, 2 * 10^9]))
        # Against Interpolations on the nonuniform grid.
        Σin = pseudorandom_complex(length(ωs), 3; seed = 4)
        cell_table = EPSpectral.cell_search_table(EP.CPUBackend(), grid)
        for col in axes(Σin, 2)
            f = itp(Σin[:, col])
            err = maximum(abs(EPSpectral._interp_linear_grid(Σin, col, x, ωs, cell_table) - f(x)) for x in xs)
            @test err <= 1e-13 * maximum(abs, Σin)
        end
    end

    @testset "graded_frequency_grid" begin
        # 1 meV cells within 0.1 eV and 20 meV cells to 0.34 eV (the SrVO3 grid), written as nested
        # windows about 0.5; the origin is the centre of the first window.
        grid = graded_frequency_grid([(0.4, 0.6, 0.001), (0.15, 0.85, 0.02)])
        @test grid isa FrequencyGrid && issorted(grid) && grid[1] ≈ 0.5 - 0.34 && grid[end] ≈ 0.5 + 0.34
        @test count(x -> 0.4 <= x <= 0.6 + 1e-12, grid) == 201 && length(grid) == 201 + 24
        @test minimum(diff(grid)) ≈ 0.001 && maximum(diff(grid)) ≈ 0.02
        # The origin is the centre of the finest window and the step its step, with integer offsets.
        @test grid.ω_origin == 0.5 && grid.dω_unit == 0.001 && grid.dω_min == 0.001 &&
            grid.offsets[1] == -340 && grid.offsets[end] == 340
        # Exact node lists: window l continues outward from the last node so far in its own steps.
        @test graded_frequency_grid([(-2.0, 2.0, 1.0), (-5.0, 5.0, 2.0), (-15.0, 15.0, 5.0)]) ==
            [-14.0, -9.0, -4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0, 9.0, 14.0]
        # Asymmetric windows about the origin 1, steps 1, 2, 5: nodes 1 + p with p in -2:2, then
        # steps of 2 out to -6 and 8, then steps of 5: 13 on the right; -11 on the left would leave [-9, 16].
        @test graded_frequency_grid([(-1.0, 3.0, 1.0), (-5.0, 9.0, 2.0), (-9.0, 16.0, 5.0)]) ==
            [-5.0, -3.0, -1.0, 0.0, 1.0, 2.0, 3.0, 5.0, 7.0, 9.0, 14.0]
        # Every cell is at most the step of the innermost window holding both of its ends, and all nodes
        # lie inside the outermost window.
        for windows_l in ([(-2.0, 2.0, 1.0), (-5.0, 5.0, 2.0), (-15.0, 15.0, 5.0)],
                [(-1.0, 3.0, 1.0), (-5.0, 9.0, 2.0), (-9.0, 16.0, 5.0)],
                [(0.4, 0.6, 0.001), (0.15, 0.85, 0.02)])
            grid_l = graded_frequency_grid(windows_l)
            @test windows_l[end][1] - 1e-12 <= grid_l[1] && grid_l[end] <= windows_l[end][2] + 1e-12
            for (j, cell) in enumerate(diff(grid_l))
                dw_inner = minimum(dw for (wmin, wmax, dw) in windows_l
                                   if wmin - 1e-12 <= grid_l[j] && grid_l[j + 1] <= wmax + 1e-12)
                @test cell <= dw_inner + 1e-12
            end
        end
        windows = [(0.0, 0.1, 0.01), (-0.2, 1.0, 0.03)]
        grid3 = graded_frequency_grid(windows)
        @test all(p -> abs(p - round(p)) < 1e-8, (grid3 .- 0.05) ./ 0.01)
        @test issorted(grid3) && allunique(round.(grid3; digits = 9))
        @test grid3[1] >= -0.2 - 1e-12 && grid3[end] <= 1.0 + 1e-12 && 1.0 - grid3[end] < 0.03
        @test grid3.dω_unit == 0.01 && grid3.ω_origin == 0.05
        # The vector helper recovers the same step and offsets from the nodes of a graded grid; its origin
        # is the first node, so its offsets differ from the centred ones by a constant.
        for g in (grid, grid3, graded_frequency_grid([(-1.0, 3.0, 1.0), (-5.0, 9.0, 2.0), (-9.0, 16.0, 5.0)]),
                graded_frequency_grid([(-2.0, 2.0, 1.0), (-5.0, 5.0, 2.0), (-15.0, 15.0, 5.0)]))
            g_vec = FrequencyGrid(collect(g))
            @test g_vec.dω_unit ≈ g.dω_unit && diff(g_vec.offsets) == diff(g.offsets) && g_vec ≈ g
        end
        # A single window is the uniform grid.
        @test graded_frequency_grid([(-1.0, 1.0, 0.25)]) ≈ -1.0:0.25:1.0
        # A single window of odd width in steps has no node at its ends: the origin is its centre.
        @test graded_frequency_grid([(0.0, 1.0, 0.2)]) ≈ 0.1:0.2:0.9
        # A finest window with fewer than two nodes (here one: 0.1 / 0.2 holds only the origin).
        @test_throws ArgumentError graded_frequency_grid([(0.0, 0.1, 0.2)])
        @test_throws ArgumentError graded_frequency_grid([(0.0, 0.1, 0.2), (-1.0, 1.0, 0.4)])
        # A non-integer ratio, windows that are not nested, a bad window, an empty list.
        @test_throws ArgumentError graded_frequency_grid([(-1.0, 1.0, 0.01), (-3.0, 3.0, 0.025)])
        @test_throws ArgumentError graded_frequency_grid([(-1.0, 1.0, 0.01), (-0.5, 3.0, 0.02)])
        @test_throws ArgumentError graded_frequency_grid([(-1.0, 1.0, 0.01), (-3.0, 0.5, 0.02)])
        @test_throws ArgumentError graded_frequency_grid([(1.0, -1.0, 0.01)])
        @test_throws ArgumentError graded_frequency_grid([(-1.0, 1.0, 0.0)])
        @test_throws ArgumentError graded_frequency_grid([])
    end
end
