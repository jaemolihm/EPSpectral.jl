# The ladder kernel on the host: the commensurability check, the shift tables, and one application
# of the kernel `apply_ladder_kernel!` against a naive loop over (i, f, ν, j) that evaluates the shifted
# vertex by exact hat-basis interpolation at ωs[j] ± ω_q, with no shift grid.

# Linear interpolation of y on the sorted grid ωs, zero outside.
function hat_value(y, ωs, x)
    (x < ωs[1] || x > ωs[end]) && return zero(y[1])
    k = min(searchsortedlast(ωs, x), length(ωs) - 1)
    t = (x - ωs[k]) / (ωs[k+1] - ωs[k])
    (1 - t) * y[k] + t * y[k+1]
end

# L(J) = -Gm (s ⊙ K[J]) Gp by loops, with the channel sums of `apply_phonon_exchange_kernel!`.
function naive_apply_ladder(ladder, J, Gm, Gp, T, ω_acoustic)
    (; g2, ωph, iq_kk, iks_i, iks_f, w_f, f_to_i, R, ωs, K) = ladder
    nm, ni, nf = size(g2)
    nω = length(ωs)
    ωq = zeros(nm, nf, ni)
    EP.gather_pair_table!(ωq, ωph, transpose(iq_kk), iks_f, iks_i)
    # The rotated vertex of every inner state.
    F = zeros(ComplexF64, nω, 4, 3, 2, nf)
    for f in 1:nf, c0 in 1:2, a in 1:3, c12 in 1:4
        F[:, c12, a, c0, f] = sum(R[a, b, f] * J[:, c12, b, c0, f_to_i[f]] for b in 1:3)
    end
    Up, Um, Vp, Vm = (zeros(ComplexF64, nω, 4, 3, 2, ni) for _ in 1:4)
    for i in 1:ni, f in 1:nf, ν in 1:nm
        ωq[ν, f, i] < ω_acoustic && continue
        w = w_f[f] * g2[ν, i, f]
        n = EPSpectral.occ_boson(ωq[ν, f, i], T)
        for c0 in 1:2, a in 1:3, c12 in 1:4, j in 1:nω
            y = view(F, :, c12, a, c0, f)
            up = w * hat_value(y, ωs, ωs[j] + ωq[ν, f, i])
            um = w * hat_value(y, ωs, ωs[j] - ωq[ν, f, i])
            Up[j, c12, a, c0, i] += up
            Um[j, c12, a, c0, i] += um
            Vp[j, c12, a, c0, i] += n * up
            Vm[j, c12, a, c0, i] += n * um
        end
    end
    KY = reshape(K * reshape((Up .- Um) ./ 2, nω, :), size(Up))
    LJ = zeros(ComplexF64, nω, 4, 3, 2, ni)
    s = [1 -1; -1 1]
    for i in 1:ni, c0 in 1:2, a in 1:3, j in 1:nω
        k11 = -im * KY[j, 1, a, c0, i] - (Up[j, 1, a, c0, i] + Um[j, 1, a, c0, i]) / 2 -
              (Vp[j, 1, a, c0, i] + Vm[j, 1, a, c0, i])
        k22 = im * KY[j, 4, a, c0, i] - (Up[j, 4, a, c0, i] + Um[j, 4, a, c0, i]) / 2 -
              (Vp[j, 4, a, c0, i] + Vm[j, 4, a, c0, i])
        k21 = -(Vp[j, 2, a, c0, i] + Um[j, 2, a, c0, i] + Vm[j, 2, a, c0, i])
        k12 = -(Up[j, 3, a, c0, i] + Vp[j, 3, a, c0, i] + Vm[j, 3, a, c0, i])
        Kab = s .* [k11 k12; k21 k22]
        LJ[j, :, a, c0, i] = -vec(reshape(Gm[j, :, i], 2, 2) * Kab * reshape(Gp[j, :, i], 2, 2))
    end
    LJ
end

@testset "ladder kernel" begin
    @testset "commensurability check" begin
        ωs = vcat(0.0:0.05:0.5, 0.52:0.02:0.8)
        @test EPSpectral.ladder_shift_step(ωs, 0.01) == 0.01
        @test EPSpectral.ladder_shift_step(ωs, 0.005) == 0.005
        # A step that does not divide the grid: the error names the compatible step.
        err = try; EPSpectral.ladder_shift_step(ωs, 0.03); catch e; e; end
        @test err isa ArgumentError
        @test occursin("dω_shift = 0.01", err.msg) && occursin("factor 3", err.msg)
        # A grid with no compatible step.
        ωs_none = [0.0, 1.0, 1.0 + sqrt(2) / 100]
        err = try; EPSpectral.ladder_shift_step(ωs_none, 1.0); catch e; e; end
        @test err isa ArgumentError && occursin("No step", err.msg)
        # The message names the node farthest off the lattice and the grid builder.
        @test occursin("ωs[3] = $(ωs_none[3])", err.msg) && occursin("graded_frequency_grid", err.msg)
    end

    @testset "duplicated and few-ulp nodes" begin
        # A duplicated node, a 1-ulp pair and a 1e-12 pair would give a zero or tiny derived step.
        for ωs in ([0.0, 0.5, 0.5, 1.0], [0.0, 0.5, nextfloat(0.5), 1.0], [0.0, 1e-12, 0.5])
            err = try; EPSpectral.ladder_shift_step(ωs, nothing); catch e; e; end
            @test err isa ArgumentError && occursin("apart", err.msg)
            @test_throws ArgumentError EPSpectral.ladder_shift_step(ωs, 0.5)
            @test_throws ArgumentError EPSpectral._check_frequency_grid(ωs)
        end
        @test_throws ArgumentError EPSpectral._check_frequency_grid([0.0])
    end

    @testset "derived shift step" begin
        # The lattice is the minimum step divided by k: k = 1 for a grid whose steps divide
        # their minimum, k = 2 for 0.05 and 0.02 steps (lattice 0.01, minimum step 0.02).
        ωs = vcat(0.0:0.05:0.5, 0.52:0.02:0.8)
        @test EPSpectral.ladder_shift_step(ωs, nothing) ≈ 0.01
        ωs_uniform = collect(0.0:0.02:1.0)
        @test EPSpectral.ladder_shift_step(ωs_uniform, nothing) === minimum(diff(ωs_uniform))
        ωs_pb = vcat(-0.5:0.05:-0.1, -0.09:0.01:0.09, 0.1:0.05:0.5)
        @test EPSpectral.ladder_shift_step(ωs_pb, nothing) ≈ 0.01
        # An explicit step is checked and returned as given.
        @test EPSpectral.ladder_shift_step(ωs, 0.005) == 0.005
        @test_throws ArgumentError EPSpectral.ladder_shift_step(ωs, 0.03)
        @test_throws ArgumentError EPSpectral.ladder_shift_step([0.0, 1.0, 1.0 + sqrt(2) / 100], nothing)
        # k > 1 is logged with its cost factor.
        @test_logs (:info, r"factor 2") EPSpectral.ladder_shift_step(ωs, nothing; verbosity = 1)
        @test_logs EPSpectral.ladder_shift_step(ωs_uniform, nothing; verbosity = 1)
        # A fine step far from zero: the minimum spacing carries accumulated round-off, and the derived
        # step is refined from the span (0.1 meV nodes around 11.83 eV in Ry, uniform and graded).
        eV = 1 / 13.605693122994
        center = 11.83 * eV
        for grid in (collect(center .+ (-2eV:1e-4eV:2eV)),
                graded_frequency_grid(center, (1e-4eV, 2e-3eV), (0.1eV, 1eV)))
            @test EPSpectral.ladder_shift_step(grid, nothing) ≈ 1e-4eV rtol = 1e-6
            @test EPSpectral.ladder_shift_step(grid, 1e-4eV) == 1e-4eV
        end
    end

    @testset "one regrouped shift against the direct interpolation" begin
        ωs = vcat(-1.0:0.1:-0.4, -0.35:0.05:0.35, 0.4:0.1:1.0)
        h, nshift = 0.05, 13
        (; klo, wlo, whi) = EPSpectral.vertex_shift_tables(ωs, h, nshift)
        y0 = pseudorandom_complex(length(ωs); seed = 0.7)
        # y vanishes at the end nodes, where the regrouping is exact on the closed table values.
        y = copy(y0); y[1] = y[end] = 0
        for (s, sign) in ((1, 1), (2, -1)), ω_q in (0.0123, 0.2237, 0.4999, 0.55)
            m = floor(Int, ω_q / h)
            r = ω_q / h - m
            shifted(m) = wlo[:, m + 1, s] .* y[klo[:, m + 1, s]] .+
                         whi[:, m + 1, s] .* y[klo[:, m + 1, s] .+ 1]
            regrouped = (1 - r) .* shifted(m) .+ r .* shifted(m + 1)
            direct = [hat_value(y, ωs, x + sign * ω_q) for x in ωs]
            @test norm(regrouped - direct) <= 1e-14 * norm(direct)
            # With nonzero end values the two agree away from the ends.
            y_ends = y0
            shifted_e(m) = wlo[:, m + 1, s] .* y_ends[klo[:, m + 1, s]] .+
                           whi[:, m + 1, s] .* y_ends[klo[:, m + 1, s] .+ 1]
            regrouped_e = (1 - r) .* shifted_e(m) .+ r .* shifted_e(m + 1)
            direct_e = [hat_value(y_ends, ωs, x + sign * ω_q) for x in ωs]
            interior = findall(x -> ωs[1] + ω_q + h < x < ωs[end] - ω_q - h, ωs)
            @test norm((regrouped_e - direct_e)[interior]) <= 1e-14 * norm(direct_e[interior])
        end
    end

    @testset "stars of inner states and chunks" begin
        pb = pb_ladder_case()
        (; inner_states_by_star, inner_star_ptr, f_to_i) = pb.ladder
        ni = pb.res.el_i.n
        @test length(inner_states_by_star) == length(f_to_i) && sort(inner_states_by_star) == eachindex(f_to_i)
        @test inner_star_ptr[1] == 1 && inner_star_ptr[end] == length(f_to_i) + 1
        @test all(1:ni) do i
            all(==(i), f_to_i[inner_states_by_star[inner_star_ptr[i]:inner_star_ptr[i + 1] - 1]])
        end
        # Every outer state is in exactly one chunk, and a chunk holds whole stars.
        for gpu_tile in (nothing, 7, 100)
            chunks = EPSpectral._ladder_workspace(pb.ladder, EP.CPUBackend(), gpu_tile, 0).chunks
            @test reduce(vcat, [collect(c.ips) for c in chunks]) == 1:ni
            @test sum(c.nf for c in chunks) == length(f_to_i)
            @test all(c -> c.nf == inner_star_ptr[last(c.ips) + 1] - inner_star_ptr[first(c.ips)], chunks)
        end
    end

    @testset "coupling on the shift grid against the sum over modes at every shift" begin
        pb = pb_ladder_case()
        # Holstein: one mode, many shifts; pb on a step h/4: several shifts, and an acoustic mode
        # below ω_acoustic that is skipped.
        for (name, case) in (("Holstein", holstein_ladder_case(; symmetry = nothing)),
                             ("pb, step h/4", with_shift_step(pb, pb.dω_shift / 4)))
            (; ladder, res) = case
            chunk = first(EPSpectral._ladder_workspace(ladder, EP.CPUBackend(), 7, 0).chunks)
            nm, ni, _ = size(ladder.g2)
            ωq = Array{Float64}(undef, nm, chunk.nf, ni)
            EP.gather_pair_table!(ωq, ladder.ωph, transpose(ladder.iq_kk), chunk.iks_f, ladder.iks_i)
            nB = EPSpectral.occ_boson.(ωq, res.Tlist[1])
            B = Array{Float64}(undef, ladder.nshift, chunk.nf, ni, 2)
            EPSpectral.bin_coupling_on_shift_grid!(B, ladder.g2, chunk.fidx, ωq, nB, chunk.w_f,
                ladder.dω_shift, EP.omega_acoustic)
            B_ref = similar(B)
            for m in axes(B, 1), f in axes(B, 2), i in axes(B, 3), type in 1:2
                acc = 0.0
                for ν in axes(ωq, 1)
                    ωq[ν, f, i] < EP.omega_acoustic && continue
                    tent = max(0.0, 1 - abs(ωq[ν, f, i] / ladder.dω_shift - (m - 1)))
                    c = ladder.g2[ν, i, chunk.fidx[f]] * tent
                    acc += type == 1 ? c : c * nB[ν, f, i]
                end
                B_ref[m, f, i, type] = chunk.w_f[f] * acc
            end
            @info "coupling on the shift grid" name nshift = ladder.nshift skipped = count(<(EP.omega_acoustic), ωq)
            # The scatter adds the same nonzero terms in the same order, so the sums are bitwise equal.
            @test isequal(B, B_ref)
            @test norm(B) > 0 && all(iszero, B[end, :, :, :])
        end
    end

    @testset "apply_ladder_kernel! against the naive loop" begin
        eV = unit_to_aru(:eV)
        pb = pb_ladder_case()
        # Cases with a shift index k >= 1: the Holstein chain without symmetry (a symmetric chain
        # has Σ_f J_f = 0, so the kernel vanishes identically) with the step below ω0, and pb on a
        # step h/4 (every pb phonon is below h).
        cases = (("Holstein, step below ω0", holstein_ladder_case(; symmetry = nothing), 0.2, 1),
                 ("pb, step h", pb, 0.123 * eV, 0),
                 ("pb, step h/4", with_shift_step(pb, pb.dω_shift / 4), 0.123 * eV, 1))
        for (name, case, Ω, kmin) in cases
            (; ladder, res) = case
            # The dense rotation maps the representative's velocity onto the inner state's, which
            # fixes its orientation independently of the naive loop below (it uses R as given).
            Rf = Array(ladder.R)
            vs_i = res.el_i.vs[Array(ladder.f_to_i)]
            vs_rotated = [Rf[:, :, f] * vs_i[f] for f in eachindex(vs_i)]
            @test maximum(norm.(vs_rotated .- case.setup.el_f.vs)) <= 1e-5
            kmax = floor(Int, maximum(ladder.ωph) / ladder.dω_shift)
            @test kmax >= kmin
            kmin == 0 && @test kmax == 0
            nω, ni = length(case.ωs), res.el_i.n
            J = pseudorandom_complex(nω, 4, 3, 2, ni; seed = 1.3)
            # The regrouping is exact for a vertex that vanishes at the end nodes.
            J[1, :, :, :, :] .= 0
            J[end, :, :, :, :] .= 0
            for iT in 1:2
                Gm, Gp = ladder_propagators(case, iT, Ω)
                LJ_ref = naive_apply_ladder(ladder, J, Gm, Gp, res.Tlist[iT], EP.omega_acoustic)
                # The comparison is not between two round-off-sized results.
                @test norm(LJ_ref) > 1e-3 * norm(J)
                for gpu_tile in (nothing, 7)
                    LJ = apply_ladder_case(case, J, iT, Ω; gpu_tile)
                    err = norm(LJ - LJ_ref) / norm(LJ_ref)
                    @info "apply_ladder_kernel! vs naive" name iT gpu_tile err kmax
                    @test err <= 1e-12
                end
            end
        end
    end

    @testset "shift step and commensurability in run_scgd0_ladder" begin
        eV = unit_to_aru(:eV)
        pb = pb_ladder_case()
        (; model, nk, window, occ, res) = pb
        run(res; kwargs...) = run_scgd0_ladder(model, nk, window, res; occ, Ωs = [0.0],
            verbosity = 0, kwargs...)
        with_ωs(ωs_new) = EPSpectral.SCGD0Result(collect(ωs_new),
            fill(complex(0.0, -0.01), length(ωs_new), res.el_i.n, length(res.Tlist)), res.μ,
            res.el_i, res.Tlist, res.nlist, res.converged, res.history)
        # The coarse step 45 meV with 20 meV cells and a 10 meV transition: the lattice is 5 meV, half
        # the minimum step, which the derived default finds (twice the cost of the minimum step).
        ωs_k2 = 11.83 * eV .+ vcat(-1.2:0.045:-0.25, -0.24:0.02:0.24, 0.25:0.045:1.2) .* eV
        res_k2 = with_ωs(ωs_k2)
        @test EPSpectral.ladder_shift_step(ωs_k2, nothing) ≈ 5e-3 * eV
        out_k2 = run(res_k2)
        @test out_k2.dω_shift ≈ 5e-3 * eV
        # An explicit minimum step is not on the lattice: the error names the compatible step.
        err = try; run(res_k2; dω_shift = 10e-3 * eV); catch e; e; end
        @test err isa ArgumentError && occursin("is compatible", err.msg)
        # A coarse step with no lattice at any k <= 64 is an error with or without an explicit step.
        res_bad = with_ωs(11.83 * eV .+ vcat(-1.2:0.0452123:-0.25, -0.24:0.02:0.24, 0.25:0.0452123:1.2) .* eV)
        for kwargs in ((;), (; dω_shift = 10e-3 * eV))
            err = try; run(res_bad; kwargs...); catch e; e; end
            @test err isa ArgumentError
            @test occursin("not commensurate", err.msg) && occursin("No step", err.msg)
        end
        @info "non-commensurate message" err.msg

        # The derived default is the minimum step on a grid whose lattice it is, so it equals the
        # explicit minimum step bitwise.
        for case in (pb, holstein_ladder_case(; nk = 8))
            run_case(res_case; kwargs...) = run_scgd0_ladder(case.model, case.nk, case.window, res_case;
                case.occ, Ωs = [0.0], verbosity = 0, rtol = 1e-6, kwargs...)
            derived = run_case(case.res)
            explicit = run_case(case.res; dω_shift = minimum(diff(case.ωs)))
            @test derived.dω_shift == minimum(diff(case.ωs))
            @test derived.σ_ladder == explicit.σ_ladder && derived.niter == explicit.niter
        end

        # The same hat-basis shifts on a table twice as fine: one application on a vertex that
        # vanishes at the end nodes is unchanged. A nonzero end value is spread over one table
        # step beyond the end by the regrouping, so the end value makes the result depend on the step.
        pb_half = with_shift_step(pb, pb.dω_shift / 2)
        nω, ni = length(pb.ωs), res.el_i.n
        J = pseudorandom_complex(nω, 4, 3, 2, ni; seed = 0.9)
        J_ends = copy(J)
        J[1, :, :, :, :] .= 0
        J[end, :, :, :, :] .= 0
        LJ, LJ_half = (apply_ladder_case(case, J, 1, 0.123 * eV) for case in (pb, pb_half))
        @test norm(LJ_half - LJ) <= 1e-12 * norm(LJ)
        LJ, LJ_half = (apply_ladder_case(case, J_ends, 1, 0.123 * eV) for case in (pb, pb_half))
        @info "end-value dependence of one application on the shift step" norm(LJ_half - LJ) / norm(LJ)

        # The solve on the finer table is the same problem; σ differs by the end tails only.
        a = run(res; rtol = 1e-10, maxiter = 60)
        b = run(res; dω_shift = pb.dω_shift / 2, rtol = 1e-10, maxiter = 60)
        @test a.dω_shift == pb.dω_shift && b.dω_shift == pb.dω_shift / 2
        @info "σ_ladder on the step h and h/2" norm(b.σ_ladder - a.σ_ladder) / norm(a.σ_ladder)
    end
end
