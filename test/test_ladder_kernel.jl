# The ladder kernel on the host: the commensurability check, the shift tables, and one application
# of the kernel `apply_ladder!` against a naive loop over (i, f, ν, j) that evaluates the shifted
# vertex by exact hat-basis interpolation at ωs[j] ± ω_q, with no shift grid.

# Linear interpolation of y on the sorted grid ωs, zero outside.
function hat_value(y, ωs, x)
    (x < ωs[1] || x > ωs[end]) && return zero(y[1])
    k = min(searchsortedlast(ωs, x), length(ωs) - 1)
    t = (x - ωs[k]) / (ωs[k+1] - ωs[k])
    (1 - t) * y[k] + t * y[k+1]
end

# L(J) = -Gm (s ⊙ K[J]) Gp by loops, with the channel sums of `combine_channels!`.
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
        @test EPSpectral.check_commensurate(ωs, 0.01) === nothing
        @test EPSpectral.check_commensurate(ωs, 0.005) === nothing
        # A step that does not divide the grid: the error names the compatible step.
        err = try; EPSpectral.check_commensurate(ωs, 0.03); catch e; e; end
        @test err isa ArgumentError
        @test occursin("dω_shift = 0.01", err.msg) && occursin("factor 3", err.msg)
        # A grid with no compatible step.
        err = try; EPSpectral.check_commensurate([0.0, 1.0, 1.0 + sqrt(2) / 100], 1.0); catch e; e; end
        @test err isa ArgumentError && occursin("No step", err.msg)
    end

    @testset "one regrouped shift against the direct interpolation" begin
        ωs = vcat(-1.0:0.1:-0.4, -0.35:0.05:0.35, 0.4:0.1:1.0)
        h, M = 0.05, 12
        (; klo, wlo, whi) = EPSpectral.shift_tables(ωs, h, M)
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
        (; fperm, fptr, f_to_i) = pb.ladder
        ni = pb.res.el_i.n
        @test length(fperm) == length(f_to_i) && sort(fperm) == eachindex(f_to_i)
        @test fptr[1] == 1 && fptr[end] == length(f_to_i) + 1
        @test all(i -> all(==(i), f_to_i[fperm[fptr[i]:fptr[i + 1] - 1]]), 1:ni)
        # Every outer state is in exactly one chunk, and a chunk holds whole stars.
        for gpu_tile in (nothing, 7, 100)
            chunks = EPSpectral._ladder_work(pb.ladder, EP.CPUBackend(), gpu_tile, 0).chunks
            @test reduce(vcat, [collect(c.ips) for c in chunks]) == 1:ni
            @test sum(c.nf for c in chunks) == length(f_to_i)
            @test all(c -> c.nf == fptr[last(c.ips) + 1] - fptr[first(c.ips)], chunks)
        end
    end

    @testset "apply_ladder! against the naive loop" begin
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
                    @info "apply_ladder! vs naive" name iT gpu_tile err kmax
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
        # The coarse step 45 meV instead of 50 meV: ωs is not on a lattice of its minimum step.
        ωs_bad = 11.83 * eV .+ vcat(-1.2:0.045:-0.25, -0.24:0.02:0.24, 0.25:0.045:1.2) .* eV
        res_bad = EPSpectral.SCGD0Result(collect(ωs_bad),
            fill(complex(0.0, -0.01), length(ωs_bad), res.el_i.n, length(res.Tlist)), res.μ,
            res.el_i, res.Tlist, res.nlist, res.converged, res.history)
        err = try; run(res_bad); catch e; e; end
        @test err isa ArgumentError
        @test occursin("not commensurate", err.msg) &&
              (occursin("is compatible", err.msg) || occursin("No step", err.msg))
        @info "non-commensurate message" err.msg

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
