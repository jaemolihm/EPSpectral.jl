# The contour Green's functions, the bare vertex and the conductivity contraction of the ladder
# (vertex-corrected) Kubo conductivity on the scGD0 self-energy.
#
# Index legend: ω = vertex frequency (node of `ωs`), c12 = column-major index of the contour pair
# (c1, c2) with 1 = forward and 2 = backward branch, i.e. 1 = 11, 2 = 21, 3 = 12, 4 = 22; c0 = the
# contour index of the current operator, cart = Cartesian direction, i = outer (IBZ) state.
#
# The vertex array is `J[ω, c12, cart, c0, i]` and the Green's functions are `G[ω, c12, i]`. All
# routines are broadcasts over backend arrays with a scalar element function, as in
# `scgd0_kernel.jl`, so the same code runs on the host and on the device.
#
# The vertex at frequency ε carries a left propagator G_m = G(ε - Ω/2) and a right propagator
# G_p = G(ε + Ω/2) for a probe frequency Ω.

"""
    _contour_green(ω, ε, Σ, μ, T, c12)

Component `c12` of the 2x2 Keldysh-contour Green's function of a state of energy `ε` with
self-energy `Σ` at the frequency `ω`. With `Gᴿ = 1/(ω - ε - Σ)`, `Gᴬ = conj(Gᴿ)` and the
fluctuation-dissipation relation `Gᴷ = (Gᴿ - Gᴬ)(1 - 2f(ω - μ))`:

    G¹¹ = (Gᴿ + Gᴬ + Gᴷ)/2    G²¹ = (Gᴿ - Gᴬ + Gᴷ)/2
    G¹² = (-Gᴿ + Gᴬ + Gᴷ)/2    G²² = (-Gᴿ - Gᴬ + Gᴷ)/2

`G²¹` is the greater and `G¹²` the lesser function.
"""
@inline function _contour_green(ω, ε, Σ, μ, T, c12)
    GR = inv(ω - ε - Σ)
    GA = conj(GR)
    GK = (GR - GA) * (1 - 2 * occ_fermion(ω - μ, T))
    if c12 == 1
        (GR + GA + GK) / 2
    elseif c12 == 2
        (GR - GA + GK) / 2
    elseif c12 == 3
        (-GR + GA + GK) / 2
    else
        (-GR - GA + GK) / 2
    end
end

"""
    contour_greens!(Gm, Gp, ε_i, Σd_m, Σd_p, ωs, Ω, μ, T) -> (Gm, Gp)

Fill the left and right propagators `Gm[ω, c12, i] = G(ωs[ω] - Ω/2)` and `Gp[ω, c12, i] =
G(ωs[ω] + Ω/2)` of the states with energies `ε_i`, from [`_contour_green`](@ref). `Σd_m[ω, i]` and
`Σd_p[ω, i]` are the self-energy at `ωs - Ω/2` and `ωs + Ω/2`, which the caller reads from `ωs`
with [`linear_interpolation_table`](@ref) and [`apply_interpolation_table!`](@ref) (flat outside the grid,
unclamped).
"""
function contour_greens!(Gm, Gp, ε_i, Σd_m, Σd_p, ωs, Ω, μ, T)
    c12 = reshape(1:4, 1, 4, 1)
    ε = reshape(ε_i, 1, 1, :)
    Gm .= _contour_green.(ωs .- Ω / 2, ε, reshape(Σd_m, size(Σd_m, 1), 1, :), μ, T, c12)
    Gp .= _contour_green.(ωs .+ Ω / 2, ε, reshape(Σd_p, size(Σd_p, 1), 1, :), μ, T, c12)
    Gm, Gp
end

# One element of J = Gm X Gp as 2x2 contour matrices: J^{c1c2} = Σ_ab Gm^{c1 a} X^{ab} Gp^{b c2},
# the sum over (a, b) written out.
@inline function _attach_greens_term(Gm, X, Gp, ω, c12, cart, c0, i)
    c1, c2 = (c12 - 1) % 2 + 1, (c12 - 1) ÷ 2 + 1
    (Gm[ω, c1, i] * X[ω, 1, cart, c0, i] * Gp[ω, 1 + 2 * (c2 - 1), i]
     + Gm[ω, c1 + 2, i] * X[ω, 2, cart, c0, i] * Gp[ω, 1 + 2 * (c2 - 1), i]
     + Gm[ω, c1, i] * X[ω, 3, cart, c0, i] * Gp[ω, 2 + 2 * (c2 - 1), i]
     + Gm[ω, c1 + 2, i] * X[ω, 4, cart, c0, i] * Gp[ω, 2 + 2 * (c2 - 1), i])
end

# One element of the bare vertex J0^{c1c2,c0} = Gm^{c1 c0} v Gp^{c0 c2}.
@inline function _attach_greens_bare_term(Gm, v, Gp, ω, c12, cart, c0, i)
    c1, c2 = (c12 - 1) % 2 + 1, (c12 - 1) ÷ 2 + 1
    Gm[ω, c1 + 2 * (c0 - 1), i] * v[cart, i] * Gp[ω, c0 + 2 * (c2 - 1), i]
end

# The index arrays of a broadcast over the dimensions of `A`: one per dimension, shaped to vary
# along that dimension only.
_broadcast_axes(A) = ntuple(d -> reshape(axes(A, d), ntuple(e -> e == d ? size(A, d) : 1, Val(ndims(A)))),
    Val(ndims(A)))

"""
    attach_greens_to_vertex!(J, Gm, X, Gp) -> J

The propagators attached to both sides of a vertex-like quantity `X[ω, c12, cart, c0, i]` on the
contour: `J^{c1c2} = Σ_ab Gm^{c1 a} X^{ab} Gp^{b c2}` with the contour pair of `X` read as a 2x2
matrix.
"""
function attach_greens_to_vertex!(J, Gm, X, Gp)
    J .= _attach_greens_term.(Ref(Gm), Ref(X), Ref(Gp), _broadcast_axes(J)...)
end

"""
    attach_greens_to_bare_vertex!(J0, Gm, v, Gp) -> J0

The bare vertex dressed by the propagators, `J0^{c1c2,c0} = Gm^{c1 c0} v_i Gp^{c0 c2}`, for the
band velocities `v[cart, i]`. The bare vertex is `V^{c1c2,c0} = δ_{c1c2} δ_{c1c0} v_i`.
"""
function attach_greens_to_bare_vertex!(J0, Gm, v, Gp)
    J0 .= _attach_greens_bare_term.(Ref(Gm), Ref(v), Ref(Gp), _broadcast_axes(J0)...)
end

"""
    conductivity_from_vertex(J, v, w_i, Wω, Ω, T) -> σ::Matrix

The 3x3 conductivity (without the factor `spin_degeneracy / volume`, and not symmetrized) of the
vertex `J[ω, c12, cart, c0, i]` at probe frequency `Ω >= 0` and temperature `T`:

    σ_ab = Re[ F(Ω)/(2πi) Σ_i w_i Σ_ω Wω[ω] conj(v_ia) (J_ib^{22,c0=1} + J_ib^{11,c0=2})(ω) ],
    F(0) = i/(4T),    F(Ω) = -1/(2iΩ (1 + 2 n_B(Ω, T))),

with the state weights `w_i`, the quadrature weights `Wω` of the grid
([`trapezoid_weights`](@ref)) and the velocities `v[cart, i]`. For `J` the bare vertex of
[`attach_greens_to_bare_vertex!`](@ref) this is the bubble
`π Σ_i w_i v_a v_b Σ_ω Wω[ω] A_i(ω - Ω/2) A_i(ω + Ω/2) (f(ω - Ω/2 - μ) - f(ω + Ω/2 - μ))/Ω`,
with `-∂f/∂ω` in place of the last factor at `Ω = 0`.
"""
function conductivity_from_vertex(J, v, w_i, Wω, Ω, T)
    F = iszero(Ω) ? im / (4T) : -1 / (2im * Ω * (1 + 2 * occ_boson(Ω, T)))
    # Y[b, i] = Σ_ω Wω[ω] (J_ib^{22,c0=1} + J_ib^{11,c0=2})(ω)
    Y = sum(reshape(Wω, :, 1, 1) .* (view(J, :, 4, :, 1, :) .+ view(J, :, 1, :, 2, :)); dims = 1)
    S = (conj.(v) .* reshape(w_i, 1, :)) * transpose(reshape(Y, size(Y, 2), size(Y, 3)))
    real.(F / (2π * im) .* Array(S))
end

# --- The D*J kernel: shift tables, the B build, the star contraction and the shift sums ---
#
# Write h = `dω_shift`, ωs[j] = ωs[1] + h p_j with integer p_j (the grid is commensurate), and let
# S_m be the hat-basis shift of a vertex by ω_m = m h, (S_m J)(ωs[j]) = J(ωs[j] ± ω_m), with J
# zero outside `ωs`. For fixed j the shifted vertex is piecewise linear in the shift with kinks at
# multiples of h, so a phonon of frequency ω_q = (m + r) h couples to the two neighbouring tables:
#
#     F_j(ω_q) = (1 - r) (S_m J)(ωs[j]) + r (S_{m+1} J)(ωs[j]).
#
# The inner states f are grouped by their IBZ representative i' (the star of i'), and the rotation
# R_f of the unfolded vertex is folded into the coupling, so that the sums U±, V± of
# `apply_phonon_exchange_kernel!` are GEMMs over (m, b, i') with no gather of the vertex per inner state:
#
#     U±_i[row] = Σ_{m,b,i'} X±[row, (m, b, i')] B̃[(m, b, i'), (a, i, 1)],   V±: type 2,
#     B[m, f, i, 1] = w_f Σ_ν g2[ν, i, f] tent(ω_νfi / h - m),    B[., ., ., 2] has n_B(ω_νfi) too,
#     B̃[m, b, i', a, i, type] = Σ_{f in star(i')} R_f[a, b] B[m, f, i, type],
#     X±[(j, c12, c0, ri), (m, b, i')] = (S_m J_{i'})^b[j] at ωs[j] ± ω_m,
#
# with tent(x) = max(0, 1 - |x|), `row` = (j, c12, c0, ri) and ri = (re, im). The result C has the
# layout (ω, c12, c0, ri, a, i, type). A jump of J at an end node of `ωs` (J there is nonzero and
# zero beyond it) sits on a multiple of h, where the regrouping is exact only for the two-sided
# limit that the closed table value is.

# The largest deviation of a node of `ωs` from the lattice `ωs[1] + h ℤ`, in units of `h`, and the
# index of that node.
function _lattice_deviation(ωs, h)
    p = (ωs .- ωs[1]) ./ h
    findmax(abs.(p .- round.(p)))
end

# The smallest k in 1, ..., 64 for which `ωs` lies on `ωs[1] + (h / k) ℤ` to `1e-8 h / k`, or `nothing`.
function _lattice_divisor(ωs, h)
    for k in 1:64
        first(_lattice_deviation(ωs, h / k)) <= 1e-8 && return k
    end
    nothing
end

# The error for a grid that is not commensurate with the step `h`: the node farthest off the
# lattice, and the largest compatible step `h / k` if there is one.
function _commensurability_error(ωs, h)
    deviation, j = _lattice_deviation(ωs, h)
    k = _lattice_divisor(ωs, h)
    advice = k === nothing ?
        "No step dω_shift / k with k <= 64 makes ωs commensurate; build ωs with " *
        "`graded_frequency_grid`, or move the node onto the lattice." :
        "dω_shift = $(h / k) (= dω_shift / $k) is compatible; the cost of the " *
        "ladder grows by a factor $k."
    ArgumentError("ωs is not commensurate with dω_shift = $h: the node ωs[$j] = $(ωs[j]) is " *
        "$deviation dω_shift off the lattice ωs[1] + dω_shift ℤ. " * advice)
end

"""
    check_frequency_grid_commensurate(ωs, dω_shift) -> nothing

Throw an `ArgumentError` unless every node of `ωs` lies on `ωs[1] + dω_shift ℤ` to `1e-8 dω_shift`,
which the shift tables of the ladder require. The message names the node farthest off the lattice and
the largest step `dω_shift / k` (k = 2, ..., 64) on which `ωs` is commensurate with the factor `k` by
which the cost grows, or says that none exists.
"""
function check_frequency_grid_commensurate(ωs, dω_shift)
    first(_lattice_deviation(ωs, dω_shift)) <= 1e-8 || throw(_commensurability_error(ωs, dω_shift))
    nothing
end

"""
    ladder_shift_step(ωs, dω_shift; verbosity = 0) -> dω_shift

The step of the shift tables of the ladder. An explicit `dω_shift` is checked with
[`check_frequency_grid_commensurate`](@ref) and returned. `dω_shift = nothing` derives the largest
step on which `ωs` is commensurate, `minimum(diff(ωs)) / k` with the smallest k <= 64, and logs `k`
at `verbosity > 0` if it is not 1, since the cost grows by that factor. Throws an `ArgumentError`
if there is no such step.
"""
function ladder_shift_step(ωs, dω_shift; verbosity = 0)
    if dω_shift !== nothing
        check_frequency_grid_commensurate(ωs, dω_shift)
        return dω_shift
    end
    h = minimum(diff(ωs))
    k = _lattice_divisor(ωs, h)
    k === nothing && throw(_commensurability_error(ωs, h))
    k > 1 && verbosity > 0 && @info "ladder: shift step dω_shift = $(h / k) (the minimum step of ωs / $k); " *
        "the cost grows by a factor $k"
    h / k
end

"""
    vertex_shift_tables(ωs, dω_shift, nshift) -> (; klo, wlo, whi)

The hat-basis shift tables of the ladder. For the sign `s = 1` (shift up) and `s = 2` (shift down),
`m = 1:nshift` (shift by `(m - 1) dω_shift`) and a node `j`, the shifted vertex is
`(S J)(ωs[j]) = wlo[j, m, s] J[klo[j, m, s]] + whi[j, m, s] J[klo[j, m, s] + 1]` for a vertex `J`
sampled on `ωs`, the linear interpolant in the cell of `ωs[j] ± (m - 1) dω_shift`. Both weights
are zero if that point lies outside `[ωs[1], ωs[end]]` (to `1e-8 dω_shift`). The (1 - r, r)
regrouping of two tables spreads a nonzero end value of the vertex over one shift step beyond the
grid (see the comment above `check_frequency_grid_commensurate`).
"""
function vertex_shift_tables(ωs, dω_shift, nshift)
    FT = eltype(ωs)
    nω = length(ωs)
    tol = FT(1e-8) * dω_shift
    klo = Array{Int32}(undef, nω, nshift, 2)
    wlo = Array{FT}(undef, nω, nshift, 2)
    whi = Array{FT}(undef, nω, nshift, 2)
    for (s, sign) in enumerate((1, -1)), m in 1:nshift, j in 1:nω
        x = ωs[j] + sign * (m - 1) * dω_shift
        # The cell of x (clamped, so that a node hit by round-off lies in the end cell).
        k = clamp(searchsortedlast(ωs, x), 1, nω - 1)
        t = clamp((x - ωs[k]) / (ωs[k+1] - ωs[k]), 0, 1)
        inside = ωs[1] - tol <= x <= ωs[end] + tol
        klo[j, m, s] = k
        wlo[j, m, s] = inside ? 1 - t : 0
        whi[j, m, s] = inside ? t : 0
    end
    (; klo, wlo, whi)
end

# Run `body(is)` over a partition of `axis`: chunks in parallel if `A` is a host array (each chunk
# writes disjoint outputs, so the result does not depend on the thread count), one call otherwise.
function _threaded_on_host(body, A, axis)
    if ElectronPhonon.on_backend(CPUBackend(), A)
        tforeach(body, index_chunks(axis; n = 2 * Threads.nthreads()); chunking = false)
    else
        body(axis)
    end
end

# Adds the modes ν of the column B[:, f, i, type] of the coupling of the shift sums into `B` in place:
# for each mode in ascending order the stored coupling `g2[ν, i, fidx[f]]` of ω_q = ωq[ν, f, i], split
# by `tent` onto the two shifts `lo = floor(ω_q / h)` and `lo + 1` (rows `lo + 1` and `lo + 2`) that
# neighbour ω_q, with the extra n_B factor `nB[ν, f, i]` for type 2. `f` is the position in the chunk
# and `fidx[f]` the inner state it holds in `g2`. A mode below `ω_acoustic` is skipped before g2 is
# read (g2 is Inf or huge at Γ-acoustic); every other shift has `tent = 0`. The written columns are
# distinct for distinct (f, i, type); the return value is only the output a broadcast needs.
@inline function _add_modes_to_coupling!(B, g2, fidx, ωq, nB, h, ω_acoustic, f, i, type)
    for ν in axes(ωq, 1)
        ω = ωq[ν, f, i]
        ω < ω_acoustic && continue
        x = ω / h
        lo = floor(Int, x)
        g = g2[ν, i, fidx[f]]
        for m in (lo, lo + 1)
            c = g * max(zero(h), 1 - abs(x - m))
            # nB is read inside the type-2 branch: with the load hoisted, the device compiler fuses
            # c * nB into the accumulation, which differs from the unfused sum by an ulp.
            B[m + 1, f, i, type] += type == 1 ? c : c * nB[ν, f, i]
        end
    end
    true
end

"""
    bin_coupling_on_shift_grid!(B, g2, fidx, ωq, nB, w_f, dω_shift, ω_acoustic) -> B

Fill `B[m, f, i, type]` (`m = 1:nshift`, shift `(m - 1) dω_shift`) of one chunk of inner states:
`B[m, f, i, 1] = w_f[f] Σ_ν g2[ν, i, fidx[f]] tent(ωq[ν, f, i] / dω_shift - (m - 1))` and the same
with a factor `nB[ν, f, i]` for `type = 2`, over the modes with `ωq >= ω_acoustic`.
`tent(x) = max(0, 1 - |x|)` is the linear split of a frequency onto its two neighbouring shifts.
Each mode is added to those two shifts only, in ascending ν. No index is guarded: `ωq` never exceeds
the largest phonon frequency, which sets `nshift`, so the last row of `B` receives at most the weight 0.
"""
function bin_coupling_on_shift_grid!(B, g2, fidx, ωq, nB, w_f, dω_shift, ω_acoustic)
    i = reshape(axes(B, 3), 1, :, 1)
    type = reshape(axes(B, 4), 1, 1, :)
    done = similar(B, Bool, size(B, 2), size(B, 3), size(B, 4))
    _threaded_on_host(B, axes(B, 2)) do fs
        fill!(view(B, :, fs, :, :), 0)
        view(done, fs, :, :) .= _add_modes_to_coupling!.(Ref(B), Ref(g2), Ref(fidx), Ref(ωq), Ref(nB),
            dω_shift, ω_acoustic, reshape(fs, :, 1, 1), i, type)
        view(B, :, fs, :, :) .*= reshape(view(w_f, fs), 1, :, 1, 1)
    end
    B
end

# One element B̃[m, b, ip, a, i, type] of the star contraction: the coupling of the inner states
# `star_ptr[ip]:star_ptr[ip + 1] - 1` of the chunk (the star of the outer state `ip`) rotated by R_f.
@inline function _star_sum_coupling_term(B, R, fidx, star_ptr, m, b, ip, a, i, type)
    acc = zero(eltype(B))
    for f in star_ptr[ip]:star_ptr[ip + 1] - 1
        acc += R[a, b, fidx[f]] * B[m, f, i, type]
    end
    acc
end

"""
    star_sum_coupling!(Bt, B, R, fidx, star_ptr) -> Bt

Fill `Bt[m, b, ip, a, i, type] = Σ_{f in star(ip)} R[a, b, fidx[f]] B[m, f, i, type]` for the outer
states `ip` of a chunk, from the coupling `B` of the chunk's inner states
([`bin_coupling_on_shift_grid!`](@ref)), the rotations `R[:, :, f]` of all inner states, `fidx[f]` (the
inner state at chunk position `f`) and the star boundaries `star_ptr` (the star of `ip` holds the chunk
positions `star_ptr[ip]:star_ptr[ip + 1] - 1`).
"""
function star_sum_coupling!(Bt, B, R, fidx, star_ptr)
    m, b, _, a, i, type = _broadcast_axes(Bt)
    _threaded_on_host(Bt, axes(Bt, 3)) do ips
        view(Bt, :, :, ips, :, :, :) .= _star_sum_coupling_term.(Ref(B), Ref(R), Ref(fidx), Ref(star_ptr),
            m, b, reshape(ips, 1, 1, :, 1, 1, 1), a, i, type)
    end
    Bt
end

# One element of the shifted vertex X[j, c12, c0, ri, m, b, ip] (see the header) of the outer
# state `i1 - 1 + ip`.
@inline function _shifted_vertex_term(J, klo, wlo, whi, i1, j, c12, c0, ri, m, b, ip)
    i = i1 - 1 + ip
    z = wlo[j, m] * J[klo[j, m], c12, b, c0, i] + whi[j, m] * J[klo[j, m] + 1, c12, b, c0, i]
    ri == 1 ? real(z) : imag(z)
end

"""
    gather_shifted_vertex!(X, J, klo, wlo, whi, i1) -> X

Fill `X[j, c12, c0, ri, m, b, ip] = (S_m J_{i1 - 1 + ip})^b[j]` (`ri` = real, imaginary part) for
the vertex `J[ω, c12, cart, c0, i]`, the outer states `i1:i1 + size(X, 7) - 1` and the shift tables
of one sign: `klo`, `wlo`, `whi` of size `(nω, nshift)` ([`vertex_shift_tables`](@ref)).
"""
function gather_shifted_vertex!(X, J, klo, wlo, whi, i1)
    j, c12, c0, ri, m, b, _ = _broadcast_axes(X)
    _threaded_on_host(X, axes(X, 7)) do ips
        view(X, :, :, :, :, :, :, ips) .= _shifted_vertex_term.(Ref(J), Ref(klo), Ref(wlo),
            Ref(whi), i1, j, c12, c0, ri, m, b, reshape(ips, 1, 1, 1, 1, 1, 1, :))
    end
    X
end

# The complex vertex-shaped value of the four channels K^{ab} (with the contour signs s_a s_b) at
# one element, from the shift sums C±[ω, c12, c0, ri, a, i, type] and the Kramers-Kronig image
# KY[ω, c12, c0, ri, a, i] of ½(U+ - U-):
#   K¹¹ = -i KK[½(U+ - U-)] - ½(U+ + U-) - (V+ + V-)      K²² = same with +i
#   K²¹ = -(V+ + U- + V-)                                  K¹² = -(U+ + V+ + V-)
# and K²¹, K¹² carry s_a s_b = -1.
@inline function _apply_phonon_exchange_kernel_term(Cp, Cm, KY, ω, c12, a, c0, i)
    sum_c(C, type) = complex(C[ω, c12, c0, 1, a, i, type], C[ω, c12, c0, 2, a, i, type])
    Up, Um, Vp, Vm = sum_c(Cp, 1), sum_c(Cm, 1), sum_c(Cp, 2), sum_c(Cm, 2)
    if c12 == 1 || c12 == 4
        ky = complex(KY[ω, c12, c0, 1, a, i], KY[ω, c12, c0, 2, a, i])
        (c12 == 1 ? -im : im) * ky - (Up + Um) / 2 - (Vp + Vm)
    elseif c12 == 2
        (Vp + Um + Vm)
    else
        (Up + Vp + Vm)
    end
end

"""
    apply_phonon_exchange_kernel!(KJ, Cp, Cm, KK, Y, KY) -> KJ

The channels `KJ[ω, c12, cart, c0, i] = s_a s_b K^{ab}[J]` of the ladder kernel (`c12` = (a, b),
`s_1 = 1`, `s_2 = -1`) from the shift sums `Cp`, `Cm` = `[U; V]` of the upward and downward shifts,
each of size `(nω, 4, 2, 2, 3, ni, 2)` (c12, c0, real/imaginary part, cart, then `type` = U, V).
The principal-value part is `KK` (the spline Kramers-Kronig matrix) applied to `½(U+ - U-)`; `Y`
and `KY` are real work arrays of size `(nω, 4, 2, 2, 3, ni)`.
"""
function apply_phonon_exchange_kernel!(KJ, Cp, Cm, KK, Y, KY)
    Y .= (view(Cp, :, :, :, :, :, :, 1) .- view(Cm, :, :, :, :, :, :, 1)) ./ 2
    mul!(reshape(KY, size(KY, 1), :), KK, reshape(Y, size(Y, 1), :))
    KJ .= _apply_phonon_exchange_kernel_term.(Ref(Cp), Ref(Cm), Ref(KY), _broadcast_axes(KJ)...)
end
