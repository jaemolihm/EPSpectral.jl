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
with [`interp_table`](@ref) and [`interp_apply!`](@ref) (flat outside the grid, unclamped).
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
@inline function _attach_term(Gm, X, Gp, ω, c12, cart, c0, i)
    c1, c2 = (c12 - 1) % 2 + 1, (c12 - 1) ÷ 2 + 1
    (Gm[ω, c1, i] * X[ω, 1, cart, c0, i] * Gp[ω, 1 + 2 * (c2 - 1), i]
     + Gm[ω, c1 + 2, i] * X[ω, 2, cart, c0, i] * Gp[ω, 1 + 2 * (c2 - 1), i]
     + Gm[ω, c1, i] * X[ω, 3, cart, c0, i] * Gp[ω, 2 + 2 * (c2 - 1), i]
     + Gm[ω, c1 + 2, i] * X[ω, 4, cart, c0, i] * Gp[ω, 2 + 2 * (c2 - 1), i])
end

# One element of the bare vertex J0^{c1c2,c0} = Gm^{c1 c0} v Gp^{c0 c2}.
@inline function _bare_vertex_term(Gm, v, Gp, ω, c12, cart, c0, i)
    c1, c2 = (c12 - 1) % 2 + 1, (c12 - 1) ÷ 2 + 1
    Gm[ω, c1 + 2 * (c0 - 1), i] * v[cart, i] * Gp[ω, c0 + 2 * (c2 - 1), i]
end

# The index arrays of a broadcast over the five dimensions of `J[ω, c12, cart, c0, i]`.
_vertex_indices(J) = (axes(J, 1), reshape(axes(J, 2), 1, :), reshape(axes(J, 3), 1, 1, :),
    reshape(axes(J, 4), 1, 1, 1, :), reshape(axes(J, 5), 1, 1, 1, 1, :))

"""
    attach!(J, Gm, X, Gp) -> J

The propagators attached to both sides of a vertex-like quantity `X[ω, c12, cart, c0, i]` on the
contour: `J^{c1c2} = Σ_ab Gm^{c1 a} X^{ab} Gp^{b c2}` with the contour pair of `X` read as a 2x2
matrix.
"""
function attach!(J, Gm, X, Gp)
    J .= _attach_term.(Ref(Gm), Ref(X), Ref(Gp), _vertex_indices(J)...)
end

"""
    bare_vertex_response!(J0, Gm, v, Gp) -> J0

The bare vertex dressed by the propagators, `J0^{c1c2,c0} = Gm^{c1 c0} v_i Gp^{c0 c2}`, for the
band velocities `v[cart, i]`. The bare vertex is `V^{c1c2,c0} = δ_{c1c2} δ_{c1c0} v_i`.
"""
function bare_vertex_response!(J0, Gm, v, Gp)
    J0 .= _bare_vertex_term.(Ref(Gm), Ref(v), Ref(Gp), _vertex_indices(J0)...)
end

"""
    ladder_conductivity(J, v, w_i, Wω, Ω, T) -> σ::Matrix

The 3x3 conductivity (without the factor `spin_degeneracy / volume`, and not symmetrized) of the
vertex `J[ω, c12, cart, c0, i]` at probe frequency `Ω >= 0` and temperature `T`:

    σ_ab = Re[ F(Ω)/(2πi) Σ_i w_i Σ_ω Wω[ω] conj(v_ia) (J_ib^{22,c0=1} + J_ib^{11,c0=2})(ω) ],
    F(0) = i/(4T),    F(Ω) = -1/(2iΩ (1 + 2 n_B(Ω, T))),

with the state weights `w_i`, the quadrature weights `Wω` of the grid
([`trapezoid_weights`](@ref)) and the velocities `v[cart, i]`. For `J` the bare vertex of
[`bare_vertex_response!`](@ref) this is the bubble
`π Σ_i w_i v_a v_b Σ_ω Wω[ω] A_i(ω - Ω/2) A_i(ω + Ω/2) (f(ω - Ω/2 - μ) - f(ω + Ω/2 - μ))/Ω`,
with `-∂f/∂ω` in place of the last factor at `Ω = 0`.
"""
function ladder_conductivity(J, v, w_i, Wω, Ω, T)
    F = iszero(Ω) ? im / (4T) : -1 / (2im * Ω * (1 + 2 * occ_boson(Ω, T)))
    # Y[b, i] = Σ_ω Wω[ω] (J_ib^{22,c0=1} + J_ib^{11,c0=2})(ω)
    Y = sum(reshape(Wω, :, 1, 1) .* (view(J, :, 4, :, 1, :) .+ view(J, :, 1, :, 2, :)); dims = 1)
    S = (conj.(v) .* reshape(w_i, 1, :)) * transpose(reshape(Y, size(Y, 2), size(Y, 3)))
    real.(F / (2π * im) .* Array(S))
end
