# Linear maps on frequency grids: the Kramers-Kronig and interpolation matrices, built once on the
# host and applied with `mul!`, and pointwise linear interpolation on a uniform grid.

xlogx(x) = iszero(x) ? zero(x) : x * log(abs(x))

"""
    linear_spline_kramers_kronig(x, x1, x2, x3)

The principal-value integral `P∫ dy b(y) / (y - x)` of the hat function `b` that is piecewise
linear with `b(x1) = b(x3) = 0` and `b(x2) = 1`.
"""
function linear_spline_kramers_kronig(x, x1, x2, x3)
    (
        + xlogx(x1 - x) / (x2 - x1)
        - xlogx(x2 - x) * (x3 - x1) / (x2 - x1) / (x3 - x2)
        + xlogx(x3 - x) / (x3 - x2)
    )
end

"""
    kk_matrix(ωs) -> K::Matrix{Float64}

The Kramers-Kronig matrix on the grid `ωs`: for `z` the imaginary part of a retarded function
sampled on `ωs`, `K * z` is its real part `1/π P∫ dω' z(ω') / (ω' - ω)` at the nodes, with `z`
replaced by its hat-basis (piecewise-linear) interpolant and the integral done exactly. Valid on
any grid. Each end value carries a hat that decays linearly to zero over one more cell, of the
width of the end cell, beyond the grid.
"""
function kk_matrix(ωs)
    N = length(ωs)
    # Pad the grid by one end cell on each side.
    x0 = ωs[1] - (ωs[2] - ωs[1])
    xN1 = ωs[end] + (ωs[end] - ωs[end-1])
    xs = vcat(x0, ωs, xN1)
    tmap(collect(Iterators.product(1:N, 1:N))) do (i, j)
        linear_spline_kramers_kronig(xs[i+1], xs[j], xs[j+1], xs[j+2]) / π
    end :: Matrix{Float64}
end

"""
    interp_matrix(ωs, ωs_dense) -> P::Matrix{ComplexF64}

The matrix of linear interpolation from the sorted grid `ωs` to the points `ωs_dense`, held flat
outside `ωs` (`Gridded(Linear())` with `Flat()` extrapolation in Interpolations.jl):
`P * y` is the interpolant of `y` at `ωs_dense`. Stored complex, because it multiplies the complex
self-energy, and a real-times-complex `mul!` has no BLAS path on the device.
"""
function interp_matrix(ωs, ωs_dense)
    FT = eltype(ωs)
    P = zeros(Complex{FT}, length(ωs_dense), length(ωs))
    for (r, x) in enumerate(ωs_dense)
        if x <= first(ωs)
            # Below the grid: the first value.
            P[r, firstindex(ωs)] = 1
        elseif x >= last(ωs)
            # Above the grid: the last value.
            P[r, lastindex(ωs)] = 1
        else
            # Inside: the two neighbours ωs[j] <= x < ωs[j+1].
            j = searchsortedlast(ωs, x)
            t = (x - ωs[j]) / (ωs[j+1] - ωs[j])
            P[r, j] = 1 - t
            P[r, j+1] = t
        end
    end
    P
end

"""
    _interp_linear_uniform(Σin, col, x, ωd0, dωd)

Linear interpolation of column `col` of `Σin` (sampled on the uniform grid `ωd0 + (j-1) dωd`,
`j in axes(Σin, 1)`) at `x`, held flat outside the grid: `BSpline(Linear())` with `Flat()`
extrapolation in Interpolations.jl. Takes isbits values, neither allocates nor throws, so it can
be called from a device broadcast.
"""
@inline function _interp_linear_uniform(Σin, col, x, ωd0, dωd)
    nωd = size(Σin, 1)
    t = (x - ωd0) / dωd
    !(t > 0) && return @inbounds Σin[1, col]   # also a NaN x
    t >= nωd - 1 && return @inbounds Σin[nωd, col]
    j = unsafe_trunc(Int, t)   # floor, since 0 < t < nωd - 1
    r = t - j
    @inbounds (1 - r) * Σin[j + 1, col] + r * Σin[j + 2, col]
end

"""
    kramers_kronig(ωs, zs) -> ys

Real part `ys = kk_matrix(ωs) * zs` of the retarded function whose imaginary part is `zs` on the
grid `ωs`; see [`kk_matrix`](@ref) for the interpolation and end conventions.
"""
kramers_kronig(ωs, zs) = kk_matrix(ωs) * zs
