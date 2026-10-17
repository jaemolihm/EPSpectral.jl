# Linear maps on frequency grids: the Kramers-Kronig and interpolation matrices, built once on the
# host and applied with `mul!`, and pointwise linear interpolation on a uniform grid.

"""
    kk_matrix(ωs) -> K::Matrix

The matrix of [`kramers_kronig`](@ref) on the grid `ωs`: `K * z == kramers_kronig(ωs, z)` up to
round-off, for any `z`. `kramers_kronig` is linear in its input, so column `j` is its value on the
`j`-th unit vector.
"""
function kk_matrix(ωs)
    stack(tmap(j -> kramers_kronig(ωs, (eachindex(ωs) .== j)), eachindex(ωs)))
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
