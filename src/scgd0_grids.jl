# The two fixed linear maps of the scGD0 iteration, built once on the host and applied with `mul!`.

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
