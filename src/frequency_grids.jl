# Linear maps on frequency grids: the Kramers-Kronig matrix, built once on the host and applied
# with `mul!`, the linear-interpolation tables applied by a gather, and pointwise linear
# interpolation on a sorted grid.

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
    K = Matrix{Float64}(undef, N, N)
    # Column j is the response to the hat centred at ωs[j]; columns are independent.
    tforeach(index_chunks(1:N; n = 4 * Threads.nthreads()); chunking = false) do js
        for j in js, i in 1:N
            K[i, j] = linear_spline_kramers_kronig(xs[i+1], xs[j], xs[j+1], xs[j+2]) / π
        end
    end
    K
end

"""
    trapezoid_weights(ωs) -> Vector

The node weights `Wω` of the trapezoid rule on the sorted grid `ωs`, `∫ y dω ≈ Σ_j Wω[j] y[j]`:
half the distance between the neighbouring nodes, with half a cell at each end node, so that
`sum(Wω) == ωs[end] - ωs[1]`.
"""
trapezoid_weights(ωs) =
    [(ωs[min(j + 1, end)] - ωs[max(j - 1, 1)]) / 2 for j in eachindex(ωs)]

"""
    interp_table(ωs, xs) -> (j::Vector{Int32}, t::Vector{eltype(ωs)})

The table of linear interpolation from the sorted grid `ωs` to the points `xs`, held flat outside
`ωs` (`Gridded(Linear())` with `Flat()` extrapolation in Interpolations.jl): `y(xs[r])` is
`(1 - t[r]) y[j[r]] + t[r] y[j[r] + 1]`. Inside, `ωs[j] <= x < ωs[j+1]` and `t` is the position in
that cell; below `ωs` the table is `(1, 0)`, above it `(length(ωs) - 1, 1)`. Apply it with
[`interp_apply!`](@ref).
"""
function interp_table(ωs, xs)
    FT = eltype(ωs)
    n = length(ωs)
    j = Vector{Int32}(undef, length(xs))
    t = Vector{FT}(undef, length(xs))
    for (r, x) in enumerate(xs)
        if x <= first(ωs)
            # Below the grid: the first value.
            j[r], t[r] = 1, 0
        elseif x >= last(ωs)
            # Above the grid: the last value.
            j[r], t[r] = n - 1, 1
        else
            # Inside: the two neighbours ωs[j] <= x < ωs[j+1].
            jr = searchsortedlast(ωs, x)
            j[r], t[r] = jr, (x - ωs[jr]) / (ωs[jr+1] - ωs[jr])
        end
    end
    j, t
end

"""
    interp_apply!(Y, Σ, j, t) -> Y

Apply the table `(j, t)` of [`interp_table`](@ref) to the columns of `Σ`:
`Y[r, i] = (1 - t[r]) Σ[j[r], i] + t[r] Σ[j[r] + 1, i]`. A gather broadcast over backend arrays, so
it runs on the host and on the device and is deterministic.
"""
function interp_apply!(Y, Σ, j, t)
    lerp(Σ, jr, tr, i) = (1 - tr) * Σ[jr, i] + tr * Σ[jr + 1, i]
    Y .= lerp.(Ref(Σ), j, t, reshape(axes(Y, 2), 1, :))
end

"""
    _interp_linear_grid(Σin, col, x, ωs)

Linear interpolation of column `col` of `Σin` (sampled on the sorted grid `ωs`, `size(Σin, 1) ==
length(ωs)`) at `x`, held flat outside the grid: `Gridded(Linear())` with `Flat()` extrapolation
in Interpolations.jl. The cell is found by bisection, `log2(length(ωs))` reads of `ωs`. Takes
isbits values and arrays, neither allocates nor throws, so it can be called from a device
broadcast.
"""
@inline function _interp_linear_grid(Σin, col, x, ωs)
    n = length(ωs)
    !(x > (@inbounds ωs[1])) && return @inbounds Σin[1, col]   # also a NaN x
    x >= (@inbounds ωs[n]) && return @inbounds Σin[n, col]
    # Bisection with ωs[lo] <= x < ωs[hi], which holds at the start since ωs[1] < x < ωs[n].
    lo, hi = 1, n
    while hi - lo > 1
        mid = (lo + hi) >>> 1
        if @inbounds ωs[mid] <= x
            lo = mid
        else
            hi = mid
        end
    end
    @inbounds begin
        t = (x - ωs[lo]) / (ωs[lo + 1] - ωs[lo])
        (1 - t) * Σin[lo, col] + t * Σin[lo + 1, col]
    end
end

"""
    kramers_kronig(ωs, zs) -> ys

Real part `ys = kk_matrix(ωs) * zs` of the retarded function whose imaginary part is `zs` on the
grid `ωs`; see [`kk_matrix`](@ref) for the interpolation and end conventions.
"""
kramers_kronig(ωs, zs) = kk_matrix(ωs) * zs
