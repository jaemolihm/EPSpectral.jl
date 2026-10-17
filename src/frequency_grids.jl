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
    kramers_kronig_matrix(ωs) -> K::Matrix{Float64}

The Kramers-Kronig matrix on the grid `ωs`: for `z` the imaginary part of a retarded function
sampled on `ωs`, `K * z` is its real part `1/π P∫ dω' z(ω') / (ω' - ω)` at the nodes, with `z`
replaced by its hat-basis (piecewise-linear) interpolant and the integral done exactly. Valid on
any grid. Each end value carries a hat that decays linearly to zero over one more cell, of the
width of the end cell, beyond the grid.
"""
function kramers_kronig_matrix(ωs)
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
    linear_interpolation_table(ωs, xs) -> (j::Vector{Int32}, t::Vector{eltype(ωs)})

The table of linear interpolation from the sorted grid `ωs` to the points `xs`, held flat outside
`ωs` (`Gridded(Linear())` with `Flat()` extrapolation in Interpolations.jl): `y(xs[r])` is
`(1 - t[r]) y[j[r]] + t[r] y[j[r] + 1]`. Inside, `ωs[j] <= x < ωs[j+1]` and `t` is the position in
that cell; below `ωs` the table is `(1, 0)`, above it `(length(ωs) - 1, 1)`. Apply it with
[`apply_interpolation_table!`](@ref).
"""
function linear_interpolation_table(ωs, xs)
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
    apply_interpolation_table!(Y, Σ, j, t) -> Y

Apply the table `(j, t)` of [`linear_interpolation_table`](@ref) to the columns of `Σ`:
`Y[r, i] = (1 - t[r]) Σ[j[r], i] + t[r] Σ[j[r] + 1, i]`. A gather broadcast over backend arrays, so
it runs on the host and on the device and is deterministic.
"""
function apply_interpolation_table!(Y, Σ, j, t)
    lerp(Σ, jr, tr, i) = (1 - tr) * Σ[jr, i] + tr * Σ[jr + 1, i]
    Y .= lerp.(Ref(Σ), j, t, reshape(axes(Y, 2), 1, :))
end

"""
    cell_search_table(backend, ωs) -> (; jtab::Vector{Int32}, x0, inv_h)

The O(1) cell search table for the sorted grid `ωs`, with `jtab` placed on `backend`. The axis is
cut into buckets of width `h < minimum(diff(ωs))` from `x0 = ωs[1]`, `bucket(x) = unsafe_trunc(Int,
(x - x0) * inv_h)`, so each bucket holds at most one node. `jtab[bucket(x) + 1]` is the number of
nodes in earlier buckets, all of which lie below `x` because `bucket` is monotone in floating point
(the table is built with the same expression), so the cell of `x` is that index or the next one.
The table has `(ωs[end] - ωs[1]) / h + 1` entries, 4 B each, which grows as the inverse of the
finest cell; more than `2^24` entries (64 MiB) throws an `ArgumentError`.
"""
function cell_search_table(backend, ωs)
    FT = eltype(ωs)
    x0 = first(ωs)
    h_min = minimum(diff(ωs))
    inv_h = 1 / (FT(1 - 1e-6) * h_min)
    nbuckets = trunc(Int, min((last(ωs) - x0) * inv_h, 2^40)) + 1
    nbuckets <= 2^24 || throw(ArgumentError(
        "bucket table of $nbuckets entries for a grid with minimum spacing $h_min over a range " *
        "$(last(ωs) - x0); use a grid without a near-coincident pair of nodes"))
    buckets = [unsafe_trunc(Int, (x - x0) * inv_h) for x in ωs]
    allunique(buckets) || throw(ArgumentError("two grid nodes fall in one bucket"))
    jtab = Int32[searchsortedfirst(buckets, b) - 1 for b in 0:last(buckets)]
    (; jtab = to_device(backend, jtab), x0, inv_h)
end

"""
    _self_energy_at(Σin, col, x, ωs, ωs_cell_table)

Linear interpolation of column `col` of `Σin` (sampled on the sorted grid `ωs`, `size(Σin, 1) ==
length(ωs)`) at `x`, held flat outside the grid: `Gridded(Linear())` with `Flat()` extrapolation
in Interpolations.jl. The cell is found from `ωs_cell_table = cell_search_table(backend, ωs)` (on the same
backend as `ωs`): one table read and one comparison. Takes isbits values and arrays, neither
allocates nor throws, so it can be called from a device broadcast.
"""
@inline function _self_energy_at(Σin, col, x, ωs, ωs_cell_table)
    (; jtab, x0, inv_h) = ωs_cell_table
    n = length(ωs)
    !(x > (@inbounds ωs[1])) && return @inbounds Σin[1, col]   # also a NaN x
    x >= (@inbounds ωs[n]) && return @inbounds Σin[n, col]
    # ωs[1] < x < ωs[n], so the bucket is in 0:bucket(ωs[n]) and j + 1 <= n.
    j = Int(@inbounds jtab[unsafe_trunc(Int, (x - x0) * inv_h) + 1])
    j += @inbounds ωs[j + 1] <= x
    @inbounds begin
        t = (x - ωs[j]) / (ωs[j + 1] - ωs[j])
        (1 - t) * Σin[j, col] + t * Σin[j + 1, col]
    end
end

"""
    kramers_kronig(ωs, zs) -> ys

Real part `ys = kramers_kronig_matrix(ωs) * zs` of the retarded function whose imaginary part is `zs` on the
grid `ωs`; see [`kramers_kronig_matrix`](@ref) for the interpolation and end conventions.
"""
kramers_kronig(ωs, zs) = kramers_kronig_matrix(ωs) * zs
