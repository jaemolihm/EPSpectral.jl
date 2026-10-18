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
    apply_interpolation_table!(Y, ys, j, t) -> Y

Apply the table `(j, t)` of [`linear_interpolation_table`](@ref) to the columns of `ys`:
`Y[r, i] = (1 - t[r]) ys[j[r], i] + t[r] ys[j[r] + 1, i]`. A gather broadcast over backend arrays,
so it runs on the host and on the device and is deterministic.
"""
function apply_interpolation_table!(Y, ys, j, t)
    lerp(ys, jr, tr, i) = (1 - tr) * ys[jr, i] + tr * ys[jr + 1, i]
    Y .= lerp.(Ref(ys), j, t, reshape(axes(Y, 2), 1, :))
end

# Throw an `ArgumentError` unless `ωs` has at least two nodes and its consecutive spacings all exceed
# `1e-8 (ωs[end] - ωs[1])`; a smaller spacing is a duplicated node or a few-ulp pair, and the step
# the ladder derives from it would size an enormous shift table.
function _check_frequency_grid(ωs)
    length(ωs) >= 2 || throw(ArgumentError("ωs must have at least two points"))
    gaps = diff(ωs)
    h, j = findmin(gaps)
    h > 1e-8 * (last(ωs) - first(ωs)) || throw(ArgumentError("ωs must be strictly increasing " *
        "with spacings above 1e-8 of its span; ωs[$j] = $(ωs[j]) and ωs[$(j + 1)] = $(ωs[j + 1]) " *
        "are $h apart"))
    nothing
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
        "cell search table of $nbuckets entries for a grid with minimum spacing $h_min over a " *
        "range $(last(ωs) - x0); use a grid without a near-coincident pair of nodes"))
    buckets = [unsafe_trunc(Int, (x - x0) * inv_h) for x in ωs]
    allunique(buckets) || throw(ArgumentError("two grid nodes fall in one bucket"))
    jtab = Int32[searchsortedfirst(buckets, b) - 1 for b in 0:last(buckets)]
    (; jtab = to_device(backend, jtab), x0, inv_h)
end

"""
    _interp_linear_grid(ys, col, x, ωs, ωs_cell_table)

Linear interpolation of column `col` of `ys` (sampled on the sorted grid `ωs`, `size(ys, 1) ==
length(ωs)`) at `x`, held flat outside the grid: `Gridded(Linear())` with `Flat()` extrapolation
in Interpolations.jl. The cell is found from `ωs_cell_table = cell_search_table(backend, ωs)`
(on the same backend as `ωs`): one table read and one comparison. Takes isbits values and arrays,
neither allocates nor throws, so it can be called from a device broadcast.
"""
@inline function _interp_linear_grid(ys, col, x, ωs, ωs_cell_table)
    (; jtab, x0, inv_h) = ωs_cell_table
    n = length(ωs)
    !(x > (@inbounds ωs[1])) && return @inbounds ys[1, col]   # also a NaN x
    x >= (@inbounds ωs[n]) && return @inbounds ys[n, col]
    # ωs[1] < x < ωs[n], so the bucket is in 0:bucket(ωs[n]) and j + 1 <= n.
    j = Int(@inbounds jtab[unsafe_trunc(Int, (x - x0) * inv_h) + 1])
    j += @inbounds ωs[j + 1] <= x
    @inbounds begin
        t = (x - ωs[j]) / (ωs[j + 1] - ωs[j])
        (1 - t) * ys[j, col] + t * ys[j + 1, col]
    end
end

"""
    kramers_kronig(ωs, zs) -> ys

Real part `ys = kramers_kronig_matrix(ωs) * zs` of the retarded function whose imaginary part is `zs` on the
grid `ωs`; see [`kramers_kronig_matrix`](@ref) for the interpolation and end conventions.
"""
kramers_kronig(ωs, zs) = kramers_kronig_matrix(ωs) * zs

"""
    graded_frequency_grid(windows) -> Vector

A grid whose step grows outward, from `windows = [(wmin, wmax, dw), ...]`: window `l` covers
`[wmin, wmax]` with step `dw`. The windows are nested from the smallest to the largest, `wmin`
non-increasing and `wmax` non-decreasing along the list, so the first window has the finest step.
No symmetry about any point is imposed.

All nodes lie on the lattice `wmin_1 + dw_1 ℤ` anchored at the lower end of the first window, as
the ladder ([`run_scgd0_ladder`](@ref)) requires of a grid, so every ratio `dw_l / dw_1` must be
an integer (to `1e-8`). Window `l` contributes the points `wmin_1 + dw_l ℤ` inside `[wmin_l,
wmax_l]`; the grid is the sorted union. Throws an `ArgumentError` for an empty list, a non-positive step, a window
with `wmin >= wmax`, windows that are not nested, or a non-integer ratio.

```julia
julia> print(graded_frequency_grid([(-2.0, 2.0, 1.0), (-6.0, 6.0, 2.0), (-15.0, 15.0, 5.0)]))
[-12.0, -7.0, -6.0, -4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 13.0]

julia> meV = unit_to_aru(:meV); μ = 0.5;   # illustrative: 1 meV within 0.1 eV of μ, 5 meV to 0.3 eV, 20 meV to 1 eV

julia> ωs = graded_frequency_grid([(μ - w * meV, μ + w * meV, dw * meV) for (w, dw) in ((100, 1), (300, 5), (1000, 20))]);
```
"""
function graded_frequency_grid(windows)
    isempty(windows) && throw(ArgumentError("windows must not be empty"))
    wmins, wmaxs, dws = (float.([w[k] for w in windows]) for k in 1:3)
    all(>(0), dws) && all(wmins .< wmaxs) || throw(ArgumentError(
        "every window needs dw > 0 and wmin < wmax, got $windows"))
    (issorted(wmins; rev = true) && issorted(wmaxs)) || throw(ArgumentError(
        "windows must be nested from the smallest to the largest (wmin non-increasing, wmax " *
        "non-decreasing), got $windows"))
    anchor, dw1 = wmins[1], dws[1]
    ps = Int[]
    for l in eachindex(windows)
        ratio = dws[l] / dw1
        abs(ratio - round(ratio)) <= 1e-8 || throw(ArgumentError(
            "dw[$l] = $(dws[l]) is not an integer multiple of dw[1] = $dw1 (ratio $ratio)"))
        r = round(Int, ratio)
        # The lattice indices p of window l are the multiples of r with wmin <= anchor + dw1 p <= wmax.
        p_lo = ceil(Int, (wmins[l] - anchor) / dw1 - 1e-8)
        p_hi = floor(Int, (wmaxs[l] - anchor) / dw1 + 1e-8)
        append!(ps, cld(p_lo, r) * r:r:p_hi)
    end
    anchor .+ dw1 .* sort!(unique!(ps))
end
