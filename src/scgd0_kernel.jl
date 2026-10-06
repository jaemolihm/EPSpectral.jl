# The Fan-Migdal sum of the scGD0 iteration and the phonon-frequency gather that feeds it.
#
# Index legend: ω = output frequency (`ωs`), ν = phonon mode, i = outer (IBZ) state, f = inner
# (full-BZ) state, ωd = dense uniform frequency (`ωs_dense`).
#
# Both are broadcasts over backend arrays with a scalar element function, so the same code runs on
# the host and on the device; there is no custom kernel. The element functions take scalars and
# isbits values only, read arrays `@inbounds` at indices established by their callers, and neither
# allocate nor throw.

"""
    _lerp_flat(Σin, col, x, ωd0, dωd)

Linear interpolation of column `col` of `Σin` (sampled on the uniform grid `ωd0 + (j-1) dωd`,
`j in axes(Σin, 1)`) at `x`, held flat outside the grid: `BSpline(Linear())` with `Flat()`
extrapolation in Interpolations.jl.
"""
@inline function _lerp_flat(Σin, col, x, ωd0, dωd)
    nωd = size(Σin, 1)
    t = (x - ωd0) / dωd
    !(t > 0) && return @inbounds Σin[1, col]   # also a NaN x
    t >= nωd - 1 && return @inbounds Σin[nωd, col]
    j = unsafe_trunc(Int, t)   # floor, since 0 < t < nωd - 1
    r = t - j
    @inbounds (1 - r) * Σin[j + 1, col] + r * Σin[j + 2, col]
end

"""
    fm_term(ω, g2, ωq, ε_f, i_f, Σin, ωd0, dωd, μ, T, w_f, ω_acoustic)

One `(ω, ν, i, f)` summand of the Fan-Migdal Im Σ_i(ω), with G self-consistent and D bare:

    w_f g2 { Im[1/(ω + ωq - ε_f - Σin_f(ω + ωq))] (n(ωq) + f(ω + ωq - μ))
           + Im[1/(ω - ωq - ε_f - Σin_f(ω - ωq))] (n(ωq) + 1 - f(ω - ωq - μ)) }.

`g2 = |g|²/(2ωq)` is the stored coupling, `Σin` the clamped input self-energy on the dense grid
with `Σin_f` its column `i_f` (the IBZ representative of `f`). There is no factor π: Im G = -πA.
A mode with `ωq < ω_acoustic` contributes zero, tested before `g2` is read, since the stored
`g2` is Inf or huge at Γ-acoustic and negative for an imaginary mode, and a multiply-by-mask would
give `Inf * 0 = NaN`.
"""
@inline function fm_term(ω, g2, ωq, ε_f, i_f, Σin, ωd0, dωd, μ, T, w_f, ω_acoustic)
    ωq < ω_acoustic && return zero(g2)
    n_q = occ_boson(ωq, T)
    Σ_abs = _lerp_flat(Σin, i_f, ω + ωq, ωd0, dωd)
    Σ_emi = _lerp_flat(Σin, i_f, ω - ωq, ωd0, dωd)
    ImG_abs = imag(inv(ω + ωq - ε_f - Σ_abs))
    ImG_emi = imag(inv(ω - ωq - ε_f - Σ_emi))
    w_f * g2 * (ImG_abs * (n_q + occ_fermion(ω + ωq - μ, T))
              + ImG_emi * (n_q + 1 - occ_fermion(ω - ωq - μ, T)))
end

"""
    fm_imsigma_tile!(ImΣ_T, g2_t, ωq_t, ωs, ε_f_t, f_to_i_t, Σin_T, ωd0, dωd, μ, T, w_f_t,
                     ω_acoustic)

Add the Fan-Migdal Im Σ of one tile of inner states `f` to `ImΣ_T[ω, i]`, at one temperature `T`
and chemical potential `μ`. `g2_t[ν, i, f]` and `ωq_t[ν, i, f]` are the tile's coupling and phonon
frequencies; `ε_f_t`, `f_to_i_t`, `w_f_t` its per-state energies, IBZ representatives and weights;
`Σin_T[ωd, i]` the clamped input self-energy on the dense grid (see [`fm_term`](@ref)).

The sum over `(ν, f)` is a `sum(…; dims = (2, 4))` over a lazy `(ω, ν, i, f)` broadcast of
`fm_term`: nothing 4D is materialized, and the reduction order is fixed, so the result is
deterministic. A host `Broadcasted` reduction runs serially, so on the host the i axis is split into
chunks reduced in parallel; chunks write disjoint i, so the result does not depend on the thread
count.
"""
function fm_imsigma_tile!(ImΣ_T, g2_t, ωq_t, ωs, ε_f_t, f_to_i_t, Σin_T, ωd0, dωd, μ, T,
        w_f_t, ω_acoustic)
    args = (ωs, ε_f_t, f_to_i_t, Σin_T, ωd0, dωd, μ, T, w_f_t, ω_acoustic)
    if ElectronPhonon.on_backend(CPUBackend(), ImΣ_T)
        # Host: thread over chunks of i.
        tforeach(index_chunks(axes(ImΣ_T, 2); n = 2 * Threads.nthreads()); chunking = false) do is
            @views _fm_imsigma_reduce!(ImΣ_T[:, is], g2_t[:, is, :], ωq_t[:, is, :], args...)
        end
    else
        # Device: one reduction over the whole tile.
        _fm_imsigma_reduce!(ImΣ_T, g2_t, ωq_t, args...)
    end
    ImΣ_T
end

function _fm_imsigma_reduce!(ImΣ_T, g2_t, ωq_t, ωs, ε_f_t, f_to_i_t, Σin_T, ωd0, dωd, μ, T,
        w_f_t, ω_acoustic)
    nω, ni = size(ImΣ_T)
    nm, nf = size(g2_t, 1), size(g2_t, 3)
    per_f(x) = reshape(x, 1, 1, 1, nf)
    bc = Broadcast.instantiate(Broadcast.broadcasted(fm_term,
        reshape(ωs, nω, 1, 1, 1), reshape(g2_t, 1, nm, ni, nf), reshape(ωq_t, 1, nm, ni, nf),
        per_f(ε_f_t), per_f(f_to_i_t), Ref(Σin_T), ωd0, dωd, μ, T, per_f(w_f_t), ω_acoustic))
    # `init` is required: a host `Broadcasted` has no `reducedim_init` method without it.
    ImΣ_T .+= reshape(sum(bc; dims = (2, 4), init = zero(eltype(ImΣ_T))), nω, ni)
    ImΣ_T
end

# Phonon frequency of mode ν at q = k_f - k_i, read from the full-grid table `ωph[ν, iq]` at
# `iq = hash + 1`. `c_i`, `c_f` are the grid coordinates of k_i and k_f reduced into `0:ng-1`, so
# their difference is in `(-ng, ng)` and one fold reduces it; the hash is `_fill_iqs!`'s, and the
# table is in `kpoints_grid` order, whose point `iq` has hash `iq - 1`.
@inline function _ωq_at_pair(ωph, ν, c_i, c_f, ng1, ng2, ng3)
    h1 = c_f[1] - c_i[1]; h1 += ifelse(h1 < 0, ng1, 0)
    h2 = c_f[2] - c_i[2]; h2 += ifelse(h2 < 0, ng2, 0)
    h3 = c_f[3] - c_i[3]; h3 += ifelse(h3 < 0, ng3, 0)
    @inbounds ωph[ν, (h1 * ng2 + h2) * ng3 + h3 + 1]
end

"""
    gather_ωq!(ωq_t, ωph, cs_i, cs_f_t, ngrid)

Fill `ωq_t[ν, i, f]` with the frequency of mode `ν` at `q = k_f - k_i` from the full-grid phonon
table `ωph[ν, iq]` (`kpoints_grid(ngrid)` order). `cs_i`, `cs_f_t` hold the per-state integer grid
coordinates of the outer states and of a tile of inner states, reduced into `0:ngrid[d]-1`.
"""
function gather_ωq!(ωq_t, ωph, cs_i, cs_f_t, ngrid)
    nm, ni, nf = size(ωq_t)
    ωq_t .= _ωq_at_pair.(Ref(ωph), reshape(1:nm, nm, 1, 1), reshape(cs_i, 1, ni, 1),
        reshape(cs_f_t, 1, 1, nf), ngrid[1], ngrid[2], ngrid[3])
end
