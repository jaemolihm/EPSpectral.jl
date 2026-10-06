# Self-consistent Fan-Migdal electron self-energy, G self-consistent and D bare (scGD0).
#
# `run_scgd0` = `_setup_scgd0` (the e-ph coupling, extracted once into host RAM, and everything else
# that does not change between iterations) + `_loop_scgd0` (the temperature-dependent iteration).
#
# Index legend: ω = output frequency (`ωs`, any sorted grid), ωd = dense uniform frequency
# (`ωs_dense`), ν = phonon mode, i = outer state (irreducible BZ, in the window), f = inner state
# (full BZ, in the window), T = temperature (`occ.Tlist`).

"""
    SCGD0Result

The result of [`run_scgd0`](@ref), one entry per temperature `T = occ.Tlist`:
- `ωs`: the output frequencies.
- `Σ[ω, i, T]`: the self-energy of the outer states `el_i` (a `BandStates` on the irreducible BZ).
- `μ[T]`: the chemical potential after the last iteration.
- `Tlist`, `nlist`: the temperatures and dopings of the `occ` it was run with.
- `converged[T]`: whether `max_{ω,i} |ΔΣ| < tol` was reached within `maxiter`.
- `history`: per iteration `n` and temperature, `err[T, n] = max_{ω,i} |ΔΣ|`, `μ[T, n]` and
  `min_spectral_weight[T, n] = min_i ∫A_i` (`NaN` once `T` has stopped iterating), `wall[n]` in
  seconds, and `iter_frozen[T]`, the iteration at which `T` converged (0 if it did not).
"""
struct SCGD0Result{FT, BS <: BandStates{FT}, H <: NamedTuple}
    ωs::Vector{FT}
    Σ::Array{Complex{FT}, 3}
    μ::Vector{FT}
    el_i::BS
    Tlist::Vector{FT}
    nlist::Vector{FT}
    converged::Vector{Bool}
    history::H
end

"""
    run_scgd0(model, nk, window; occ, ωs, kwargs...) -> SCGD0Result

Self-consistent Fan-Migdal self-energy Σ_i(ω) with G self-consistent and D bare (scGD0), and the
chemical potential μ, for the electron states in the energy `window` on the `nk` grid (outer
states reduced to the irreducible BZ by `symmetry`), at every temperature of `occ`.

Each iteration computes Im Σ at the current μ from the Fan-Migdal sum over the window states, with
the input Σ linearly interpolated onto a uniform dense grid `ωs_dense` and clamped to
`Im Σ <= -η_min`; Re Σ by [`kramers_kronig`](@ref); averages Σ over degenerate multiplets
(`electron_degen_cutoff`); and then re-solves μ from the spectral functions of the unclamped Σ for
the carrier count `occ.nlist + occ.nelec`. A temperature stops iterating once
`max_{ω,i} |ΔΣ| < tol`; at `maxiter` the last Σ and μ are returned, unconverged ones flagged.

Keywords:
- `occ`: `ElectronOccupationParams` with `occ_type = :FermiDirac`. `occ.μlist`, if given, is the
  starting μ; otherwise μ starts from the bare window bands. `occ` is not modified.
- `ωs`: output grid, strictly increasing. `dω_dense = minimum(diff(ωs))` is the step of
  `ωs_dense = range(extrema(ωs)...; step = dω_dense)`.
- `η_min = 0` (scalar or per temperature), `η_init = 5 meV`: Σ starts from `-im * η_init`, so
  `maxiter = 1` is the one-shot G0D0.
- `maxiter = 20`, `tol = 0.05 meV`.
- `fix_μ = false`: `true` keeps μ at `occ.μlist`, which must then be given.
- `Σ_init`: `nothing`, or a starting `Σ[ω, i, T]` on the same `ωs`, states and temperatures (the
  `Σ` of an `SCGD0Result`), for a restart together with `occ.μlist`.
- `backend = CPUBackend()`, `symmetry = model.symmetry`, `verbosity = 1`.
- `gpu_tile`: inner states per chunk of the Fan-Migdal sum; `nothing` takes chunks of at most
  1 GB of phonon frequencies.
- `ω_acoustic`: modes below it are skipped. `electron_degen_cutoff`: multiplet tolerance.
- Other keywords are passed to `run_eph_over_k_and_kq`.
"""
function run_scgd0(model::Model{FT}, nk, window; occ::ElectronOccupationParams, ωs,
        dω_dense = minimum(diff(ωs)), η_min = 0, η_init = 5 * unit_to_aru(:meV), maxiter = 20,
        tol = 0.05 * unit_to_aru(:meV), backend = CPUBackend(), symmetry = model.symmetry,
        verbosity = 1, gpu_tile = nothing, Σ_init = nothing, fix_μ = false,
        ω_acoustic = ElectronPhonon.omega_acoustic,
        electron_degen_cutoff = ElectronPhonon.electron_degen_cutoff, ep_kwargs...) where {FT}
    nk isa NTuple{3, Integer} ||
        throw(ArgumentError("nk must be a grid size (n1, n2, n3), got $nk"))
    occ.occ_type === :FermiDirac || throw(ArgumentError(
        "run_scgd0 uses the Fermi-Dirac occupation only, got occ.occ_type = $(occ.occ_type)"))
    occ.type ∈ (:Metal, :Semiconductor) || throw(ArgumentError(
        "occ.type must be :Metal or :Semiconductor, got $(occ.type)"))
    (length(ωs) >= 2 && all(>(0), diff(ωs))) || throw(ArgumentError(
        "ωs must be strictly increasing with at least two points"))
    fix_μ && !chemical_potential_is_computed(occ) && throw(ArgumentError(
        "fix_μ = true keeps μ at occ.μlist, which is not given"))
    η_min = η_min isa Number ? fill(FT(η_min), length(occ)) : collect(FT, η_min)
    length(η_min) == length(occ) || throw(ArgumentError("η_min must be a scalar or have one " *
        "entry per temperature ($(length(occ))), got $(length(η_min))"))
    (gpu_tile === nothing || gpu_tile >= 1) || throw(ArgumentError(
        "gpu_tile must be nothing or a positive number of inner states, got $gpu_tile"))
    maxiter >= 1 || throw(ArgumentError("maxiter must be at least 1, got $maxiter"))

    setup = _setup_scgd0(model, nk, window, occ; ωs, dω_dense, backend, symmetry, verbosity,
        electron_degen_cutoff, ep_kwargs...)
    (; Σ, μ, converged, history) = _loop_scgd0(setup, occ; η_min, η_init, maxiter, tol, backend,
        gpu_tile, Σ_init, fix_μ, ω_acoustic, verbosity)
    SCGD0Result(collect(FT, ωs), Σ, μ, setup.el_i, collect(FT, occ.Tlist), collect(FT, occ.nlist),
        converged, history)
end

# Everything the iteration needs that does not change between iterations, on `backend` where the
# iteration reads it on the device: the states, the coupling g2[ν, i, f] (host), the full-grid
# phonon frequencies, the grids and maps, the multiplets and the starting μ.
function _setup_scgd0(model::Model{FT}, nk, window, occ; ωs, dω_dense, backend, symmetry,
        verbosity, electron_degen_cutoff, ep_kwargs...) where {FT}
    # Outer states on the irreducible BZ, inner states on the full BZ, coupling extracted once.
    # Each run diagonalizes on its own backend, so the electron gauge inside a multiplet is the
    # backend's.
    sel_i = filter_electron_states(nk, model, window; symmetry, backend)
    sel_i.n > 0 ||
        throw(ArgumentError("the window $window holds no electron state on the grid $nk"))
    calc = ElectronPhonon.G2Calculator{FT}(; model.nmodes, store_ωq = false)
    run_eph_over_k_and_kq(model, sel_i, unfold_band_states(sel_i, symmetry); ep_kwargs...,
        calculators = [calc], symmetry = nothing, backend, verbosity)
    (; el_i, el_f, g2) = calc
    f_to_i = find_unfolding_indices(el_i, el_f, symmetry)

    # Phonon frequencies on the full q grid, read by the kernel at the integer hash of k_f - k_i.
    qpts = kpoints_grid(nk)
    all(iq -> ElectronPhonon._hash_xk(qpts.vectors[iq], nk, zero(qpts.vectors[iq])) == iq - 1,
        1:qpts.n) || error("kpoints_grid order is not the integer-hash order the ωq gather assumes")
    ωph = compute_phonon_states_batched(model, qpts, [:e]; backend).e
    grid_coords(el) = to_device(backend,
        [ElectronPhonon._grid_coords_reduced(xk, nk, el_i.kpts.shift) for xk in state_xks(el)])

    # Degenerate multiplets of the outer states, with at least two members: at one k, the levels
    # chained by gaps below `electron_degen_cutoff`.
    multiplets = Vector{Int}[]
    order = sortperm(collect(zip(el_i.iks, el_i.es)))
    group = [first(order)]
    for i in @view order[2:end]
        j = group[end]
        if el_i.iks[i] == el_i.iks[j] && el_i.es[i] - el_i.es[j] < electron_degen_cutoff
            # Degenerate with the previous level.
            push!(group, i)
        else
            # A new level: close the previous group.
            length(group) > 1 && push!(multiplets, group)
            group = [i]
        end
    end
    length(group) > 1 && push!(multiplets, group)

    ωs_dense = range(extrema(ωs)...; step = dω_dense)

    # Starting μ: given, or from the bare window bands on a copy, so `occ` is not modified.
    μ_start = if chemical_potential_is_computed(occ)
        collect(FT, occ.μlist)
    else
        ElectronPhonon.bte_compute_μ!(deepcopy(occ), el_i; do_print = verbosity > 0)
    end

    # On a host backend `to_device` aliases the states' own arrays; the loop only reads them.
    (; el_i, g2, ωph, cs_i = grid_coords(el_i), cs_f = grid_coords(el_f), ngrid = nk,
       ε_i = to_device(backend, el_i.es), w_i = to_device(backend, state_weights(el_i)),
       ε_f = to_device(backend, el_f.es), w_f = to_device(backend, state_weights(el_f)),
       f_to_i = to_device(backend, f_to_i), W = sum(state_weights(el_i)),
       nstates_base = el_i.nstates_base, multiplets,
       ωs = to_device(backend, collect(FT, ωs)), ωs_dense,
       K = to_device(backend, kk_matrix(ωs)), P = to_device(backend, interp_matrix(ωs, ωs_dense)),
       μ_start)
end

# The scGD0 iteration on the setup `s` (see `_setup_scgd0`; the fields read here can also be built
# by hand), starting from `s.μ_start`. Returns the host `Σ[ω, i, T]`, `μ[T]`, `converged[T]` and the
# `history` of `SCGD0Result`.
function _loop_scgd0(s, occ; η_min, η_init, maxiter, tol, backend, gpu_tile, Σ_init, fix_μ,
        ω_acoustic, verbosity)
    (; g2, ωph, cs_i, cs_f, ngrid, ε_i, w_i, ε_f, w_f, f_to_i, W, nstates_base, multiplets, ωs,
       ωs_dense, K, P) = s
    FT = eltype(g2)
    nm, ni, nf = size(g2)
    nω, nωd, nT = length(ωs), length(ωs_dense), length(occ)
    ωd0, dωd = first(ωs_dense), step(ωs_dense)
    bracket = extrema(ωs_dense) .+ (-FT(0.1), FT(0.1))   # μ search interval (Ry)

    if Σ_init === nothing
        # The scFM start Σ = -iη_init: iteration 1 is the one-shot G0D0.
        Σ = fill!(alloc(backend, Complex{FT}, nω, ni, nT), -im * FT(η_init))
    else
        # A restart.
        size(Σ_init) == (nω, ni, nT) || throw(ArgumentError(
            "Σ_init must have size (nω, n_i, nT) = $((nω, ni, nT)), got $(size(Σ_init))"))
        Σ = to_device_copy(backend, Σ_init)
    end
    Σ_new = copy(Σ)
    Σd = alloc(backend, Complex{FT}, nωd, ni)          # P·Σ of one temperature
    Σin = alloc(backend, Complex{FT}, nωd, ni, nT)     # clamped P·Σ, the Fan-Migdal input
    ImΣ = alloc(backend, FT, nω, ni, nT)
    ReΣ = alloc(backend, FT, nω, ni)
    A = alloc(backend, FT, nωd, ni)                    # spectral functions A_i(ωd)
    D = alloc(backend, FT, nωd)                        # D_T(ωd) = Σ_i w_i A_i(ωd) dωd
    # One partition of the inner states for every iteration: chunks of at most 1 GB of ωq.
    nf_chunk = gpu_tile === nothing ? clamp(fld(2^30, sizeof(FT) * nm * ni), 1, nf) :
                                      min(gpu_tile, nf)
    ωq_buf = alloc(backend, FT, nm, ni, nf_chunk)
    # Read-only for the whole run, so a host alias of the caller's g2 is safe.
    g2_dev = to_device(backend, g2)

    μ = collect(FT, s.μ_start)
    active = trues(nT)
    hist_err, hist_μ, hist_minA = (fill(FT(NaN), nT, maxiter) for _ in 1:3)
    wall = zeros(maxiter)
    iter_frozen = zeros(Int, nT)
    niter = 0
    for iter in 1:maxiter
        t0 = time()
        iTs = findall(active)

        # 1. Input self-energy on the dense grid, clamped to Im Σ <= -η_min.
        for iT in iTs
            mul!(Σd, P, view(Σ, :, :, iT))
            view(Σin, :, :, iT) .= complex.(real.(Σd), min.(imag.(Σd), -η_min[iT]))
        end
        # 2-3. Im Σ: the Fan-Migdal sum, f-chunks outside so a chunk's ωq is gathered once.
        fill!(ImΣ, 0)
        for fs in Iterators.partition(1:nf, nf_chunk)
            ωq_t = view(ωq_buf, :, :, 1:length(fs))
            gather_ωq!(ωq_t, ωph, cs_i, view(cs_f, fs), ngrid)
            g2_t = view(g2_dev, :, :, fs)
            for iT in iTs
                fm_imsigma_tile!(view(ImΣ, :, :, iT), g2_t, ωq_t, ωs, view(ε_f, fs),
                    view(f_to_i, fs), view(Σin, :, :, iT), ωd0, dωd, μ[iT], occ.Tlist[iT],
                    view(w_f, fs), ω_acoustic)
            end
        end
        # 4. Re Σ by Kramers-Kronig.
        for iT in iTs
            mul!(ReΣ, K, view(ImΣ, :, :, iT))
            view(Σ_new, :, :, iT) .= complex.(ReΣ, view(ImΣ, :, :, iT))
        end
        # 5. Average over degenerate multiplets, on the host.
        if !isempty(multiplets)
            Σ_host = Array(Σ_new)
            for iT in iTs, m in multiplets
                @views Σ_host[:, m, iT] .= sum(Σ_host[:, m, iT]; dims = 2) ./ length(m)
            end
            copyto!(Σ_new, Σ_host)
        end
        for iT in iTs
            Σ_new_T = view(Σ_new, :, :, iT)
            # 6. Spectral DOS from the unclamped Σ.
            mul!(Σd, P, Σ_new_T)
            A .= .-imag.(inv.(ωs_dense .- transpose(ε_i) .- Σd)) ./ π
            mul!(D, A, w_i, dωd, 0)
            hist_minA[iT, iter] = minimum(sum(A; dims = 1)) * dωd
            # 7. Chemical potential, on the host.
            if !fix_μ
                D_host = Array(D)
                T = occ.Tlist[iT]
                count_holes = occ.type === :Semiconductor && occ.nlist[iT] < 0
                n_target = occ.nlist[iT] + occ.nelec
                excess(μ_) = compute_ncarrier_spectral(μ_, T, ωs_dense, D_host, W; count_holes,
                    nstates_base, occ.spin_degeneracy) - n_target
                excess(bracket[1]) * excess(bracket[2]) > 0 && throw(ArgumentError(
                    "no chemical potential in $bracket gives $n_target electrons per cell at " *
                    "T = $T: the window holds too few states on one side of the gap. Pass " *
                    "occ.μlist and fix_μ = true."))
                μ[iT] = find_zero(excess, bracket)
            end
            # 8-9. Convergence; a converged temperature keeps its Σ and μ and stops iterating.
            hist_err[iT, iter] = maximum(abs.(Σ_new_T .- view(Σ, :, :, iT)))
            hist_μ[iT, iter] = μ[iT]
            view(Σ, :, :, iT) .= Σ_new_T
            if hist_err[iT, iter] < tol
                active[iT] = false
                iter_frozen[iT] = iter
            end
        end
        wall[iter] = time() - t0
        niter = iter
        verbosity > 0 &&
            @info "scGD0 iteration $iter" err = hist_err[iTs, iter] μ = μ[iTs] wall = wall[iter]
        any(active) || break
    end
    # 10. Unconverged temperatures keep the last Σ and μ. `maxiter = 1` is the one-shot.
    any(active) && maxiter > 1 && @warn "scGD0 not converged after maxiter = $maxiter " *
        "iterations at T = $(occ.Tlist[active])"
    history = (; err = hist_err[:, 1:niter], μ = hist_μ[:, 1:niter],
        min_spectral_weight = hist_minA[:, 1:niter], wall = wall[1:niter], iter_frozen)
    (; Σ = Array(Σ), μ, converged = Vector{Bool}(.!active), history)
end

"""
    compute_ncarrier_spectral(μ, T, ωs_dense, D_T, W; count_holes, nstates_base, spin_degeneracy)

Electrons per cell at chemical potential `μ` and temperature `T`, from the spectral DOS
`D_T[ωd] = Σ_i w_i A_i(ωd) dω` of the window states on `ωs_dense`, with `W = Σ_i w_i`:

    count_holes = false:  spin_degeneracy · (Σ_ωd D_T f(ωd - μ)           + nstates_base)
    count_holes = true:   spin_degeneracy · (W - Σ_ωd D_T (1 - f(ωd - μ)) + nstates_base)

The second form, for a hole-doped semiconductor, counts the window states as filled minus their
holes, so the result does not assume ∫A_i = 1.
"""
function compute_ncarrier_spectral(μ, T, ωs_dense, D_T, W; count_holes, nstates_base,
        spin_degeneracy)
    n_window = count_holes ? W - compute_ncarrier_hole(μ, T, ωs_dense, D_T) :
                             compute_ncarrier(μ, T, ωs_dense, D_T)
    (n_window + nstates_base) * spin_degeneracy
end

"""
    save_scgd0(filename, res::SCGD0Result) -> filename

Write `res` to a JLD2 file: `el_i` as the `BandStates` itself, the other fields as arrays, and
`history` as the group `history`. Read it back with [`load_scgd0`](@ref).
"""
function save_scgd0(filename, res::SCGD0Result)
    jldopen(filename, "w") do file
        for name in (:ωs, :Σ, :μ, :el_i, :Tlist, :nlist, :converged)
            file[String(name)] = getfield(res, name)
        end
        for (name, value) in pairs(res.history)
            file["history/$name"] = value
        end
    end
    filename
end

"""
    load_scgd0(filename) -> SCGD0Result

Read an [`SCGD0Result`](@ref) written by [`save_scgd0`](@ref).
"""
function load_scgd0(filename)
    jldopen(filename, "r") do file
        history = (; (Symbol(name) => file["history/$name"]
                      for name in ("err", "μ", "min_spectral_weight", "wall", "iter_frozen"))...)
        SCGD0Result(file["ωs"], file["Σ"], file["μ"], file["el_i"], file["Tlist"], file["nlist"],
            file["converged"], history)
    end
end
