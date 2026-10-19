# Self-consistent Fan-Migdal electron self-energy, G self-consistent and D bare (scGD0).
#
# `run_scgd0` = `_setup_scgd0` (the e-ph coupling, extracted once into host RAM, and everything else
# that does not change between iterations) + `_loop_scgd0` (the temperature-dependent iteration).
#
# Index legend: ω = output frequency (node of the `FrequencyGrid` `ωs`), ωa = node of the
# `FrequencyGrid` `ωs_spectral` on which the spectral functions are integrated, ν = phonon mode,
# i = outer state (irreducible BZ, in the window), f = inner state (full BZ, in the window),
# T = temperature (`occ.Tlist`).

"""
    SCGD0Result

The result of [`run_scgd0`](@ref), one entry per temperature `T = occ.Tlist`:
- `ωs`, `ωs_spectral`: the output frequencies and the frequencies on which the spectral functions
  were integrated, [`FrequencyGrid`](@ref)s.
- `Σ[ω, i, T]`: the self-energy of the outer states `el_i` (a `BandStates` on the irreducible BZ).
- `μ[T]`: the chemical potential after the last iteration.
- `Tlist`, `nlist`: the temperatures and dopings of the `occ` it was run with.
- `converged[T]`: whether `max_{ω,i} |ΔΣ| < tol` was reached within `maxiter`.
- `history`: per iteration `n` and temperature, `err[T, n] = max_{ω,i} |ΔΣ|`, `μ[T, n]` and
  `min_spectral_weight[T, n] = min_i ∫A_i` (`NaN` once `T` has stopped iterating), `wall[n]` in
  seconds, and `iter_frozen[T]`, the iteration at which `T` converged (0 if it did not).
"""
struct SCGD0Result{FT, BS <: BandStates{FT}, H <: NamedTuple}
    ωs::FrequencyGrid{FT}
    ωs_spectral::FrequencyGrid{FT}
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
the input Σ clamped to `Im Σ <= -η_min` and read at ω ± ωq by linear interpolation on `ωs`; Re Σ by
the spline Kramers-Kronig map [`kramers_kronig`](@ref); averages Σ over degenerate multiplets
(`electron_degen_cutoff`); and then re-solves μ from the spectral functions of the unclamped Σ,
integrated on `ωs_spectral` with the trapezoid rule, for the carrier count
`occ.nlist + occ.nelec`. A temperature stops iterating once `max_{ω,i} |ΔΣ| < tol`; at
`maxiter` the last Σ and μ are returned, unconverged ones flagged.

Keywords:
- `occ`: `ElectronOccupationParams` with `occ_type = :FermiDirac`. `occ.μlist`, if given, is the
  starting μ; otherwise μ starts from the bare window bands. `occ` is not modified.
- `ωs`: output grid, a [`FrequencyGrid`](@ref) ([`graded_frequency_grid`](@ref) builds a graded one;
  `FrequencyGrid(ωs::AbstractVector)` converts a vector of nodes).
- `ωs_spectral = nothing`: the [`FrequencyGrid`](@ref) on which the spectral functions A_i(ω) that
  give μ and `min_i ∫A_i` are integrated, with Σ linearly interpolated onto it and held flat
  outside `ωs`. Any grid, graded too, that covers the span of `ωs` to within `ωs.dω_min`; a grid
  reaching past `ωs` is accepted with a warning.
  `nothing` takes the uniform grid on the span of `ωs` with step `ωs.dω_min`.
- `η_min = 0` (scalar or per temperature), `η_init = 5 meV`: Σ starts from `-im * η_init`, so
  `maxiter = 1` is the one-shot G0D0.
- `maxiter = 20`, `tol = 0.05 meV`.
- `fix_μ = false`: `true` keeps μ at `occ.μlist`, which must then be given.
- `Σ_init`: `nothing`, or a starting `Σ[ω, i, T]` on the same `ωs`, states and temperatures (the
  `Σ` of an `SCGD0Result`), for a restart together with `occ.μlist`.
- `backend = CPUBackend()`, `symmetry = model.symmetry`, `verbosity = 1`.
- `gpu_tile`: inner states per chunk of the Fan-Migdal sum; `nothing` takes chunks of at most
  1 GB of phonon frequencies and keeps g2 on the device if it fits. On a device backend an Int
  also keeps g2 in host RAM, copying one chunk at a time.
- `ω_acoustic`: modes below it are skipped. `electron_degen_cutoff`: multiplet tolerance.
- Other keywords are passed to `run_eph_over_k_and_kq`.
"""
function run_scgd0(model::Model{FT}, nk, window; occ::ElectronOccupationParams, ωs,
        ωs_spectral = nothing, η_min = 0, η_init = 5 * unit_to_aru(:meV), maxiter = 20,
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
    ωs isa FrequencyGrid{FT} || throw(ArgumentError("ωs must be a FrequencyGrid{$FT}, got a " *
        "$(typeof(ωs)); build one with `graded_frequency_grid` or `FrequencyGrid(ωs::AbstractVector)`"))
    if ωs_spectral === nothing
        # The uniform grid on the span of ωs with its smallest spacing.
        nω_spectral = length(range(extrema(ωs)...; step = ωs.dω_min))
        ωs_spectral = FrequencyGrid(first(ωs), ωs.dω_min, 0:nω_spectral - 1)
    else
        # A given grid: of the right type, and covering ωs (else μ drops spectral weight).
        ωs_spectral isa FrequencyGrid{FT} || throw(ArgumentError("ωs_spectral must be a " *
            "FrequencyGrid{$FT} or nothing, got a $(typeof(ωs_spectral))"))
        (first(ωs_spectral) <= first(ωs) + ωs.dω_min &&
            last(ωs_spectral) >= last(ωs) - ωs.dω_min) || throw(ArgumentError(
            "ωs_spectral spans [$(first(ωs_spectral)), $(last(ωs_spectral))], which does not " *
            "cover ωs = [$(first(ωs)), $(last(ωs))] to within ωs.dω_min"))
        # Allowed past the ends of ωs, where Σ is held at its end value.
        (first(ωs_spectral) < first(ωs) - ωs.dω_min || last(ωs_spectral) > last(ωs) + ωs.dω_min) &&
            @warn "ωs_spectral spans [$(first(ωs_spectral)), $(last(ωs_spectral))], past ωs = " *
                "[$(first(ωs)), $(last(ωs))]; Σ is held at its end values outside ωs"
    end
    fix_μ && !chemical_potential_is_computed(occ) && throw(ArgumentError(
        "fix_μ = true keeps μ at occ.μlist, which is not given"))
    η_min = η_min isa Number ? fill(FT(η_min), length(occ)) : collect(FT, η_min)
    length(η_min) == length(occ) || throw(ArgumentError("η_min must be a scalar or have one " *
        "entry per temperature ($(length(occ))), got $(length(η_min))"))
    (gpu_tile === nothing || gpu_tile >= 1) || throw(ArgumentError(
        "gpu_tile must be nothing or a positive number of inner states, got $gpu_tile"))
    maxiter >= 1 || throw(ArgumentError("maxiter must be at least 1, got $maxiter"))

    setup = _setup_scgd0(model, nk, window, occ; ωs, ωs_spectral, backend, symmetry,
        verbosity, electron_degen_cutoff, ep_kwargs...)
    (; Σ, μ, converged, history) = _loop_scgd0(setup, occ; η_min, η_init, maxiter, tol, backend,
        gpu_tile, Σ_init, fix_μ, ω_acoustic, verbosity)
    SCGD0Result(ωs, ωs_spectral, Σ, μ, setup.el_i, collect(FT, occ.Tlist), collect(FT, occ.nlist),
        converged, history)
end

# Everything the iteration needs that does not change between iterations, on `backend` where the
# iteration reads it on the device: the states, the coupling g2[ν, i, f] (host), the run's phonon
# table and q index with the states' k indices, the grids and maps, the multiplets and the
# starting μ.
function _setup_scgd0(model::Model{FT}, nk, window, occ; ωs::FrequencyGrid,
        ωs_spectral::FrequencyGrid, backend, symmetry, verbosity, electron_degen_cutoff,
        ep_kwargs...) where {FT}
    # Outer states on the irreducible BZ, inner states on the full BZ, coupling extracted once.
    # Each run diagonalizes on its own backend, so the electron gauge inside a multiplet is the
    # backend's. The extraction decides whether its output fits the device from the free device
    # memory, which an earlier call's pooled buffers hide until they are returned to the driver.
    ElectronPhonon.reclaim_device_memory(backend)
    sel_i = filter_electron_states(nk, model, window; symmetry, backend)
    sel_i.n > 0 ||
        throw(ArgumentError("the window $window holds no electron state on the grid $nk"))
    calc = ElectronPhonon.G2Calculator{FT}(; model.nmodes)
    run_eph_over_k_and_kq(model, sel_i, unfold_band_states(sel_i, symmetry); ep_kwargs...,
        calculators = [calc], symmetry = nothing, backend, verbosity)
    (; el_i, el_f, g2) = calc
    f_to_i = find_unfolding_indices(el_i, el_f, symmetry)

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

    ωs_cell_table = cell_search_table(backend, ωs)
    j_spectral, t_spectral = linear_interpolation_table(ωs, ωs_spectral)

    # Starting μ: given, or from the bare window bands on a copy, so `occ` is not modified.
    μ_start = if chemical_potential_is_computed(occ)
        collect(FT, occ.μlist)
    else
        ElectronPhonon.bte_compute_μ!(deepcopy(occ), el_i; do_print = verbosity > 0)
    end

    # On a host backend `to_device` aliases the states' own arrays; the loop only reads them.
    (; el_i, el_f, g2, ωph = to_device(backend, calc.ωph), iq_kk = to_device(backend, calc.iq_kk),
       iks_i = to_device(backend, el_i.iks), iks_f = to_device(backend, el_f.iks),
       ε_i = to_device(backend, el_i.es), w_i = to_device(backend, state_weights(el_i)),
       ε_f = to_device(backend, el_f.es), w_f = to_device(backend, state_weights(el_f)),
       f_to_i = to_device(backend, f_to_i), W = sum(state_weights(el_i)),
       nstates_base = el_i.nstates_base, multiplets,
       ωs = to_device(backend, collect(FT, ωs)), ωs_cell_table,
       ωs_spectral, ωs_spectral_dev = to_device(backend, collect(FT, ωs_spectral)),
       ωs_spectral_weights = to_device(backend, trapezoid_weights(ωs_spectral)),
       j_spectral = to_device(backend, j_spectral), t_spectral = to_device(backend, t_spectral),
       KK = to_device(backend, kramers_kronig_matrix(ωs)), μ_start)
end

# The scGD0 iteration on `setup` (see `_setup_scgd0`; the fields read here can also be built by
# hand), starting from `setup.μ_start`. Returns the host `Σ[ω, i, T]`, `μ[T]`, `converged[T]` and
# the `history` of `SCGD0Result`.
function _loop_scgd0(setup, occ; η_min, η_init, maxiter, tol, backend, gpu_tile, Σ_init, fix_μ,
        ω_acoustic, verbosity)
    (; g2, ωph, iq_kk, iks_i, iks_f, ε_i, w_i, ε_f, w_f, f_to_i, W, nstates_base, multiplets, ωs,
       ωs_cell_table, ωs_spectral, ωs_spectral_dev, ωs_spectral_weights, j_spectral,
       t_spectral, KK) = setup
    FT = eltype(g2)
    nm, ni, nf = size(g2)
    nω, nωa, nT = length(ωs), length(ωs_spectral), length(occ)
        bracket = extrema(ωs_spectral) .+ (-FT(0.1), FT(0.1))   # μ search interval (Ry)

    # One partition of the inner states for every iteration, for both residencies: chunks of at
    # most `CHUNK_BYTES` of ωq, or `gpu_tile` inner states.
    nf_chunk = gpu_tile === nothing ? clamp(fld(CHUNK_BYTES, sizeof(FT) * nm * ni), 1, nf) :
                                      min(gpu_tile, nf)

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
    Σ_spectral = alloc(backend, Complex{FT}, nωa, ni)  # Σ of one temperature on ωs_spectral
    Σin = alloc(backend, Complex{FT}, nω, ni, nT)      # clamped Σ, the Fan-Migdal input
    ImΣ = alloc(backend, FT, nω, ni, nT)
    ReΣ = alloc(backend, FT, nω, ni)
    A = alloc(backend, FT, nωa, ni)                    # spectral functions A_i(ωa)
    D = alloc(backend, FT, nωa)                        # D_T(ωa) = Σ_i w_i A_i(ωa) × trapezoid weight of ωa
    # One chunk of ωq and of g2 in the (ν f, i) layout of `add_fan_migdal_imsigma!`, flat so that a
    # shorter last chunk is a contiguous prefix.
    ωq_buf = alloc(backend, FT, nm * nf_chunk * ni)
    g2p_buf = alloc(backend, FT, nm * nf_chunk * ni)
    # Per-f arrays repeated over ν, in the same fused νf order.
    per_νf(x) = vec(repeat(reshape(x, 1, nf), nm))
    ε_νf, f_to_i_νf, w_νf = per_νf(ε_f), per_νf(f_to_i), per_νf(w_f)
    # g2 is either resident on the device, or streamed: kept in host RAM with each chunk's slab
    # copied over.
    streamed = if ElectronPhonon.is_host(backend)
        false   # no streaming on the host: a chunk is a view of g2
    elseif gpu_tile !== nothing
        true    # an Int `gpu_tile` forces streaming
    else
        # Resident iff g2 fits next to the buffers above with 4 GB to spare.
        ElectronPhonon.reclaim_device_memory(backend)
        ElectronPhonon.should_stream_per_batch(backend, sizeof(g2) + RESIDENCY_HEADROOM_BYTES)
    end
    if verbosity > 0 && !ElectronPhonon.is_host(backend)
        @info "scGD0: g2 ($(round(sizeof(g2) / 1e9, digits = 2)) GB) " *
              (streamed ? "streamed from host RAM in chunks of $nf_chunk inner states" :
                          "resident on the device")
    end
    # Read-only for the whole run, so a host alias of the caller's g2 is safe.
    g2_dev = streamed ? alloc(backend, FT, nm, ni, nf_chunk) : to_device(backend, g2)

    μ = collect(FT, setup.μ_start)
    active = trues(nT)
    hist_err, hist_μ, hist_minA = (fill(FT(NaN), nT, maxiter) for _ in 1:3)
    wall = zeros(maxiter)
    iter_frozen = zeros(Int, nT)
    niter = 0
    for iter in 1:maxiter
        t0 = time()
        iTs = findall(active)

        # 1. Input self-energy, clamped to Im Σ <= -η_min.
        for iT in iTs
            Σ_T = view(Σ, :, :, iT)
            view(Σin, :, :, iT) .= complex.(real.(Σ_T), min.(imag.(Σ_T), -η_min[iT]))
        end
        # 2-3. Im Σ: the Fan-Migdal sum, f-chunks outside so a chunk's ωq is gathered and its
        # g2 permuted to (ν, f, i) once.
        fill!(ImΣ, 0)
        @views for fs in Iterators.partition(1:nf, nf_chunk)
            nνf = nm * length(fs)
            # ωq[ν, f, i]: the pair gather with the roles of the two state lists swapped and
            # the q index transposed, so the inner states come first.
            ωq_t = reshape(ωq_buf[1:nνf * ni], nm, length(fs), ni)
            gather_pair_table!(ωq_t, ωph, transpose(iq_kk), iks_f[fs], iks_i)
            g2_slab = if streamed
                # The chunk's contiguous slab g2[:, :, fs], copied to the device.
                copyto!(g2_dev, 1, g2, nm * ni * (first(fs) - 1) + 1, nm * ni * length(fs))
                g2_dev[:, :, 1:length(fs)]
            else
                # Resident: a view.
                g2_dev[:, :, fs]
            end
            g2_t = reshape(g2p_buf[1:nνf * ni], nm, length(fs), ni)
            permutedims!(g2_t, g2_slab, (1, 3, 2))
            νfs = nm * (first(fs) - 1) + 1:nm * last(fs)
            for iT in iTs
                add_fan_migdal_imsigma!(ImΣ[:, :, iT], reshape(g2_t, nνf, ni),
                    reshape(ωq_t, nνf, ni), ωs, ωs_cell_table, ε_νf[νfs], f_to_i_νf[νfs],
                    Σin[:, :, iT], μ[iT], occ.Tlist[iT], w_νf[νfs],
                    ω_acoustic)
            end
        end
        # 4. Re Σ by Kramers-Kronig.
        for iT in iTs
            mul!(ReΣ, KK, view(ImΣ, :, :, iT))
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
            # 6. Spectral DOS from the unclamped Σ, on the spectral grid.
            apply_interpolation_table!(Σ_spectral, Σ_new_T, j_spectral, t_spectral)
            A .= .-imag.(inv.(ωs_spectral_dev .- transpose(ε_i) .- Σ_spectral)) ./ π
            mul!(D, A, w_i)
            D .*= ωs_spectral_weights
            hist_minA[iT, iter] = minimum(transpose(A) * ωs_spectral_weights)
            # 7. Chemical potential, on the host.
            if !fix_μ
                D_host = Array(D)
                T = occ.Tlist[iT]
                count_holes = occ.type === :Semiconductor && occ.nlist[iT] < 0
                n_target = occ.nlist[iT] + occ.nelec
                excess(μ_) = compute_ncarrier_spectral(μ_, T, ωs_spectral, D_host, W; count_holes,
                    nstates_base, occ.spin_degeneracy) - n_target
                excess(bracket[1]) * excess(bracket[2]) > 0 && throw(ArgumentError(
                    "no chemical potential in $bracket gives $n_target electrons per cell at " *
                    "T = $T: the window holds too few states on one side of the gap, or the " *
                    "spectral weight it loses outside ωs_spectral moves the count past the " *
                    "target. " *
                    "Pass occ.μlist and fix_μ = true, or widen the window or ωs."))
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
    compute_ncarrier_spectral(μ, T, ωs_spectral, D_T, W; count_holes, nstates_base, spin_degeneracy)

Electrons per cell at chemical potential `μ` and temperature `T`, from the spectral DOS
`D_T[ωa] = Σ_i w_i A_i(ωa) Wω[ωa]` of the window states on the nodes `ωs_spectral`, with the
trapezoid weights `Wω = trapezoid_weights(ωs_spectral)` and `W = Σ_i w_i`:

    count_holes = false:  spin_degeneracy · (Σ_ωa D_T f(ωa - μ)           + nstates_base)
    count_holes = true:   spin_degeneracy · (W - Σ_ωa D_T (1 - f(ωa - μ)) + nstates_base)

The second form, for a hole-doped semiconductor, counts the window states as filled minus their
holes, so the result does not assume ∫A_i = 1.
"""
function compute_ncarrier_spectral(μ, T, ωs_spectral, D_T, W; count_holes, nstates_base,
        spin_degeneracy)
    n_window = count_holes ? W - compute_ncarrier_hole(μ, T, ωs_spectral, D_T) :
                             compute_ncarrier(μ, T, ωs_spectral, D_T)
    (n_window + nstates_base) * spin_degeneracy
end

"""
    save_scgd0(filename, res::SCGD0Result) -> filename

Write `res` to a JLD2 file: `el_i` as the `BandStates` itself, the other fields as arrays, `history`
as the group `history`, and `ωs` and `ωs_spectral` as groups with the keys `ω_origin`, `dω_unit`,
`offsets` (the field names of [`FrequencyGrid`](@ref)). Read it back with [`load_scgd0`](@ref).
"""
function save_scgd0(filename, res::SCGD0Result)
    jldopen(filename, "w") do file
        write_frequency_grid(file, "ωs", res.ωs)
        write_frequency_grid(file, "ωs_spectral", res.ωs_spectral)
        for name in (:Σ, :μ, :el_i, :Tlist, :nlist, :converged)
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
        group = file["history"]
        history = (; (Symbol(name) => group[name] for name in keys(group))...)
        SCGD0Result(read_frequency_grid(file, "ωs"), read_frequency_grid(file, "ωs_spectral"),
            file["Σ"], file["μ"], file["el_i"], file["Tlist"], file["nlist"],
            file["converged"], history)
    end
end

# A `FrequencyGrid` as the JLD2 group `name`, with one key per constructor argument, named as the
# struct field.
function write_frequency_grid(file, name, grid::FrequencyGrid)
    for key in (:ω_origin, :dω_unit, :offsets)
        file["$name/$key"] = getfield(grid, key)
    end
end

read_frequency_grid(file, name) =
    FrequencyGrid((file["$name/$key"] for key in (:ω_origin, :dω_unit, :offsets))...)
