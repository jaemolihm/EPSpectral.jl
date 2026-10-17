# Vertex-corrected (ladder) Kubo conductivity on the scGD0 self-energy: the application of the
# ladder kernel to a vertex, the solve of J = J0 + L(J) at every temperature and probe frequency,
# and the public driver.
#
# `run_scgd0_ladder` = `_setup_scgd0` (the shared e-ph extraction) + `_setup_ladder` (rotations of
# the vertex, shift tables, quadrature weights, velocities) + `_loop_ladder` (the temperature and
# frequency loops). Index legend as in `ladder_kernel.jl`, plus f = inner (full-BZ) state,
# ν = phonon mode, m = shift index, T = temperature, Ω = probe frequency.

"""
    SCGD0LadderResult

The result of [`run_scgd0_ladder`](@ref), for the probe frequencies `Ωs` and the temperatures
`Tlist` of the scGD0 result it was computed from. In atomic Rydberg units:
- `μ[T]`: the chemical potential of the scGD0 result.
- `σ_bubble[a, b, Ω, T]`, `σ_ladder[a, b, Ω, T]`: the electron-only Kubo conductivity from the
  bare and from the dressed current vertex (Re σ(Ω), symmetrized).
- `converged[Ω, T]`, `niter[Ω, T]`: whether the solve reached `rtol`, and its Krylov or fixed-point
  iteration count.
- `history[Ω, T]`: a `NamedTuple` `(; norms, σ_iter)`. For `solver = :gmres`, `norms` holds the
  residual norms of the Krylov iteration and `σ_iter` is empty; for `:fixed_point`, `σ_iter` holds
  the unsymmetrized conductivity (without `spin_degeneracy / volume`) after every iteration and
  `norms` is empty.
- `dω_shift`: the shift-grid step used.
- `J[ω, c12, cart, c0, i, Ω, T]`: the dressed vertex, or `nothing` unless `store_vertex = true`.
"""
struct SCGD0LadderResult{FT}
    Ωs::Vector{FT}
    Tlist::Vector{FT}
    μ::Vector{FT}
    σ_bubble::Array{FT, 4}
    σ_ladder::Array{FT, 4}
    converged::Matrix{Bool}
    niter::Matrix{Int}
    history::Matrix{NamedTuple{(:norms, :σ_iter), Tuple{Vector{FT}, Vector{Matrix{FT}}}}}
    dω_shift::FT
    J::Union{Nothing, Array{Complex{FT}, 7}}
end

"""
    run_scgd0_ladder(model, nk, window, scgd0::SCGD0Result; occ, Ωs = [0.0], solver = :gmres,
        rtol = 1e-2, maxiter = 30, dω_shift = minimum(diff(scgd0.ωs)), store_vertex = false,
        kwargs...) -> SCGD0LadderResult

The electron-only Kubo conductivity at every temperature of `scgd0` and every probe frequency
`Ωs` (`0` is the DC limit), from the bare current vertex (`σ_bubble`) and from the vertex dressed
by the ladder of the scGD0 coupling, `J = J0 + L(J)` on the 2x2 Keldysh contour (`σ_ladder`).
Self-energy, chemical potential and temperatures come from `scgd0`; the e-ph coupling is extracted
again for the same `model`, `nk` and `window`, which must give the same states as `scgd0.el_i`.

`ωs = scgd0.ωs` must be commensurate with `dω_shift` (every node on `ωs[1] + dω_shift ℤ`); the
cost grows as `1 / dω_shift`. `occ` supplies `spin_degeneracy` and `volume`, and its `Tlist` must
equal `scgd0.Tlist`. The linear problem is solved for each `(T, Ω)` in turn by
`solver = :gmres` (warm-started from the previous `Ω` at the same `T`; residual below `rtol` times
the norm of `J0`, at most `maxiter` iterations) or `solver = :fixed_point` (from `J0`,
`J ← J0 + L(J)` until the relative change of σ is below `rtol`). An unconverged solve returns its last σ, flagged in `converged`, with one
warning.

Keywords as in [`run_scgd0`](@ref): `backend`, `symmetry`, `verbosity`, `gpu_tile` (inner states
per chunk), `ω_acoustic`, `electron_degen_cutoff`; the others go to `run_eph_over_k_and_kq`.
"""
function run_scgd0_ladder(model::Model{FT}, nk, window, scgd0::SCGD0Result;
        occ::ElectronOccupationParams, Ωs = [zero(FT)], solver = :gmres, rtol = 1e-2, maxiter = 30,
        dω_shift = minimum(diff(scgd0.ωs)), store_vertex = false, backend = CPUBackend(),
        symmetry = model.symmetry, gpu_tile = nothing, verbosity = 1,
        ω_acoustic = ElectronPhonon.omega_acoustic,
        electron_degen_cutoff = ElectronPhonon.electron_degen_cutoff, ep_kwargs...) where {FT}
    solver ∈ (:gmres, :fixed_point) || throw(ArgumentError(
        "solver must be :gmres or :fixed_point, got $solver"))
    maxiter >= 1 || throw(ArgumentError("maxiter must be at least 1, got $maxiter"))
    (gpu_tile === nothing || gpu_tile >= 1) || throw(ArgumentError(
        "gpu_tile must be nothing or a positive number of inner states, got $gpu_tile"))
    isempty(Ωs) && throw(ArgumentError("Ωs must hold at least one probe frequency"))
    (length(occ.Tlist) == length(scgd0.Tlist) && occ.Tlist ≈ scgd0.Tlist) || throw(ArgumentError(
        "occ.Tlist = $(occ.Tlist) must equal the temperatures of scgd0, $(scgd0.Tlist)"))

    setup = _setup_scgd0(model, nk, window, occ; ωs = scgd0.ωs, dω_dense = minimum(diff(scgd0.ωs)),
        backend, symmetry, verbosity, electron_degen_cutoff, ep_kwargs...)
    el_new, el_old = setup.el_i, scgd0.el_i
    (el_new.iks == el_old.iks && el_new.ibands == el_old.ibands &&
     isapprox(el_new.es, el_old.es; rtol = 1e-10)) || throw(ArgumentError(
        "the states of the new extraction differ from scgd0.el_i: run_scgd0_ladder needs the " *
        "same model, nk, window and symmetry as the run that produced scgd0"))
    ladder = _setup_ladder(setup, scgd0, symmetry; dω_shift, backend)
    res = _loop_ladder(ladder, scgd0.Σ, scgd0.μ, scgd0.Tlist, collect(FT, Ωs); solver, rtol,
        maxiter, store_vertex, backend, gpu_tile, ω_acoustic, verbosity)

    # The sum over the irreducible states, symmetrized, with the spin degeneracy and the volume.
    function to_symmetrized_σ(σ)
        out = similar(σ)
        for iT in axes(σ, 4), iΩ in axes(σ, 3)
            out[:, :, iΩ, iT] = occ.spin_degeneracy / occ.volume *
                ElectronPhonon.symmetrize(ElectronPhonon.Mat3{FT}(σ[:, :, iΩ, iT]), symmetry)
        end
        out
    end
    SCGD0LadderResult(collect(FT, Ωs), collect(FT, scgd0.Tlist), collect(FT, scgd0.μ),
        to_symmetrized_σ(res.σ_bubble), to_symmetrized_σ(res.σ_ladder), res.converged, res.niter, res.history,
        FT(dω_shift), res.J)
end

# What the ladder needs on top of the `_setup_scgd0` fields: the rotation R[:, :, f] and IBZ
# representative of every inner state (the averaged vector unfolding of ElectronPhonon, TR-odd, as
# a dense array), the hat-basis shift tables on the commensurate `ωs`, the quadrature weights and
# the band velocities v[cart, i]. `symmetry === nothing` has no rotation.
function _setup_ladder(setup, scgd0, symmetry; dω_shift, backend)
    (; el_i, el_f, ωph) = setup
    FT = eltype(setup.g2)
    ωs = scgd0.ωs
    check_commensurate(ωs, dω_shift)

    nf = el_f.n
    f_to_i = Array(setup.f_to_i)
    R = zeros(FT, 3, 3, nf)
    if symmetry === nothing
        # Identity rotation: el_f is el_i itself.
        for a in 1:3
            R[a, a, :] .= 1
        end
    else
        # One nonzero per row, in the column of the representative: the map zeroes an inner state
        # without a match silently, so check.
        rows, cols, vals = findnz(ElectronPhonon.vector_field_unfold_and_interpolate_map(
            el_i, el_f, symmetry))
        (length(rows) == nf && allunique(rows)) || error("the vector unfolding map has " *
            "$(length(rows)) entries for $nf inner states: every inner state needs exactly one " *
            "representative")
        cols[sortperm(rows)] == f_to_i || error("the vector unfolding map and " *
            "find_unfolding_indices disagree on the representatives of the inner states")
        for (f, val) in zip(rows, vals)
            R[:, :, f] .= val
        end
    end

    M = ceil(Int, maximum(ωph) / dω_shift) + 1
    klo, wlo, whi = shift_tables(ωs, FT(dω_shift), M)
    (; setup..., R = to_device(backend, R), klo = to_device(backend, klo),
       wlo = to_device(backend, wlo), whi = to_device(backend, whi), M, dω_shift = FT(dω_shift),
       Wω = to_device(backend, trapezoid_weights(ωs)),
       v = to_device(backend, reduce(hcat, el_i.vs)))
end

# The work arrays of `apply_ladder!` and the chunking of the inner states: chunks of `gpu_tile`
# states, or of at most `CHUNK_BYTES` for the larger of the coupling B, the shifted vertex X and
# the phonon frequencies. g2 is resident on the device or streamed from host RAM, as in
# `_loop_scgd0`.
function _ladder_work(ladder, backend, gpu_tile, verbosity)
    (; g2, ωs, M) = ladder
    FT = eltype(g2)
    nm, ni, nf = size(g2)
    nω = length(ωs)
    nrow = nω * 4 * 3 * 2 * 2
    bytes_per_f = sizeof(FT) * max(2 * (M + 1) * ni, nrow * (M + 1), nm * ni)
    nf_chunk = gpu_tile === nothing ? clamp(fld(CHUNK_BYTES, bytes_per_f), 1, nf) : min(gpu_tile, nf)
    work = (; nf_chunk,
        ωq_buf = alloc(backend, FT, nm * nf_chunk * ni), nB_buf = alloc(backend, FT, nm * nf_chunk * ni),
        B_buf = alloc(backend, FT, 2 * (M + 1) * nf_chunk * ni),
        X_buf = alloc(backend, FT, nrow * (M + 1) * nf_chunk),
        Cp = alloc(backend, FT, nrow, 2ni), Cm = alloc(backend, FT, nrow, 2ni),
        Y = alloc(backend, FT, nω, 4, 3, 2, 2, ni), KY = alloc(backend, FT, nω, 4, 3, 2, 2, ni),
        KJ = alloc(backend, Complex{FT}, nω, 4, 3, 2, ni))
    streamed = if ElectronPhonon.is_host(backend)
        false   # no streaming on the host: a chunk is a view of g2
    elseif gpu_tile !== nothing
        true    # an Int `gpu_tile` forces streaming
    else
        # Resident iff g2 fits next to the buffers above with the headroom to spare.
        ElectronPhonon.reclaim_device_memory(backend)
        ElectronPhonon.should_stream_per_batch(backend, sizeof(g2) + RESIDENCY_HEADROOM_BYTES)
    end
    if verbosity > 0 && !ElectronPhonon.is_host(backend)
        @info "ladder: g2 ($(round(sizeof(g2) / 1e9, digits = 2)) GB) " *
              (streamed ? "streamed from host RAM in chunks of $nf_chunk inner states" :
                          "resident on the device")
    end
    # Read-only for the whole run, so a host alias of the caller's g2 is safe.
    (; work..., streamed, g2_dev = streamed ? alloc(backend, FT, nm, ni, nf_chunk) :
                                              to_device(backend, g2))
end

"""
    apply_ladder!(LJ, J, ladder, work, Gm, Gp, T; ω_acoustic) -> LJ

The ladder kernel on the vertex `J[ω, c12, cart, c0, i]` at temperature `T` and the propagators
`Gm`, `Gp` of one probe frequency: `L(J) = -Gm (s ⊙ K[J]) Gp`, where `K^{ab}[J]_i(ε) = Σ_{f,ν} w_f
g2 ∫dω d^{ba}(ω) R_f J_{i(f)}^{ab}(ε + ω)` with the bare phonon propagator `d`, evaluated as the
four shifted sums U±, V± in chunks of inner states and their combination
([`combine_channels!`](@ref)). `ladder` is the output of `_setup_ladder` and `work` that of
`_ladder_work`. Modes below `ω_acoustic` are skipped.
"""
@views function apply_ladder!(LJ, J, ladder, work, Gm, Gp, T; ω_acoustic)
    (; g2, ωph, iq_kk, iks_i, iks_f, w_f, f_to_i, R, klo, wlo, whi, K, ωs, M, dω_shift) = ladder
    (; nf_chunk, streamed, g2_dev, ωq_buf, nB_buf, B_buf, X_buf, Cp, Cm, Y, KY, KJ) = work
    nm, ni, nf = size(g2)
    nω = length(ωs)
    nrow = size(Cp, 1)
    fill!(Cp, 0)
    fill!(Cm, 0)
    for fs in Iterators.partition(1:nf, nf_chunk)
        nfs = length(fs)
        # ωq[ν, f, i] and n_B(ωq), as in the Fan-Migdal sum.
        ωq = reshape(ωq_buf[1:nm * nfs * ni], nm, nfs, ni)
        gather_pair_table!(ωq, ωph, transpose(iq_kk), iks_f[fs], iks_i)
        nB = reshape(nB_buf[1:nm * nfs * ni], nm, nfs, ni)
        _over_chunks(nB, axes(nB, 2)) do cs
            nB[:, cs, :] .= occ_boson.(ωq[:, cs, :], T)
        end
        g2_slab = if streamed
            # The chunk's contiguous slab g2[:, :, fs], copied to the device.
            copyto!(g2_dev, 1, g2, nm * ni * (first(fs) - 1) + 1, nm * ni * nfs)
            g2_dev[:, :, 1:nfs]
        else
            # Resident: a view.
            g2_dev[:, :, fs]
        end
        B = reshape(B_buf[1:2 * (M + 1) * nfs * ni], M + 1, nfs, ni, 2)
        compute_shift_coupling!(B, g2_slab, ωq, nB, w_f[fs], dω_shift, ω_acoustic)
        X = reshape(X_buf[1:nrow * (M + 1) * nfs], nω, 4, 3, 2, 2, M + 1, nfs)
        for (s, C) in ((1, Cp), (2, Cm))
            gather_shifted_vertex!(X, J, R[:, :, fs], f_to_i[fs], klo[:, :, s], wlo[:, :, s],
                whi[:, :, s])
            # C += X B, threaded over row blocks of C on the host.
            X2, B2 = reshape(X, nrow, (M + 1) * nfs), reshape(B, (M + 1) * nfs, 2ni)
            _over_chunks(C, axes(C, 1)) do rows
                mul!(C[rows, :], X2[rows, :], B2, 1, 1)
            end
        end
    end
    combine_channels!(KJ, reshape(Cp, nω, 4, 3, 2, 2, ni, 2), reshape(Cm, nω, 4, 3, 2, 2, ni, 2),
        K, Y, KY)
    attach!(LJ, Gm, KJ, Gp)
    LJ .*= -1
end

# The loops of the ladder on `ladder` (see `_setup_ladder`; its fields can be replaced, e.g. `v`
# by a density vertex), for the host self-energy `Σ[ω, i, T]`, chemical potentials `μ[T]`,
# temperatures `Tlist` and probe frequencies `Ωs`. Returns the unsymmetrized conductivities
# without `spin_degeneracy / volume`, `σ_bubble` and `σ_ladder[a, b, Ω, T]`, `converged[Ω, T]`,
# `niter[Ω, T]`, `history[Ω, T]` and the vertex `J` (`nothing` unless `store_vertex`).
function _loop_ladder(ladder, Σ, μ, Tlist, Ωs; solver, rtol, maxiter, store_vertex, backend,
        gpu_tile, ω_acoustic, verbosity)
    (; ωs, ε_i, w_i, Wω, v) = ladder
    FT = eltype(ladder.g2)
    nω, ni, nT = size(Σ)
    nΩ = length(Ωs)
    work = _ladder_work(ladder, backend, gpu_tile, verbosity)
    Σ_dev = to_device(backend, Σ)
    Σd_m, Σd_p = (alloc(backend, Complex{FT}, nω, ni) for _ in 1:2)
    Gm, Gp = (alloc(backend, Complex{FT}, nω, 4, ni) for _ in 1:2)
    J0, J, LJ = (alloc(backend, Complex{FT}, nω, 4, 3, 2, ni) for _ in 1:3)
    σ_bubble, σ_ladder = (zeros(FT, 3, 3, nΩ, nT) for _ in 1:2)
    converged, niter = fill(false, nΩ, nT), zeros(Int, nΩ, nT)
    history = fill((; norms = FT[], σ_iter = Matrix{FT}[]), nΩ, nT)
    J_all = store_vertex ? zeros(Complex{FT}, nω, 4, 3, 2, ni, nΩ, nT) : nothing
    # Σ at ωs - Ω/2 and ωs + Ω/2, flat outside the grid, unclamped.
    function shifted_Σ!(Σd, Σ_T, xs)
        j, t = interp_table(Array(ωs), xs)
        interp_apply!(Σd, Σ_T, to_device(backend, j), to_device(backend, t))
    end

    for iT in 1:nT
        T = Tlist[iT]
        for (iΩ, Ω) in enumerate(Ωs)
            t0 = time()
            Σ_T = view(Σ_dev, :, :, iT)
            shifted_Σ!(Σd_m, Σ_T, Array(ωs) .- Ω / 2)
            shifted_Σ!(Σd_p, Σ_T, Array(ωs) .+ Ω / 2)
            contour_greens!(Gm, Gp, ε_i, Σd_m, Σd_p, ωs, Ω, μ[iT], T)
            bare_vertex_response!(J0, Gm, v, Gp)
            σ_bubble[:, :, iΩ, iT] = ladder_conductivity(J0, v, w_i, Wω, Ω, T)
            apply!(LJ_, J_) = apply_ladder!(LJ_, J_, ladder, work, Gm, Gp, T; ω_acoustic)

            if solver === :gmres
                # (I - L) J = J0, warm-started from the previous Ω at this T.
                op = LinearMap{Complex{FT}}(length(J0); ismutating = true) do y, x
                    apply!(LJ, reshape(x, size(J0)))
                    y .= x .- vec(LJ)
                end
                args = iΩ == 1 ? (op, vec(J0)) : (op, vec(J0), vec(copy(J)))
                # Stop at rtol * ||J0|| for every Ω (Krylov's rtol would scale with ||b - A x0||).
                x, stats = Krylov.gmres(args...; atol = FT(rtol) * norm(J0), rtol = zero(FT),
                    itmax = maxiter, history = true)
                copyto!(vec(J), x)
                converged[iΩ, iT], niter[iΩ, iT] = stats.solved, stats.niter
                history[iΩ, iT] = (; norms = FT.(stats.residuals), σ_iter = Matrix{FT}[])
            else
                # J ← J0 + L(J) from J0; stop when σ changes by less than rtol (not at iteration 1).
                copyto!(J, J0)
                σ_iter = Matrix{FT}[]
                for iter in 1:maxiter
                    apply!(LJ, J)
                    J .= J0 .+ LJ
                    push!(σ_iter, ladder_conductivity(J, v, w_i, Wω, Ω, T))
                    niter[iΩ, iT] = iter
                    if iter > 1 && norm(σ_iter[end] - σ_iter[end-1]) < rtol * norm(σ_iter[end])
                        converged[iΩ, iT] = true
                        break
                    end
                end
                history[iΩ, iT] = (; norms = FT[], σ_iter)
            end
            σ_ladder[:, :, iΩ, iT] = ladder_conductivity(J, v, w_i, Wω, Ω, T)
            store_vertex && (J_all[:, :, :, :, :, iΩ, iT] = Array(J))
            verbosity > 0 && @info "ladder T = $T, Ω = $Ω" solver niter = niter[iΩ, iT] converged = (
                converged[iΩ, iT]) wall = time() - t0
        end
    end
    all(converged) || @warn "ladder not converged (rtol = $rtol, maxiter = $maxiter) at " *
        "(Ω, T) = $([(Ωs[iΩ], Tlist[iT]) for iΩ in 1:nΩ, iT in 1:nT if !converged[iΩ, iT]])"
    (; σ_bubble, σ_ladder, converged, niter, history, J = J_all)
end

"""
    save_scgd0_ladder(filename, res::SCGD0LadderResult) -> filename

Write `res` to a JLD2 file, with `J` only if it is stored and `history` as the groups
`history/norms` and `history/σ_iter`. Read it back with [`load_scgd0_ladder`](@ref).
"""
function save_scgd0_ladder(filename, res::SCGD0LadderResult)
    jldopen(filename, "w") do file
        for name in (:Ωs, :Tlist, :μ, :σ_bubble, :σ_ladder, :converged, :niter, :dω_shift)
            file[String(name)] = getfield(res, name)
        end
        for iΩ in axes(res.history, 1), iT in axes(res.history, 2)
            file["history/norms/$(iΩ)_$(iT)"] = res.history[iΩ, iT].norms
            file["history/σ_iter/$(iΩ)_$(iT)"] = res.history[iΩ, iT].σ_iter
        end
        res.J === nothing || (file["J"] = res.J)
    end
    filename
end

"""
    load_scgd0_ladder(filename) -> SCGD0LadderResult

Read an [`SCGD0LadderResult`](@ref) written by [`save_scgd0_ladder`](@ref).
"""
function load_scgd0_ladder(filename)
    jldopen(filename, "r") do file
        Ωs, Tlist = file["Ωs"], file["Tlist"]
        history = [(; norms = file["history/norms/$(iΩ)_$(iT)"],
                      σ_iter = file["history/σ_iter/$(iΩ)_$(iT)"])
                   for iΩ in eachindex(Ωs), iT in eachindex(Tlist)]
        SCGD0LadderResult(Ωs, Tlist, file["μ"], file["σ_bubble"], file["σ_ladder"],
            file["converged"], file["niter"], history, file["dω_shift"],
            haskey(file, "J") ? file["J"] : nothing)
    end
end
