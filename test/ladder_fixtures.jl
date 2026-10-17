# The converged-or-not scGD0 cases that the ladder tests run on, and the ladder's own setup on
# them: the 1D Holstein chain (any ω0, so a shift step below ω0 is possible) and the pb window.

# `run_scgd0` result `res` and the ladder setup for `model`; `symmetry` is the one the ladder uses
# (the scGD0 run uses it too).
function ladder_case(model, nk, window, occ, ωs, res; symmetry, dω_shift = minimum(diff(ωs)))
    backend = EP.CPUBackend()
    setup = EPSpectral._setup_scgd0(model, nk, window, occ; ωs, dω_dense = minimum(diff(ωs)),
        backend, symmetry, verbosity = 0, electron_degen_cutoff = EP.electron_degen_cutoff)
    ladder = EPSpectral._setup_ladder(setup, res, symmetry; dω_shift, backend)
    (; model, nk, window, occ, ωs, res, setup, ladder, symmetry, dω_shift)
end

function holstein_ladder_case(; nk = 16, ω0 = 0.173, t = 1.0, λ = 0.5, symmetry = :model,
        ωs = vcat(-4.0:0.04:-1.0, -0.98:0.02:0.98, 1.0:0.04:4.0), Tlist = [0.1, 0.2])
    model = EP.holstein_model(; t, λ, ω₀ = ω0, dimension = 1, verbose = false)
    symmetry = symmetry === :model ? model.symmetry : symmetry
    occ = ElectronOccupationParams(; Tlist, nlist = 0.0, nelec = 1, volume = model.volume,
        spin_degeneracy = 2, type = :Metal)
    res = run_scgd0(model, (nk, 1, 1), (-Inf, Inf); occ, ωs, symmetry, η_init = 0.05, tol = 1e-6,
        maxiter = 100, verbosity = 0)
    ladder_case(model, (nk, 1, 1), (-Inf, Inf), occ, ωs, res; symmetry)
end

function pb_ladder_case(; maxiter = 2)
    (; model, nk, window, ωs, occ, η_init, η_min) = pb_scgd0_fixture()
    res = @test_logs (:warn, r"not converged") match_mode = :any run_scgd0(model, nk, window;
        occ, ωs, η_init, η_min, maxiter, tol = 0.0, verbosity = 0)
    ladder_case(model, nk, window, occ, ωs, res; symmetry = model.symmetry)
end

# The same case on the shift step `dω_shift`.
with_shift_step(case, dω_shift) = ladder_case(case.model, case.nk, case.window, case.occ, case.ωs,
    case.res; case.symmetry, dω_shift)

# The propagators Gm, Gp of probe frequency Ω at temperature index iT.
function ladder_propagators(case, iT, Ω)
    (; res, ωs) = case
    Σd_m, Σd_p = (similar(res.Σ[:, :, iT]) for _ in 1:2)
    EPSpectral.apply_interpolation_table!(Σd_m, res.Σ[:, :, iT], EPSpectral.linear_interpolation_table(ωs, ωs .- Ω / 2)...)
    EPSpectral.apply_interpolation_table!(Σd_p, res.Σ[:, :, iT], EPSpectral.linear_interpolation_table(ωs, ωs .+ Ω / 2)...)
    Gm, Gp = (zeros(ComplexF64, length(ωs), 4, res.el_i.n) for _ in 1:2)
    EPSpectral.contour_greens!(Gm, Gp, res.el_i.es, Σd_m, Σd_p, ωs, Ω, res.μ[iT], res.Tlist[iT])
    Gm, Gp
end

# The work arrays and `apply_ladder_kernel!` of a case, with chunks of `gpu_tile` inner states.
function apply_ladder_case(case, J, iT, Ω; gpu_tile = nothing)
    (; ladder, res) = case
    Gm, Gp = ladder_propagators(case, iT, Ω)
    workspace = EPSpectral._ladder_workspace(ladder, EP.CPUBackend(), gpu_tile, 0)
    LJ = similar(J)
    EPSpectral.apply_ladder_kernel!(LJ, J, ladder, workspace, Gm, Gp, res.Tlist[iT]; ω_acoustic = EP.omega_acoustic)
end
