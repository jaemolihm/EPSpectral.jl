# `run_scgd0` on the pb artifact (irreducible BZ, phonon frequencies gathered from the
# calculator's table, unfolding, multiplet average) against the v2 solver's math transcribed as plain
# loops on the full BZ (`compute_self_energy!`, `get_interpolated_self_energy`,
# `compute_occupation!`, `update_chemical_potential!` of `ab_initio_solver_new.jl`): no symmetry,
# f -> i the identity, no multiplet average, Interpolations and Roots as the old code used them.
# Its inputs (states, g2, ωq, nstates_base) come from its own `G2Calculator` run with
# `symmetry = nothing`, the per-pair ωq read off that run's phonon table by a plain loop; nothing is
# shared with the run under test.
using Pkg: Pkg

# The v2 scFM iteration (`run_scFM_v2` with wfpt = false), `niter` iterations from Σ = -iη_init.
function scgd0_v2_reference(el, g2, ωq, ωs, ωs_dense, occ, ηlist, η_init, niter, ω_acoustic)
    nω, n, nT, nmodes = length(ωs), el.n, length(occ), size(g2, 1)
    occ_f(e, T) = 1 / (exp(e / T) + 1)
    occ_b(e, T) = 1 / expm1(e / T)
    on_dense(Σs) = [extrapolate(interpolate((ωs,), Σs[:, i, iT], Gridded(Linear())), Flat())(ω)
                    for ω in ωs_dense, i in 1:n, iT in 1:nT]
    Σs = fill(complex(0.0, -η_init), nω, n, nT)
    μlist = copy(occ.μlist)
    for _ in 1:niter
        Σs_input = on_dense(Σs)
        Σs_new = zeros(ComplexF64, nω, n, nT)
        for iT in 1:nT, f in 1:n
            T, μ = occ.Tlist[iT], μlist[iT]
            Σ_f = Σs_input[:, f, iT]
            Σ_clamped = @. complex(real(Σ_f), min(imag(Σ_f), -ηlist[iT]))
            Σ_itp = extrapolate(scale(interpolate(Σ_clamped, BSpline(Linear())), ωs_dense), Flat())
            ekq, wkq = el.es[f], el.weights[f]
            for i in 1:n, ν in 1:nmodes
                ω_q = ωq[ν, i, f]
                ω_q < ω_acoustic && continue
                nq = occ_b(ω_q, T)
                for (iω, ω) in enumerate(ωs)
                    A1 = imag(1 / (ω + ω_q - ekq - Σ_itp(ω + ω_q))) * (nq + occ_f(ω + ω_q - μ, T))
                    A2 = imag(1 / (ω - ω_q - ekq - Σ_itp(ω - ω_q))) *
                         (nq + 1 - occ_f(ω - ω_q - μ, T))
                    Σs_new[iω, i, iT] += (A1 + A2) * g2[ν, i, f] * wkq * im
                end
            end
        end
        for iT in 1:nT, i in 1:n
            Σs_new[:, i, iT] .+= kramers_kronig(ωs, imag.(Σs_new[:, i, iT]))
        end
        Σs = Σs_new
        # Metal branch of `compute_occupation!`.
        Σs_dense = on_dense(Σs)
        dω = step(ωs_dense)
        for iT in 1:nT
            T = occ.Tlist[iT]
            function nocc(μ)
                fermi = [occ_f(ω - μ, T) for ω in ωs_dense]
                s = 0.0
                for i in 1:n
                    As = @. -imag(1 / (ωs_dense - el.es[i] - Σs_dense[:, i, iT])) / π
                    s += sum(As .* fermi) * dω * el.weights[i]
                end
                (s + el.nstates_base) * occ.spin_degeneracy
            end
            n_target = occ.nlist[iT] + occ.nelec
            μlist[iT] = find_zero(μ -> nocc(μ) - n_target, extrema(ωs_dense) .+ (-0.1, 0.1))
        end
    end
    Σs, μlist
end

@testset "scGD0 pb against the v2 reference" begin
    folder = Pkg.Artifacts.ensure_artifact_installed("pb", joinpath(@__DIR__, "Artifacts.toml"))
    model = EP.load_model_from_epw_new(folder, "temp", "pb"; epmat_outer_momentum = "el")
    eV, K = unit_to_aru(:eV), unit_to_aru(:K)
    nk = (6, 6, 6)
    e_F = 11.83 * eV   # 4 electrons in the 4 Wannier bands on this grid
    window = (e_F - 1.0eV, e_F + 1.0eV)
    ωs = e_F .+ vcat(-1.2:0.05:-0.25, -0.24:0.02:0.24, 0.25:0.05:1.2) .* eV
    ωs_dense = range(extrema(ωs)...; step = minimum(diff(ωs)))
    η_init = 0.05eV
    η_min = [0.0, 0.03eV]   # one per temperature
    occ = ElectronOccupationParams(; Tlist = [300.0, 600.0] .* K, nlist = 0.0, nelec = 4,
        volume = model.volume, spin_degeneracy = 2, type = :Metal)

    # Two iterations with the μ update, never converged (tol = 0).
    res = @test_logs (:warn, r"not converged") match_mode = :any run_scgd0(model, nk, window;
        occ, ωs, η_init, η_min, maxiter = 2, tol = 0.0, verbosity = 0)

    sel = EP.filter_electron_states(nk, model, window; symmetry = nothing)
    calc = EP.G2Calculator{Float64}(; model.nmodes)
    EP.run_eph_over_k_and_kq(model, sel, sel; calculators = [calc], symmetry = nothing,
        verbosity = 0)
    el, el_f = calc.el_i, calc.el_f
    ωq = [calc.ωph[ν, calc.iq_kk[el.iks[i], el_f.iks[f]]]
          for ν in 1:model.nmodes, i in 1:el.n, f in 1:el_f.n]
    occ_ref = deepcopy(occ)
    EP.bte_compute_μ!(occ_ref, el; do_print = false)
    Σ_ref, μ_ref = scgd0_v2_reference(el, calc.g2, ωq, ωs, ωs_dense, occ_ref, η_min, η_init,
        2, EP.omega_acoustic)

    # The window holds a degenerate pair (at X, split by 2e-7 eV), and the average makes its two
    # self-energies identical.
    el_i = res.el_i
    pairs = [(i, j) for i in 1:el_i.n, j in 1:el_i.n if i < j && el_i.iks[i] == el_i.iks[j] &&
             abs(el_i.es[i] - el_i.es[j]) < EP.electron_degen_cutoff]
    @test length(pairs) == 1
    @test all(res.Σ[:, i, :] == res.Σ[:, j, :] for (i, j) in pairs)
    # The irreducible states cover the full-BZ window states.
    @test sum(state_weights(res.el_i)) ≈ sum(state_weights(el)) rtol = 1e-12
    js = [state_index(el, res.el_i[i]) for i in 1:res.el_i.n]
    @test all(>(0), js)
    scale_Σ = maximum(abs, Σ_ref)
    err_Σ = maximum(abs, res.Σ .- Σ_ref[:, js, :])
    err_μ = maximum(abs, res.μ .- μ_ref)
    @info "scGD0 pb vs v2 reference" err_Σ / scale_Σ err_μ / scale_Σ scale_Σ
    @test err_Σ <= 1e-3 * scale_Σ
    @test err_μ <= 1e-3 * scale_Σ
end
