# Bubble and ladder conductivity of the 1D Holstein chain on its scGD0 self-energy (the fixture of
# test/test_scgd0_holstein.jl, on a coarser k grid). The Holstein ladder correction vanishes for the
# current vertex, so σ_ladder = σ_bubble. Re-runnable top to bottom in one REPL session.
using Revise
using ElectronPhonon
using EPSpectral

# --- Step P1: model ---
begin
    # Energies in the model's units (t = 1).
    model = ElectronPhonon.holstein_model(; t = 1.0, λ = 0.5, ω₀ = 0.2, dimension = 1)
end;

# --- Step P2: run settings ---
begin
    use_gpu = false
    nk = 32
    window = (-Inf, Inf)
    ωs = FrequencyGrid(-4.0:0.02:4.0)
    occ = ElectronOccupationParams(; Tlist = [0.1, 0.2], nlist = 0.0, nelec = 1,
        volume = model.volume, spin_degeneracy = 2, type = :Metal)
    Ωs = [0.0, 0.1, 0.3]
    cfg = (; backend = ElectronPhonon.backend_from(use_gpu), η_init = 0.05, tol = 1e-6,
        maxiter = 100)
end;

# --- Step P3: scGD0 ---
begin
    res = run_scgd0(model, (nk, 1, 1), window; occ, ωs, cfg...)
    save_scgd0("scgd0_holstein_1d_nk$(nk).jld2", res)
    println("converged = $(res.converged), μ = $(res.μ)")
end;

# --- Step P4: ladder conductivity on that self-energy ---
begin
    out = run_scgd0_ladder(model, (nk, 1, 1), window, res; occ, Ωs, solver = :gmres,
        backend = cfg.backend)
    save_scgd0_ladder("scgd0_ladder_holstein_1d_nk$(nk).jld2", out)
    for (iT, T) in enumerate(out.Tlist), (iΩ, Ω) in enumerate(out.Ωs)
        println("T = $T, Ω = $Ω: σ_bubble,xx = $(out.σ_bubble[1, 1, iΩ, iT]), " *
                "σ_ladder,xx = $(out.σ_ladder[1, 1, iΩ, iT]), niter = $(out.niter[iΩ, iT])")
    end
end;
