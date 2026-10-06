# scGD0 spectral function of the 1D Holstein chain (the fixture of test/test_scgd0_holstein.jl).
# Re-runnable top to bottom in one REPL session.
using Revise
using ElectronPhonon
using EPSpectral
includet(joinpath(@__DIR__, "compute_scgd0.jl"))

# --- Step P1: model ---
begin
    # Energies in the model's units (t = 1).
    model = ElectronPhonon.holstein_model(; t = 1.0, λ = 0.5, ω₀ = 0.2, dimension = 1)
end;

# --- Step P2: run settings ---
begin
    use_gpu = false
    nk = 64
    window = (-Inf, Inf)
    ωs = collect(-4.0:0.02:4.0)
    occ = ElectronOccupationParams(; Tlist = [0.1], nlist = 0.0, nelec = 1, volume = model.volume,
        spin_degeneracy = 2, type = :Metal)
    # η_init and tol in the model's units: the defaults are meV-scale.
    cfg = (; backend = ElectronPhonon.backend_from(use_gpu), η_init = 0.05, tol = 1e-6)
end;

# --- Step P3: scGD0 ---
begin
    res = compute_scgd0(model, (nk, 1, 1), window, ωs, occ; cfg...)
    save_scgd0("scgd0_holstein_1d_nk$(nk).jld2", res)
    println("converged = $(res.converged) after $(size(res.history.err, 2)) iterations, " *
            "μ = $(res.μ)")
end;
