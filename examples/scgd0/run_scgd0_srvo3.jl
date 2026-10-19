# scGD0 spectral function of SrVO3 (wfpt = false: active-space Fan-Migdal only, no Debye-Waller).
# Re-runnable top to bottom in one REPL session.
using Revise
using CUDA
using ElectronPhonon
using EPSpectral

# --- Step P1: model and material ---
begin
    folder = "/mnt/ceph/users/jlihm/scGD/SrVO3/1_epw.pbe"
    model = ElectronPhonon.load_model_from_epw_new(folder, "temp", "SrVO3";
        epmat_outer_momentum = "el")
    eV, K = unit_to_aru(:eV), unit_to_aru(:K)
    eF = 12.65 * eV
end;

# --- Step P2: run settings ---
begin
    use_gpu = true
    nk = 80
    window = (eF - 0.15eV, eF + 0.15eV)
    ωs = graded_frequency_grid([(eF - 0.1eV, eF + 0.1eV, 0.001eV),
        (eF - 0.35eV, eF + 0.35eV, 0.02eV)])
    occ = ElectronOccupationParams(; Tlist = [100.0, 150, 200, 250, 300] .* K, nlist = 1.0,
        nelec = 0, volume = model.volume, spin_degeneracy = 2, type = :Metal)
    cfg = (; backend = ElectronPhonon.backend_from(use_gpu))
end;

# --- Step P3: scGD0 ---
begin
    res = run_scgd0(model, (nk, nk, nk), window; occ, ωs, cfg...)
    save_scgd0("scgd0_nk$(nk).jld2", res)
    println("μ - E_F (meV) = $((res.μ .- eF) ./ (eV / 1000))")
end;
