# scGD0 spectral function of SrVO3 (wfpt = false: active-space Fan-Migdal only, no Debye-Waller).
# Re-runnable top to bottom in one REPL session.
using Revise
using CUDA
using ElectronPhonon
using EPSpectral
includet(joinpath(@__DIR__, "compute_scgd0.jl"))

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
    ωs = eF .+ vcat(-0.35:0.02:-0.11, -0.099:0.001:0.100, 0.12:0.02:0.34) .* eV
    occ = ElectronOccupationParams(; Tlist = [100.0, 150, 200, 250, 300] .* K, nlist = 1.0,
        nelec = 0, volume = model.volume, spin_degeneracy = 2, type = :Metal)
    cfg = (; backend = ElectronPhonon.backend_from(use_gpu))
end;

# --- Step P3: scGD0 ---
begin
    res = compute_scgd0(model, (nk, nk, nk), window, ωs, occ; cfg...)
    save_scgd0("scgd0_nk$(nk).jld2", res)
    println("μ - E_F (meV) = $((res.μ .- eF) ./ (eV / 1000))")
end;
