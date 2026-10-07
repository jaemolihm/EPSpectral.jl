# The pb artifact window shared by `test_scgd0_pb.jl` and `test_scgd0_gpu.jl`: nk = 6³, E_F ± 1 eV
# (8 irreducible / 120 full-BZ states, a degenerate pair at X), two temperatures with different
# η_min. The artifact entry is a copy of ElectronPhonon's `test/Artifacts.toml` one.
using Pkg: Pkg

function pb_scgd0_fixture()
    folder = Pkg.Artifacts.ensure_artifact_installed("pb", joinpath(@__DIR__, "Artifacts.toml"))
    model = EP.load_model_from_epw_new(folder, "temp", "pb"; epmat_outer_momentum = "el")
    eV, K = unit_to_aru(:eV), unit_to_aru(:K)
    e_F = 11.83 * eV   # 4 electrons in the 4 Wannier bands on this grid
    ωs = e_F .+ vcat(-1.2:0.05:-0.25, -0.24:0.02:0.24, 0.25:0.05:1.2) .* eV
    occ = ElectronOccupationParams(; Tlist = [300.0, 600.0] .* K, nlist = 0.0, nelec = 4,
        volume = model.volume, spin_degeneracy = 2, type = :Metal)
    (; model, nk = (6, 6, 6), window = (e_F - 1.0eV, e_F + 1.0eV), ωs,
       ωs_dense = range(extrema(ωs)...; step = minimum(diff(ωs))), occ,
       η_init = 0.05eV, η_min = [0.0, 0.03eV])
end
