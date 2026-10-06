module EPSpectral

using LinearAlgebra: mul!
using JLD2: jldopen
using OhMyThreads: tforeach, tmap, index_chunks
using Roots: find_zero
# EP names are imported one by one: EP exports `occ_fermion` / `occ_boson`, which must not shadow
# the device-compilable local copies in `utils.jl`.
using ElectronPhonon: ElectronPhonon, Model, BandStates, ElectronOccupationParams, CPUBackend,
    alloc, to_device, to_device_copy, unit_to_aru, kpoints_grid, filter_electron_states,
    unfold_band_states, find_unfolding_indices, run_eph_over_k_and_kq,
    compute_phonon_states_batched, state_weights, state_xks, chemical_potential_is_computed,
    compute_ncarrier, compute_ncarrier_hole

include("utils.jl")
include("kramers_kronig.jl")
include("scgd0_grids.jl")
include("scgd0_kernel.jl")
include("scgd0.jl")

export run_scgd0, SCGD0Result, save_scgd0, load_scgd0, kramers_kronig

end
