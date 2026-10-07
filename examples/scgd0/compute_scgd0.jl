# Compute file of the scGD0 drivers (`run_scgd0_*.jl`): plain functions, nothing runs on include.
# The material, the settings and the output belong to the run scripts.
using ElectronPhonon
using EPSpectral

# The scGD0 self-energy and μ at every temperature of `occ`. Extra keywords go to `run_scgd0`;
# the wall time per iteration is in `res.history.wall`.
function compute_scgd0(model, nk, window, ωs, occ; backend, kwargs...)
    run_scgd0(model, nk, window; occ, ωs, backend, kwargs...)
end
