# EPSpectral

[![Build Status](https://github.com/jaemolihm/EPSpectral.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/jaemolihm/EPSpectral.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/jaemolihm/EPSpectral.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/jaemolihm/EPSpectral.jl)

Electron spectral functions from the self-consistent Fan-Migdal self-energy with G self-consistent
and D bare (scGD0), on top of [ElectronPhonon.jl](https://github.com/jaemolihm/ElectronPhonon.jl).

`run_scgd0(model, nk, window; occ, ωs)` returns, for each temperature of `occ`, the self-energy
Σ_i(ω) of the electron states `i` in an energy `window` on the irreducible part of the `nk` k grid,
and the chemical potential μ, as an `SCGD0Result`:

1. The e-ph coupling `g2[ν, i, f] = |g|²/(2ω)` between the window states (`i` irreducible, `f` full
   Brillouin zone) is extracted once with ElectronPhonon's `G2Calculator` and kept in host memory.
2. Each iteration computes Im Σ from the Fan-Migdal sum with the current Σ in the inner Green's
   function, Re Σ by Kramers-Kronig, averages Σ over degenerate multiplets, and re-solves μ for the
   carrier density from the spectral functions. Σ starts from `-iη_init`, so `maxiter = 1` is the
   one-shot G0D0.
3. A temperature stops iterating once `max |ΔΣ| < tol`.

The Fan-Migdal sum is a reduction over a lazy broadcast of one scalar function, written for any
array backend (`backend` keyword). `save_scgd0` / `load_scgd0` write and read a result
with JLD2; a run restarts from one through the `Σ_init` keyword and `occ.μlist`.

`run_scgd0_ladder(model, nk, window, res; occ, Ωs)` takes a converged `SCGD0Result` and returns the
electron-only Kubo conductivity at every temperature and probe frequency `Ωs` (`0` is the DC limit),
from the bare current vertex (`σ_bubble`) and from the vertex dressed by the ladder of the scGD0
coupling on the 2x2 Keldysh contour (`σ_ladder`), as an `SCGD0LadderResult`. The ladder equation
is solved by GMRES (default, `rtol = 1e-2`) or by a fixed-point iteration. The output grid `ωs` of the
scGD0 run must be commensurate with the shift step `dω_shift`, derived as the largest step on which it
is (`minimum(diff(ωs)) / k`) and reported in the result; the cost grows as `1 / dω_shift`. `graded_frequency_grid([(wmin, wmax, dw), ...])` builds graded
grids from nested windows that are commensurate by construction. With `backend = gpu_backend()` the ladder runs on the device by the same
code as on the host (g2 resident or streamed in chunks of `gpu_tile` inner states, as in scGD0).
`save_scgd0_ladder` / `load_scgd0_ladder` write and read a result with JLD2.

Example drivers are in `examples/scgd0/`: the 1D Holstein chain (`run_scgd0_holstein_1d.jl`,
`run_scgd0_ladder_holstein_1d.jl`) and SrVO3 (`run_scgd0_srvo3.jl`).
