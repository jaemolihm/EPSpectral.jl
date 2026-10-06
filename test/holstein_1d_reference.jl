# The 1D Holstein chain with constant coupling g2 and phonon frequency ω0, ε_j = -2t cos(2πj/nk),
# j = 0, ..., nk-1: independent references that do not call EPSpectral's iteration, then the
# hand-built input that runs it directly.
using Roots: find_zero

holstein_1d_energies(nk, t) = [-2t * cospi(2j / nk) for j in 0:nk-1]

# Im Σ of the one-shot G0D0 at chemical potential μ, with Σ_in ≡ -iη, in closed form.
function holstein_1d_oneshot_imsigma(ωs; nk, t, ω0, g2, T, μ, η)
    n_B = 1 / expm1(ω0 / T)
    f(x) = 1 / (exp(x / T) + 1)
    εs = holstein_1d_energies(nk, t)
    map(ωs) do ω
        -g2 / nk * sum(η / ((ω + ω0 - ε)^2 + η^2) * (n_B + f(ω + ω0 - μ))
                     + η / ((ω - ω0 - ε)^2 + η^2) * (n_B + 1 - f(ω - ω0 - μ)) for ε in εs)
    end
end

# The scGD0 iteration as a scalar problem in ω: with constant g2 and ω0, Σ is k-independent and
# the Fan-Migdal sum only needs the local G. Same grids, clamp, flat linear interpolation,
# Kramers-Kronig, rectangle-rule spectral weight and bracketed μ root as the package, each written
# out here.
# Returns Σ on ωs and μ after `niter` iterations.
function holstein_1d_local_scgd0(ωs, ωs_dense; nk, t, ω0, g2, T, η_init, η_min = 0.0, μ_start,
        niter, fix_μ, n_target, spin_degeneracy)
    εs = holstein_1d_energies(nk, t)
    n_B = 1 / expm1(ω0 / T)
    f(x) = 1 / (exp(x / T) + 1)
    dω = step(ωs_dense)
    on_dense(Σ) = extrapolate(interpolate((ωs,), Σ, Gridded(Linear())), Flat()).(ωs_dense)
    Σ = fill(complex(0.0, -η_init), length(ωs))
    μ = μ_start
    for _ in 1:niter
        Σin = [complex(real(z), min(imag(z), -η_min)) for z in on_dense(Σ)]
        Σin_itp = extrapolate(scale(interpolate(Σin, BSpline(Linear())), ωs_dense), Flat())
        ImG_loc(x) = imag(sum(1 / (x - ε - Σin_itp(x)) for ε in εs) / nk)
        ImΣ = [g2 * (ImG_loc(ω + ω0) * (n_B + f(ω + ω0 - μ))
                     + ImG_loc(ω - ω0) * (n_B + 1 - f(ω - ω0 - μ))) for ω in ωs]
        Σ = complex.(kramers_kronig(ωs, ImΣ), ImΣ)
        fix_μ && continue
        Σd = on_dense(Σ)
        D = [sum(-imag(1 / (ω - ε - Σd[r])) / π for ε in εs) / nk * dω
             for (r, ω) in enumerate(ωs_dense)]
        μ = find_zero(μ_ -> spin_degeneracy * sum(D .* f.(ωs_dense .- μ_)) - n_target,
            extrema(ωs_dense) .+ (-0.1, 0.1))
    end
    Σ, μ
end

# Bare-band chemical potential of the chain for `n_target` electrons per cell.
function holstein_1d_bare_μ(; nk, t, T, n_target, spin_degeneracy)
    εs = holstein_1d_energies(nk, t)
    find_zero(μ -> spin_degeneracy * sum(1 / (exp((ε - μ) / T) + 1) for ε in εs) / nk - n_target,
        (minimum(εs) - 1, maximum(εs) + 1))
end

# The input of `EPSpectral._loop_scgd0` for the chain, built by hand instead of by `_setup_scgd0`:
# outer states on the irreducible wedge j = 0, ..., nk/2 of k -> -k, inner states on the full grid,
# one mode of frequency ω0 everywhere, constant coupling g2.
function holstein_1d_loop_setup(; nk, t, ω0, g2, ωs, μ_start)
    ni = nk ÷ 2 + 1
    εs = holstein_1d_energies(nk, t)
    ωs_dense = range(extrema(ωs)...; step = minimum(diff(ωs)))
    (; g2 = fill(g2, 1, ni, nk), ωph = fill(ω0, 1, nk),
       cs_i = [EP.Vec3(j, 0, 0) for j in 0:ni-1], cs_f = [EP.Vec3(j, 0, 0) for j in 0:nk-1],
       ngrid = (nk, 1, 1), ε_i = εs[1:ni],
       w_i = [(j == 0 || 2j == nk ? 1 : 2) / nk for j in 0:ni-1],
       ε_f = εs, w_f = fill(1 / nk, nk), f_to_i = [min(j, nk - j) + 1 for j in 0:nk-1],
       W = 1.0, nstates_base = 0.0, multiplets = Vector{Int}[], ωs, ωs_dense,
       K = EPSpectral.kk_matrix(ωs), P = EPSpectral.interp_matrix(ωs, ωs_dense), μ_start)
end

# `_loop_scgd0` with the keywords `run_scgd0` would pass for a CPU run.
run_loop_scgd0(setup, occ; η_min = zeros(length(occ)), η_init, tol, maxiter = 20, fix_μ = false,
    gpu_tile = nothing, Σ_init = nothing) =
    EPSpectral._loop_scgd0(setup, occ; η_min, η_init, maxiter, tol, backend = EP.CPUBackend(),
        gpu_tile, Σ_init, fix_μ, ω_acoustic = EP.omega_acoustic, verbosity = 0)
