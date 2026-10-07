# The carrier count of the μ solve: n(μ) from the spectral DOS D_T against the old per-state
# formula (`compute_occupation!` of the v2 solver), transcribed below.
@testset "scGD0 carrier count" begin
    # n(μ) of the old code: per state, the rectangle-rule ∫A f (electrons) or ∫A (1 - f) (holes,
    # counted as filled minus holes) on the dense grid, summed with the state weights.
    function n_old(μ, T, ωs_dense, Σd, ε_i, w_i, nstates_base, spin_degeneracy, count_holes)
        dω = step(ωs_dense)
        fermi = count_holes ? [1 / (exp((μ - ω) / T) + 1) for ω in ωs_dense] :
                              [1 / (exp((ω - μ) / T) + 1) for ω in ωs_dense]
        nocc = 0.0
        nocc_base = 0.0
        for i in eachindex(ε_i)
            As = @. -imag(1 / (ωs_dense - ε_i[i] - Σd[:, i])) / π
            spectral_occ = sum(As .* fermi) * dω
            if count_holes
                nocc -= spectral_occ * w_i[i]
                nocc_base += w_i[i]
            else
                nocc += spectral_occ * w_i[i]
            end
        end
        (nocc + nocc_base + nstates_base) * spin_degeneracy
    end

    # D_T as the iteration builds it.
    function spectral_dos(ωs_dense, Σd, ε_i, w_i)
        A = .-imag.(inv.(ωs_dense .- transpose(ε_i) .- Σd)) ./ π
        A * w_i .* step(ωs_dense)
    end

    @testset "compute_ncarrier_spectral" begin
        ωs_dense = range(-1.0, 1.0; step = 0.002)
        ni = 9
        ε_i = 0.7 .* pseudorandom(ni; seed = 12)
        w_i = 0.05 .+ 0.1 .* abs.(pseudorandom(ni; seed = 13))
        Σd = complex.(0.05 .* pseudorandom(length(ωs_dense), ni; seed = 14),
                      -0.01 .- 0.03 .* abs.(pseudorandom(length(ωs_dense), ni; seed = 15)))
        D = spectral_dos(ωs_dense, Σd, ε_i, w_i)
        W, nstates_base, spin_degeneracy = sum(w_i), 0.37, 2
        # μ in the band (metal), above it (electron count of a nearly full window) and below it
        # (hole count of a nearly empty one), at two temperatures.
        for μ in (0.1, 0.9, -0.95), T in (0.01, 0.05), count_holes in (false, true)
            n = EPSpectral.compute_ncarrier_spectral(μ, T, ωs_dense, D, W; count_holes,
                nstates_base, spin_degeneracy)
            n_ref = n_old(μ, T, ωs_dense, Σd, ε_i, w_i, nstates_base, spin_degeneracy, count_holes)
            @test n ≈ n_ref rtol = 1e-12
        end
    end

    @testset "branch per occ.type and doping" begin
        # The μ the loop solves for satisfies the old formula of the selected branch, and not the
        # other one: electron count for :Metal and for :Semiconductor with nlist >= 0, hole count
        # for :Semiconductor with nlist < 0. The two differ by the spectral weight the window
        # misses on the dense grid, W - Σ D_T, 3.7e-5 electrons here.
        nk, t, ω0, g2, T = 64, 1.0, 0.2, 0.2, 0.1
        ωs = collect(-4.0:0.02:4.0)
        s = holstein_1d_loop_setup(; nk, t, ω0, g2, ωs, μ_start = [0.0])
        for (type, nlist, count_holes) in ((:Metal, -0.3, false), (:Semiconductor, 0.3, false),
                                           (:Semiconductor, -0.3, true))
            occ = ElectronOccupationParams(; Tlist = [T], nlist, nelec = 1, volume = 1.0,
                spin_degeneracy = 2, type)
            res = run_loop_scgd0(s, occ; η_init = 0.05, tol = 1e-6, maxiter = 1)
            Σd = stack(extrapolate(interpolate((ωs,), res.Σ[:, i, 1], Gridded(Linear())), Flat()
                ).(s.ωs_dense) for i in axes(res.Σ, 2))
            n_of(holes) = n_old(res.μ[1], T, s.ωs_dense, Σd, s.ε_i, s.w_i, s.nstates_base, 2, holes)
            @test n_of(count_holes) ≈ 1 + nlist rtol = 1e-12
            @test abs(n_of(!count_holes) - (1 + nlist)) > 1e-5
        end
    end
end
