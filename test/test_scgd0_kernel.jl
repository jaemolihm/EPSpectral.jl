@testset "scGD0 kernel" begin
    @testset "add_fan_migdal_imsigma!" begin
        # The broadcast reduction (threaded over i on the host) against a plain loop over
        # (ω, ν, i, f) on non-constant data: per-pair ωq, one mode below ω_acoustic with an
        # infinite g2 (skipped, so no NaN), a random f -> i map and a complex Σin.
        nm, ni, nf = 3, 13, 17
        # A graded ωs: 0.1-wide cells in the middle, 0.02-wide on the outside.
        ωs = vcat(-1.0:0.02:-0.6, -0.5:0.1:0.5, 0.6:0.02:1.0)
        nω = length(ωs)
        Σin = complex.(0.1 .* pseudorandom(nω, ni; seed = 7),
                       -0.05 .- 0.05 .* abs.(pseudorandom(nω, ni; seed = 8)))
        g2 = abs.(pseudorandom(nm, ni, nf; seed = 9))
        ωq = 0.05 .+ 0.2 .* abs.(pseudorandom(nm, ni, nf; seed = 10))
        ω_acoustic = 1e-3
        ωq[2, 3, :] .= 0.0
        g2[2, 3, :] .= Inf
        ε_f = 0.8 .* pseudorandom(nf; seed = 11)
        f_to_i = [mod(5f, ni) + 1 for f in 1:nf]
        w_f = fill(1 / nf, nf)
        μ, T = 0.07, 0.03
        Σ_itps = [extrapolate(interpolate((ωs,), Σin[:, i], Gridded(Linear())), Flat())
                  for i in 1:ni]

        # The (ν f, i) layout of `add_fan_migdal_imsigma!`, from the [ν, i, f] arrays of the reference.
        fused(x) = reshape(permutedims(x, (1, 3, 2)), nm * nf, ni)
        per_νf(x) = vec(repeat(reshape(x, 1, nf), nm))
        ImΣ = fill(0.5, nω, ni)   # accumulated into
        ωs_cell_table = EPSpectral.cell_search_table(EP.CPUBackend(), ωs)
        EPSpectral.add_fan_migdal_imsigma!(ImΣ, fused(g2), fused(ωq), ωs, ωs_cell_table,
            per_νf(ε_f), per_νf(f_to_i), Σin,
            μ, T, per_νf(w_f), ω_acoustic)
        ImΣ_ref = fill(0.5, nω, ni)
        fd(x) = 1 / (exp(x / T) + 1)
        for f in 1:nf, i in 1:ni, ν in 1:nm, (iω, ω) in enumerate(ωs)
            ωq[ν, i, f] < ω_acoustic && continue
            n = 1 / expm1(ωq[ν, i, f] / T)
            Σp = Σ_itps[f_to_i[f]](ω + ωq[ν, i, f])
            Σm = Σ_itps[f_to_i[f]](ω - ωq[ν, i, f])
            ImΣ_ref[iω, i] += w_f[f] * g2[ν, i, f] *
                (imag(1 / (ω + ωq[ν, i, f] - ε_f[f] - Σp)) * (n + fd(ω + ωq[ν, i, f] - μ))
               + imag(1 / (ω - ωq[ν, i, f] - ε_f[f] - Σm)) * (n + 1 - fd(ω - ωq[ν, i, f] - μ)))
        end
        @test all(isfinite, ImΣ)
        @test norm(ImΣ - ImΣ_ref) <= 1e-13 * norm(ImΣ_ref)
    end
end
