@testset "scGD0 kernel" begin
    @testset "fm_imsigma_tile!" begin
        # The broadcast reduction (threaded over i on the host) against a plain loop over
        # (ω, ν, i, f) on non-constant data: per-pair ωq, one mode below ω_acoustic with an
        # infinite g2 (skipped, so no NaN), a random f -> i map and a complex Σin.
        nω, nm, ni, nf = 23, 3, 13, 17
        ωs = collect(range(-1.0, 1.0; length = nω))
        ωs_dense = range(-1.0, 1.0; step = 0.013)
        Σin = complex.(0.1 .* pseudorandom(length(ωs_dense), ni; seed = 7),
                       -0.05 .- 0.05 .* abs.(pseudorandom(length(ωs_dense), ni; seed = 8)))
        g2 = abs.(pseudorandom(nm, ni, nf; seed = 9))
        ωq = 0.05 .+ 0.2 .* abs.(pseudorandom(nm, ni, nf; seed = 10))
        ω_acoustic = 1e-3
        ωq[2, 3, :] .= 0.0
        g2[2, 3, :] .= Inf
        ε_f = 0.8 .* pseudorandom(nf; seed = 11)
        f_to_i = [mod(5f, ni) + 1 for f in 1:nf]
        w_f = fill(1 / nf, nf)
        μ, T = 0.07, 0.03
        ωd0, dωd = first(ωs_dense), step(ωs_dense)

        ImΣ = fill(0.5, nω, ni)   # accumulated into
        EPSpectral.fm_imsigma_tile!(ImΣ, g2, ωq, ωs, ε_f, f_to_i, Σin, ωd0, dωd, μ, T, w_f,
            ω_acoustic)
        ImΣ_ref = fill(0.5, nω, ni)
        fd(x) = 1 / (exp(x / T) + 1)
        for f in 1:nf, i in 1:ni, ν in 1:nm, (iω, ω) in enumerate(ωs)
            ωq[ν, i, f] < ω_acoustic && continue
            n = 1 / expm1(ωq[ν, i, f] / T)
            Σp = EPSpectral._lerp_flat(Σin, f_to_i[f], ω + ωq[ν, i, f], ωd0, dωd)
            Σm = EPSpectral._lerp_flat(Σin, f_to_i[f], ω - ωq[ν, i, f], ωd0, dωd)
            ImΣ_ref[iω, i] += w_f[f] * g2[ν, i, f] *
                (imag(1 / (ω + ωq[ν, i, f] - ε_f[f] - Σp)) * (n + fd(ω + ωq[ν, i, f] - μ))
               + imag(1 / (ω - ωq[ν, i, f] - ε_f[f] - Σm)) * (n + 1 - fd(ω - ωq[ν, i, f] - μ)))
        end
        @test all(isfinite, ImΣ)
        @test norm(ImΣ - ImΣ_ref) <= 1e-13 * norm(ImΣ_ref)
    end
end
