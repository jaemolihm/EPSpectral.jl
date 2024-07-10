# Phonon self-energy for the Frohlich model

function compute_electron_susceptibility(
        S :: ElectronPhononSolver{FrohlichModel},
        q :: SVector{3, Float64},
        ω :: Float64,
        kpts :: Kpoints,
        Σ_itp,
        ;
        nν = 100
    )

    model = S.model
    μ = model.μ
    νs = range(μ - ω, μ; length=nν)
    dν = νs[2] - νs[1]

    χ0_imag = tmapreduce(+, 1:length(kpts)) do ik
        k, weight = kpts[ik]
        kk = norm(k)
        kq = norm(k + q)
        εkk = get_εk(k, model)
        εkq = get_εk(k + q, model)

        # ∫_{μ-ω}^{μ} dν Gkk(ν) Gkq(ν + ω)
        # (μ-ω, μ) is the integration range for Gkk
        # (μ, μ+ω) is the integration range for Gkq
        val = 0.0
        for ν in νs
            Gkk = imag(1 / (ν - εkk - Σ_itp(ν, kk)))
            Gkq = imag(1 / (ν + ω - εkq - Σ_itp(ν + ω, kq)))
            val += Gkk * Gkq
        end
        val * dν / -π * weight
    end
    χ0_imag *= 2 / (2π)^3  # 2 comes from the spin degeneracy

    χ0_imag
end


function compute_electron_susceptibility(
        S :: ElectronPhononSolver{FrohlichModel},
        q :: SVector{3, Float64},
        ωs :: AbstractVector{Float64},
        kpts :: Kpoints,
        ;
        nν = 100
    )

    if !(ωs ≈ .-reverse(ωs))
        throw(ArgumentError("ωs must be symmetric about zero"))
    end

    Σ_itp = linear_interpolation((S.ωs, S.ks), S.Σs .- im .* S.η; extrapolation_bc = Flat());
    χ0_imag = zeros(length(ωs))
    
    # First compute Im χ0(ω) for ω >= 0
    for (iω, ω) in enumerate(ωs)
        if ω >= 0
            χ0_imag[iω] = compute_electron_susceptibility(S, q, ω, kpts, Σ_itp; nν)
        end
    end

    # Then fill in Im χ0(ω) for ω < 0 using the relation Im χ0(ω) = -Im χ0(-ω)
    for (iω, ω) in enumerate(ωs)
        if ω < 0
            χ0_imag[iω] = -χ0_imag[end - iω + 1]
        end
    end

    χ0_real = kramers_kronig(ωs, χ0_imag)

    χ0 = χ0_real .+ im .* χ0_imag
    χ0
end
