mutable struct FrohlichSolver
    model :: FrohlichModel
    η :: Float64
    T :: Float64
    occupation :: Float64

    ωs :: StepRangeLen{Float64, Base.TwicePrecision{Float64}, Base.TwicePrecision{Float64}, Int64}
    ks :: StepRangeLen{Float64, Base.TwicePrecision{Float64}, Base.TwicePrecision{Float64}, Int64}
    ωs_dense :: StepRangeLen{Float64, Base.TwicePrecision{Float64}, Base.TwicePrecision{Float64}, Int64}
    ks_dense :: StepRangeLen{Float64, Base.TwicePrecision{Float64}, Base.TwicePrecision{Float64}, Int64}
    Σs :: Matrix{ComplexF64}
    As :: Matrix{Float64}
    Σs_dense :: Matrix{ComplexF64}
    As_dense :: Matrix{Float64}

    spectral_sum :: Vector{Float64}
    spectral_occ :: Vector{Float64}
end

function FrohlichSolver(model, ωs, ks; occupation, ωs_dense = ωs, ks_dense = ks, η, T = 1e-5)
    Σs = zeros(ComplexF64, length(ωs), length(ks))
    As = zeros(length(ωs), length(ks))
    Σs_dense = zeros(ComplexF64, length(ωs_dense), length(ks_dense))
    As_dense = zeros(length(ωs_dense), length(ks_dense))
    spectral_sum = zeros(length(ks))
    spectral_occ = zeros(length(ks))

    FrohlichSolver(model, η, T, occupation, ωs, ks, ωs_dense, ks_dense,
        Σs, As, Σs_dense, As_dense, spectral_sum, spectral_occ)
end

function Base.show(io :: IO, S :: FrohlichSolver)
    print(io, "FrohlichSolver(", S.model, ", η = ", S.η, ", T = ", S.T, ")\n")
    print(io, "ωs       : ", length(S.ωs), " points, ", S.ωs, "\n")
    print(io, "ks       : ", length(S.ks), " points, ", S.ks, "\n")
    print(io, "ωs_dense : ", length(S.ωs_dense), " points, ", S.ωs_dense, "\n")
    print(io, "ks_dense : ", length(S.ks_dense), " points, ", S.ks_dense)
end

function compute_self_energy_analytic!(S :: FrohlichSolver)
    for (ik, k) in enumerate(S.ks)
        S.Σs[:, ik] .= get_Σ_analytic.(k, S.ωs .+ im * S.η, S.model)
    end

    return S
end

function compute_spectral_function!(S :: FrohlichSolver)
    # Interpolate the self-energy from (ωs, ks) to (ωs_dense, ks_dense)
    Σ_itp = linear_interpolation((S.ωs, S.ks), S.Σs; extrapolation_bc = Flat());

    # Compute spectral function
    @views for (ik, k) in enumerate(S.ks_dense)
        εk = get_εk(k, S.model)
        @. S.As_dense[:, ik] = -imag(1 / (S.ωs_dense - εk - Σ_itp(S.ωs_dense, k))) / π
    end

    return S
end


function compute_occupation!(S :: FrohlichSolver)
    dω = S.ωs_dense[2] - S.ωs_dense[1]
    fermi = @. 1 / (exp((S.ωs_dense - S.model.μ) / S.T) + 1)

    @views for (ik, k) in enumerate(S.ks_dense)
        # Compute integral of the spectral function (should be 1 when exact)
        S.spectral_sum[ik] = sum(S.As_dense[:, ik]) * dω

        # Compute occupation
        S.spectral_occ[ik] = sum(S.As_dense[:, ik] .* fermi) * dω
    end

    # Linearly interpolate the occupations
    occ_itp = linear_interpolation(S.ks_dense, S.spectral_occ)

    # Integrate over the Brillouin zone
    n = quadgk(k -> 1 / 2π^2 * k^2 * occ_itp(k), extrema(S.ks_dense)...)[1]

    return n
end

function plot_spectral_function!(ax, S :: FrohlichSolver;
    bare_band = true,
    chemical_potential = true,
    kwargs_plot...
    )

    ks = S.ks_dense
    ωs = S.ωs_dense
    dk = ks[2] - ks[1]
    dω = ωs[2] - ωs[1]
    extent = [ks[1] - dk/2, ks[end] + dk/2, ωs[1] - dω/2, ωs[end] + dω/2]

    img = ax.imshow(S.As_dense; origin="lower", extent, aspect="auto", kwargs_plot...)

    if bare_band
        ax.plot(S.ks_dense, get_εk.(S.ks_dense, S.model), c="grey", ls="--", lw=1)
    end
    if chemical_potential
        ax.axhline(S.model.μ, c="r", ls="--", lw=1)
    end

    return img
end

function update_chemical_potential!(S)

    function compute_occupation(S :: FrohlichSolver, μ)
        # Same as compute_occupation!, but do not update S.spectral_occ

        dω = S.ωs_dense[2] - S.ωs_dense[1]
        fermi = @. 1 / (exp((S.ωs_dense - μ) / S.T) + 1)

        spectral_occ = zero(S.spectral_occ)

        @views for (ik, k) in enumerate(S.ks_dense)
            # Compute occupation
            spectral_occ[ik] = sum(S.As_dense[:, ik] .* fermi) * dω
        end

        # Linearly interpolate the occupations
        occ_itp = linear_interpolation(S.ks_dense, spectral_occ)

        # Integrate over the Brillouin zone
        n = quadgk(k -> 1 / 2π^2 * k^2 * occ_itp(k), extrema(S.ks_dense)...)[1]

        return n
    end

    μ_new = find_zero(μ -> compute_occupation(S, μ) - S.occupation, extrema(S.ωs_dense))

    S.model = FrohlichModel(S.model.α, S.model.ω₀, S.model.m, μ_new)

    return S
end
