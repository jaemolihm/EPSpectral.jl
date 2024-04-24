# Frohlich model for the electron-phonon interaction
# We use the atomic Hartree units (ħ = mₑ = e² = 4πε0 = 1).
#
# References
# [1] N. Kandolf et al., PRB 105, 085148 (2022)

struct FrohlichModel
    α  :: Float64
    ω₀ :: Float64
    m  :: Float64
    μ  :: Float64
end

get_εk(k, param::FrohlichModel) = norm(k)^2 / 2 / param.m

Base.Broadcast.broadcastable(param::FrohlichModel) = Ref(param)


"""
    get_Σ_analytic(k, ω, param::FrohlichModel)

The retarded self-energy of the undoped Frohlich model (equals the greater self-energy),
computed using the analytic formula.
Implements Eq.(28) of Ref.[1]. (The π in the denominator is a typo and is removed.)
"""
function get_Σ_analytic(k, ω, param::FrohlichModel)
    if param.μ !== -Inf
        @warn "Analytic formula is only implemented for μ = -∞. Using the μ = -∞ formula."
    end

    εk = get_εk(k, param)
    (; ω₀, α) = param
    if εk < eps(typeof(εk))
        # Case k = 0
        -im * α * ω₀^1.5 / sqrt(ω - ω₀)
    else
        # Case k /= 0
        -im * α * ω₀^1.5 / (2 * sqrt(εk)) * log((sqrt(ω - ω₀) + sqrt(εk)) / (sqrt(ω - ω₀) - sqrt(εk)))
    end
end


"""
    get_Σ_mesh(k, ω, qpts :: Kpoints, param :: FrohlichModel)

Retarded self-energy of the undoped Frohlich model (equals the greater self-energy),
computed by numerical summation on the mesh `qpts`.
Implements Eq.(13) of Ref.[1] with `g(q)` from Eq.(2).
"""
function get_Σ_mesh(k :: T, ω, qpts :: Kpoints{T}, param :: FrohlichModel) where {T}
    (; α, ω₀, m, μ) = param

    Σ = tmapreduce(+, 1:length(qpts)) do iq
        q, weight = qpts[iq]

        val = zero(complex(ω))

        if norm(q) > sqrt(eps(Float64))
            εkq = get_εk(k .+ q, param)
            factor = 1 / norm(q)^2 * weight
            val = zero(complex(ω))
            if real(ω) > μ + ω₀
                val += 1 / (ω - εkq - ω₀) * factor
            elseif real(ω) < μ - ω₀
                val += 1 / (ω - εkq + ω₀) * factor
            end
            if isfinite(μ)
                val += imag(log((εkq - μ) / (ω - μ - ω₀)) / (ω - εkq - ω₀)) / π * factor
                val -= imag(log((εkq - μ) / (ω - μ + ω₀)) / (ω - εkq + ω₀)) / π * factor
            end
        end

        val
    end
    Σ *= α * sqrt(ω₀^3 / 2m) / (2 * π^2)
    Σ
end


function get_Σ_mesh(k :: T, ωs :: AbstractVector, qpts :: Kpoints{T}, param :: FrohlichModel) where {T}
    (; α, ω₀, m, μ) = param

    Σ = tmapreduce(.+, chunks(1:length(qpts); n = 2 * Threads.nthreads()); chunking = false) do iqs
        Σ_tmp = zeros(ComplexF64, length(ωs))

        for iq in iqs
            q, weight = qpts[iq]

            if norm(q) < sqrt(eps(Float64))
                continue
            end

            εkq = get_εk(k .+ q, param)
            factor = 1 / norm(q)^2 * weight

            for (iω, ω) in enumerate(ωs)
                if real(ω) > μ + ω₀
                    Σ_tmp[iω] += imag(1 / (ω - εkq - ω₀)) * factor
                elseif real(ω) < μ - ω₀
                    Σ_tmp[iω] += imag(1 / (ω - εkq + ω₀)) * factor
                end
                # Σ_tmp[iω] += imag(log((εkq - μ) / (ω - μ - ω₀)) / (ω - εkq - ω₀)) / π * factor
                # Σ_tmp[iω] -= imag(log((εkq - μ) / (ω - μ + ω₀)) / (ω - εkq + ω₀)) / π * factor
            end
        end

        Σ_tmp
    end
    Σ *= α * sqrt(ω₀^3 / 2m) / (2 * π^2)

    Σ_real = kramers_kronig(real.(ωs), Σ; tail = false)

    return Σ_real .+ im .* Σ
end
