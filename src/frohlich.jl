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

struct FrohlichModel_2D
    α  :: Float64
    ω₀ :: Float64
    m  :: Float64
    μ  :: Float64
end

Base.Broadcast.broadcastable(param::FrohlichModel) = Ref(param)
Base.Broadcast.broadcastable(param::FrohlichModel_2D) = Ref(param)

get_εk(k, param::FrohlichModel) = norm(k)^2 / 2 / param.m
get_εk(k, param::FrohlichModel_2D) = norm(k)^2 / 2 / param.m

get_vk(k, param::FrohlichModel) = k / param.m

function get_eph_g(q, model :: FrohlichModel)
    (; α, ω₀, m) = model
    if norm(q) == 0
        return 0.0
    else
        return sqrt(4π * α * sqrt(ω₀^3 / 2m)) / norm(q)
    end
end

function L(z1, z2)
    # Eq.(41) of Ref.[1]. (Typos on the sign of second and third terms fixed)
    li2((1 + z1) / (1 + z2)) + li2((1 + z1) / (1 - z2)) - li2((1 - z1) / (1 + z2)) - li2((1 - z1) / (1 - z2))
end


"""
    get_Σ_analytic(k, ω, param::FrohlichModel)

The retarded self-energy of the undoped Frohlich model (equals the greater self-energy),
computed using the analytic formula.
For μ < 0, use Eq.(28) of Ref.[1]. (The π in the denominator is a typo and is removed.)
For μ > 0, use Eq.(39-42) of Ref.[1]. (The π in the denominator is a typo and is removed.)
"""
function get_Σ_analytic(k, ω, param::FrohlichModel, T = 0.0)
    (; ω₀, α, μ) = param
    εk = get_εk(k, param)

    nq = occ_boson(ω₀, T)

    if μ < 0
        if εk < eps(typeof(εk))
            # Case k = 0
            Σ_emi = -im * α * ω₀^1.5 / √(ω - ω₀)
            Σ_abs = -im * α * ω₀^1.5 / √(ω + ω₀)
        else
            # Case k /= 0
            Σ_emi = -im * α * ω₀^1.5 / (2 * √(εk)) * log((√(ω - ω₀) + √(εk)) / (√(ω - ω₀) - √(εk)))
            Σ_abs = -im * α * ω₀^1.5 / (2 * √(εk)) * log((√(ω + ω₀) + √(εk)) / (√(ω + ω₀) - √(εk)))
        end

        return Σ_emi * (nq + 1) + Σ_abs * nq

    else
        T > 0 && throw(ArgumentError("μ > 0 and T > 0 not implemented"))
        if εk < eps(typeof(εk))
            # Eq.(B9) of Ref.[1]
            Σles = log((√(conj(ω) + ω₀) + √(μ)) / (√(conj(ω) + ω₀) - √(μ))) / √(conj(ω) + ω₀)

            # Eq.(B11) of Ref.[1]
            Σgtr = -(log((√(ω - ω₀) + √(μ)) / (√(ω - ω₀) - √(μ))) + im * π) / √(ω - ω₀)

            return (conj(Σles) + Σgtr) * α * ω₀^1.5 / π
        else
            # Eq.(39) of Ref.[1] without the last Σ(E_F) term
            Σles = -(
                L(√(μ / εk), √((conj(ω) + ω₀) / εk))
                + log((conj(ω) + ω₀ - μ) / (conj(ω) + ω₀ - εk)) * log(abs((√(μ) + √(εk)) / (√(μ) - √(εk))))
            )

            # Eq.(42) of Ref.[1] without the last Σ(E_F) term
            # (Typo on the sign of the denominator in the second term fixed)
            Σgtr = (
                L(√(μ / εk), √((ω - ω₀) / εk))
                + log((ω - ω₀ - μ) / (ω - ω₀ - εk)) * log(abs((√(μ) + √(εk)) / (√(μ) - √(εk))))
                - im * π * log((√(ω - ω₀) + √(εk)) / (√(ω - ω₀) - √(εk)))
            )

            return (conj(Σles) + Σgtr) * α * ω₀^1.5 / 2π / √(εk)
        end

    end
end

@inline function retarded_self_energy_single_pole(ω :: Float64, zkq :: ComplexF64, ωq :: Float64, μ :: Float64)
    Σ = 0.0im
    if real(ω) > μ + ωq
        Σ += 1 / (ω - zkq - ωq)
    elseif real(ω) < μ - ωq
        Σ += 1 / (ω - zkq + ωq)
    end
    if isfinite(μ)
        Σ += imag(log((zkq - μ) / (ω - μ - ωq)) / (ω - zkq - ωq)) / π
        Σ -= imag(log((zkq - μ) / (ω - μ + ωq)) / (ω - zkq + ωq)) / π
    end
    Σ
end


"""
    get_Σ_mesh(k, ω, qpts :: Kpoints, param :: FrohlichModel)

Retarded self-energy of the undoped Frohlich model (equals the greater self-energy),
computed by numerical summation on the mesh `qpts`.
Implements Eq.(13) of Ref.[1] with `g(q)` from Eq.(2).
"""
function get_Σ_mesh(k :: T, ω, qpts :: Kpoints{T}, param :: FrohlichModel; linewidth_on_electron = true) where {T}
    (; α, ω₀, m, μ) = param

    Σ = tmapreduce(+, 1:length(qpts)) do iq
        q, weight = qpts[iq]

        Σq = zero(complex(ω))

        if norm(q) > sqrt(eps(Float64))
            εkq = get_εk(k .+ q, param)
            factor = 1 / norm(q)^2 * weight

            if linewidth_on_electron
                # imag(ω) is linewidth of electrons
                Σq = retarded_self_energy_single_pole(real(ω), εkq - im * imag(ω), ω₀, μ)

            else
                # imag(ω) is linewidth of phonons
                if εkq > μ
                    Σq += 1 / (ω - εkq - ω₀)
                elseif εkq < μ
                    Σq += 1 / (ω - εkq + ω₀)
                end
            end

            Σq *= factor
        end

        Σq
    end
    Σ *= α * sqrt(ω₀^3 / 2m) / (2 * π^2)
    Σ
end


function get_Σ_mesh(k :: T, ωs :: AbstractVector, qpts :: Kpoints{T}, param :: FrohlichModel; linewidth_on_electron = true) where {T}
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
                if linewidth_on_electron
                    # imag(ω) is linewidth of electrons
                    Σ_tmp[iω] += retarded_self_energy_single_pole(real(ω), εkq - im * imag(ω), ω₀, μ) * factor

                else
                    # imag(ω) is linewidth of phonons
                    if εkq > μ
                        Σ_tmp[iω] += 1 / (ω - εkq - ω₀) * factor
                    elseif εkq < μ
                        Σ_tmp[iω] += 1 / (ω - εkq + ω₀) * factor
                    end
                end
            end
        end

        Σ_tmp
    end
    Σ *= α * sqrt(ω₀^3 / 2m) / (2 * π^2)

    return Σ
end
