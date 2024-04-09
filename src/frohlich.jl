# Frohlich model for the electron-phonon interaction
# We use the atomic Hartree units (ħ = mₑ = e² = 4πε0 = 1).
#
# References
# [1] N. Kandolf et al., PRB 105, 085148 (2022)

struct FrohlichModel
    α  :: Float64
    ω₀ :: Float64
    m  :: Float64
end

get_εk(k, param::FrohlichModel) = norm(k)^2 / 2 / param.m


"""
    get_Σ_analytic(k, ω, param::FrohlichModel)

The retarded self-energy of the undoped Frohlich model (equals the greater self-energy),
computed using the analytic formula.
Implements Eq.(28) of Ref.[1]. (The π in the denominator is a typo and is removed.)
"""
function get_Σ_analytic(k, ω, param::FrohlichModel)
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
    (; α, ω₀, m) = param

    Σ = tmapreduce(+, 1:length(qpts)) do iq
        q, weight = qpts[iq]

        if norm(q) < sqrt(eps(Float64))
            zero(complex(ω))
        else
            εkq = get_εk(k .+ q, param)
            1 / (ω - ω₀ - εkq) / norm(q)^2 * weight
        end
    end
    Σ *= α * sqrt(ω₀^3 / 2m) / (2 * π^2)
    Σ
end
