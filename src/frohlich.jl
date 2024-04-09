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
Retarded self-energy of the undoped Frohlich model (equals the greater self-energy)
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
