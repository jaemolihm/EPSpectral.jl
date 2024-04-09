module EPSpectral

using LinearAlgebra

include("frohlich.jl")

export
    FrohlichModel,
    get_εk,
    get_Σ_analytic

end
