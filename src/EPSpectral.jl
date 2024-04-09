module EPSpectral

using LinearAlgebra
using StaticArrays
using OhMyThreads

include("kpoints.jl")
include("frohlich.jl")

export
    Kpoints,
    polynomial_grid_1d,
    polynomial_grid_3d,
    FrohlichModel,
    get_εk,
    get_Σ_analytic,
    get_Σ_mesh

end
