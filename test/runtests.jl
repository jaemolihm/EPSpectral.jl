using Test
using EPSpectral
using ElectronPhonon
using LinearAlgebra: norm
using Interpolations: interpolate, extrapolate, scale, Gridded, BSpline, Linear, Flat
const EP = ElectronPhonon

# Optional GPU dependency (skipped if absent). No testset uses it yet.
const HAVE_CUDA = try
    @eval using CUDA
    CUDA.functional()
catch
    false
end
HAVE_CUDA && CUDA.allowscalar(false)

# Deterministic test data in [-1, 1], without a Random dependency.
pseudorandom(dims...; seed) = reshape(sin.(seed .* (1:prod(dims)) .^ 1.5), dims)
pseudorandom_complex(dims...; seed) =
    complex.(pseudorandom(dims...; seed), pseudorandom(dims...; seed = seed + 0.5))

include("holstein_1d_reference.jl")

@testset "EPSpectral.jl" begin
    include("test_kramers_kronig.jl")
    include("test_scgd0_grids.jl")
    include("test_scgd0_kernel.jl")
    include("test_scgd0_mu.jl")
    include("test_scgd0_loop.jl")
    include("test_scgd0_holstein.jl")
end
