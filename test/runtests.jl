using Test
using EPSpectral
using ElectronPhonon
using LinearAlgebra: norm
using JLD2: jldopen
using Interpolations: interpolate, extrapolate, Gridded, Linear, Flat
const EP = ElectronPhonon

# Optional GPU dependency for the CPU-vs-GPU testsets (skipped if absent).
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
include("pb_fixture.jl")
include("ladder_fixtures.jl")

# `Pkg.test(; test_args = ["test_frequency_grids.jl", ...])` runs only the named files.
const CPU_TEST_FILES = ["test_kramers_kronig.jl", "test_frequency_grids.jl", "test_scgd0_kernel.jl",
    "test_scgd0_mu.jl", "test_scgd0_loop.jl", "test_scgd0_holstein.jl",
    "test_scgd0_spectral_grid.jl", "test_scgd0_pb.jl", "test_ladder_bubble.jl",
    "test_ladder_kernel.jl", "test_ladder_holstein.jl", "test_ladder_pb.jl"]
const GPU_TEST_FILES = ["test_scgd0_gpu.jl", "test_ladder_gpu.jl"]
for file in ARGS
    file in CPU_TEST_FILES || file in GPU_TEST_FILES || error("unknown test file $file")
end
selected(file) = isempty(ARGS) || file in ARGS

@testset "EPSpectral.jl" begin
    for file in filter(selected, CPU_TEST_FILES)
        include(file)
    end
    if HAVE_CUDA
        for file in filter(selected, GPU_TEST_FILES)
            include(file)
        end
    else
        @info "CUDA not functional: skipping the CPU-vs-GPU testsets"
    end
end
