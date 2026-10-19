# Evaluated at the start of every test item with `setup = [Common]`: the imports, deterministic
# test data and the fixtures.
@testsnippet Common begin
    using ElectronPhonon
    using LinearAlgebra: norm
    using JLD2: jldopen
    using Interpolations: interpolate, extrapolate, Gridded, Linear, Flat
    const EP = ElectronPhonon

    # Deterministic test data in [-1, 1], without a Random dependency.
    pseudorandom(dims...; seed) = reshape(sin.(seed .* (1:prod(dims)) .^ 1.5), dims)
    pseudorandom_complex(dims...; seed) =
        complex.(pseudorandom(dims...; seed), pseudorandom(dims...; seed = seed + 0.5))

    include("holstein_1d_reference.jl")
    include("pb_fixture.jl")
    include("ladder_fixtures.jl")
end
