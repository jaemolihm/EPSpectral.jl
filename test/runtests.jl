using EPSpectral
using Test

@testset "EPSpectral.jl" begin
    include("test_frohlich.jl")
    include("test_kramers_kronig.jl")
    include("test_cumulant.jl")
end
