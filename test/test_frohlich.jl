using EPSpectral
using StaticArrays
using LinearAlgebra
using Test

@testset "Kpoints" begin
    using EPSpectral: get_dimension

    kmax = 3.0
    for n in [4, 7]
        kpts = polynomial_grid_1d(kmax, n, 1)
        @test get_dimension(kpts) == 1
        @test kpts.vectors ≈ range(-kmax, kmax, length=n)
        @test kpts.weights ≈ fill(2kmax / (n - 1), n)
    end

    kpts = polynomial_grid_1d(kmax, 4, 2)
    @test get_dimension(kpts) == 1
    @test kpts.vectors ≈ kmax .* [-1, -1/9, 1/9, 1]
    @test kpts.weights ≈ kmax .* [5/9, 5/9, 5/9, 5/9]

    kpts = polynomial_grid_1d(kmax, 5, 2)
    @test get_dimension(kpts) == 1
    @test kpts.vectors ≈ kmax .* [-1, -1/4, 0, 1/4, 1]
    @test kpts.weights ≈ kmax .* [1/2, 1/2, 1/4, 1/2, 1/2]

    kpts = polynomial_grid_3d(kmax, 6, 3)
    @test get_dimension(kpts) == 3
    @test kpts isa EPSpectral.Kpoints{SVector{3, Float64}}
    @test all(norm.(kpts.vectors) .<= kmax)
end

@testset "Frohlich" begin
    α, ω0, m, μ, T = 2.0, 1.0, 0.5, -Inf, 0.
    model = FrohlichModel(α, ω0, m, μ, T)

    k = (0.5, 0.3, 0.2)
    @test get_εk(0.0, model) ≈ 0.0^2 / 2m
    @test get_εk(1.0, model) ≈ 1.0^2 / 2m
    @test get_εk(2.0, model) ≈ 2.0^2 / 2m
    @test get_εk(k, model) ≈ sum(abs2.(k)) / 2m

    # Test analytic formula

    @test real(get_Σ_analytic(0., 0. + 1e-5im, model)) ≈ -α
    @test real(get_Σ_analytic(1e-5, 1e-10/2m + 1e-5im, model)) ≈ -α
    @test get_Σ_analytic(0.5, 0.6 + 0.1im, model) ≈ -2.642829175728291 - 0.23803240733301528im
    @test get_Σ_analytic(0.5, 2.0 + 0.1im, model) ≈ -0.13209525264524624 - 2.18513635988446im

    # Test numerical integration on the mesh

    qpts = polynomial_grid_3d(15.0, 100, 4)

    k = 0.0
    ω = k^2 / 2m + 0.05im
    @test get_Σ_mesh(SVector(k, 0, 0), ω, qpts, model) ≈ -1.9432408368379066 - 0.05108721482214386im
    @test get_Σ_mesh(SVector(k, 0, 0), ω, qpts, model) ≈ get_Σ_analytic(k, ω, model) atol=1e-1

    k = 2.0
    ω = k^2 / 2m + 0.2im
    @test get_Σ_mesh(SVector(k, 0, 0), ω, qpts, model) ≈ -1.3822449385409892 - 1.3298515254158005im
    @test get_Σ_mesh(SVector(k, 0, 0), ω, qpts, model) ≈ get_Σ_analytic(k, ω, model) atol=1e-1
end
