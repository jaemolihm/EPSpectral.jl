using EPSpectral
using Test

@testset "Frohlich" begin
    α, ω0, m = 2.0, 1.0, 0.5
    model = FrohlichModel(α, ω0, m)

    k = (0.5, 0.3, 0.2)
    @test get_εk(0.0, model) ≈ 0.0^2 / 2m
    @test get_εk(1.0, model) ≈ 1.0^2 / 2m
    @test get_εk(2.0, model) ≈ 2.0^2 / 2m
    @test get_εk(k, model) ≈ sum(abs2.(k)) / 2m


    @test real(get_Σ_analytic(0., 0. + 1e-5im, model)) ≈ -α
    @test real(get_Σ_analytic(1e-6, 1e-12/2m + 1e-5im, model)) ≈ -α
    @test get_Σ_analytic(0.5, 0.6 + 0.1im, model) ≈ -2.642829175728291 - 0.23803240733301528im
    @test get_Σ_analytic(0.5, 2.0 + 0.1im, model) ≈ -0.13209525264524624 - 2.18513635988446im
end
