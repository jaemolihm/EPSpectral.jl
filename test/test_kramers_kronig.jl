using EPSpectral
using Test

@testset "kramers_kronig" begin
    α = 2.0
    m = 0.5
    ω₀ = 1.0
    η = 0.1
    μ = 0.5
    model = FrohlichModel(α, ω₀, m, μ)

    ωs = range(-100., 100., step=0.04)
    k = 0.2

    Σs_anal = get_Σ_analytic.(k, ωs .+ im * η, model)

    Σs_kk_re = kramers_kronig(ωs, imag.(Σs_anal))
    Σs_kk_im = kramers_kronig(ωs, real.(Σs_anal)) .* -1

    inds = abs.(ωs) .< 5.0
    @test maximum(abs.(Σs_kk_re[inds] .- real.(Σs_anal[inds]))) < 0.14
    @test maximum(abs.(Σs_kk_im[inds] .- imag.(Σs_anal[inds]))) < 0.14
end
