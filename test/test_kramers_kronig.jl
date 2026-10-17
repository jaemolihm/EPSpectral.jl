@testset "kramers_kronig" begin
    # A single pole f(ω) = 1/(ω - ε + iη), analytic in the upper half plane: Re f and Im f are
    # each the Kramers-Kronig transform of the other (up to sign). The grid truncation at ±100 and
    # the linear-in-cell quadrature leave an error of 1.1e-2 near the pole, where |f| = 1/η = 10.
    ε, η = 0.5, 0.1
    ωs = range(-100., 100., step = 0.02)
    f = @. 1 / (ωs - ε + im * η)

    re_kk = kramers_kronig(ωs, imag.(f))
    im_kk = -kramers_kronig(ωs, real.(f))

    inds = abs.(ωs) .< 5.0
    @test maximum(abs.(re_kk[inds] .- real.(f[inds]))) < 0.02
    @test maximum(abs.(im_kk[inds] .- imag.(f[inds]))) < 0.02
end
