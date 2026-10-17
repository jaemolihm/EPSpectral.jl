@testset "kramers_kronig single pole" begin
    # A single pole f(ω) = 1/(ω - ε + iη), analytic in the upper half plane: Re f and Im f are
    # each the Kramers-Kronig transform of the other (up to sign). The grid truncation at ±40 and
    # the piecewise-linear interpolation of f near the pole (error ~ dω^2 |f''| / 8 = 0.1 at the
    # peak, where |f| = 1/η = 10) leave errors of 5.5e-2 and 8.1e-2.
    ε, η = 0.5, 0.1
    ωs = range(-40., 40., step = 0.02)
    f = @. 1 / (ωs - ε + im * η)

    K = EPSpectral.kk_matrix(ωs)
    re_kk = K * imag.(f)
    im_kk = -K * real.(f)

    inds = abs.(ωs) .< 5.0
    @test maximum(abs.(re_kk[inds] .- real.(f[inds]))) < 0.1
    @test maximum(abs.(im_kk[inds] .- imag.(f[inds]))) < 0.1
end
