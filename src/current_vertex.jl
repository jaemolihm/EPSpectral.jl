function get_D_map_real(basis :: LinearSplineBasis, ω₀)
    (; N, xs) = basis
    D = zeros(N, N)
    for j in 1:N, i in 1:N
        D[i, j] += linear_spline_kramers_kronig(xs[i+1] + ω₀, xs[j], xs[j+1], xs[j+2])
        D[i, j] -= linear_spline_kramers_kronig(xs[i+1] - ω₀, xs[j], xs[j+1], xs[j+2])
    end

    D
end

function get_D_map_imag(basis :: LinearSplineBasis, ω₀)
    (; N, xs) = basis
    D = zeros(N, N)
    for j in 1:N, i in 1:N
        y = xs[i+1] + ω₀
        x1, x2, x3 = xs[j], xs[j+1], xs[j+2]
        if x1 <= y <= x2
            D[i, j] += π * (y - x1) / (x2 - x1)
        elseif x2 <= y <= x3
            D[i, j] += π * (x3 - y) / (x3 - x2)
        end
    end
    D
end

function get_G_contour(ω, ε, Σ, μ, T)
    f = EPSpectral.occ_fermion(ω - μ, T)
    Gᴿ = 1 / (ω - ε - Σ)
    Gᴬ = Gᴿ'
    Gᴷ = (Gᴿ - Gᴬ) * (1 - 2f)
    [Gᴿ + Gᴬ + Gᴷ ; Gᴿ - Gᴬ + Gᴷ ;; -Gᴿ + Gᴬ + Gᴷ; -Gᴿ - Gᴬ + Gᴷ] ./ 2
end
