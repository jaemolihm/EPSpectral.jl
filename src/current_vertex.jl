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
    SMatrix{2, 2}(Gᴿ + Gᴬ + Gᴷ, Gᴿ - Gᴬ + Gᴷ, -Gᴿ + Gᴬ + Gᴷ, -Gᴿ - Gᴬ + Gᴷ) / 2
end

function get_D_map(basis :: LinearSplineBasis, ω₀, T)
    nocc = occ_boson(ω₀, T)

    ReD = get_D_map_real(basis, ω₀) ./ (2π * im)
    ImD_emi = get_D_map_imag(basis, +ω₀) ./ (2π * im)
    ImD_abs = get_D_map_imag(basis, -ω₀) ./ (2π * im)

    D_contour = [zeros(ComplexF64, basis.N, basis.N) for _ in 1:2, _ in 1:2]
    @. D_contour[1, 1] =  ReD - im * (2nocc + 1) * (ImD_emi + ImD_abs)
    @. D_contour[2, 2] = -ReD - im * (2nocc + 1) * (ImD_emi + ImD_abs)
    @. D_contour[2, 1] = -2im * ((nocc + 1) * ImD_emi + nocc * ImD_abs)
    @. D_contour[1, 2] = -2im * (nocc * ImD_emi + (nocc + 1) * ImD_abs)

    D_contour
end


function compute_occupation_dilute(S, basis, ω_cutoff)
    model = S.model
    Σs_itp = get_Σ_itp_dense(S)
    T = model.T

    basis_sum = @. (basis.xs[3:end] .- basis.xs[1:end-2]) / 2
    ωs = grid_points(basis)

    # Occupation function f(ω) = df_FD(ω, μ) / dμ * exp(-μ/T) in the μ -> -Inf limit
    fermi = @. exp(-ωs / T) / T
    fermi[ωs .< ω_cutoff] .= 0

    function _compute_occupation_k(k)
        εk = get_εk(k, model)
        As = .-imag.(1 ./ (ωs .- εk .- Σs_itp.(ωs, k))) ./ π
        occ_k = sum(basis_sum .*  fermi .* As)
        occ_k
    end
    quadgk(k -> k^2 * _compute_occupation_k(k), 0, Inf)[1] * 4π / (2π)^3
end
