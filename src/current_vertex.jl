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


function compute_occupation_dilute(S, ωs, ω_cutoff)
    model = S.model
    T = S.model.T
    Σs_itp = get_Σ_itp_dense(S)

    sum_factor = vcat(
        (ωs[2] - ωs[1]) / 2,
        (ωs[3:end] .- ωs[1:end-2]) ./ 2,
        (ωs[end] - ωs[end-1]) / 2
    )

    # Occupation function f(ω) = df_FD(ω, μ) / dμ * exp(-μ/T) in the μ -> -Inf limit
    fermi = @. exp(-ωs / T) / T
    fermi[ωs .< ω_cutoff] .= 0

    sum_factor .*= fermi

    function _compute_occupation_k(k)
        εk = get_εk(k, model)
        tmapreduce(+, eachindex(ωs)) do iω
            ω = ωs[iω]
            A = -imag(1 / (ω - εk - Σs_itp(ω, k))) / π
            sum_factor[iω] * A
        end
    end

    occ_k = _compute_occupation_k.(S.ks_dense)
    occ = quadgk(k -> k^2 * _compute_occupation_k(k), 0, Inf)[1] * 4π / (2π)^3

    (; occ, occ_k)
end
