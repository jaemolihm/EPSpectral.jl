function xlogabsx(x :: Real)
    iszero(x) ? zero(x) : x * log(abs(x))
end


function get_G_contour(ω, ε, Σ, μ, T)
    Gᴿ = 1 / (ω - ε - Σ)
    retarded_to_contour(Gᴿ, ω, μ, T)
end

"""
    retarded_to_contour(Gᴿ, ω, μ, T; is_green = true)
For a retarded Fermionic Green function Gᴿ(ω), compute the contour-ordered Green
function using the fluctuation-dissipation theorem.
NOTE : We shift the frequency by μ (to make the spectral functions centered around
ω = 0). So the FDT involves f(ω - μ) instead of f(ω).
- is_green == true  : treat as a Green function.
- is_green == false : treat as a self-energy. (-+ and +- components have opposite sign)
"""
function retarded_to_contour(Gᴿ, ω, μ, T; is_green = true)
    f = EPSpectral.occ_fermion(ω - μ, T)
    Gᴬ = Gᴿ'
    Gᴷ = (Gᴿ - Gᴬ) * (1 - 2f)
    if is_green
        SMatrix{2, 2}(Gᴿ + Gᴬ + Gᴷ, Gᴿ - Gᴬ + Gᴷ, -Gᴿ + Gᴬ + Gᴷ, -Gᴿ - Gᴬ + Gᴷ) / 2
    else
        SMatrix{2, 2}(Gᴿ + Gᴬ + Gᴷ, -Gᴿ + Gᴬ - Gᴷ, Gᴿ - Gᴬ - Gᴷ, -Gᴿ - Gᴬ + Gᴷ) / 2
    end
end

function get_D_map_real(basis :: LinearSplineBasis, ω₀, i, j)
    # 1 / (x - ω₀) - 1 / (x + ω₀)
    (; xs) = basis
    ( EPSpectral.linear_spline_kramers_kronig(xs[i+1] + ω₀, xs[j], xs[j+1], xs[j+2])
    - EPSpectral.linear_spline_kramers_kronig(xs[i+1] - ω₀, xs[j], xs[j+1], xs[j+2]))
end

function get_D_map_imag(basis :: LinearSplineBasis, ω₀, i, j)
    # π * δ(x - ω₀)
    (; xs) = basis
    y = xs[i+1] + ω₀
    x1, x2, x3 = xs[j], xs[j+1], xs[j+2]
    if x1 <= y <= x2
        return π * (y - x1) / (x2 - x1)
    elseif x2 <= y <= x3
        return π * (x3 - y) / (x3 - x2)
    else
        return 0.0
    end
end

function get_D_map!(D_c, basis :: LinearSplineBasis, ω₀, T)
    # Computes g(ν) = ∫dω/2πi D(ω) * f(ν + ω) as a matrix multiplication g = D * f.
    nocc = occ_boson(ω₀, T)

    for j in 1:basis.N, i in 1:basis.N
        ReD = get_D_map_real(basis, ω₀, i, j) / (2π * im)
        ImD_emi = get_D_map_imag(basis, +ω₀, i, j) / (2π * im)
        ImD_abs = get_D_map_imag(basis, -ω₀, i, j) / (2π * im)

        D_c[1, 1][i, j] =  ReD - im * (2nocc + 1) * (ImD_emi + ImD_abs)
        D_c[2, 2][i, j] = -ReD - im * (2nocc + 1) * (ImD_emi + ImD_abs)
        D_c[2, 1][i, j] = -2im * ((nocc + 1) * ImD_emi + nocc * ImD_abs)
        D_c[1, 2][i, j] = -2im * (nocc * ImD_emi + (nocc + 1) * ImD_abs)
    end

    D_c
end

function get_D_map_opt!(D_c, basis :: LinearSplineBasis, ω₀, T, xlogabsx_buffer)
    # Same as get_D_map!, but use optimized implementation with pre-computing xlogabsx.
    (; xs) = basis
    for D in D_c
        D .= 0
    end

    # Compute real part
    for i in 1:basis.N
        xlogabsx_buffer .= xlogabsx.(xs[i+1] + ω₀ .- xs) .- xlogabsx.(xs[i+1] - ω₀ .- xs)

        for j in 1:basis.N
            ReD =  ( - xlogabsx_buffer[j]   / (xs[j+1] - xs[j])
                     + xlogabsx_buffer[j+1] * (xs[j+2] - xs[j]) / (xs[j+1] - xs[j]) / (xs[j+2] - xs[j+1])
                     - xlogabsx_buffer[j+2] / (xs[j+2] - xs[j+1]) ) / (2π * im)

            D_c[1, 1][i, j] += ReD
            D_c[2, 2][i, j] -= ReD
        end
    end
    
    # Compute imaginary part
    nocc = occ_boson(ω₀, T)
    for j in 1:basis.N, i in 1:basis.N
        ImD_emi = get_D_map_imag(basis, +ω₀, i, j) / (2π * im)
        ImD_abs = get_D_map_imag(basis, -ω₀, i, j) / (2π * im)

        D_c[1, 1][i, j] +=  -im * (2nocc + 1) * (ImD_emi + ImD_abs)
        D_c[2, 2][i, j] +=  -im * (2nocc + 1) * (ImD_emi + ImD_abs)
        D_c[2, 1][i, j] += -2im * ((nocc + 1) * ImD_emi + nocc * ImD_abs)
        D_c[1, 2][i, j] += -2im * (nocc * ImD_emi + (nocc + 1) * ImD_abs)
    end

    D_c
end

function get_D_map_opt_buffers(basis :: LinearSplineBasis)
    return zeros(length(basis.xs))
end

function get_D_map_opt_buffers_v2(basis :: LinearSplineBasis)
    return (
        zeros(ComplexF64, basis.N),
        zeros(ComplexF64, basis.N),
        zeros(ComplexF64, basis.N),
        zeros(basis.N, basis.N+2),
    )
end

function get_D_map_opt_v2!(D_c, basis :: LinearSplineBasis, ω₀, T, buffers)
    # Same as get_D_map!, but use optimized implementation with pre-computing xlogabsx.
    (; xs) = basis
    for D in D_c
        D .= 0
    end

    inv_fac_1 = buffers[1]
    inv_fac_2 = buffers[2]
    inv_fac_3 = buffers[3]
    xlogabsx_buffer = buffers[4]

    for j in 1:basis.N
        inv_fac_1[j] = 1 / (xs[j+1] - xs[j]) / (2π * im)
        inv_fac_2[j] = (xs[j+2] - xs[j]) / (xs[j+1] - xs[j]) / (xs[j+2] - xs[j+1]) / (2π * im)
        inv_fac_3[j] = 1 / (xs[j+2] - xs[j+1]) / (2π * im)
    end

    for j in 1:basis.N+2
        for i in 1:basis.N
            xlogabsx_buffer[i, j] = xlogabsx(xs[i+1] + ω₀ - xs[j]) - xlogabsx(xs[i+1] - ω₀ - xs[j])
        end
    end

    # Compute real part
    @views for j in 1:basis.N
        D_c[1, 1][:, j] .-= inv_fac_1[j] .* xlogabsx_buffer[:, j+0]
        D_c[1, 1][:, j] .+= inv_fac_2[j] .* xlogabsx_buffer[:, j+1]
        D_c[1, 1][:, j] .-= inv_fac_3[j] .* xlogabsx_buffer[:, j+2]
    end
    D_c[2, 2] .= .-D_c[1, 1]

    # Compute imaginary part
    nocc = occ_boson(ω₀, T)
    for j in 1:basis.N, i in 1:basis.N
        ImD_emi = get_D_map_imag(basis, +ω₀, i, j) / 2π
        ImD_abs = get_D_map_imag(basis, -ω₀, i, j) / 2π

        if ImD_emi != 0 || ImD_abs != 0
            D_c[1, 1][i, j] += -(2nocc + 1) * (ImD_emi + ImD_abs)
            D_c[2, 2][i, j] += -(2nocc + 1) * (ImD_emi + ImD_abs)
            D_c[2, 1][i, j] += -2 * ((nocc + 1) * ImD_emi + nocc * ImD_abs)
            D_c[1, 2][i, j] += -2 * (nocc * ImD_emi + (nocc + 1) * ImD_abs)
        end
    end

    D_c
end


function get_D_map_opt_v2_array!(D_c :: AbstractArray{ComplexF64, 4}, basis :: LinearSplineBasis, ω₀, T, buffers)
    # Same as get_D_map!, but use optimized implementation with pre-computing xlogabsx.
    (; xs) = basis
    D_c .= 0

    inv_fac_1 = buffers[1]
    inv_fac_2 = buffers[2]
    inv_fac_3 = buffers[3]
    xlogabsx_buffer = buffers[4]

    for j in 1:basis.N
        inv_fac_1[j] = 1 / (xs[j+1] - xs[j]) / (2π * im)
        inv_fac_2[j] = (xs[j+2] - xs[j]) / (xs[j+1] - xs[j]) / (xs[j+2] - xs[j+1]) / (2π * im)
        inv_fac_3[j] = 1 / (xs[j+2] - xs[j+1]) / (2π * im)
    end

    for j in 1:basis.N+2
        for i in 1:basis.N
            xlogabsx_buffer[i, j] = xlogabsx(xs[i+1] + ω₀ - xs[j]) - xlogabsx(xs[i+1] - ω₀ - xs[j])
        end
    end

    # Compute real part
    @views for j in 1:basis.N
        D_c[:, j, 1, 1] .-= inv_fac_1[j] .* xlogabsx_buffer[:, j+0]
        D_c[:, j, 1, 1] .+= inv_fac_2[j] .* xlogabsx_buffer[:, j+1]
        D_c[:, j, 1, 1] .-= inv_fac_3[j] .* xlogabsx_buffer[:, j+2]
    end
    @views D_c[:, :, 2, 2] .= .-D_c[:, :, 1, 1]

    # Compute imaginary part
    nocc = occ_boson(ω₀, T)
    for j in 1:basis.N, i in 1:basis.N
        ImD_emi = get_D_map_imag(basis, +ω₀, i, j) / 2π
        ImD_abs = get_D_map_imag(basis, -ω₀, i, j) / 2π

        if ImD_emi != 0 || ImD_abs != 0
            D_c[i, j, 1, 1] += -(2nocc + 1) * (ImD_emi + ImD_abs)
            D_c[i, j, 2, 2] += -(2nocc + 1) * (ImD_emi + ImD_abs)
            D_c[i, j, 2, 1] += -2 * ((nocc + 1) * ImD_emi + nocc * ImD_abs)
            D_c[i, j, 1, 2] += -2 * (nocc * ImD_emi + (nocc + 1) * ImD_abs)
        end
    end

    D_c
end


"""
    get_D_map(basis :: LinearSplineBasis, ω₀, T)
Return a 2*2 matrix of `nbasis*nbasis` matrices, such that for a quantity `f` defined on
`basis`, `D[c2, c1] * f` yields the convolution ``(Df)(ε) = ∫dω/2πi D^{c2, c1}(ω) * f(ε + ω)``,
where `D^{c2, c1}(ω)` is the contour-ordered phonon Green's function.
"""
function get_D_map(basis :: LinearSplineBasis, ω₀, T)
    D_contour = [zeros(ComplexF64, basis.N, basis.N) for _ in 1:2, _ in 1:2]
    get_D_map!(D_contour, basis, ω₀, T)
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
        mapreduce(+, eachindex(ωs)) do iω
            ω = ωs[iω]
            A = -imag(1 / (ω - εk - Σs_itp(ω, k))) / π
            sum_factor[iω] * A
        end
    end

    occ_k = _compute_occupation_k.(S.ks_dense)

    occ_k_itp = linear_interpolation(S.ks_dense, occ_k)
    occ = quadgk(k -> k^2 * occ_k_itp(k), 0, S.ks_dense[end])[1] * 4π / (2π)^3

    (; occ, occ_k)
end
