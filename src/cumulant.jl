using QuadGK
using FFTW
using FourierTools
using SpecialFunctions

function get_Cs_t(βs_itp, ts)
    ωmin, ωmax = βs_itp.itp.knots[1][1], βs_itp.itp.knots[1][end]
    Cs_t = tmap(ts) do t
        val = quadgk(ω -> βs_itp(ω) * (cis(-ω*t) +im * ω * t - 1) / ω^2, ωmin, 0)[1]
        val += quadgk(ω -> βs_itp(ω) * (cis(-ω*t) +im * ω * t - 1) / ω^2, 0, ωmax)[1]
        val
    end
    Cs_t
end

function get_Cs_t_exact(ts, εk, k, model, η)
    Cs_t = tmap(ts) do t
        val = quadgk(ω -> -imag(get_Σ_analytic(k, ω + εk + im * η, model)) / π * (cis(-ω*t) + im * ω * t - 1) / ω^2, -Inf, 0)[1]
        val += quadgk(ω -> -imag(get_Σ_analytic(k, ω + εk + im * η, model)) / π * (cis(-ω*t) + im * ω * t - 1) / ω^2, 0, Inf)[1]
        val
    end
    Cs_t
end

function fftfreq_t2w(ts)
    # Frequency mesh for the Fourier transformation of y(t) to y(ω).
    # ts includes only 0 and positive t values.
    @assert ts[1] == 0.0
    dt = ts[2] - ts[1]
    ωmax = π / dt
    ωs = range(-ωmax, ωmax, length = 2length(ts)+1)[1:end-1]
    ωs
end

function fft_t2w(ts, ys_t)
    # Fourier transform of y(t) to y(ω).
    # Assume ys(t) is nonzero only for t >= 0 and only t >= 0 data are used.
    @assert ts[1] == 0.0
    @assert length(ts) == length(ys_t)
    dt = ts[2] - ts[1]
    ωmax = π / dt
    ωs = range(-ωmax, ωmax, length = 2length(ts)+1)[1:end-1]

    # Pad t < 0 data with zeros.
    ys_t_pad = vcat(zeros(length(ts)), ys_t)
    ys = fftshift(bfft(fftshift(ys_t_pad))) .* dt
    (; ys, ωs)
end

using SpecialFunctions

function _cumulant_int0(x, t)
    # ∫ dω (exp(-iωt) - 1 + iωt) / ω^2
    @assert t >= 0
    if x > 0
        -im * t * (im * (-π/2 - sinint(t * x) ) + cosint(t * x)) - cis(-t * x) / x + im * t * log(x) + 1 / x
    else
        -im * t * (im * (-π/2 + sinint(-t * x) ) + cosint(-t * x)) - cis(-t * x) / x + im * t * log(-x) + 1 / x
    end
end
function _cumulant_int1(x, t)
    # ∫ dω ω * (exp(-iωt) - 1 + iωt) / ω^2
    @assert t >= 0
    if x > 0
        im * (-π/2 - sinint(t * x) ) + cosint(t * x)  + im * t * x - log(x)
    else
        im * (-π/2 + sinint(-t * x) ) + cosint(-t * x)  + im * t * x - log(-x)
    end
end

# Evaluate ∫ b(ω; a, b, c) * (exp(-iωt) - 1 + iωt) / ω^2 for a linear spline function b
function _cumulant_integral(t, a, b, c)
    if t == 0.
        return 0.0im
    end
    a == 0 && (a += 1e-6)
    b == 0 && (b += 1e-6)
    c == 0 && (c += 1e-6)

    int0_a = _cumulant_int0(a, t)
    int0_b = _cumulant_int0(b, t)
    int0_c = _cumulant_int0(c, t)
    int1_a = _cumulant_int1(a, t)
    int1_b = _cumulant_int1(b, t)
    int1_c = _cumulant_int1(c, t)

    z  = -a / (b - a) * (int0_b - int0_a)
    z +=  1 / (b - a) * (int1_b - int1_a)
    z +=  c / (c - b) * (int0_c - int0_b)
    z += -1 / (c - b) * (int1_c - int1_b)
    z
end

function apply_cumulant_integral(basis :: LinearSplineBasis, βs, t)
    tmapreduce(+, 1:basis.N) do i
        a, b, c = basis.xs[i], basis.xs[i+1], basis.xs[i+2]
        _cumulant_integral(t, a, b, c) * βs[i]
    end
end

function compute_cumulant_matrix(ts, basis)
    t_and_ib = collect(Iterators.product(ts, 1:basis.N))
    tmap(t_and_ib) do (t, ib)
        a, b, c = basis.xs[ib], basis.xs[ib+1], basis.xs[ib+2]
        EPSpectral._cumulant_integral(t, a, b, c)
    end
end

function run_cumulant(basis, βs, εk, ts_small, ts, cumulant_matrix = nothing)
    # Step 1-1: Perform integration to get C(t) from Σ(ω)
    if cumulant_matrix !== nothing
        # Reuse cumulant integral coefficients
        @assert size(cumulant_matrix) == (length(ts_small), basis.N)
        Cs_t_small = cumulant_matrix * βs
    else
        Cs_t_small = tmap(ts_small) do t
            apply_cumulant_integral(basis, βs, t)
        end
    end

    # Step 1-2: Compute the constant and linear terms of C(t), subtract from C
    Σ0_real = -1 * apply_Kramers_Kronig(basis, βs, 0.0)
    Σ0_imag = -π * basis(0.0, βs)
    Σ0 = Σ0_real + im * Σ0_imag

    δω = 1e-5
    dΣ0_real = -1 * (apply_Kramers_Kronig(basis, βs, δω) - apply_Kramers_Kronig(basis, βs, -δω)) / 2δω
    dΣ0_imag = -π * (basis(δω, βs) - basis(-δω, βs)) / 2δω
    dΣ0 = dΣ0_real + im * dΣ0_imag

    @. Cs_t_small -= -im * Σ0 * ts_small + dΣ0

    # Step 1-3: Interpolation C(t) and append linear extrapolation
    Cs_t_itp = linear_interpolation(ts_small, Cs_t_small; extrapolation_bc=0)
    Cs_t = @. Cs_t_itp(ts) - im * Σ0 * ts + dΣ0


    # Step 2: Perform Fourier transformation to get A_cum(t) from C(t)
    Gs_cum_t = @. -im * cis(-εk * ts) * exp(Cs_t)
    Gs_cum_t[1] /= 2  # Divide t=0 term by 2
    Gs_cum, ωs = fft_t2w(ts, Gs_cum_t)

    As_cum = @. -imag(Gs_cum) / π

    As_cum
end


function run_cumulant_analytic(model :: FrohlichModel, k, η, ts_, ts)
    εk = get_εk(k, model)

    Cs_t_ = get_Cs_t_exact(ts_, εk, k, model, η)

    Σ0 = get_Σ_analytic(k, εk + im * η, model)
    δω = 1e-5
    dΣ0 = (get_Σ_analytic(k, εk + δω + im * η, model) - get_Σ_analytic(k, εk - δω + im * η, model)) / 2δω

    @. Cs_t_ -= -im * Σ0 * ts_+ dΣ0
    Cs_t_itp = linear_interpolation(ts_, Cs_t_; extrapolation_bc=0)
    Cs_t = @. Cs_t_itp(ts) - im * Σ0 * ts + dΣ0

    Gs_cum_t = -im .* cis.(-εk .* ts) .* exp.(Cs_t)
    Gs_cum_t[1] /= 2  # Divide t=0 term by 2

    Gs_cum, ωs = fft_t2w(ts, Gs_cum_t)

    As_cum = .-imag.(Gs_cum) ./ π
    As_cum
end
