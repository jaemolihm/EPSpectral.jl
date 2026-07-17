using EPSpectral
using EPSpectral: run_cumulant, run_cumulant_analytic, grid_points, fftfreq_t2w,
    apply_Kramers_Kronig, ElectronPhononSolver, Kpoints, compute_self_energy!,
    compute_spectral_function!, get_Σ_itp_dense
using Test

@testset "cumulant (Frohlich)" begin
    α, ω0, m, μ, T = 2.0, 1.0, 0.5, -Inf, 0.0
    model = FrohlichModel(α, ω0, m, μ, T)
    k = 0.5
    η = 0.05
    εk = get_εk(k, model)

    # Linear-spline representation of -Im Σ(εk + ω + iη) / π on an ω grid relative to εk
    xs = collect(range(-60.0, 60.0, step=0.1))
    basis = LinearSplineBasis(xs)
    βs = [-imag(get_Σ_analytic(k, εk + ω + im*η, model)) / π for ω in grid_points(basis)]

    # Coarse time grid for the C(t) integration, fine grid for the FFT to A(ω)
    ts_small = collect(range(0.0, 40.0, step=0.4))
    ts = collect(range(0.0, 40.0, step=0.05))
    ωs = fftfreq_t2w(ts)
    dω = ωs[2] - ωs[1]

    As_ana = run_cumulant_analytic(model, k, η, ts_small, ts)
    As_num = run_cumulant(basis, βs, εk, ts_small, ts)

    # Spectral sum rule ∫ A dω = 1
    @test sum(As_ana) * dω ≈ 1 atol = 1e-3
    @test sum(As_num) * dω ≈ 1 atol = 1e-3

    # Spectral function is (approximately) non-negative
    @test minimum(As_ana) > -0.05
    @test minimum(As_num) > -0.05

    # Quasiparticle peak sits at εk + Re Σ(εk + iη)
    Σ0_ana = get_Σ_analytic(k, εk + im * η, model)
    @test ωs[argmax(As_ana)] ≈ εk + real(Σ0_ana) atol = 0.15

    # For the spline pipeline the real part comes from the Kramers-Kronig transform
    Σ0_num_real = -apply_Kramers_Kronig(basis, βs, 0.0)
    @test ωs[argmax(As_num)] ≈ εk + Σ0_num_real atol = 0.15

    # Both pipelines carry the same quasiparticle weight near the peak
    qp(A) = sum(A[abs.(ωs .- (εk + real(Σ0_ana))) .< 2.0]) * dω
    @test qp(As_num) ≈ qp(As_ana) atol = 0.1

    # Hardcoded regression values at a few frequencies (analytic, spline)
    ref = [(0.0, 0.17855763, 0.16352528),
           (1.0, 0.10383923, 0.11217547),
           (2.0, 0.05800431, 0.06418802)]
    for (ωt, a_ref, n_ref) in ref
        i = argmin(abs.(ωs .- ωt))
        @test As_ana[i] ≈ a_ref rtol = 1e-4
        @test As_num[i] ≈ n_ref rtol = 1e-4
    end
end

@testset "cumulant (Holstein)" begin
    # G0D0 Holstein self-energy (following Holstein/1_spectral.jl), then cumulant spectral function
    t = 1.0
    ω₀ = 1.0
    λ = 0.5
    g = sqrt(2t * ω₀ * λ)
    μ = -Inf
    T = 0.0
    η = 0.05
    model = HolsteinLatticeModel(1.0, g, ω₀, t, μ, T)

    ωs = collect(range(-10.0, 10.0, step=0.05))
    ks = collect(range(-Float64(π), Float64(π), length=11))
    # Brillouin-zone average over a uniform periodic q grid (as in run_Holstein): nq
    # points, endpoint excluded, equal weights summing to 1.
    nq = 256
    qpts = Kpoints(Vector(range(-Float64(π), Float64(π), length=nq+1)[1:end-1]), fill(1/nq, nq))

    S = ElectronPhononSolver(model, ωs, ks, qpts; occupation=0.0, η=η)
    compute_self_energy!(S)          # non-self-consistent: Σ inside G is zero (bare G0)
    compute_spectral_function!(S)

    # Cumulant spectral function at k = 0
    Σ_itp = get_Σ_itp_dense(S, S.η)
    basis = LinearSplineBasis(S.ωs)
    kc = 0.0
    εk = get_εk(kc, model)
    βs = [-imag(Σ_itp(ω + εk, kc)) / π for ω in basis.xs[2:end-1]]

    tmax = π / minimum(diff(S.ωs))
    dt = π / S.ωs[end]
    ts = collect(range(0.0, 2tmax, step=dt))
    ωs_cum = fftfreq_t2w(ts)
    dω = ωs_cum[2] - ωs_cum[1]

    As_cum = run_cumulant(basis, βs, εk, ts, ts)

    # Spectral sum rule and (approximate) non-negativity
    @test sum(As_cum) * dω ≈ 1 atol = 1e-3
    @test minimum(As_cum) > -0.05

    # Quasiparticle peak sits at εk + Re Σ0 (Kramers-Kronig of the spline)
    @test ωs_cum[argmax(As_cum)] ≈ εk - apply_Kramers_Kronig(basis, βs, 0.0) atol = 0.05

    # Hardcoded regression values at a few frequencies
    ref = [(-2.0, 0.08006561),
           ( 0.0, 0.03634884),
           ( 2.0, 0.01222282)]
    for (ωt, a_ref) in ref
        i = argmin(abs.(ωs_cum .- ωt))
        @test As_cum[i] ≈ a_ref rtol = 1e-4
    end
end
