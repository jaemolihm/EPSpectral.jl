# The contour assembly of the ladder conductivity (`src/ladder_kernel.jl`): the bare-vertex
# response of the Keldysh contour against the spectral-function bubble, on converged-or-not scGD0
# self-energies of the 1D Holstein chain and of the pb window, at Ω = 0 and Ω > 0 and two
# temperatures each. The reference reads Σ with
# Interpolations.jl and forms A_i = -Im Gᴿ/π and the Fermi functions itself, so nothing is shared
# with the contour Green's functions beyond the input Σ.

# The bubble from spectral functions: σ_ab = π Σ_i w_i v_a v_b Σ_ε Wω[ε] A_i(ε - Ω/2) A_i(ε + Ω/2)
# (f(ε - Ω/2 - μ) - f(ε + Ω/2 - μ))/Ω, with -∂f/∂ε = f(1 - f)/T at Ω = 0.
function spectral_bubble(res, iT, Ω)
    (; ωs, el_i) = res
    T, μ = res.Tlist[iT], res.μ[iT]
    f(x) = 1 / (exp(x / T) + 1)
    Wω = EPSpectral.trapezoid_weights(ωs)
    σ = zeros(3, 3)
    for i in 1:el_i.n
        Σ = extrapolate(interpolate((ωs,), res.Σ[:, i, iT], Gridded(Linear())), Flat())
        A(ω) = -imag(1 / (ω - el_i.es[i] - Σ(ω))) / π
        window = sum(Wω[j] * A(ωs[j] - Ω / 2) * A(ωs[j] + Ω / 2) *
            (iszero(Ω) ? f(ωs[j] - μ) * (1 - f(ωs[j] - μ)) / T :
                          (f(ωs[j] - Ω / 2 - μ) - f(ωs[j] + Ω / 2 - μ)) / Ω)
            for j in eachindex(ωs))
        σ .+= π * state_weights(el_i)[i] * window .* (el_i.vs[i] * el_i.vs[i]')
    end
    σ
end

# σ_bubble through the package: Σ gathered at ωs ∓ Ω/2, contour G, J0, contraction.
function contour_bubble(res, iT, Ω)
    (; ωs, el_i) = res
    ni = el_i.n
    Σd_m, Σd_p = (similar(res.Σ[:, :, iT]) for _ in 1:2)
    EPSpectral.interp_apply!(Σd_m, res.Σ[:, :, iT], EPSpectral.interp_table(ωs, ωs .- Ω / 2)...)
    EPSpectral.interp_apply!(Σd_p, res.Σ[:, :, iT], EPSpectral.interp_table(ωs, ωs .+ Ω / 2)...)
    Gm, Gp = (zeros(ComplexF64, length(ωs), 4, ni) for _ in 1:2)
    EPSpectral.contour_greens!(Gm, Gp, el_i.es, Σd_m, Σd_p, ωs, Ω, res.μ[iT], res.Tlist[iT])
    v = reduce(hcat, el_i.vs)
    J0 = zeros(ComplexF64, length(ωs), 4, 3, 2, ni)
    EPSpectral.bare_vertex_response!(J0, Gm, v, Gp)
    Wω = EPSpectral.trapezoid_weights(ωs)
    EPSpectral.ladder_conductivity(J0, v, state_weights(el_i), Wω, Ω, res.Tlist[iT])
end

@testset "ladder bubble from the contour" begin
    @testset "contour Green's function identities" begin
        f(x, T) = 1 / (exp(x / T) + 1)
        for (ω, ε, Σ, μ, T) in ((0.3, -0.1, -0.02 - 0.05im, 0.05, 0.1),
                                (-0.7, 0.4, 0.03 - 0.2im, -0.1, 0.25),
                                (1.1, 0.2, -0.1 - 0.01im, 0.3, 0.05))
            GR = 1 / (ω - ε - Σ)
            GA = conj(GR)
            G11, G21, G12, G22 = (EPSpectral._contour_green(ω, ε, Σ, μ, T, c) for c in 1:4)
            fω = f(ω - μ, T)
            tol = 1e-14 * abs(GR - GA)
            @test abs(G11 - G12 - GR) <= tol
            @test abs(G21 - G22 - GR) <= tol
            @test abs(G11 - G21 - GA) <= tol
            @test abs(G21 - (1 - fω) * (GR - GA)) <= tol
            @test abs(G12 + fω * (GR - GA)) <= tol
        end
    end

    @testset "trapezoid weights" begin
        ωs = [-1.0, -0.7, -0.6, 0.0, 0.5, 2.0]
        Wω = EPSpectral.trapezoid_weights(ωs)
        @test sum(Wω) ≈ ωs[end] - ωs[1] atol = 1e-14
        @test Wω[1] ≈ (ωs[2] - ωs[1]) / 2 && Wω[end] ≈ (ωs[end] - ωs[end-1]) / 2
        # Exact for a linear integrand.
        @test sum(Wω .* ωs) ≈ (ωs[end]^2 - ωs[1]^2) / 2 atol = 1e-14
    end

    @testset "attach! and the bare vertex against 2x2 matrices" begin
        nω, ni = 5, 3
        Gm = pseudorandom_complex(nω, 4, ni; seed = 1.1)
        Gp = pseudorandom_complex(nω, 4, ni; seed = 2.3)
        X = pseudorandom_complex(nω, 4, 3, 2, ni; seed = 3.7)
        J = zeros(ComplexF64, nω, 4, 3, 2, ni)
        EPSpectral.attach!(J, Gm, X, Gp)
        # The pair index (c1, c2) is column-major, so reshaping a four-vector to 2x2 gives the
        # contour matrix.
        J_ref = zeros(ComplexF64, nω, 4, 3, 2, ni)
        for i in 1:ni, c0 in 1:2, a in 1:3, ω in 1:nω
            J_ref[ω, :, a, c0, i] = vec(reshape(Gm[ω, :, i], 2, 2) *
                reshape(X[ω, :, a, c0, i], 2, 2) * reshape(Gp[ω, :, i], 2, 2))
        end
        @test norm(J - J_ref) <= 1e-14 * norm(J_ref)

        # The bare vertex is attach! of V^{c1c2,c0} = δ_{c1c2} δ_{c1c0} v_i.
        v = pseudorandom(3, ni; seed = 4.1)
        V = zeros(ComplexF64, nω, 4, 3, 2, ni)
        for c0 in 1:2
            V[:, c0 + 2 * (c0 - 1), :, c0, :] .= reshape(v, 1, 3, ni)
        end
        J_V = similar(J)
        EPSpectral.attach!(J_V, Gm, V, Gp)
        J0 = similar(J)
        EPSpectral.bare_vertex_response!(J0, Gm, v, Gp)
        @test norm(J0 - J_V) <= 1e-14 * norm(J0)
        @test norm(J0) > 0
    end

    @testset "bubble identity" begin
        t, ω0, nk = 1.0, 0.2, 64
        model_h = EP.holstein_model(; t, λ = 0.5, ω₀ = ω0, dimension = 1, verbose = false)
        occ_h = ElectronOccupationParams(; Tlist = [0.1, 0.2], nlist = 0.0, nelec = 1,
            volume = model_h.volume, spin_degeneracy = 2, type = :Metal)
        holstein = run_scgd0(model_h, (nk, 1, 1), (-Inf, Inf); occ = occ_h, ωs = -4.0:0.02:4.0,
            η_init = 0.05, tol = 1e-6, verbosity = 0)
        (; model, nk, window, ωs, occ, η_init, η_min) = pb_scgd0_fixture()
        pb = @test_logs (:warn, r"not converged") match_mode = :any run_scgd0(model, nk, window;
            occ, ωs, η_init, η_min, maxiter = 2, tol = 0.0, verbosity = 0)
        eV = unit_to_aru(:eV)
        for (name, res, Ωs) in (("Holstein", holstein, (0.0, 0.3)), ("pb", pb, (0.0, 0.123 * eV)))
            for Ω in Ωs, iT in eachindex(res.Tlist)
                σ = contour_bubble(res, iT, Ω)
                σ_ref = spectral_bubble(res, iT, Ω)
                err = norm(σ - σ_ref) / norm(σ_ref)
                @info "bubble identity" name Ω iT err norm(σ_ref)
                @test σ_ref[1, 1] > 0
                @test err <= 1e-12
            end
        end
    end
end
