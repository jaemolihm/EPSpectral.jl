using HCubature

function frohlich_compute_angle_factor(kmax, nkk, nzk)
    # Compute integration factor g(k) = ∫d³k' f(k') / |k-k'|² for constant interpolation
    # basis function f(k') = f(|k'|, z = cosθ) (azimuthal symmetry is assumed.)
    # For k, we do constant interpolation with nkk points on [0, kmax].
    # For z, we do constant interpolation with nzk points on [-1, 1].
    # If nzk == 1, we assume a spherically symmetric function f(k') = f(|k'|).
    function _integrand(x, xk)
        # return q² / |xk - xq|² dq
        q, cosθ, ϕ = x
        sinθ = sqrt(1 - cosθ^2)
        xq = q * SVector(cosθ, sinθ * cos(ϕ), sinθ * sin(ϕ))
        return q^2 / norm(xk - xq)^2
    end

    ks_1d = range(0, kmax, length = nkk)
    zs_1d = nzk == 1 ? (1.0:1.0) : range(-1., 1., length = nzk)

    # Shift the integration region slightly to avoid singularity at x = xk
    dk = sqrt(eps(Float64))

    angle_fac_array = tmap(collect(Iterators.product(1:nkk, 1:nzk, 1:nkk, 1:nzk))) do (ikp, izp, ik, iz)
        kp = ks_1d[ikp]
        cosθp = zs_1d[izp]
        sinθp = sqrt(1 - cosθp^2)
        xk = kp * SVector(cosθp, sinθp, 0)

        k1 = ik == 1   ? ks_1d[1]   : (ks_1d[ik-1] + ks_1d[ik]) / 2
        k2 = ik == nkk ? ks_1d[end] : (ks_1d[ik] + ks_1d[ik+1]) / 2
        if nzk == 1
            z1 = -1.0
            z2 = +1.0
        else
            z1 = iz == 1   ? zs_1d[1]   : (zs_1d[iz-1] + zs_1d[iz]) / 2
            z2 = iz == nzk ? zs_1d[end] : (zs_1d[iz] + zs_1d[iz+1]) / 2
        end
        # @info xk, k1, k2, z1, z2

        if k1 < kp < k2
            res1 = hcubature(x -> _integrand(x, xk), (k1, z1, 0.), (kp - dk, z2, 2π))
            res2 = hcubature(x -> _integrand(x, xk), (kp + dk, z1, 0.), (k2, z2, 2π))
            res = res1 .+ res2
        else
            k1 ≈ kp && (k1 += dk)
            k2 ≈ kp && (k2 -= dk)
            res = hcubature(x -> _integrand(x, xk), (k1, z1, 0.), (k2, z2, 2π))
        end

        res[1] / (2π)^3
    end :: Array{Float64, 4}
    angle_fac_matrix = reshape(angle_fac_array, nkk*nzk, nkk*nzk)

    (; ks_1d, zs_1d, angle_fac_matrix)
end;

function frohlich_compute_angle_factor_for_velocity(kmax, nkk)
    # Compute integration factor g(k) = ∫d³k' cos(θ') f(k') / |k-k'|² for constant interpolation
    # basis function f(k') = f(|k'|) (sphreical symmetry is assumed.)
    # For k, we do constant interpolation with nkk points on [0, kmax].
    # (Difference with frohlich_compute_angle_factor is the cos(θ') factor.)
    function _integrand(x, xk)
        # return q² / |xk - xq|² dq
        q, cosθ = x
        sinθ = sqrt(1 - cosθ^2)
        xq = q * SVector(cosθ, sinθ, 0)
        return 2π * cosθ * q^2 / norm(xk - xq)^2
    end

    ks_1d = range(0, kmax, length = nkk)

    # Shift the integration region slightly to avoid singularity at x = xk
    dk = sqrt(eps(Float64))

    angle_fac_matrix = tmap(collect(Iterators.product(1:nkk, 1:nkk))) do (ikp, ik)
        kp = ks_1d[ikp]
        kp < 1e-10 && return 0.0  # At k = 0, the velocity is 0.

        xk = kp * SVector(1, 0, 0)

        k1 = ik == 1   ? ks_1d[1]   : (ks_1d[ik-1] + ks_1d[ik]) / 2
        k2 = ik == nkk ? ks_1d[end] : (ks_1d[ik] + ks_1d[ik+1]) / 2
        z1 = -1.0
        z2 = +1.0

        if k1 < kp < k2
            res1 = hcubature(x -> _integrand(x, xk), (k1, z1), (kp - dk, z2))
            res2 = hcubature(x -> _integrand(x, xk), (kp + dk, z1), (k2, z2))
            res = res1 .+ res2
        else
            k1 ≈ kp && (k1 += dk)
            k2 ≈ kp && (k2 -= dk)
            res = hcubature(x -> _integrand(x, xk), (k1, z1), (k2, z2))
        end

        res[1] / (2π)^3
    end :: Array{Float64, 2}

    (; ks_1d, angle_fac_matrix)
end;


function compute_self_energy_analytic!(S :: ElectronPhononSolver)
    for (ik, k) in enumerate(S.ks)
        S.Σs[:, ik] .= get_Σ_analytic.(k, S.ωs .+ im * S.η, S.model)
    end

    return S
end

function get_Σ_itp(S :: ElectronPhononSolver, η = S.η)
    Σs_with_η = copy(S.Σs)

    # Old version: Add -im * η to all elements
    # Σs_with_η .+= -im * η

    # New version: make |Im Σ| at least η
    inds = @. -imag(Σs_with_η) < η
    Σs_with_η[inds] .= real.(Σs_with_η[inds]) .- im * η

    linear_interpolation((S.ωs, S.ks), Σs_with_η; extrapolation_bc = Flat())
end

function get_Σ_itp_dense(S :: ElectronPhononSolver, η = S.η)
    Σs_with_η = copy(S.Σs_dense)
    if η != S.η
        # S.Σs_dense is set with S.η, so we need to recompute it.
        Σ_itp_coarse = get_Σ_itp(S, η)
        @views for (ik, k) in enumerate(S.ks_dense)
            @. Σs_with_η[:, ik] = Σ_itp_coarse(S.ωs_dense, k)
        end
    end

    # New version: make |Im Σ| at least η
    inds = @. -imag(Σs_with_η) < η
    Σs_with_η[inds] .= real.(Σs_with_η[inds]) .- im * η

    linear_interpolation((S.ωs_dense, S.ks_dense), Σs_with_η; extrapolation_bc = Flat())
end

"""
    compute_spectral_function!(S :: AbstractSolver; spectral_smearing = S.η)
"""
function compute_spectral_function!(S :: AbstractSolver; spectral_smearing = S.η)
    # Interpolate the self-energy from (ωs, ks) to (ωs_dense, ks_dense)
    Σ_itp = get_Σ_itp(S, spectral_smearing)

    # Compute spectral function
    @views for (ik, k) in enumerate(S.ks_dense)
        εk = get_εk(k, S.model)
        @. S.Σs_dense[:, ik] = Σ_itp(S.ωs_dense, k)
        @. S.As_dense[:, ik] = -imag(1 / (S.ωs_dense - εk - Σ_itp(S.ωs_dense, k))) / π
    end

    return S
end


function compute_occupation!(S :: ElectronPhononSolver)
    dω = S.ωs_dense[2] - S.ωs_dense[1]
    fermi = @. 1 / (exp((S.ωs_dense - S.model.μ) / S.model.T) + 1)

    @views for (ik, k) in enumerate(S.ks_dense)
        # Compute integral of the spectral function (should be 1 when exact)
        S.spectral_sum[ik] = sum(S.As_dense[:, ik]) * dω

        # Compute occupation
        S.spectral_occ[ik] = sum(S.As_dense[:, ik] .* fermi) * dω
    end

    # Compute total occupation
    if isfinite(S.model.μ)
        # Linearly interpolate the occupations
        occ_itp = linear_interpolation(S.ks_dense, S.spectral_occ)

        # Integrate over the Brillouin zone
        if S.model isa HolsteinLatticeModel || S.model isa PeierlsLatticeModel
            n = quadgk(occ_itp, -π, π)[1] / 2π
        elseif S.model isa FrohlichModel
            n = quadgk(k -> 1 / 2π^2 * k^2 * occ_itp(k), extrema(S.ks_dense)...)[1]
        else
            error("Model not supported")
        end
    else
        n = zero(S.model.μ)
    end

    return n
end


function compute_occupation(S :: ElectronPhononSolver, μ; ω_cutoff = -Inf)
    # Same as compute_occupation!, but do not update S.spectral_occ

    dω = step(S.ωs_dense)
    fermi = @. occ_fermion(S.ωs_dense - μ, S.model.T)
    fermi[S.ωs_dense .< ω_cutoff] .= 0

    spectral_occ = (fermi' * S.As_dense)' .* dω

    # Integrate over the Brillouin zone
    if S.model isa HolsteinLatticeModel || S.model isa PeierlsLatticeModel
        # Linearly interpolate the occupations
        occ_itp = linear_interpolation(S.ks_dense, spectral_occ)
        n = quadgk(occ_itp, -π, π)[1] / 2π
    elseif S.model isa FrohlichModel
        # Constant interpolation
        # Integration weights for \int d^3k / (2π)^3 f(k) = \int_a^b dk k^2 f(k) / (2π^2)
        ks_midpoints = vcat(0, (S.ks_dense[1:end-1] .+ S.ks_dense[2:end]) ./ 2, S.ks_dense[end])
        weights_occ = diff(ks_midpoints.^3) ./ 3 ./ 2π^2
        n = sum(weights_occ .* spectral_occ)
    else
        error("Model not supported")
    end

    return n
end

function compute_occupation_MaxwellBoltzmann(S :: ElectronPhononSolver)
    # Same as compute_occupation!, but do not update S.spectral_occ

    dω = S.ωs_dense[2] - S.ωs_dense[1]
    fermi = @. exp(-S.ωs_dense / S.model.T)

    spectral_occ = zero(S.spectral_occ)

    @views for (ik, k) in enumerate(S.ks_dense)
        # Compute occupation
        spectral_occ[ik] = sum(S.As_dense[:, ik] .* fermi) * dω
    end

    # Linearly interpolate the occupations
    occ_itp = linear_interpolation(S.ks_dense, spectral_occ)

    # Integrate over the Brillouin zone
    if S.model isa HolsteinLatticeModel || S.model isa PeierlsLatticeModel
        n = quadgk(occ_itp, -π, π)[1] / 2π
    elseif S.model isa FrohlichModel
        n = quadgk(k -> 1 / 2π^2 * k^2 * occ_itp(k), extrema(S.ks_dense)...)[1]
    else
        error("Model not supported")
    end

    return n
end

function plot_spectral_function!(ax, S :: ElectronPhononSolver;
    bare_band = true,
    chemical_potential = true,
    yscale_fac = 1.0,
    kwargs_plot...
    )

    ks = S.ks_dense
    ωs = S.ωs_dense
    dk = ks[2] - ks[1]
    dω = ωs[2] - ωs[1]
    extent = [ks[1] - dk/2, ks[end] + dk/2, (ωs[1] - dω/2) * yscale_fac, (ωs[end] + dω/2) * yscale_fac]

    img = ax.imshow(S.As_dense; origin="lower", extent, aspect="auto", kwargs_plot...)

    if bare_band
        ax.plot(S.ks_dense, get_εk.(S.ks_dense, S.model) .* yscale_fac, c="grey", ls="--", lw=1)
    end
    if chemical_potential
        ax.axhline(S.model.μ * yscale_fac, c="r", ls="--", lw=1)
    end

    return img
end

function plot_self_energy!(ax, S :: ElectronPhononSolver, term = :real;
    bare_band = true,
    chemical_potential = true,
    yscale_fac = 1.0,
    kwargs_plot...
    )

    Σ_itp = get_Σ_itp(S, 0)

    ks = S.ks_dense
    ωs = S.ωs_dense
    dk = ks[2] - ks[1]
    dω = ωs[2] - ωs[1]
    extent = [ks[1] - dk/2, ks[end] + dk/2, (ωs[1] - dω/2) * yscale_fac, (ωs[end] + dω/2) * yscale_fac]
    Σs = [Σ_itp(ω, k) for ω in ωs, k in ks]

    func = term == :real ? real : imag

    img = ax.imshow(func.(Σs); origin="lower", extent, aspect="auto", kwargs_plot...)

    if bare_band
        ax.plot(S.ks_dense, get_εk.(S.ks_dense, S.model) .* yscale_fac, c="grey", ls="--", lw=1)
    end
    if chemical_potential
        ax.axhline(S.model.μ * yscale_fac, c="r", ls="--", lw=1)
    end

    return img
end

function update_chemical_potential!(S)

    if S.occupation == 0
        μ_new = -Inf
    else
        μ_new = find_zero(μ -> compute_occupation(S, μ) - S.occupation, extrema(S.ωs_dense))
    end

    S.model = FrohlichModel(S.model.α, S.model.ω₀, S.model.m, μ_new, S.model.T)

    return S
end


# function compute_self_energy!(S :: ElectronPhononSolver)
#     (; ω₀, μ, T) = S.model
#     dim = get_dimension(S.qpts)
#     Σ_itp = get_Σ_itp_dense(S, S.η)

#     Σs_all = tmap(S.ks) do k
#         Σs_imag_k = tmapreduce(.+, chunks(1:length(S.qpts); n = 2 * Threads.nthreads()); chunking = false) do iqs
#             Σs_imag_q = zeros(length(S.ωs))

#             @views for iq in iqs
#                 q, weight = S.qpts[iq]
#                 εkq = get_εk(SVector(k, 0, 0) + q, S.model)
#                 kq = norm(SVector(k, 0, 0) + q)
#                 if norm(kq) > S.ks[end]
#                     # Keep k' = k+q only inside the k range, skip k+q points outside.
#                     continue
#                 end

#                 ωq = ω₀
#                 nq = occ_boson(ωq, T)
#                 gq = get_eph_g(q, S.model)

#                 factor = weight * abs2(gq)

#                 for (iω, ω) in enumerate(S.ωs)
#                     if nq < sqrt(eps(ω₀))
#                         if real(ω) > μ + ωq
#                             Σs_imag_q[iω] += imag(1 / (ω - ωq - εkq - Σ_itp(ω - ωq, kq))) * factor
#                         elseif real(ω) < μ - ωq
#                             Σs_imag_q[iω] += imag(1 / (ω + ωq - εkq - Σ_itp(ω + ωq, kq))) * factor
#                         end
#                     else
#                         # Finite-temperature case
#                         fac1 = imag(1 / (ω + ωq - εkq - Σ_itp(ω + ωq, kq)))
#                         fac2 = imag(1 / (ω - ωq - εkq - Σ_itp(ω - ωq, kq)))
#                         Σs_imag_q[iω] += ( fac1 * (nq + occ_fermion(ω + ωq - μ, T))
#                                          + fac2 * (nq + 1 - occ_fermion(ω - ωq - μ, T)) ) * factor
#                     end
#                 end
#             end

#             Σs_imag_q
#         end :: Vector{Float64}

#         Σs_imag_k ./= (2π)^dim
#         Σs_real_k = kramers_kronig(real.(S.ωs), Σs_imag_k)

#         Σs_real_k .+ im .* Σs_imag_k
#     end

#     for ik in eachindex(S.ks)
#         S.Σs[:, ik] .= Σs_all[ik] .+ S.Σs_rest[ik]
#     end

#     return S
# end

function compute_self_energy!_old(S :: ElectronPhononSolver{FrohlichModel})
    (; ω₀, μ, T) = S.model
    dim = get_dimension(S.qpts)
    Σ_itp = get_Σ_itp_dense(S, S.η)

    # Since ωq is constant, we only need Σ_itp(ω ± ω₀, k). So we precompute those.
    Σm = zeros(ComplexF64, length(S.ωs), length(S.ks_dense))
    Σp = zeros(ComplexF64, length(S.ωs), length(S.ks_dense))
    for (ik, k) in enumerate(S.ks_dense)
        @. Σm[:, ik] = Σ_itp(S.ωs - ω₀, k)
        @. Σp[:, ik] = Σ_itp(S.ωs + ω₀, k)
    end

    fac_m = zeros(length(S.ωs))
    fac_p = zeros(length(S.ωs))
    if T < 1e-10
        # Zero-temperature case
        @. fac_m = float(S.ωs - ω₀ > μ)
        @. fac_p = float(S.ωs + ω₀ < μ)
    else
        # Finite-temperature case
        nq = occ_boson(ω₀, T)
        @. fac_m = nq + 1 - occ_fermion(S.ωs - ω₀ - μ, T)
        @. fac_p = nq + occ_fermion(S.ωs + ω₀ - μ, T)
    end

    Σs_all = tmap(S.ks) do k
        Σs_imag_k = tmapreduce(.+, chunks(1:length(S.qpts); n = Threads.nthreads()); chunking = false) do iqs
            Σs_imag_q_m = zeros(length(S.ωs))
            Σs_imag_q_p = zeros(length(S.ωs))
            Σp_kq = zeros(ComplexF64, length(S.ωs))
            Σm_kq = zeros(ComplexF64, length(S.ωs))

            @views for iq in iqs
                q, weight = S.qpts[iq]
                εkq = get_εk(SVector(k, 0, 0) + q, S.model)
                kq = norm(SVector(k, 0, 0) + q)
                if norm(kq) > S.ks[end]
                    # Keep k' = k+q only inside the k range, skip k+q points outside.
                    continue
                end

                ωq = ω₀
                gq = get_eph_g(q, S.model)

                factor = weight * abs2(gq)

                # Compute Σp_kq, Σm_kq = Σ(ωs + ω₀, kq) using linear interpolation weight
                # of kq in S.ks_dense
                if kq <= S.ks_dense[1]
                    @. Σp_kq = Σp[:, 1]
                    @. Σm_kq = Σm[:, 1]
                elseif kq >= S.ks_dense[end]
                    @. Σp_kq = Σp[:, end]
                    @. Σm_kq = Σm[:, end]
                else
                    ikq = searchsortedfirst(S.ks_dense, kq)  # kq <= S.ks_dense[ikq]
                    k1, k2 = S.ks_dense[ikq-1], S.ks_dense[ikq]
                    c1 = (k2 - kq) / (k2 - k1)
                    c2 = (kq - k1) / (k2 - k1)
                    @. Σp_kq = c1 * Σp[:, ikq-1] + c2 * Σp[:, ikq]
                    @. Σm_kq = c1 * Σm[:, ikq-1] + c2 * Σm[:, ikq]
                end

                if T < 1e-10 && μ == -Inf
                    # Only phonon emission in T=0 undoped case
                    @. Σs_imag_q_m += imag(Σm_kq) / ((S.ωs - ωq - εkq - real(Σm_kq))^2 + imag(Σm_kq)^2) * factor
                else
                    # Phonon emission and absorption
                    @. Σs_imag_q_m += imag(Σm_kq) / ((S.ωs - ωq - εkq - real(Σm_kq))^2 + imag(Σm_kq)^2) * factor
                    @. Σs_imag_q_p += imag(Σp_kq) / ((S.ωs + ωq - εkq - real(Σp_kq))^2 + imag(Σp_kq)^2) * factor
                end
            end

            @. Σs_imag_q_m * fac_m + Σs_imag_q_p * fac_p
        end :: Vector{Float64}

        Σs_imag_k ./= (2π)^dim
        Σs_real_k = kramers_kronig(real.(S.ωs), Σs_imag_k)

        Σs_real_k .+ im .* Σs_imag_k
    end

    for ik in eachindex(S.ks)
        S.Σs[:, ik] .= Σs_all[ik] .+ S.Σs_rest[ik]
    end

    return S
end


function compute_self_energy!(S :: ElectronPhononSolver{FrohlichModel})
    (; ω₀, μ, T) = S.model

    fac_m = zeros(length(S.ωs))
    fac_p = zeros(length(S.ωs))
    if T < 1e-10
        # Zero-temperature case
        @. fac_m = float(S.ωs - ω₀ > μ)
        @. fac_p = float(S.ωs + ω₀ < μ)
    else
        # Finite-temperature case
        nq = occ_boson(ω₀, T)
        @. fac_m = nq + 1 - occ_fermion(S.ωs - ω₀ - μ, T)
        @. fac_p = nq + occ_fermion(S.ωs + ω₀ - μ, T)
    end

    Σs_imag = [zeros(length(S.ωs)) for _ in eachindex(S.ks)]
    Σs_imag_integrand = zeros(length(S.ωs))

    ωq = ω₀

    for ikq in eachindex(S.ks)
        kq = S.ks[ikq]
        εkq = get_εk(SVector(kq, 0, 0), S.model)

        Σkq_with_η = copy(S.Σs[:, ikq])

        # Make |Im Σ| at least η
        inds = @. -imag(Σkq_with_η) < S.η
        Σkq_with_η[inds] .= real.(Σkq_with_η[inds]) .- im * S.η

        Σkq_itp = linear_interpolation((S.ωs,), Σkq_with_η; extrapolation_bc = Flat())

        for (iω, ω) in enumerate(S.ωs)
            Σp_kq = Σkq_itp(ω + ω₀)
            Σm_kq = Σkq_itp(ω - ω₀)

            if T < 1e-10 && μ == -Inf
                # Only phonon emission in T=0 undoped case
                Σs_imag_q_m = imag(Σm_kq) / ((ω - ωq - εkq - real(Σm_kq))^2 + imag(Σm_kq)^2)
                Σs_imag_q_p = 0.0
            else
                # Phonon emission and absorption
                Σs_imag_q_m = imag(Σm_kq) / ((ω - ωq - εkq - real(Σm_kq))^2 + imag(Σm_kq)^2)
                Σs_imag_q_p = imag(Σp_kq) / ((ω + ωq - εkq - real(Σp_kq))^2 + imag(Σp_kq)^2)
            end

            Σs_imag_integrand[iω] = Σs_imag_q_m * fac_m[iω] + Σs_imag_q_p * fac_p[iω]
        end

        for ik in eachindex(S.ks)
            @. Σs_imag[ik] += Σs_imag_integrand * S.q_integral_matrix[ik, ikq]
        end
    end

    Σs_real = tmap(Σs_imag) do Σs_imag_k
        S.KK_map * Σs_imag_k
    end :: Vector{Vector{Float64}}

    for ik in eachindex(S.ks)
        @. S.Σs[:, ik] = Σs_real[ik] + im * Σs_imag[ik] .+ S.Σs_rest[ik]
    end

    return S
end



function compute_self_energy!(S :: ElectronPhononSolver{FrohlichModel_2D})
    (; α, ω₀, m, μ) = S.model
    Σ_itp = linear_interpolation((S.ωs, S.ks), S.Σs .- im .* S.η; extrapolation_bc = Flat());

    for (ik, k) in enumerate(S.ks)
        Σs_imag = tmapreduce(.+, chunks(1:length(S.qpts); n = 2 * Threads.nthreads()); chunking = false) do iqs
            Σs_imag_q = zeros(length(S.ωs))

            for iq in iqs
                q, weight = S.qpts[iq]
                εkq = get_εk(SVector(k, 0) + q, S.model)
                kq = norm(SVector(k, 0) + q)

                factor = weight / norm(q)

                for (iω, ω) in enumerate(S.ωs)
                    if real(ω) > μ + ω₀
                        Σs_imag_q[iω] += imag(1 / (ω - ω₀ - εkq - Σ_itp(real(ω) - ω₀, kq))) * factor
                    elseif real(ω) < μ - ω₀
                        Σs_imag_q[iω] += imag(1 / (ω + ω₀ - εkq - Σ_itp(real(ω) + ω₀, kq))) * factor
                    end
                end
            end

            Σs_imag_q
        end :: Vector{Float64}

        Σs_imag .*= α * sqrt(ω₀^3 / 2m) * 4π / (2π)^2
        Σs_real = kramers_kronig(real.(S.ωs), Σs_imag)

        S.Σs[:, ik] .= Σs_real .+ im .* Σs_imag
        S.Σs[:, ik] .+= S.Σs_rest[ik]
    end

    return S
end
