# Frohlich model in the dilute limit. Compute quantities under infinitesimal doping.
# f(ω) = 0 -> f(ω) = δf(ω) = exp(-ω / T) / T

function compute_self_energy_dilute(S, dΣs, ω_cutoff)
    (; ω₀, μ, T) = S.model
    @assert T > 0
    @assert μ == -Inf

    dim = get_dimension(S.qpts)
    Σ_itp = get_Σ_itp_dense(S, S.η)

    dΣ_itp = linear_interpolation((S.ωs, S.ks), dΣs; extrapolation_bc = 0)

    dΣs_out = zero.(S.Σs)

    for (ik, k) in enumerate(S.ks)
        Σs_imag = tmapreduce(.+, chunks(1:length(S.qpts); n = 2 * Threads.nthreads()); chunking = false) do iqs
            Σs_imag_q = zeros(length(S.ωs))

            for iq in iqs
                q, weight = S.qpts[iq]
                εkq = get_εk(SVector(k, 0, 0) + q, S.model)
                kq = norm(SVector(k, 0, 0) + q)
                ωq = ω₀
                nq = occ_boson(ωq, T)
                gq = get_eph_g(q, S.model)

                factor = weight * abs2(gq)

                for (iω, ω) in enumerate(S.ωs)
                    # Contribution from df
                    A1 = imag(1 / (ω + ωq - εkq - Σ_itp(ω + ωq, kq)))
                    A2 = imag(1 / (ω - ωq - εkq - Σ_itp(ω - ωq, kq)))
                    df1 = ω + ωq < ω_cutoff ? 0.0 : +exp(-(ω + ωq) / T) / T
                    df2 = ω - ωq < ω_cutoff ? 0.0 : -exp(-(ω - ωq) / T) / T
                    Σs_imag_q[iω] += (A1 * df1 + A2 * df2) * factor

                    # Contribution from dΣ
                    dA1 = imag(dΣ_itp(ω + ωq, kq) / (ω + ωq - εkq - Σ_itp(ω + ωq, kq))^2)
                    dA2 = imag(dΣ_itp(ω - ωq, kq) / (ω - ωq - εkq - Σ_itp(ω - ωq, kq))^2)
                    f1 = nq + occ_fermion(ω + ωq - μ, T)
                    f2 = nq + 1 - occ_fermion(ω - ωq - μ, T)
                    Σs_imag_q[iω] += (dA1 * f1 + dA2 * f2) * factor
                end
            end

            Σs_imag_q
        end :: Vector{Float64}

        Σs_imag .*= 1 / (2π)^dim
        Σs_real = kramers_kronig(real.(S.ωs), Σs_imag)

        dΣs_out[:, ik] .= Σs_real .+ im .* Σs_imag
    end

    dΣs_out
end;

function solve_self_energy_dilute(S, ω_cutoff)
    # TODO: Interpolation to dense grid
    
    function _fixed_point!(R, x)
        dΣs_ = reshape(x, length(S.ωs), length(S.ks))
        dΣs_new = compute_self_energy_dilute(S, dΣs_, ω_cutoff)
        R .= vec(dΣs_new .- dΣs_)
        nothing
    end
    
    dΣs = zeros(ComplexF64, length(S.ωs), length(S.ks))
    res = nlsolve((R, x) -> _fixed_point!(R, x), vec(dΣs),
        method = :anderson,
        iterations = 100,
        ftol = 1e-4,
        beta = 0.7,
        m = 30,
        show_trace = true,
    )
    
    dΣs .= reshape(res.zero, size(dΣs))
    dΣs
end
