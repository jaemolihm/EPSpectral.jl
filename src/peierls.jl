# Peierls model for the electron-phonon interaction
# See Eqs.(1-4) of arXiv:2501.05054.
# g(k, q) = -2i * g * (sin(k+q) - sin(k))

struct PeierlsLatticeModel
    g  :: Float64
    ω₀ :: Float64
    t  :: Float64
    μ  :: Float64
    T  :: Float64
end

Base.Broadcast.broadcastable(model::PeierlsLatticeModel) = Ref(model)

get_εk(k, model::PeierlsLatticeModel) = -2 * model.t * cos(k)

get_vk(k, model::PeierlsLatticeModel) = 2 * model.t * sin(k)

get_eph_g(k, q, model :: PeierlsLatticeModel) = -2 * im * model.g * (sin(k + q) - sin(k))

# Return dg/dk
get_eph_g_der(k, q, model :: PeierlsLatticeModel) = -2 * im * model.g * (cos(k + q) - cos(k))


function compute_self_energy!(S :: ElectronPhononSolver{PeierlsLatticeModel})
    (; ω₀, μ, T) = S.model
    # dim = get_dimension(S.qpts)
    Σ_itp = get_Σ_itp_dense(S, S.η)

    focc_p = occ_fermion.(S.ωs .+ ω₀ .- μ, T)
    focc_m = occ_fermion.(S.ωs .- ω₀ .- μ, T)

    # Holstein model self-energy is local, so compute only for one k point.
    Σs_all = tmap(S.ks) do k

        Σs_imag_k = tmapreduce(.+, chunks(1:length(S.qpts); n = 2 * Threads.nthreads()); chunking = false) do iqs
            Σs_imag_q = zeros(length(S.ωs))

            for iq in iqs
                q, weight = S.qpts[iq]
                εkq = get_εk(k + q, S.model)
                kq = k + q
                kq = mod(kq + π, 2π) - π  # periodic boundary condition, [-π, π]
                ωq = ω₀
                nq = occ_boson(ωq, T)
                gq = get_eph_g(k, q, S.model)

                factor = weight * abs2(gq)

                for (iω, ω) in enumerate(S.ωs)
                    if nq < sqrt(eps(ω₀))
                        if real(ω) > μ + ωq
                            Σs_imag_q[iω] += imag(1 / (ω - ωq - εkq - Σ_itp(ω - ωq, kq))) * factor
                        elseif real(ω) < μ - ωq
                            Σs_imag_q[iω] += imag(1 / (ω + ωq - εkq - Σ_itp(ω + ωq, kq))) * factor
                        end
                    else
                        # Finite-temperature case
                        fac1 = imag(1 / (ω + ωq - εkq - Σ_itp(ω + ωq, kq)))
                        fac2 = imag(1 / (ω - ωq - εkq - Σ_itp(ω - ωq, kq)))
                        Σs_imag_q[iω] += ( fac1 * (nq + focc_p[iω])
                                         + fac2 * (nq + 1 - focc_m[iω]) ) * factor
                    end
                end
            end

            Σs_imag_q
        end :: Vector{Float64}

        Σs_real_k = kramers_kronig(real.(S.ωs), Σs_imag_k)

        Σs_real_k .+ im .* Σs_imag_k
    end

    @views for ik in eachindex(S.ks)
        S.Σs[:, ik] .= Σs_all[ik] .+= S.Σs_rest[ik]
    end

    return S
end

function update_chemical_potential!(S :: ElectronPhononSolver{PeierlsLatticeModel})

    if S.occupation == 0
        μ_new = -Inf
    else
        # μ_new = find_zero(μ -> compute_occupation(S, μ) - S.occupation, extrema(S.ωs_dense))
        μ_new = find_zero(μ -> compute_occupation(S, μ) - S.occupation, (-500.0, 100.0))
    end

    S.model = PeierlsLatticeModel(S.model.g, S.model.ω₀, S.model.t, μ_new, S.model.T)

    return S
end
