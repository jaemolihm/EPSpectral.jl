abstract type AbstractSolver end


mutable struct ElectronPhononSolver{MT, VT} <: AbstractSolver
    iter :: Int

    model :: MT
    η :: Float64
    occupation :: Float64

    qpts :: Kpoints{VT}

    # ωs and ks are a Vector so they can be nonuniform.
    # ωs_dense and ks_dense are a StepRangeLen so linear interpolation over them are efficient.

    # ωs :: StepRangeLen{Float64, Base.TwicePrecision{Float64}, Base.TwicePrecision{Float64}, Int64}
    ωs :: Vector{Float64}
    ks :: Vector{Float64}
    ωs_dense :: StepRangeLen{Float64, Base.TwicePrecision{Float64}, Base.TwicePrecision{Float64}, Int64}
    ks_dense :: StepRangeLen{Float64, Base.TwicePrecision{Float64}, Base.TwicePrecision{Float64}, Int64}
    Σs :: Matrix{ComplexF64}
    As :: Matrix{Float64}
    Σs_dense :: Matrix{ComplexF64}
    As_dense :: Matrix{Float64}

    Σs_rest :: Vector{ComplexF64}

    spectral_sum :: Vector{Float64}
    spectral_occ :: Vector{Float64}

    # Precomputed integral weights
    # Kramers-Kronig transformation matrix for ωs (Re Σ = KK_map * Im Σ)
    KK_map :: Matrix{Float64}
    # Integration weight for f(k) = ∫d³k' g(k') / |k - k'|² becomes
    # f[ik] = ∑_ikq q_integral_matrix[ik, ikq] * g[ikq] where f[ik] and g[ik] are linear
    # spline coefficients
    q_integral_matrix :: Matrix{Float64}
end


function ElectronPhononSolver(model, ωs_, ks_, qpts; occupation, ωs_dense = nothing, ks_dense = nothing, η, Σs_rest = nothing)
    # Convert ranges to vectors
    ωs = Vector(ωs_)
    ks = Vector(ks_)

    # Default values for dense grids: same range, but 5 times denser
    if ωs_dense === nothing
        ωs_dense = range(extrema(ωs)..., step = minimum(diff(ωs)) / 5)
    end
    if ks_dense === nothing
        ks_dense = range(extrema(ks)..., step = minimum(diff(ks)) / 5)
    end

    Σs = zeros(ComplexF64, length(ωs), length(ks))
    As = zeros(length(ωs), length(ks))
    Σs_dense = zeros(ComplexF64, length(ωs_dense), length(ks_dense))
    As_dense = zeros(length(ωs_dense), length(ks_dense))
    spectral_sum = zeros(length(ks_dense))
    spectral_occ = zeros(length(ks_dense))

    if Σs_rest === nothing
        Σs_rest = zeros(ComplexF64, length(ks))
    end

    KK_map = get_Kramers_Kronig_map(ωs) ./ π

    # For efficient q integral, we use linear spline basis. (qpts is no longer used.)
    # This requires a uniform k mesh.
    # (For nonuniform k mesh, frohlich_compute_angle_factor must be updated.)
    if model isa FrohlichModel
        ks_tmp, _, q_integral_matrix = frohlich_compute_angle_factor(maximum(ks), length(ks), 1);
        q_integral_matrix .*= abs2(get_eph_g(1.0, model))
        @assert all(diff(ks) .≈ diff(ks)[1])
        @assert ks ≈ ks_tmp
    else
        q_integral_matrix = zeros(length(ks), length(ks))
    end

    ElectronPhononSolver(0, model, η, occupation, qpts, ωs, ks, ωs_dense, ks_dense,
        Σs, As, Σs_dense, As_dense, Σs_rest, spectral_sum, spectral_occ,
        KK_map, q_integral_matrix)
end

function Base.show(io :: IO, S :: ElectronPhononSolver)
    print(io, "ElectronPhononSolver(", S.model, ", η = ", S.η, ")\n")
    print(io, "qpts     : ", length(S.qpts), " points, ", "\n")
    print(io, "ωs       : ", length(S.ωs), " points, ", S.ωs, "\n")
    print(io, "ks       : ", length(S.ks), " points, ", S.ks, "\n")
    print(io, "ωs_dense : ", length(S.ωs_dense), " points, ", S.ωs_dense, "\n")
    print(io, "ks_dense : ", length(S.ks_dense), " points, ", S.ks_dense)
end



function flatten(S :: AbstractSolver)
    return flatten!(S, zeros(eltype(S.Σs), length(S.Σs)))
end

function flatten!(S :: AbstractSolver, x)
    x[1:length(S.Σs)] .= S.Σs[:]
    return x
end

function unflatten!(S :: AbstractSolver, x)
    S.Σs .= reshape(x[1:length(S.Σs)], size(S.Σs))

    # Impose Im Σ < 0
    inds = imag.(S.Σs) .>= 0
    S.Σs[inds] .= real.(S.Σs[inds]) .+ 0.0im

    # Update S.Σs_dense and S.As_dense
    compute_spectral_function!(S)

    return S
end


# fixed-point equation
function fixed_point!(
    R :: Vector{Q},
    x :: Vector{Q},
    S :: AbstractSolver,
    ;
    verbose :: Bool = true,
    callback = identity,
    kwargs_solver...
    ) :: Nothing where {Q}

    # update S from input vector
    unflatten!(S, x)

    # Iterate the solver
    iterate_solver!(S; callback, verbose, kwargs_solver...)

    # calculate residue
    flatten!(S, R)
    R .-= x

    return nothing
end


# run the solver
function ep_solve!(
    S :: AbstractSolver,
    ;
    maxiter  :: Int64 = 100,
    tol      :: Float64 = 1e-3,
    δ        :: Float64 = 0.85,
    mem      :: Int64 = 20,
    verbose  :: Bool = true,
    η_init   :: Float64 = 0.05,
    kwargs_solver...
    )

    # Initialize self-energy with constant imaginary part
    S.Σs .= -im .* η_init
    S.Σs_dense .= -im .* η_init

    verbose && println("Converging electron-phonon self-energies.")

    S.iter = 0

    ti = time()
    res = nlsolve((R, x) -> fixed_point!(R, x, S; verbose, kwargs_solver...), flatten(S),
        method=:anderson,
        iterations=maxiter,
        ftol=tol,
        beta=δ,
        m=mem,
        show_trace = verbose,
        store_trace = true,
    )

    # # Faster by ~50% due to no NLSolve overhead.
    # # Not used since (i) speedup is not that much, and (ii) NLsolve printing is good.
    # Σs_prev = copy(S.Σs)
    # for iter in 1:maxiter
    #     iterate_solver!(S; verbose, kwargs_solver...)

    #     err = norm(S.Σs .- Σs_prev) / norm(S.Σs)
    #     verbose && println("Iteration $iter, relative error $err")
    #     if err < tol
    #         verbose && println("Converged in $iter iterations, tol = $tol")
    #         break
    #     end
    #     if iter == maxiter
    #         println("Not converged in $maxiter iterations, err = $err > tol = $tol")
    #     end

    #     Σs_prev .= S.Σs
    # end
    # res = (; )

    verbose && println("Done. Calculation took $(round(time() - ti, digits = 3)) seconds.")

    if !verbose && !(res.f_converged)
        (; fnorm, stepnorm) = res.trace[res.iterations]
        println("Convergence not reached. Inf-norm $fnorm, Step 2-norm $stepnorm")
    end

    return res
end

function iterate_solver!(
    S :: AbstractSolver
    ;
    verbose :: Bool = true,
    callback = identity,
    update_μ = false,
    kwargs_self_energy...
    )

    S.iter += 1

    compute_self_energy!(S; kwargs_self_energy...)
    compute_spectral_function!(S)

    callback(S)

    if update_μ
        compute_occupation!(S)
        update_chemical_potential!(S)
    end

    # if verbose
    #     @info "occupation = $nocc, μ = $(S.model.μ)"
    # end

    S
end


function G0D0_truncation_window(e1, e2, T, doping)
    if doping == :Electron
        return (e1 + e2) / 2 - 2T - sqrt(4T^2 + (e1 - e2)^2 / 4)
    elseif doping == :Hole
        return (e1 + e2) / 2 + 2T + sqrt(4T^2 + (e1 - e2)^2 / 4)
    else
        throw(ArgumentError("doping should be :Hole or :Electron, but got $doping"))
    end
end
