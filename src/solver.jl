abstract type AbstractSolver end


function flatten(S :: ElectronPhononSolver)
    return flatten!(S, zeros(eltype(S.Σs), length(S.Σs)))
end

function flatten!(S :: ElectronPhononSolver, x)
    x[1:length(S.Σs)] .= S.Σs[:]
    return x
end

function unflatten!(S :: ElectronPhononSolver, x)
    S.Σs .= reshape(x[1:length(S.Σs)], size(S.Σs))

    # Impose Im Σ < 0
    inds = imag.(S.Σs) .>= 0
    S.Σs[inds] .= real.(S.Σs[inds]) .+ 0.0im

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
    kwargs_solver...
    )

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
    )

    verbose && println("Done. Calculation took $(round(time() - ti, digits = 3)) seconds.")

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

    if verbose
        @time compute_self_energy!(S; kwargs_self_energy...)
    else
        compute_self_energy!(S; kwargs_self_energy...)
    end
    compute_spectral_function!(S)
    nocc = compute_occupation!(S)

    callback(S)

    if update_μ
        update_chemical_potential!(S)
    end

    if verbose
        @info "occupation = $nocc, μ = $(S.model.μ)"
    end

    S
end
