__precompile__(true)
module EPSpectral

    using PrecompileTools

    @recompile_invalidations begin
        using LinearAlgebra
        using StaticArrays
        using OhMyThreads
        using ChunkSplitters
        using PolyLog
        using Interpolations
        using QuadGK
        using Roots
        using NLsolve
        using Base.Threads
    end

    include("kpoints.jl")
    include("kramers_kronig.jl")
    include("frohlich.jl")
    include("solver.jl")
    include("solver_frohlich.jl")

    @compile_workload begin
        α = 1.0
        m = 0.5
        ω₀ = 1.0
        μ = -Inf
        model = FrohlichModel(α, ω₀, m, μ)

        qmax = 5.0
        nq = 10
        order = 4
        qpts = polynomial_grid_3d(qmax, nq, order)

        ω = 3.0 + 0.5im
        k = SVector(2.0, 0., 0.)

        get_Σ_analytic(k, ω, model)
        get_Σ_mesh(k, ω, qpts, model)
    end


    export
        Kpoints,
        polynomial_grid_1d,
        polynomial_grid_3d,
        kramers_kronig,
        FrohlichModel,
        get_εk,
        get_Σ_analytic,
        get_Σ_mesh,
        ElectronPhononSolver,
        compute_self_energy_analytic!, compute_spectral_function!, compute_occupation!,
        plot_spectral_function!, update_chemical_potential!, plot_self_energy!

end
