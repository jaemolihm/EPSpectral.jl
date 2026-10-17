# The ladder on the GPU against the host, on the pb window: first on one shared setup, then end to
# end. Runs under `CUDA.allowscalar(false)`, so every step is a device-generic broadcast, reduction
# or GEMM.
@testset "ladder CPU vs GPU" begin
    eV = unit_to_aru(:eV)
    pb = pb_ladder_case()
    (; model, nk, window, occ, res) = pb
    gpu = EP.gpu_backend()
    Ωs = [0.0, 0.1 * eV]
    loop_kwargs = (; solver = :gmres, rtol = 1e-2, maxiter = 30, store_vertex = true,
        ω_acoustic = EP.omega_acoustic, verbosity = 0)
    # The loop on a ladder setup, on its backend.
    run_loop(ladder, backend; gpu_tile = nothing, Ωs = Ωs, kwargs...) = EPSpectral._loop_ladder(
        ladder, res.Σ, res.μ, res.Tlist, Ωs; loop_kwargs..., kwargs..., backend, gpu_tile)
    rel(a, b) = norm(a - b) / norm(b)

    # The setup of the host extraction with its arrays on the device, except the ones the loops
    # keep on the host (g2, the states).
    on_device(name, value) = name in (:g2, :el_i, :el_f, :multiplets) || !(value isa Array) ?
        value : EP.to_device(gpu, value)
    names = keys(pb.setup)
    setup_device = NamedTuple{names}(map(on_device, names, values(pb.setup)))
    device_ladder(dω_shift) = EPSpectral._setup_ladder(setup_device, res, pb.symmetry; dω_shift,
        backend = gpu)

    @testset "loop on one shared setup" begin
        # The only place a device loop sees host-extracted data, which `run_scgd0_ladder` never
        # does. Covers g2 resident and streamed, the default shift step (the shift index of every
        # mode is 0 or -1 on this fixture) and a quarter of it (up to 3), and run-to-run
        # reproducibility of each. The device and host iterations stop after the same number of
        # steps; the solutions then differ by round-off (measured below 2e-15, on an A100).
        for dω_shift in (pb.dω_shift, pb.dω_shift / 4)
            host = run_loop(with_shift_step(pb, dω_shift).ladder, EP.CPUBackend())
            ladder_dev = device_ladder(dω_shift)
            for gpu_tile in (nothing, 7)
                # `gpu_tile` forces streaming, and the small g2 is resident otherwise.
                workspace = EPSpectral._ladder_workspace(ladder_dev, gpu, gpu_tile, 0)
                @test workspace.streamed == (gpu_tile !== nothing)
                # The solver's own device memory counts against the resident g2.
                @test EPSpectral._ladder_workspace(ladder_dev, gpu, nothing, 0; solver_bytes = 10^15).streamed
                dev = run_loop(ladder_dev, gpu; gpu_tile)
                @test dev.niter == host.niter && all(dev.converged)
                @test rel(dev.σ_bubble, host.σ_bubble) <= 1e-10
                @test rel(dev.σ_ladder, host.σ_ladder) <= 1e-10
                @test rel(dev.J, host.J) <= 1e-10
                # The correction is not negligible, so the agreement is not a trivial one.
                @test norm(host.σ_ladder - host.σ_bubble) > 1e-2 * norm(host.σ_bubble)
                again = run_loop(ladder_dev, gpu; gpu_tile)
                @test again.J == dev.J && again.σ_ladder == dev.σ_ladder &&
                    again.σ_bubble == dev.σ_bubble && again.history == dev.history
            end
        end
        # The fixed point, streamed. Ω > 0 only: it diverges at the DC point of this fixture.
        fixed_kwargs = (; solver = :fixed_point, rtol = 1e-6, maxiter = 200, Ωs = Ωs[2:2])
        host = run_loop(pb.ladder, EP.CPUBackend(); fixed_kwargs...)
        dev = run_loop(device_ladder(pb.dω_shift), gpu; gpu_tile = 7, fixed_kwargs...)
        @test dev.niter == host.niter && all(dev.converged)
        @test rel(dev.σ_ladder, host.σ_ladder) <= 1e-10
    end

    @testset "end to end" begin
        # Each run extracts g2 and diagonalizes on its own backend, so the electron gauge inside a
        # multiplet differs between backends. The pb multiplets are split by up to 2.4e-7 eV, so
        # the gate is 1e-6; the measured difference is below 2e-15 (A100).
        run_l(backend; kwargs...) = run_scgd0_ladder(model, nk, window, res; occ, Ωs, verbosity = 0,
            backend, kwargs...)
        host = run_l(EP.CPUBackend())
        dev = run_l(gpu)
        @test dev.niter == host.niter && all(dev.converged)
        @test rel(dev.σ_bubble, host.σ_bubble) <= 1e-6
        @test rel(dev.σ_ladder, host.σ_ladder) <= 1e-6
        @test norm(host.σ_ladder - host.σ_bubble) > 1e-2 * norm(host.σ_bubble)
        # Two GPU runs, each with its own extraction and eigensolve, and the streamed one.
        again = run_l(gpu)
        @test again.σ_ladder == dev.σ_ladder && again.σ_bubble == dev.σ_bubble
        streamed = run_l(gpu; gpu_tile = 7)
        @test rel(streamed.σ_ladder, dev.σ_ladder) <= 1e-10
    end

    @testset "device memory left by an earlier call" begin
        # A device of `budget` bytes, on which the memory CUDA.jl's pool holds counts as used
        # (mirroring `free_bytes` in ElectronPhononCUDAExt.jl). A call before this one leaves
        # freed buffers in the pool, and the extraction's `plan_batch` refuses a device on which
        # its committed bytes exceed the free bytes, so each call must return them to the driver
        # first. The pb g2 output is far below the committed bytes, so the extraction's
        # resident-or-streamed fallback is not reachable on this fixture; the check is that the
        # leftovers do not make a call throw and that a call behind another gives the same result.
        budget = 5_000_000_000
        @eval ElectronPhonon free_bytes(::GPUBackend) = max(0, $budget - $(CUDA.cached_memory)())
        try
            results = map(1:2) do _
                CUDA.unsafe_free!(CUDA.zeros(UInt8, budget + 10^8))
                Base.invokelatest(run_scgd0_ladder, model, nk, window, res; occ, Ωs = Ωs[1:1],
                    verbosity = 0, backend = gpu)
            end
            @test results[1].σ_ladder == results[2].σ_ladder
        finally
            @eval ElectronPhonon free_bytes(::GPUBackend) = $(CUDA.free_memory)()
        end
    end
end
