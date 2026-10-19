using TestItemRunner

# `Pkg.test(; test_args = ["test_frequency_grids.jl", ...])` runs only the test items in the named
# files. The items tagged `:gpu` run only when CUDA is functional (CI drops it from the test target).
for file in ARGS
    isfile(joinpath(@__DIR__, file)) || error("unknown test file $file")
end
const HAVE_CUDA = try
    @eval using CUDA
    CUDA.functional()
catch
    false
end
HAVE_CUDA || @info "CUDA not functional: skipping the test items tagged :gpu"

@run_package_tests filter = ti -> (isempty(ARGS) || basename(ti.filename) in ARGS) &&
    (HAVE_CUDA || :gpu ∉ ti.tags)
