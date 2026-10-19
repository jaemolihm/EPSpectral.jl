using TestItemRunner

# `Pkg.test(; test_args = ["test_frequency_grids.jl", ...])` runs only the test items in the named
# files (paths relative to test/). The items tagged `:gpu` run only when CUDA is functional (CI
# drops it from the test target).
const TEST_FILES = normpath.(ARGS)
for file in TEST_FILES
    path = joinpath(@__DIR__, file)
    isfile(path) || error("unknown test file $file")
    occursin(r"^@testitem"m, read(path, String)) || error("no @testitem in $file")
end
const HAVE_CUDA = try
    @eval using CUDA
    CUDA.functional()
catch
    false
end
HAVE_CUDA || @info "CUDA not functional: skipping the test items tagged :gpu"

@run_package_tests filter = ti -> (isempty(TEST_FILES) || relpath(ti.filename, @__DIR__) in TEST_FILES) &&
    (HAVE_CUDA || :gpu ∉ ti.tags)
