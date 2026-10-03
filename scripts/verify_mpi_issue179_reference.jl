# Optional independent Julia worker. Uses the exact staged Rust input, not Rust output.
using MVMCOptimizers
using LinearAlgebra
using MPI
VERSION == v"1.13.1" || error("issue179 requires Julia 1.13.1")
length(ARGS) == 3 || error("usage: <namelist> <real|cmp|fsz> <output-dir>")
println("Julia=", VERSION, " BLAS=", BLAS.get_config(), " threads=", Threads.nthreads())
try
    result = MVMCOptimizers.run_para_opt_from_namelist(
        ARGS[1]; nsteps=1, nsmp=1, mode=Symbol(ARGS[2]), output_dir=ARGS[3])
    println("reference status=", result.status)
catch err
    showerror(stderr, err, catch_backtrace())
    if MPI.Initialized() && !MPI.Finalized()
        MPI.Abort(MPI.COMM_WORLD, 1)
    end
    rethrow()
end
