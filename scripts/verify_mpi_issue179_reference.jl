# Optional independent Julia worker. Uses the exact staged Rust input, not Rust output.
using MVMCOptimizers
using LinearAlgebra
using MPI
using MVMCExpertModeParsers, SFMT
VERSION == v"1.13.1" || error("issue179 requires Julia 1.13.1")
length(ARGS) == 3 || error("usage: <namelist> <real|cmp|fsz> <output-dir>")
println("Julia=", VERSION, " BLAS=", BLAS.get_config(), " threads=", Threads.nthreads())
ENV["JULIA_MVMC_MPI"] = "1"
include(joinpath(@__DIR__,"mpi_issue179_reference_provenance.jl"))
issue179_reference_provenance(ARGS[3],ARGS[1],[MVMCOptimizers,MVMCExpertModeParsers,SFMT])
try
    opttrans = any(line -> occursin(r"^\s*OptTrans\s"i, line), eachline(ARGS[1]))
    result = MVMCOptimizers.run_para_opt_from_namelist(
        ARGS[1]; nsteps=1, nsmp=1, mode=Symbol(ARGS[2]), output_dir=ARGS[3],
        initial_def=opttrans ? :none : :auto)
    println("reference status=", result.status)
    result.status == 0 || error("reference optimization failed: $(result.status)")
catch err
    showerror(stderr, err, catch_backtrace())
    if MPI.Initialized() && !MPI.Finalized()
        MPI.Abort(MPI.COMM_WORLD, 1)
    end
    rethrow()
end
