# Use MPI.jl's selected launcher/ABI, rather than mixing Open MPI and MPICH.
using MPI
VERSION == v"1.13.1" || error("Julia 1.13.1 required")
length(ARGS) >= 2 || error("usage: <ranks> <worker> [worker arguments]")
println(MPI.Get_library_version())
run(`$(MPI.mpiexec()) -n $(parse(Int, ARGS[1])) $(Base.julia_cmd()) --project=$(Base.active_project()) $(ARGS[2:end])`)
