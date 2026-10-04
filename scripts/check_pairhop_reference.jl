# The unchanged canonical C-reference checks supplement deterministic Julia gates.
using Test, LinearAlgebra
VERSION == v"1.13.1" || error("PairHop reference check requires Julia 1.13.1")
BLAS.set_num_threads(1)
@info "PairHop C-reference check" julia=VERSION blas=BLAS.get_config() threads=BLAS.get_num_threads()
include(joinpath(@__DIR__,"..","extern","Julia-mVMC","test","integration","pairhop_equivalent.jl"))
