"""
PfaPack.jl

A Julia wrapper around Pfaffian and LTL decomposition routines for
dense skew-symmetric matrices. The package combines pure-Julia
implementations (BSD-3) with C++ reference implementations and Julia
ports under MPL-2.0 from xrq-phys/Pfaffine. See per-file headers and
`THIRD_PARTY_LICENSES.md` for the license map.
"""

module PfaPack

using LinearAlgebra
using Libdl: dlext

include("pfaffian.jl")          # BSD-3 (Wimmer Python derivative)
include("ltl_decomposition.jl") # BSD-3 (Wimmer Fortran derivative)
include("utu2.jl")              # MPL-2.0 (xrq-phys/Pfaffine derivative)
include("c_wrapper.jl")         # BSD-3 (Terasaki ccall to MPL libltl2inv)
include("fortran_wrapper.jl")   # BSD-3 (Terasaki ccall to Wimmer libzsktf2/libdsktf2)

using .Pfaffian
export pfaffian_ltl!

using .LTLDecomposition
export julia_zsktf2!, julia_dsktf2!, julia_zsktf2_turbo!

using .Utu2
export utu2pfa, utu2inv!

# Fortran wrappers
export fimpl_zsktf2!, fimpl_dsktf2!

# C++ wrapper (ccall to libltl2inv)
export cimpl_utu2inv!

end # module PfaPack
