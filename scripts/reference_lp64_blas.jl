# Optional ABI-only adapter. Julia's unmodified solver calls the same system
# BLAS/LAPACK as Rust, with ILP64 dimensions converted to checked LP64 scalars.
# This is reference generation only; no Rust build/test loads this library.
module ReferenceLP64BLAS
using LinearAlgebra, Libdl, SHA
const HANDLE = Ref{Ptr{Cvoid}}(C_NULL)
function install!(library)
    HANDLE[] = Libdl.dlopen(library)
    ccall(Libdl.dlsym(HANDLE[], :mvmc_reference_blas_init), Cvoid, ())
    for routine in ("dpotrf", "dpotrs", "dgemv", "dsyrk", "dgemm", "dtrtri", "ztrtri", "dtrmm", "ztrmm", "zgemv", "zgeru", "zscal", "zaxpy")
        address = Libdl.dlsym(HANDLE[], Symbol("mvmc_reference_" * routine))
        BLAS.lbt_set_forward(routine * "_", address, :ilp64) == 0 || error("Cannot forward $routine")
    end
    config = unsafe_string(ccall(Libdl.dlsym(HANDLE[], :mvmc_reference_blas_config), Cstring, ()))
    core = unsafe_string(ccall(Libdl.dlsym(HANDLE[], :mvmc_reference_blas_core), Cstring, ()))
    return "# Independent Julia solver with ABI-only LP64 system BLAS adapter; $config; core=$core; threads=1; bridge sha256=$(bytes2hex(sha256(read(library))))\n"
end
end
