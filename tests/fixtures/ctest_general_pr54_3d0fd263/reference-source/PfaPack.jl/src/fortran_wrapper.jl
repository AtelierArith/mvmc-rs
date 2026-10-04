using Libdl

# Get the package directory using @__DIR__ (works at compile time)
const libzsktf2 = joinpath(dirname(@__DIR__), "deps", "libzsktf2.$(dlext)")
const libdsktf2 = joinpath(dirname(@__DIR__), "deps", "libdsktf2.$(dlext)")

"""
    fimpl_zsktf2!(A::AbstractMatrix{ComplexF64}, iPiv::Vector{Cint}; uplo::Char='U', mode::Char='P')

Call Fortran ZSKTF2 routine for LTL decomposition of skew-symmetric matrix.

# Arguments
- `A`: Skew-symmetric `ComplexF64` matrix (input/output). The bundled Fortran
       kernel is built for `DOUBLE COMPLEX` only; other element types are
       intentionally not accepted to avoid silent ABI mismatches.
- `iPiv`: Pivot array (output)
- `uplo`: 'U' for upper triangular, 'L' for lower triangular (default: 'U')
- `mode`: 'N' for normal mode, 'P' for Pfaffian mode (default: 'P')

# Returns
- `info`: Return code (0 = success, >0 = error)
"""
function fimpl_zsktf2!(A::AbstractMatrix{ComplexF64}, iPiv::Vector{Cint}; uplo::Char='U', mode::Char='P')
    n = size(A, 1)
    lda = size(A, 1)
    info = Ref{Cint}(0)

    # Fortran uses underscore suffix by default with gfortran
    # Symbol name is zsktf2_ (Fortran adds trailing underscore, macOS adds leading underscore internally)
    # Use ccall with library path constant and symbol name
    # For SubArray, we need to ensure we have a contiguous pointer
    #
    # Fortran CHARACTER arguments require hidden length arguments at the end
    # gfortran passes these as Csize_t (size_t) for each CHARACTER argument
    ccall(
        (:zsktf2_, libzsktf2),
        Cvoid,
        (Ref{Cchar}, Ref{Cchar}, Ref{Cint}, Ptr{ComplexF64}, Ref{Cint}, Ptr{Cint}, Ref{Cint}, Csize_t, Csize_t),
        Cchar(uplo),
        Cchar(mode),
        Ref{Cint}(n),
        A,
        Ref{Cint}(lda),
        iPiv,
        info,
        Csize_t(1),  # Hidden length argument for UPLO (1 character)
        Csize_t(1)   # Hidden length argument for MODE (1 character)
    )

    return info[]
end

"""
    fimpl_dsktf2!(A::AbstractMatrix{Float64}, iPiv::Vector{Cint}; uplo::Char='U', mode::Char='N') -> Int

Call Fortran DSKTF2 routine for LTL decomposition of real skew-symmetric matrix.

# Arguments
- `A`: Real skew-symmetric matrix (input/output)
- `iPiv`: Pivot array (output)
- `uplo`: 'U' for upper triangular, 'L' for lower triangular (default: 'U')
- `mode`: 'N' for normal mode, 'P' for Pfaffian mode (default: 'N')

# Returns
- `info`: Return code (0 = success, >0 = error)
"""
function fimpl_dsktf2!(A::AbstractMatrix{Float64}, iPiv::Vector{Cint}; uplo::Char='U', mode::Char='N')
    n = size(A, 1)
    lda = size(A, 1)
    info = Ref{Cint}(0)

    # Fortran uses underscore suffix by default with gfortran
    # Fortran CHARACTER arguments require hidden length arguments at the end
    ccall(
        (:dsktf2_, libdsktf2),
        Cvoid,
        (Ref{Cchar}, Ref{Cchar}, Ref{Cint}, Ptr{Float64}, Ref{Cint}, Ptr{Cint}, Ref{Cint}, Csize_t, Csize_t),
        Cchar(uplo),
        Cchar(mode),
        Ref{Cint}(n),
        A,
        Ref{Cint}(lda),
        iPiv,
        info,
        Csize_t(1),  # Hidden length argument for UPLO (1 character)
        Csize_t(1)   # Hidden length argument for MODE (1 character)
    )

    return info[]
end