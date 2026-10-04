using Libdl

# Get the package directory using @__DIR__ (works at compile time)
const libltl2inv = joinpath(dirname(@__DIR__), "deps", "libltl2inv.$(dlext)")

"""
    cimpl_utu2inv!(n::Int, A::Matrix{T}, ldA::Int, iPiv::Vector{Int},
               vT::Vector{T}, M::Matrix{T}, ldM::Int) where T

Compute inverse matrix from an upper triangular LTL-decomposed matrix.

This function calls the C++ implementation via ccall.
C++ function: utu2inv_z (for Complex{Float64}) or utu2inv_d (for Float64)

# Arguments
- `n`: Size of the matrix
- `A`: Upper triangular LTL-decomposed matrix (input/output, column-major)
- `ldA`: Leading dimension of A
- `iPiv`: Pivot array (1-based indexing)
- `vT`: Work vector for tridiagonal elements (length n-1)
- `M`: Work matrix (n x n, column-major)
- `ldM`: Leading dimension of M

# Reference
- C++: ltl2inv/invert.tcc:76-118 (utu2inv function)
- C interface: ltl2inv/ltl2inv.cc (utu2inv_z, utu2inv_d)
"""
function cimpl_utu2inv!(n::Int, A::AbstractMatrix{Complex{Float64}}, ldA::Int,
                    iPiv::Vector{<:Integer}, vT::Vector{Complex{Float64}},
                    M::AbstractMatrix{Complex{Float64}}, ldM::Int)
    # Convert iPiv to Cint array (1-based indexing is preserved)
    iPiv_c = Cint.(iPiv)

    # Call C++ function utu2inv_z
    ccall(
        (:utu2inv_z, libltl2inv),
        Cvoid,
        (Cint, Ptr{Complex{Float64}}, Cint, Ptr{Cint},
         Ptr{Complex{Float64}}, Ptr{Complex{Float64}}, Cint),
        Cint(n),
        A,
        Cint(ldA),
        iPiv_c,
        vT,
        M,
        Cint(ldM)
    )

    return nothing
end

function cimpl_utu2inv!(n::Int, A::AbstractMatrix{Float64}, ldA::Int,
                    iPiv::Vector{<:Integer}, vT::Vector{Float64},
                    M::AbstractMatrix{Float64}, ldM::Int)
    # Convert iPiv to Cint array (1-based indexing is preserved)
    iPiv_c = Cint.(iPiv)

    # Call C++ function utu2inv_d
    ccall(
        (:utu2inv_d, libltl2inv),
        Cvoid,
        (Cint, Ptr{Float64}, Cint, Ptr{Cint},
         Ptr{Float64}, Ptr{Float64}, Cint),
        Cint(n),
        A,
        Cint(ldA),
        iPiv_c,
        vT,
        M,
        Cint(ldM)
    )

    return nothing
end
