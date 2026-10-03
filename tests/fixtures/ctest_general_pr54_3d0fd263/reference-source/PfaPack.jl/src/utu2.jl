# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# The routines in this file are Julia ports of the C++ "ltl2inv"
# implementations originally written by RuQing Xu (@xrq-phys) and
# released as part of xrq-phys/Pfaffine
# (https://github.com/xrq-phys/Pfaffine) under the Mozilla Public
# License 2.0. The Julia port preserves the original MPL-2.0 license
# per MPL §1.10 (Modifications). See `THIRD_PARTY_LICENSES.md` for the
# per-function provenance map.

"""
LTL-form helpers: Pfaffian and inverse extraction from a skew-symmetric
matrix that has already been LTL-decomposed (e.g. by `julia_zsktf2!`).

Wimmer's PfaPack 2014-09 Python reference (`python/pfaffian.py`)
provides `pfaffian_LTL` (LTL decomposition + Pfaffian in one pass) but
does not ship post-decomposition routines for re-extracting the
Pfaffian or computing the inverse from an existing LTL form. Those
were added by RuQing Xu in his C++ extension `xrq-phys/Pfaffine`
(MPL-2.0) and bundled at `deps/pfaffian.tcc` / `deps/invert.tcc`. The
Julia functions below are direct translations of those C++ routines
and therefore inherit the upstream MPL-2.0 license.
"""

module Utu2

using LinearAlgebra
using LinearAlgebra.LAPACK: trtri!

export utu2pfa, utu2inv!

"""
    utu2pfa(n, A, ldA, iPiv)

Compute the Pfaffian from a matrix that has already been put into
upper-triangular LTL form (so the formula reduces to a product of
off-diagonal entries with the pivot sign).

# Reference
- C++ source: `deps/pfaffian.tcc: utu2pfa()` (xrq-phys/Pfaffine, MPL-2.0).
"""
function utu2pfa(n::Int, A::AbstractMatrix{T}, ldA::Int, iPiv::Vector{<:Integer}) where T
    if n == 0
        return one(T)
    end

    pfaff = one(T)
    for i in 1:2:(n-1)
        pfaff *= A[i, i+1]
    end

    # Sign from the LTL pivot permutation
    sign = 1
    for i in 1:n
        if iPiv[i] != i
            sign = -sign
        end
    end

    return T(sign) * pfaff
end

"""
    sktdsmx!(n, vT, B, C)

Skew-symmetric tridiagonal system solver (internal helper for
`utu2inv!`). Solves `T * C = B` where `T` is skew-tridiagonal with
sub-/super-diagonal entries in `vT`.

# Reference
- C++ source: `deps/invert.tcc: sktdsmx()` (xrq-phys/Pfaffine, MPL-2.0).
"""
function sktdsmx!(n::Int, vT::Vector{T}, B::AbstractMatrix{T}, C::AbstractMatrix{T}) where T
    # Forward pass — fill C[2], C[4], …
    inv_minus_vT_1 = inv(-vT[1])
    @inbounds for j in 1:n
        C[2, j] = B[1, j] * inv_minus_vT_1
    end

    i_cpp = 2
    while i_cpp < n
        inv_minus_vT_i_cpp_1 = inv(-vT[i_cpp+1])
        vT_i_cpp = vT[i_cpp]
        @inbounds for j in 1:n
            # C[i+2, j] = (B[i+1, j] - C[i, j] * vT[i]) / -vT[i+1]
            C[i_cpp + 2, j] = (B[i_cpp + 1, j] - C[i_cpp, j] * vT_i_cpp) * inv_minus_vT_i_cpp_1
        end
        i_cpp += 2
    end

    # Backward pass — fill C[n-1], C[n-3], …
    vT_n_1 = inv(vT[n - 1])
    @inbounds for j in 1:n
        C[n - 1, j] = B[n, j] * vT_n_1
    end

    i_cpp = n - 3
    while i_cpp >= 1
        inv_vT_i_cpp = inv(vT[i_cpp])
        vT_i_cpp_1 = vT[i_cpp + 1]
        @inbounds for j in 1:n
            # C[i, j] = (B[i+1, j] + C[i+2, j] * vT[i+1]) / vT[i]
            C[i_cpp, j] = (B[i_cpp + 1, j] + C[i_cpp + 2, j] * vT_i_cpp_1) * inv_vT_i_cpp
        end
        i_cpp -= 2
    end
end

"""
    utu2inv!(n, A, ldA, iPiv, vT, M, ldM)

Compute the inverse of a skew-symmetric matrix from its upper-triangular
LTL form. The eight-step pipeline (trtri → lacpy → tridiagonal solve →
column permute → trmm → row permute) follows the upstream C++
implementation; `M` and `vT` are workspaces.

# Reference
- C++ source: `deps/invert.tcc:76-118` (xrq-phys/Pfaffine, MPL-2.0).
"""
function utu2inv!(n::Int, A::AbstractMatrix{T}, ldA::Int, iPiv::Vector{<:Integer},
                  vT::Vector{T}, M::AbstractMatrix{T}, ldM::Int) where T
    # Step 1: M ← I
    fill!(M, zero(T))
    for i in 1:n
        M[i, i] = one(T)
    end

    # Step 2: trtri — invert the unit upper-triangular submatrix A[1:n-1, 2:n]
    if n > 1
        trtri!('U', 'U', @view(A[1:n-1, 2:n]))
    end

    # Step 3: lacpy — copy upper-triangular block A[1:n-2, 3:n] → M[1:n-2, 2:n-1]
    if n > 2
        @inbounds for j_rel in 1:n-2
            for i_rel in 1:j_rel  # upper triangular
                M[i_rel, j_rel + 1] = A[i_rel, j_rel + 2]
            end
        end
    end

    # Step 4: extract negated tridiagonal: vT[i] = -A[i, i+1]
    for i in 1:n-1
        vT[i] = -A[i, i+1]
    end

    # Step 5: solve skew-tridiagonal system (input M, output A)
    sktdsmx!(n, vT, M, A)

    # Step 6: column permutation by iPiv (forward direction)
    for j in 1:n
        target = iPiv[j]
        if target != j
            @inbounds for i in 1:n
                A[i, j], A[i, target] = A[i, target], A[i, j]
            end
        end
    end

    # Step 7: A ← M^T * A (trmm, M unit upper-triangular)
    BLAS.trmm!('L', 'U', 'T', 'U', 1.0, M, A)

    # Step 8: row permutation by iPiv (forward direction, sequential)
    for i in 1:n
        target = iPiv[i]
        if target != i
            @inbounds for j in 1:n
                A[i, j], A[target, j] = A[target, j], A[i, j]
            end
        end
    end

    return nothing
end

end # module Utu2
