"""
Pfaffian calculation for skew-symmetric matrices.

Implements `pfaffian_ltl!`: the Pfaffian via the Parlett–Reid LTL
algorithm. This is a line-by-line Julia port of `pfaffian_LTL` from
upstream Wimmer PfaPack 2014-09, `python/pfaffian.py:247-308`. The
upstream `LapackLicence` (root of the PfaPack release) is the
applicable license; it is BSD-3-compatible, and we redistribute this
function under the package's primary BSD-3-Clause license accordingly.
"""

module Pfaffian

using LinearAlgebra

export pfaffian_ltl!

"""
    pfaffian_ltl!(A::AbstractMatrix{T}; overwrite_a=true) where T

Compute the Pfaffian of a skew-symmetric matrix `A` using the
Parlett–Reid algorithm. `A` is overwritten when `overwrite_a=true`.

Returns `zero(T)` when `n` is odd.

# Arguments
- `A`: skew-symmetric matrix (`A = -A^T`); must be square.
- `overwrite_a`: if `false`, a copy of `A` is used internally.

# Reference
- Python implementation: upstream Wimmer PfaPack 2014-09 release,
  `python/pfaffian.py:247-308` (`pfaffian_LTL`). The per-line
  `# Python: …` comments below trace back to the corresponding
  source lines.
- Algorithm: Parlett–Reid algorithm for skew-symmetric matrices
  (M. Wimmer, *Algorithm 923*, ACM TOMS 38 (2012), 30:1–30:17).
"""
function pfaffian_ltl!(A::AbstractMatrix{T}; overwrite_a::Bool=true) where T
    n = size(A, 1)

    if size(A, 1) != size(A, 2)
        throw(ArgumentError("Matrix must be square"))
    end

    # Quick return if odd-sized
    if n % 2 == 1
        return zero(T)
    end

    if !overwrite_a
        A = copy(A)
    end

    pfaffian_val = one(T)

    # Parlett-Reid algorithm
    # Python: for k in xrange(0, n-1, 2):
    for k in 0:2:(n-2)
        # Python: kp = k+1+np.abs(A[k+1:,k]).argmax()
        k1 = k + 1  # 1-based index of Python's k
        k2 = k + 2  # 1-based index of Python's k+1

        # Find pivot: max |A[i, k1]| for i in k2..n
        max_abs2 = abs2(A[k2, k1])
        max_idx = k2
        for i in (k2+1):n
            val_abs2 = abs2(A[i, k1])
            if val_abs2 > max_abs2
                max_abs2 = val_abs2
                max_idx = i
            end
        end

        # Pivot if necessary
        if max_idx != k2
            # Interchange rows k2 and max_idx
            @inbounds for j in 1:n
                A[k2, j], A[max_idx, j] = A[max_idx, j], A[k2, j]
            end

            # Interchange columns k2 and max_idx
            @inbounds for i in 1:n
                A[i, k2], A[i, max_idx] = A[i, max_idx], A[i, k2]
            end

            # Each interchange flips det(P)
            pfaffian_val = -pfaffian_val
        end

        # Python: if A[k+1,k] != 0.0:
        if iszero(A[k1, k2])
            return zero(T)
        end

        # Form the Gauss vector tau
        # Python: tau = A[k,k+2:].copy() / A[k,k+1]
        if k2 + 1 <= n
            tau = @view(A[k1, (k2+1):n])

            # Python: pfaffian_val *= A[k,k+1]
            pfaffian_val *= A[k1, k2]

            # Python: A[k+2:,k+2:] += np.outer(tau, A[k+2:,k+1])
            #                       - np.outer(A[k+2:,k+1], tau)
            y = @view A[(k2+1):n, k2]
            Asub = @view A[(k2+1):n, (k2+1):n]

            BLAS.ger!(inv(A[k1, k2]), tau, y, Asub)
            BLAS.ger!(-inv(A[k1, k2]), y, tau, Asub)
        else
            # Last iteration: just multiply by the off-diagonal element
            pfaffian_val *= A[k1, k2]
        end
    end

    return pfaffian_val
end

end # module Pfaffian
