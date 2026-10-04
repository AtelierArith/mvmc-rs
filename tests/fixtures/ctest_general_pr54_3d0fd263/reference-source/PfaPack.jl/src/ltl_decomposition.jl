"""
LTL decomposition functions for skew-symmetric matrices.

This module implements LTL decomposition (Parlett-Reid algorithm) for
skew-symmetric matrices, which is used as a preprocessing step for
Pfaffian and inverse matrix calculations.

Reference implementations:
- Fortran: upstream Wimmer PfaPack 2014-09, `fortran/zsktf2.f` (ZSKTF2);
  the same file is bundled at `deps/zsktf2.f`.
"""

module LTLDecomposition

using LinearAlgebra
using LoopVectorization
using StructArrays

export julia_zsktf2!, julia_dsktf2!, julia_zsktf2_turbo!

# BLAS 1-norm for complex numbers: |Re(z)| + |Im(z)|
# This matches IZAMAX behavior in BLAS
@inline blas1norm(z::Complex) = abs(real(z)) + abs(imag(z))
@inline blas1norm(x::Real) = abs(x)

"""
    julia_zsktf2!(A::AbstractMatrix{T}, iPiv::Vector{Int}) where T

Perform LTL decomposition of a skew-symmetric matrix A in upper triangular form.
This is a faithful translation of the Fortran ZSKTF2 routine (UPLO='U', MODE='N').

# Arguments
- `A`: Skew-symmetric matrix (A = -A^T). Must be square. Will be overwritten.
- `iPiv`: Output pivot array (1-based indexing). Must be pre-allocated with size n.

# Returns
- `info::Int`: Return code (0 = success, >0 = error indicates zero pivot at that row)

# Reference
- Fortran: zsktf2.f, UPLO='U', MODE='N'
- For UPLO='U': iterations go from K=N down to K=2
- iPiv(K-1) = KP means row/column K-1 was interchanged with row/column KP
"""
function julia_zsktf2! end  # Forward declaration for docstring

"""
    julia_dsktf2!(A::AbstractMatrix{Float64}, iPiv::Vector{<:Integer})

Optimized LTL decomposition for Float64 skew-symmetric matrices.
Uses skew-symmetric rank-2 update (DSKR2) that only updates upper triangular part.

This is equivalent to Fortran DSKTF2 with UPLO='U', MODE='N'.
"""
function julia_dsktf2!(A::AbstractMatrix{Float64}, iPiv::Vector{<:Integer})
    n = size(A, 1)

    # Check if matrix is square
    if size(A, 1) != size(A, 2)
        throw(ArgumentError("Matrix must be square"))
    end

    info = 0

    # Initialize pivot array
    @inbounds for i in 1:n
        iPiv[i] = i
    end

    # Iterate K from N down to 2 (Fortran: DO K=N, 2, -1)
    @inbounds for k in n:-1:2
        # Find pivot: max|A[j, k]| for j = 1..k-1
        # IDAMAX returns 1-based index of element with max absolute value
        kp = 1
        colmax = abs(A[1, k])
        for j in 2:(k-1)
            val_abs = abs(A[j, k])
            if val_abs > colmax
                colmax = val_abs
                kp = j
            end
        end

        kk = k - 1  # The row/column to swap with kp

        if colmax == 0.0
            # Column is zero - record error but continue
            if info == 0
                info = k - 1
            end
            # Set pivot to k-1 (no swap)
            iPiv[kk] = kk
            continue
        end

        # Swap rows and columns kk and kp if needed
        if kp != kk
            # Swap columns kk and kp in rows 1:kp-1
            for j in 1:(kp-1)
                A[j, kk], A[j, kp] = A[j, kp], A[j, kk]
            end

            # Swap A[kp+1:kk-1, kk] with A[kp, kp+1:kk-1]
            for j in (kp+1):(kk-1)
                A[j, kk], A[kp, j] = A[kp, j], A[j, kk]
            end

            # Swap rows kk and kp in columns k:n
            for j in k:n
                A[kk, j], A[kp, j] = A[kp, j], A[kk, j]
            end

            # Negate elements
            for j in kp:(kk-1)
                A[j, kk] = -A[j, kk]
            end
            for j in (kp+1):(kk-1)
                A[kp, j] = -A[kp, j]
            end
        end

        # Store pivot
        iPiv[kk] = kp

        # Skew-symmetric rank-2 update of leading submatrix A[1:k-2, 1:k-2]
        # DSKR2: A := A + alpha*(x*y' - y*x') for upper triangular only
        if k >= 3
            alpha = 1.0 / A[kk, k]

            # Optimized skew-symmetric rank-2 update (upper triangular only)
            # This matches Fortran DSKR2 exactly
            for j in 1:k-2
                temp1 = alpha * A[j, kk]  # alpha * y[j]
                temp2 = alpha * A[j, k]   # alpha * x[j]

                @simd for i in 1:j-1
                    # A[i,j] += x[i]*temp1 - y[i]*temp2
                    A[i, j] += A[i, k] * temp1 - A[i, kk] * temp2
                end

                # Diagonal must be zero for skew-symmetric
                A[j, j] = 0.0
            end

            # Store L(k+1) in A(k): scale column k by alpha
            for j in 1:k-2
                A[j, k] *= alpha
            end
        end
    end

    return info
end

# Keep the original julia_zsktf2! for Real as a fallback (uses BLAS.ger!)
function julia_zsktf2!(A::AbstractMatrix{<:Real}, iPiv::Vector{<:Integer})
    # For Float64, delegate to optimized version
    if eltype(A) === Float64
        return julia_dsktf2!(A, iPiv)
    end

    n = size(A, 1)

    # Check if matrix is square
    if size(A, 1) != size(A, 2)
        throw(ArgumentError("Matrix must be square"))
    end

    info = 0

    # Initialize pivot array
    for i in 1:n
        iPiv[i] = i
    end

    # IPIV(N) = N (last element)
    iPiv[n] = n

    # Iterate K from N down to 2 (Fortran: DO K=N, 2, -1)
    for k in n:-1:2
        # Find pivot: max|A[j, k]| for j = 1..k-1
        kp = 1
        colmax = blas1norm(A[1, k])
        for j in 2:(k-1)
            val_abs = blas1norm(A[j, k])
            if val_abs > colmax
                colmax = val_abs
                kp = j
            end
        end

        kk = k - 1

        if colmax == zero(colmax)
            if info == 0
                info = k - 1
            end
            iPiv[kk] = kk
            continue
        end

        if kp != kk
            @inbounds for j in 1:(kp-1)
                A[j, kk], A[j, kp] = A[j, kp], A[j, kk]
            end

            @inbounds for j in (kp+1):(kk-1)
                A[j, kk], A[kp, j] = A[kp, j], A[j, kk]
            end

            @inbounds for j in k:n
                A[kk, j], A[kp, j] = A[kp, j], A[kk, j]
            end

            for j in kp:(kk-1)
                A[j, kk] = -A[j, kk]
            end

            for j in (kp+1):(kk-1)
                A[kp, j] = -A[kp, j]
            end
        end

        iPiv[kk] = kp

        if k >= 3
            alpha = one(eltype(A)) / A[kk, k]

            x = @view A[1:k-2, k]
            y = @view A[1:k-2, kk]
            Asub = @view A[1:k-2, 1:k-2]

            BLAS.ger!(alpha, x, y, Asub)
            BLAS.ger!(-alpha, y, x, Asub)

            @inbounds for i in 1:k-2
                A[i, i] = zero(eltype(A))
            end

            BLAS.scal!(k-2, alpha, @view(A[1:k-2, k]), 1)
        end
    end

    return info
end

function julia_zsktf2!(A::AbstractMatrix{<:Complex}, iPiv::Vector{<:Integer})
    n = size(A, 1)

    # Check if matrix is square
    if size(A, 1) != size(A, 2)
        throw(ArgumentError("Matrix must be square"))
    end

    info = 0

    # Initialize pivot array
    @inbounds for i in 1:n
        iPiv[i] = i
    end

    # Iterate K from N down to 2 (Fortran: DO K=N, 2, -1)
    @inbounds for k in n:-1:2
        # Find pivot: max|A[j, k]| for j = 1..k-1
        # IZAMAX returns 1-based index of element with max |Re| + |Im| (BLAS 1-norm)
        kp = 1
        colmax = blas1norm(A[1, k])
        for j in 2:(k-1)
            val_abs = blas1norm(A[j, k])
            if val_abs > colmax
                colmax = val_abs
                kp = j
            end
        end

        kk = k - 1  # The row/column to swap with kp

        if colmax == zero(colmax)
            # Column is zero - record error but continue
            if info == 0
                info = k - 1
            end
            # Set pivot to k-1 (no swap)
            iPiv[kk] = kk
            continue
        end

        # Swap rows and columns kk and kp if needed
        if kp != kk
            # Swap columns kk and kp in rows 1:kp-1
            for j in 1:(kp-1)
                A[j, kk], A[j, kp] = A[j, kp], A[j, kk]
            end

            # Swap A[kp+1:kk-1, kk] with A[kp, kp+1:kk-1]
            for j in (kp+1):(kk-1)
                A[j, kk], A[kp, j] = A[kp, j], A[j, kk]
            end

            # Swap rows kk and kp in columns k:n
            for j in k:n
                A[kk, j], A[kp, j] = A[kp, j], A[kk, j]
            end

            # Negate elements
            for j in kp:(kk-1)
                A[j, kk] = -A[j, kk]
            end
            for j in (kp+1):(kk-1)
                A[kp, j] = -A[kp, j]
            end
        end

        # Store pivot
        iPiv[kk] = kp

        # Skew-symmetric rank-2 update of leading submatrix A[1:k-2, 1:k-2]
        # ZSKR2: A := A + alpha*(x*y' - y*x') for upper triangular only
        if k >= 3
            alpha = one(eltype(A)) / A[kk, k]

            # Optimized skew-symmetric rank-2 update (upper triangular only)
            # This matches Fortran ZSKR2 exactly
            for j in 1:k-2
                temp1 = alpha * A[j, kk]  # alpha * y[j]
                temp2 = alpha * A[j, k]   # alpha * x[j]

                for i in 1:j-1
                    # A[i,j] += x[i]*temp1 - y[i]*temp2
                    A[i, j] += A[i, k] * temp1 - A[i, kk] * temp2
                end

                # Diagonal must be zero for skew-symmetric
                A[j, j] = zero(eltype(A))
            end

            # Store L(k+1) in A(k): scale column k by alpha
            for j in 1:k-2
                A[j, k] *= alpha
            end
        end
    end

    return info
end

"""
    julia_zsktf2_turbo!(A::AbstractMatrix{ComplexF64}, iPiv::Vector{<:Integer})

Optimized LTL decomposition for ComplexF64 skew-symmetric matrices using LoopVectorization.
Uses StructArrays to separate real/imaginary parts into contiguous arrays for SIMD vectorization.

This is equivalent to Fortran ZSKTF2 with UPLO='U', MODE='N'.
"""
function julia_zsktf2_turbo!(A::AbstractMatrix{ComplexF64}, iPiv::Vector{<:Integer})
    n = size(A, 1)

    # Check if matrix is square
    if size(A, 1) != size(A, 2)
        throw(ArgumentError("Matrix must be square"))
    end

    info = 0

    # Convert to StructArray for contiguous real/imaginary access
    # This creates separate contiguous arrays for real and imaginary parts
    A_soa = StructArray(A)
    A_re = A_soa.re
    A_im = A_soa.im

    # Initialize pivot array
    @inbounds for i in 1:n
        iPiv[i] = i
    end

    # Iterate K from N down to 2 (Fortran: DO K=N, 2, -1)
    @inbounds for k in n:-1:2
        # Find pivot: max|A[j, k]| for j = 1..k-1
        # IZAMAX returns 1-based index of element with max |Re| + |Im| (BLAS 1-norm)
        kp = 1
        colmax = abs(A_re[1, k]) + abs(A_im[1, k])
        for j in 2:(k-1)
            val_abs = abs(A_re[j, k]) + abs(A_im[j, k])
            if val_abs > colmax
                colmax = val_abs
                kp = j
            end
        end

        kk = k - 1  # The row/column to swap with kp

        if colmax == 0.0
            # Column is zero - record error but continue
            if info == 0
                info = k - 1
            end
            # Set pivot to k-1 (no swap)
            iPiv[kk] = kk
            continue
        end

        # Swap rows and columns kk and kp if needed
        if kp != kk
            # Swap columns kk and kp in rows 1:kp-1
            for j in 1:(kp-1)
                A_re[j, kk], A_re[j, kp] = A_re[j, kp], A_re[j, kk]
                A_im[j, kk], A_im[j, kp] = A_im[j, kp], A_im[j, kk]
            end

            # Swap A[kp+1:kk-1, kk] with A[kp, kp+1:kk-1]
            for j in (kp+1):(kk-1)
                A_re[j, kk], A_re[kp, j] = A_re[kp, j], A_re[j, kk]
                A_im[j, kk], A_im[kp, j] = A_im[kp, j], A_im[j, kk]
            end

            # Swap rows kk and kp in columns k:n
            for j in k:n
                A_re[kk, j], A_re[kp, j] = A_re[kp, j], A_re[kk, j]
                A_im[kk, j], A_im[kp, j] = A_im[kp, j], A_im[kk, j]
            end

            # Negate elements A[j, kk] for j in kp:(kk-1)
            for j in kp:(kk-1)
                A_re[j, kk] = -A_re[j, kk]
                A_im[j, kk] = -A_im[j, kk]
            end

            # Negate elements A[kp, j] for j in (kp+1):(kk-1)
            for j in (kp+1):(kk-1)
                A_re[kp, j] = -A_re[kp, j]
                A_im[kp, j] = -A_im[kp, j]
            end
        end

        # Store pivot
        iPiv[kk] = kp

        # Skew-symmetric rank-2 update of leading submatrix A[1:k-2, 1:k-2]
        # ZSKR2: A := A + alpha*(x*y' - y*x') for upper triangular only
        if k >= 3
            # alpha = 1 / A[kk, k] = conj(A[kk,k]) / |A[kk,k]|^2
            denom = A_re[kk, k]^2 + A_im[kk, k]^2
            alpha_re = A_re[kk, k] / denom
            alpha_im = -A_im[kk, k] / denom

            # Optimized skew-symmetric rank-2 update (upper triangular only)
            for j in 1:k-2
                # temp1 = alpha * A[j, kk] (complex multiplication)
                Ajkk_re = A_re[j, kk]
                Ajkk_im = A_im[j, kk]
                temp1_re = alpha_re * Ajkk_re - alpha_im * Ajkk_im
                temp1_im = alpha_re * Ajkk_im + alpha_im * Ajkk_re

                # temp2 = alpha * A[j, k] (complex multiplication)
                Ajk_re = A_re[j, k]
                Ajk_im = A_im[j, k]
                temp2_re = alpha_re * Ajk_re - alpha_im * Ajk_im
                temp2_im = alpha_re * Ajk_im + alpha_im * Ajk_re

                # A[i,j] += A[i,k]*temp1 - A[i,kk]*temp2 for i in 1:j-1
                # Using @turbo for SIMD on contiguous arrays
                @turbo for i in 1:j-1
                    # A[i,k] * temp1 - A[i,kk] * temp2
                    A_re[i, j] += (A_re[i, k] * temp1_re - A_im[i, k] * temp1_im) -
                                  (A_re[i, kk] * temp2_re - A_im[i, kk] * temp2_im)
                    A_im[i, j] += (A_re[i, k] * temp1_im + A_im[i, k] * temp1_re) -
                                  (A_re[i, kk] * temp2_im + A_im[i, kk] * temp2_re)
                end

                # Diagonal must be zero for skew-symmetric
                A_re[j, j] = 0.0
                A_im[j, j] = 0.0
            end

            # Store L(k+1) in A(k): scale column k by alpha
            @turbo for j in 1:k-2
                Ajk_re = A_re[j, k]
                Ajk_im = A_im[j, k]
                A_re[j, k] = alpha_re * Ajk_re - alpha_im * Ajk_im
                A_im[j, k] = alpha_re * Ajk_im + alpha_im * Ajk_re
            end
        end
    end

    # Copy results back to original array
    @inbounds for j in 1:n
        @simd for i in 1:n
            A[i, j] = Complex(A_re[i, j], A_im[i, j])
        end
    end

    return info
end

end # module LTLDecomposition
