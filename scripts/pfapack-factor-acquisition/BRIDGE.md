# Conditional factor telescope → the actual inverse factor representation

Let B_k be the logical reduced skew matrix: its current active upper block is
the actual buffer, eliminated columns are mathematical zeros except the final
T band. The physical packed upper also stores previous multipliers; those are
NOT entries of B_k. The stream validates their exact prescribed swaps and
unchanged storage separately. Strict lower padding is never a logical operand.

At the next stage, P_k swaps kp with k-1. The actual active update and stored
l_i=x_i/p define G_k=I+l*e_(k-1)^T (l_i nonzero only for i<k-1).
The prospective, independently derived local ceiling bounds
P_k^T B_before P_k - G_k B_after G_k^T. The reconstruction has
A_ij=Schur_ij-l_i*y_j+y_i*l_j and A_i,k=l_i*p. Previously eliminated
T-band links touch neither the target k-1 nor the shear's earlier rows, so
their extension contributes no unmodelled local defect.

H starts at I. In chronological order it becomes H←H P_k, then H←H G_k.
Error contributions are H E_k H^T using H after permutation and before shear.
Therefore A=H_final T H_final^T+E_total, with
||E_total||inf <= Σ ||H_k||inf ||E_k||inf ||H_k^T||inf.
Only one column of H changes per shear. Outward dyadic capsules bound proof
representation rounding, not a globally uncertain production pivot trajectory.

Subsequent pivots permute the already stored multipliers' rows exactly as the
original algorithm swaps upper entries in columns above the active block.
Induction gives H_final=P_total U_packed, where U_ii=1,
U_ij=F[i,j+1] for i<j<n-1, and the last column is the last identity column.
P_total is the chronological product of the recorded transpositions. This is
the SAME unit-upper submatrix interpreted by actual utu2inv Step2, including
the implicitly unit diagonal and the identity workspace's final column.
The streaming consumer independently checks H_final versus P_total U_packed
within the predetermined proof-representation radii; a failed bridge is not
accepted by inflating them or measuring a final residual.

bridge-controls.mjs supplies n4 and n6 complete hand matrices with two actual
pivots, two nonzero dyadic shears, and independently specified H/T/U/P.
Both verify exact rational A=H T H^T and H=P U. n6 includes an invertible
disjoint final pair with the original intermediate zero-column INFO=4.
These are SOURCE controls pending execution, not evidence from Rust outputs.

This bridge does not prove inverse TRTRI/solve/TRMM/provider or Pf acceptance.
Those remain explicit obligations; a complete stream and factor ceiling alone
cannot declare the historical four-size regression or full models passing.
