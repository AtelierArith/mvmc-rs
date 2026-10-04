/* Original optional fixed-operand driver; no extracted upstream implementation.
 * Operands: actual normalized step14 SR observer matrix/RHS. See README.md.
 * Contract: authoritative stcopt_dposv.c:33-49 calls DPOSV and returns INFO.
 * Not full native C sampler validation or a Rust runtime dependency.
 */
#include <cblas.h>
#include <math.h>
#include <stdio.h>

_Static_assert(sizeof(blasint) == sizeof(int), "LP64 BLAS/LAPACK required");
extern void dposv_(const char *, const int *, const int *, double *, const int *,
                   double *, const int *, int *);

int main(void) {
    const char upper = 'U';
    const int n = 2, nrhs = 1;
    double a[] = {8.32675595141552149e-17, -2.77555756156289135e-17,
                  -2.77555756156289135e-17, 6.93896329284626790e-18};
    double b[] = {8.88178419700125251e-18, -3.33066907387546950e-18};
    const double original[] = {b[0], b[1]};
    int info = -999;
    dposv_(&upper, &n, &nrhs, a, &n, b, &n, &info);
    const int unchanged = b[0] == original[0] && b[1] == original[1];
    printf("C_STEP14_DPOSV dimension=2 info=%d rhs_unchanged=%d finite=%d\n",
           info, unchanged, isfinite(b[0]) && isfinite(b[1]));
    return info != 2 || !unchanged || !isfinite(b[0]) || !isfinite(b[1]);
}
