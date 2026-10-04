/* Optional fixed-operand diagnostic, not a native C sampler or Cargo dependency.
 * Original driver following stcopt_dposv.c:33-49 (DPOSV/INFO contract).
 * Operands are unmodified read-only SR observer output from
 * issue179-grouped20-current.QXcDew/repeats/r2-real-s2-cg0-store0/
 * prefix20/w1/repeat1/rank-0.txt, sr-system-000004 matrix and RHS.
 */
#include <cblas.h>
#include <math.h>
#include <stdio.h>

_Static_assert(sizeof(blasint) == sizeof(int), "LP64 required");
extern void dposv_(const char *, const int *, const int *, double *, const int *,
                   double *, const int *, int *);

int main(void) {
    const char upper = 'U';
    const int n = 2, nrhs = 1;
    double a[] = {6.93896329284626790e-18, -1.73472347597680709e-18,
                  -1.73472347597680709e-18, 1.08421301450722936e-19};
    double b[] = {-1.38777878078144570e-19, 1.73472347597680713e-20};
    const double original[] = {b[0], b[1]};
    int info = -999;
    dposv_(&upper, &n, &nrhs, a, &n, b, &n, &info);
    int unchanged = b[0] == original[0] && b[1] == original[1];
    printf("GROUPED_STEP4_DPOSV info=%d rhs_unchanged=%d finite=%d\n",
           info, unchanged, isfinite(b[0]) && isfinite(b[1]));
    return info != 2 || !unchanged;
}
