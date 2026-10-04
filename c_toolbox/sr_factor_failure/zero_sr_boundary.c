/* Original optional fixed-operand driver; no extracted upstream implementation.
 * Contract: mVMC stcopt_dposv.c:33-49 returns DPOSV INFO, and stcopt.c
 * updates parameters only for INFO==0. See README.md for provenance.
 * Literal dimension-ten zero fixture, not a native C sampler trajectory.
 */
#include <cblas.h>
#include <math.h>
#include <stdio.h>

_Static_assert(sizeof(blasint) == sizeof(int), "LP64 BLAS/LAPACK required");
extern void dposv_(const char *, const int *, const int *, double *, const int *,
                   double *, const int *, int *);
extern void dpotrf_(const char *, const int *, double *, const int *, int *);
extern void dpotrs_(const char *, const int *, const int *, const double *,
                    const int *, double *, const int *, int *);

int main(void) {
    const char upper = 'U';
    const int n = 10, nrhs = 1;
    double s[100] = {0}, rhs[10] = {0};
    int info = -999;
    dposv_(&upper, &n, &nrhs, s, &n, rhs, &n, &info);
    int finite_rhs = 0;
    int unchanged = 1;
    for (int i = 0; i < n; ++i) {
        finite_rhs += isfinite(rhs[i]) != 0;
        unchanged &= rhs[i] == 0.0;
    }
    printf("C_DPOSV info=%d finite_rhs=%d dimension=%d rhs_unchanged=%d\n",
           info, finite_rhs, n, unchanged);
    if (info != 1 || finite_rhs != n || !unchanged) return 1;

    double old_s[100] = {0}, old_rhs[10] = {0};
    int factor = -999, solve = -999;
    dpotrf_(&upper, &n, old_s, &n, &factor);
    dpotrs_(&upper, &n, &nrhs, old_s, &n, old_rhs, &n, &solve);
    int nonfinite = 0;
    for (int i = 0; i < n; ++i) nonfinite += !isfinite(old_rhs[i]);
    printf("OLD_POSITIVE_INFO_IGNORED factor=%d solve=%d nonfinite_rhs=%d\n",
           factor, solve, nonfinite);
    return factor != 1 || solve != 0 || nonfinite != n;
}
