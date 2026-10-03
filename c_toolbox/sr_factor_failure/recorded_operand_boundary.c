/* Optional original fixed-operand DPOSV driver; not a native sampler.
 * Contract follows extern/mVMC-1.3.0/src/mVMC/stcopt_dposv.c:33-49.
 * stdin: dimension, column-major matrix, RHS. No Rust/Cargo dependency.
 */
#include <cblas.h>
#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
_Static_assert(sizeof(blasint) == sizeof(int), "LP64 required");
extern void dposv_(const char *, const int *, const int *, double *, const int *,
                   double *, const int *, int *);
int main(int argc, char **argv) {
    if (argc != 3) return 2;
    char *end;
    long expected_n = strtol(argv[1], &end, 10);
    if (!*argv[1] || *end || expected_n < 1 || expected_n > 1024) return 2;
    long expected_info = strtol(argv[2], &end, 10);
    if (!*argv[2] || *end || expected_info < 1 || expected_info > expected_n) return 2;
    int n, info = -999;
    if (scanf("%d", &n) != 1 || n != expected_n) return 2;
    double *a = calloc((size_t)n * n, sizeof(double));
    double *b = calloc(n, sizeof(double)), *original = calloc(n, sizeof(double));
    if (!a || !b || !original) return 2;
    for (int i = 0; i < n * n; ++i) if (scanf("%lf", &a[i]) != 1 || !isfinite(a[i])) return 2;
    for (int i = 0; i < n; ++i) if (scanf("%lf", &b[i]) != 1 || !isfinite(b[i])) return 2;
    int next;
    while ((next = getchar()) != EOF) if (!isspace((unsigned char)next)) return 2;
    if (ferror(stdin)) return 2;
    memcpy(original, b, (size_t)n * sizeof(double));
    const char upper = 'U';
    const int nrhs = 1;
    dposv_(&upper, &n, &nrhs, a, &n, b, &n, &info);
    int unchanged = memcmp(original, b, (size_t)n * sizeof(double)) == 0;
    printf("dimension=%d UPLO=U NRHS=1 INFO=%d rhs_bits_unchanged=%d\n", n, info, unchanged);
    free(a); free(b); free(original);
    return info != expected_info || !unchanged;
}
