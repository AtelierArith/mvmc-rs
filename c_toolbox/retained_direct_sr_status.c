/* Standalone retained-system diagnostic; no mVMC algorithm/body is patched.
 * C authority: stcopt_dposv.c 44-45, UPLO='U', NRHS=1, original DPOSV.
 * See retained_direct_sr_status.md for source/compiler/reproduction provenance.
 */
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

extern void dposv_(const char *, const int *, const int *, double *,
                   const int *, double *, const int *, int *, size_t);

static void read_values(const char *path, double *values, size_t count) {
    FILE *file = fopen(path, "r");
    if (!file) { perror(path); exit(2); }
    for (size_t i = 0; i < count; ++i) {
        if (fscanf(file, "%lf", &values[i]) != 1 || !isfinite(values[i])) {
            fprintf(stderr, "invalid/short retained input: %s\n", path); exit(2);
        }
    }
    char extra;
    if (fscanf(file, " %c", &extra) != EOF) {
        fprintf(stderr, "extra retained input: %s\n", path); exit(2);
    }
    if (fclose(file)) { perror(path); exit(2); }
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: %s dimension matrix rhs\n", argv[0]); return 2; }
    char *end;
    errno = 0;
    long parsed = strtol(argv[1], &end, 10);
    if (errno || *end || parsed < 1 || parsed > 4096) return 2;
    int n = (int)parsed, nrhs = 1, info = -999;
    size_t count = (size_t)n * (size_t)n;
    double *a = malloc(count * sizeof(*a)), *b = malloc((size_t)n * sizeof(*b));
    if (!a || !b) return 2;
    read_values(argv[2], a, count);
    read_values(argv[3], b, (size_t)n);
    if (n >= 2) {
        long double minor = (long double)a[0] * a[n + 1]
                          - (long double)a[n] * a[n];
        printf("leading_minor_2x2=%.21Le\n", minor);
    }
    char uplo = 'U';
    dposv_(&uplo, &n, &nrhs, a, &n, b, &n, &info, 1);
    printf("dimension=%d UPLO=U NRHS=1 LAPACK_INFO=%d\n", n, info);
    free(a); free(b);
    /* Exit zero means diagnostic completion, NOT successful factorization. */
    return info < 0 ? 2 : 0;
}
