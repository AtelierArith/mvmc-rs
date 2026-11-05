/* Optional standalone probe; never compiled or read by Cargo. GPL-3.0-or-later (the
 * included helpers are verbatim from mVMC 1.4.0 src/mVMC/average.c).
 * Regenerates crates/mvmc-core/tests/fixtures/normalization_c_literals.tsv: the
 * single-process branch of weightAverageReduce_real/_fcmp for weights 3, 10, 100 and
 * inputs .3-.7i, -.7+.3i, 13.486979846002933. See README.md. */
#include <complex.h>
#include <inttypes.h>
#include <stdio.h>
#include <string.h>

typedef int MPI_Comm;
static double complex Wc;
/* Size-one stubs: the probe exercises only the size == 1 branch. */
static int MPI_Comm_rank(MPI_Comm c, int *r) { (void)c; *r = 0; return 0; }
static int MPI_Comm_size(MPI_Comm c, int *s) { (void)c; *s = 1; return 0; }
static void RequestWorkSpaceComplex(int n) { (void)n; }
static double complex *GetWorkSpaceComplex(int n) { (void)n; return 0; }
static void ReleaseWorkSpaceComplex(void) {}
static void RequestWorkSpaceDouble(int n) { (void)n; }
static double *GetWorkSpaceDouble(int n) { (void)n; return 0; }
static void ReleaseWorkSpaceDouble(void) {}
static void SafeMpiReduce_fcmp(double complex *a, double complex *b, int n, MPI_Comm c) {
  (void)a; (void)b; (void)n; (void)c;
}
static void SafeMpiReduce(double *a, double *b, int n, MPI_Comm c) {
  (void)a; (void)b; (void)n; (void)c;
}

#include "average_helpers.inc"

static uint64_t bits(double x) { uint64_t u; memcpy(&u, &x, sizeof u); return u; }

int main(void) {
  const double weights[] = {3.0, 10.0, 100.0};
  const double complex inputs[] = {0.3 - 0.7 * I, -0.7 + 0.3 * I, 13.486979846002933};
  printf("weight_index\tvalue_index\treal_bits\tcomplex_re_bits\tcomplex_im_bits\n");
  for (int w = 0; w < 3; w++) {
    for (int v = 0; v < 3; v++) {
      Wc = weights[w];
      double real = creal(inputs[v]);
      double complex cmp = inputs[v];
      weightAverageReduce_real(1, &real, 0);
      weightAverageReduce_fcmp(1, &cmp, 0);
      printf("%d\t%d\t%016" PRIx64 "\t%016" PRIx64 "\t%016" PRIx64 "\n", w, v, bits(real),
             bits(creal(cmp)), bits(cimag(cmp)));
    }
  }
  return 0;
}
