/* Optional #367 probe: unmodified C negative-DSROptStepDt normalization
 * (readdef.c:749-755), the SRinfo header selection (initfile.c:47-54) and the
 * direct-solver system construction stcOptInit (stcopt_dposv.c), all extracted
 * verbatim into negative_stepdt_upstream.inc, followed by LAPACK dposv exactly
 * as stcOptMain calls it. Standalone kernel check on fixed well-conditioned
 * operands: no MPI, no sampling, no full mVMC executable. Neither Cargo nor
 * Rust tests compile, invoke or read this probe.
 *
 * Usage: negative_stepdt <DSROptStepDt> <DSROptStaDel>
 */
#include <complex.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef int MPI_Comm; /* headers declare MPI prototypes; no MPI is linked. */
#include "global.h"
void stcOptInit(double *const S, double *const g, const int nSmat, const int *const smatToParaIdx);
extern void dposv_(const char *uplo, const int *n, const int *nrhs, double *a, const int *lda,
                   double *b, const int *ldb, int *info);
#include "negative_stepdt_upstream.inc"

#define NP 3
static const double OO[NP + 1][NP + 1] = {
    {1.0, 0.3, -0.2, 0.1},
    {0.3, 2.09, 0.44, 0.13},
    {-0.2, 0.44, 3.04, 0.18},
    {0.1, 0.13, 0.18, 1.51},
};
static const double HO[NP + 1] = {-1.5, 0.7, -0.4, 0.25};

int main(int argc, char **argv) {
  if (argc != 3) return 2;
  DSROptStepDt = strtod(argv[1], NULL);
  DSROptStaDel = strtod(argv[2], NULL);
  int rank = 1; /* silence the rank-0 stderr remark */
  NormalizeStepDt(rank);
  printf("RESULT srflag %d\n", SRFlag);
  printf("RESULT dt %.17g\n", DSROptStepDt);
  printf("RESULT header %s\n",
         SRFlag == 0 ? "#Npara Msize optCut diagCut sDiagMax  sDiagMin    absRmax       imax"
                     : "#Npara Msize optCut diagCut sEigenMax  sEigenMin    absRmax       imax");

  SROptSize = NP + 1;
  SROptOO = calloc((size_t)(2 * SROptSize) * (2 * SROptSize), sizeof(double complex));
  SROptHO = calloc((size_t)(2 * SROptSize), sizeof(double complex));
  SROptHO[0] = HO[0];
  for (int p = 0; p < NP; ++p) {
    SROptOO[p + 2] = OO[0][p + 1];
    SROptHO[p + 2] = HO[p + 1];
    for (int q = 0; q < NP; ++q)
      SROptOO[(p + 2) * (2 * SROptSize) + (q + 2)] = OO[p + 1][q + 1];
  }
  int idx[NP] = {0, 1, 2};
  double S[NP * NP], g[NP];
  stcOptInit(S, g, NP, idx);
  int n = NP, nrhs = 1, info;
  dposv_("U", &n, &nrhs, S, &n, g, &n, &info);
  printf("RESULT info %d\n", info);
  printf("RESULT solution");
  for (int i = 0; i < NP; ++i) printf(" %.17g", g[i]);
  printf("\n");
  return 0;
}
