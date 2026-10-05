/* Issue #342 native negative-INFO probe (optional developer tool).
 *
 * Runs the UNMODIFIED original shared initializer (makeInitialSample from
 * vmcmake.c, extracted by extract_initializers.pl; the real sampler calls the
 * same function, vmcmake_real.c:69) against the UNMODIFIED original matrix.c
 * (CalculateMAll_fcmp / CalculateMAll_real), original projection.c
 * (MakeProjCnt), original SFMT.c and the real pfapack Fortran
 * dsktrf/zsktrf + OpenBLAS (whose XERBLA reports and RETURNS).
 *
 * Only passive observation is added: the shared loop's CalculateMAll_fcmp call is wrapped by a function
 * that forwards to the original and logs its INFO. The globals normally set by
 * readdef.c/setmemory.c are set directly with the same formulas
 * (Nsize = 2*Ne, Nsite2 = 2*Nsite, readdef.c:757-758; LapackLWork from the
 * original getLWork_fcmp, vmcmain.c:259).
 *
 * usage: probe real|complex ne0|ne1
 */
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <complex.h>
#include <math.h>
#include <omp.h>
#include "global.h"
#include "blas_externs.h"
#include "projection.h"
#include "matrix.c"   /* original; pulls workspace.c */
#include "projection.c"
#include "sfmt/SFMT.h"

static int logged_info[256];
static int n_logged;

static int probe_calc_fcmp(const int *eleIdx, const int qpStart, const int qpEnd) {
  int info = CalculateMAll_fcmp(eleIdx, qpStart, qpEnd);
  logged_info[n_logged++] = info;
  return info;
}

/* vmcmake_real.c:69 and vmcmake.c:70 both call this complex shared initializer. */
#define CalculateMAll_fcmp probe_calc_fcmp
#include "initializer-complex.inc"
#undef CalculateMAll_fcmp

static void print_ints(const char *label, const int *v, int n) {
  printf("%s=", label);
  for (int i = 0; i < n; i++) printf("%s%d", i ? " " : "", v[i]);
  printf("\n");
}

int main(int argc, char **argv) {
  if (argc != 3) return 2;
  const int complex_mode = strcmp(argv[1], "complex") == 0;
  const int ne0 = strcmp(argv[2], "ne0") == 0;
  MPI_Init(&argc, &argv);

  NThread = omp_get_max_threads(); /* vmcmain.c:75 */
  initializeWorkSpaceAll();         /* setmemory.c:462 */
  Nsite = 2;
  Ne = ne0 ? 0 : 1;
  Nsize = 2 * Ne;     /* readdef.c:757 */
  Nsite2 = 2 * Nsite; /* readdef.c:758 */
  NProj = 0;
  NQPFull = 1;
  LocSpn = calloc(Nsite, sizeof(int)); /* itinerant electrons only */
  LapackLWork = getLWork_fcmp();      /* vmcmain.c:259, issues XERBLA for Nsize=0 */

  const int nsq = Nsize * Nsize;
  const int slater_n = Nsite2 * Nsite2;
  SlaterElm = calloc(slater_n, sizeof(double complex));
  SlaterElm_real = calloc(slater_n, sizeof(double));
  InvM = calloc(nsq > 0 ? nsq : 1, sizeof(double complex));
  InvM_real = calloc(nsq > 0 ? nsq : 1, sizeof(double));
  PfM = calloc(1, sizeof(double complex));
  PfM_real = calloc(1, sizeof(double));
  PfM[0] = 12345.0 + 0.0 * I; /* sentinels: the kernel must not publish */
  PfM_real[0] = 12345.0;
  if (!ne0) {
    for (int up = 0; up < Nsite; up++) {
      for (int down = 0; down < Nsite; down++) {
        SlaterElm[up * Nsite2 + down + Nsite] = 1.0;
        SlaterElm[(down + Nsite) * Nsite2 + up] = -1.0;
        SlaterElm_real[up * Nsite2 + down + Nsite] = 1.0;
        SlaterElm_real[(down + Nsite) * Nsite2 + up] = -1.0;
      }
    }
  }

  int eleIdx[2] = {-7, -7}, eleCfg[4] = {-7, -7, -7, -7}, eleNum[4] = {-7, -7, -7, -7};
  int eleProjCnt[1] = {-7};
  init_gen_rand(1);
  int ret = makeInitialSample(eleIdx, eleCfg, eleNum, eleProjCnt, 0, 1, MPI_COMM_SELF);
  const int pf_shared_unchanged = creal(PfM[0]) == 12345.0 && cimag(PfM[0]) == 0.0;
  /* Separate caller setup: vmcmake.c:110 (complex) / vmcmake_real.c:102 (real). */
  const int setup_info = complex_mode ? CalculateMAll_fcmp(eleIdx, 0, 1) : CalculateMAll_real(eleIdx, 0, 1);
  uint32_t after[4];
  for (int i = 0; i < 4; i++) after[i] = gen_rand32();
  init_gen_rand(1);
  uint32_t fresh[4];
  for (int i = 0; i < 4; i++) fresh[i] = gen_rand32();

  printf("mode=%s\ncase=%s\n", argv[1], argv[2]);
  printf("Nsite=%d\nNe=%d\nNsize=%d\nNsite2=%d\nqp=1\nseed=1\n", Nsite, Ne, Nsize, Nsite2);
  print_ints("kernel_infos", logged_info, n_logged);
  printf("attempts=%d\n", n_logged);
  printf("returned=%d\n", ret);
  print_ints("eleIdx", eleIdx, Nsize);
  print_ints("eleCfg", eleCfg, Nsite2);
  print_ints("eleNum", eleNum, Nsite2);
  printf("pf_shared_sentinel_unchanged=%d\n", pf_shared_unchanged);
  printf("setup_info=%d\n", setup_info);
  printf("pf_setup_sentinel_unchanged=%d\n",
         complex_mode ? (creal(PfM[0]) == 12345.0 && cimag(PfM[0]) == 0.0) : (PfM_real[0] == 12345.0));
  printf("rng_next4_after=%u %u %u %u\n", after[0], after[1], after[2], after[3]);
  printf("rng_next4_fresh=%u %u %u %u\n", fresh[0], fresh[1], fresh[2], fresh[3]);
  MPI_Finalize();
  return 0;
}
