/* Issue #176 native real-FSZ setup/initializer probe (optional developer tool).
 *
 * Links the UNMODIFIED original matrix.c (CalculateMAll_fsz_real and its child:
 * spin-indexed assembly, M_DSKTRF, utu2pfa_d, utu2inv_d, sign flip), projection.c
 * (MakeProjCnt), SFMT.c, the extracted original makeInitialSample_fsz_real and
 * the real pfapack Fortran dsktrf + OpenBLAS. Only passive observation is added:
 * CalculateMAll_fsz_real is wrapped to log its INFO, and MPI_Abort (the 101-attempt
 * failure exit) is replaced by a longjmp that records the abort. Globals normally
 * set by readdef.c/setmemory.c are set directly with the same formulas
 * (Nsize = 2*Ne, Nsite2 = 2*Nsite).
 *
 * The C kernel stores results at local index qp-qpStart; this probe prints them
 * under the absolute QP number used by the Rust layout.
 *
 * usage: probe matrix <setup.txt>   |   probe init
 */
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <complex.h>
#include <math.h>
#include <setjmp.h>
#include <stdint.h>
#include <omp.h>
#include "global.h"
#include "blas_externs.h"
#include "projection.h"
#include "matrix.c"
#include "projection.c"
#include "sfmt/SFMT.h"

static int logged_info[256];
static int n_logged;
static jmp_buf abort_jump;
static int aborted;

static int probe_calc(const int *eleIdx, const int *eleSpn, const int qpStart, const int qpEnd) {
  int info = CalculateMAll_fsz_real(eleIdx, eleSpn, qpStart, qpEnd);
  if (n_logged < 256) logged_info[n_logged++] = info;
  return info;
}
#define CalculateMAll_fsz_real probe_calc
#define MPI_Abort(comm, code) (aborted = 1, longjmp(abort_jump, 1), 0)
#include "initializer-fsz-real.inc"
#undef MPI_Abort
#undef CalculateMAll_fsz_real

static uint64_t bits(double x) { uint64_t u; memcpy(&u, &x, 8); return u; }
static double from_bits(uint64_t u) { double x; memcpy(&x, &u, 8); return x; }
static const double SENTINEL = 23.0;

static void setup_globals(int nsite, int ne, int nqp, int twosz) {
  NThread = omp_get_max_threads();
  initializeWorkSpaceAll();
  Nsite = nsite;
  Ne = ne;
  Nsize = 2 * Ne;
  Nsite2 = 2 * Nsite;
  NProj = 0;
  NQPFull = nqp;
  TwoSz = twosz;
  NGutzwillerIdx = NJastrowIdx = NDoublonHolon2siteIdx = NDoublonHolon4siteIdx = 0;
  LocSpn = calloc(Nsite, sizeof(int));
  LapackLWork = 2 * Nsize * Nsize + 16;
  SlaterElm_real = calloc((size_t)nqp * Nsite2 * Nsite2, sizeof(double));
  InvM_real = calloc((size_t)nqp * Nsize * Nsize + 1, sizeof(double));
  PfM_real = calloc(nqp, sizeof(double));
  for (int i = 0; i < nqp * Nsize * Nsize + 1; i++) InvM_real[i] = SENTINEL;
  for (int i = 0; i < nqp; i++) PfM_real[i] = SENTINEL;
}

static void print_matrix_state(const char *label, int nqp_local, int qp_offset) {
  for (int k = 0; k < nqp_local; k++) {
    printf("%s_pf qp=%d %016llx\n", label, qp_offset + k, (unsigned long long)bits(PfM_real[k]));
    printf("%s_inv qp=%d", label, qp_offset + k);
    for (int i = 0; i < Nsize * Nsize; i++)
      printf(" %016llx", (unsigned long long)bits(InvM_real[k * Nsize * Nsize + i]));
    printf("\n");
  }
}

/* Matrix cases: operands are the literal (idx, spins, complex Slater hex) records of the
 * Julia fixture; C's vmcmain.c:372 copies creal(SlaterElm) into SlaterElm_real before
 * CalculateMAll_fsz_real, so only real parts enter the native kernel. */
static int run_matrix(const char *path) {
  FILE *fp = fopen(path, "r");
  if (!fp) return 2;
  static char line[1 << 22];
  for (int c = 0; c < 16; c++) {
    int ns, ne, a, b;
    do {
      if (!fgets(line, sizeof line, fp)) return 3;
    } while (line[0] == '#');
    if (sscanf(line, "%d %d %d %d", &ns, &ne, &a, &b) != 4) return 4;
    int idx[16], spn[16];
    for (int pass = 0; pass < 2; pass++) {
      if (!fgets(line, sizeof line, fp)) return 5;
      char *p = line;
      for (int i = 0; i < 2 * ne; i++) (pass ? spn : idx)[i] = (int)strtol(p, &p, 10);
    }
    setup_globals(ns, ne, 4, -1);
    if (!fgets(line, sizeof line, fp)) return 6;
    char *p = line;
    const int plane = Nsite2 * Nsite2;
    for (int qp = 0; qp < 4; qp++)
      for (int i = 0; i < plane; i++) {
        uint64_t re = strtoull(p, &p, 16);
        (void)strtoull(p, &p, 16);
        SlaterElm_real[qp * plane + i] = from_bits(re);
      }
    for (int skip = 0; skip < 4; skip++)
      if (!fgets(line, sizeof line, fp)) return 7;
    n_logged = 0;
    /* Rust layout planes 0 and 3 stay untouched; C range [1,3) is local 0..2. */
    const int info = probe_calc(idx, spn, 1, 3);
    printf("matrix_case %d ns=%d ne=%d polarized=%d complex_input=%d info=%d\n", c, ns, ne, a, b,
           info);
    print_matrix_state("matrix", 2, 1);
  }
  fclose(fp);
  return 0;
}

static void print_ints(const char *label, const int *v, int n) {
  printf("%s=", label);
  for (int i = 0; i < n; i++) printf("%s%d", i ? " " : "", v[i]);
  printf("\n");
}

static void init_case(const char *kind) {
  const int nsite = 3, ne = 1;
  const int zero = !strcmp(kind, "zero");
  setup_globals(nsite, ne, 1, strcmp(kind, "magnetized") == 0 ? 2 : -1);
  const int n2 = Nsite2;
  for (int i = 0; i < n2 * n2; i++) SlaterElm_real[i] = zero ? 0.0 : NAN;
  if (!strcmp(kind, "retry")) {
    SlaterElm_real[0 * n2 + 4] = 1.25;
    SlaterElm_real[4 * n2 + 0] = -1.25;
    for (int i = 0; i < n2; i++) SlaterElm_real[i * n2 + i] = 0.0;
  }
  if (!strcmp(kind, "localspin") || !strcmp(kind, "magnetized")) {
    for (int i = 0; i < n2; i++) {
      SlaterElm_real[i * n2 + i] = 0.0;
      for (int j = i + 1; j < n2; j++) {
        double z = (double)(i + 2 * j + 1) / 7.0;
        SlaterElm_real[i * n2 + j] = z;
        SlaterElm_real[j * n2 + i] = -z;
      }
    }
    if (!strcmp(kind, "localspin")) LocSpn[0] = 1;
  }
  int eleIdx[2], eleSpn[2], eleCfg[6], eleNum[6], eleProjCnt[1] = {0};
  init_gen_rand(1);
  n_logged = 0;
  aborted = 0;
  if (setjmp(abort_jump) == 0) makeInitialSample_fsz_real(eleIdx, eleCfg, eleNum, eleProjCnt, eleSpn, 0, 1, MPI_COMM_SELF);
  printf("init_case %s attempts=%d aborted=%d\n", kind, n_logged, aborted);
  print_ints("init_infos", logged_info, n_logged);
  print_ints("eleIdx", eleIdx, 2);
  print_ints("eleCfg", eleCfg, 6);
  print_ints("eleNum", eleNum, 6);
  print_ints("eleProjCnt", eleProjCnt, 0);
  print_ints("eleSpn", eleSpn, 2);
  print_matrix_state("init", 1, 0);
  printf("rng624=");
  for (int i = 0; i < 624; i++) printf("%s%u", i ? " " : "", gen_rand32());
  printf("\n");
}

int main(int argc, char **argv) {
  if (argc < 2) return 2;
  MPI_Init(&argc, &argv);
  int rc = 0;
  if (!strcmp(argv[1], "matrix") && argc == 3) {
    rc = run_matrix(argv[2]);
  } else if (!strcmp(argv[1], "init")) {
    const char *kinds[] = {"retry", "failure", "zero", "localspin", "magnetized"};
    for (int k = 0; k < 5; k++) init_case(kinds[k]);
  } else {
    rc = 2;
  }
  MPI_Finalize();
  return rc;
}
