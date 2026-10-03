/* Optional fixed-Sz real Green/InterAll oracle; no Cargo dependency. */
#include <assert.h>
#include <complex.h>
#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

int Nsite = 4, Nsite2 = 8, Ne = 2, Nsize = 4, NQPFull = 2;
int NProj = 2, NGutzwillerIdx = 1, NJastrowIdx = 1;
int NDoublonHolon2siteIdx = 0, NDoublonHolon4siteIdx = 0;
int gutz[4], jast[4][4], *GutzwillerIdx = gutz;
int *JastrowIdx[4], **DoublonHolon2siteIdx, **DoublonHolon4siteIdx;
double complex Proj[2], QPFullWeight[2] = {1, -0.375};
double SlaterElm_real[128], InvM_real[32], PfM_real[2];

/* MPI_COMM_SELF plumbing only: the extracted overlap sees one process. */
typedef int MPI_Comm;
#define MPI_COMM_SELF 0
#define MPI_DOUBLE 0
#define MPI_SUM 0
void MPI_Comm_size(MPI_Comm comm, int *size) { (void)comm; *size = 1; }
void MPI_Allreduce(const void *a, void *b, int n, int kind, int op, MPI_Comm comm) {
  (void)a; (void)b; (void)n; (void)kind; (void)op; (void)comm; assert(0);
}
void calculateNewPfMTwo_child_real(const int, const int, const int, const int,
                                 double *, const int *, const int, const int,
                                 const int, double *, double *);
/* Preserve inline body verbatim and give the standalone driver local linkage. */
#define inline static inline
#include "interall_real_upstream.inc"
#undef inline

static double read_double(void) {
  union { uint64_t bits; double value; } v;
  assert(scanf("%" SCNx64, &v.bits) == 1);
  return v.value;
}
static void print_double(double value) {
  union { uint64_t bits; double value; } v = {.value = value};
  printf("%016" PRIx64 "\n", v.bits);
}

int main(void) {
  int myEleIdx[4], eleCfg[8], myEleNum[8], eleProjCnt[2], myProjCntNew[2];
  for (int i = 0; i < 4; i++) assert(scanf("%d", &myEleIdx[i]) == 1);
  for (int i = 0; i < 8; i++) assert(scanf("%d", &eleCfg[i]) == 1);
  for (int i = 0; i < 8; i++) assert(scanf("%d", &myEleNum[i]) == 1);
  for (int i = 0; i < 2; i++) assert(scanf("%d", &eleProjCnt[i]) == 1);
  for (int i = 0; i < 2; i++) Proj[i] = read_double();
  for (int i = 0; i < 128; i++) SlaterElm_real[i] = read_double();
  for (int i = 0; i < 2; i++) PfM_real[i] = read_double();
  for (int i = 0; i < 32; i++) InvM_real[i] = read_double();
  double ip = read_double();
  for (int i = 0; i < 4; i++) {
    JastrowIdx[i] = jast[i];
    for (int j = 0; j < 4; j++) jast[i][j] = i == j ? -1 : 0;
  }
  int NInterAll = 1024, InterAll[1024][8];
  double complex ParaInterAll[1024];
  double myBuffer[10];
  int originalIdx[4], originalNum[8];
  memcpy(originalIdx, myEleIdx, sizeof(originalIdx));
  memcpy(originalNum, myEleNum, sizeof(originalNum));
  for (int idx = 0; idx < NInterAll; idx++) {
    int *r = InterAll[idx];
    assert(scanf("%d%d%d%d%d%d", r, r+2, r+4, r+6, r+3, r+7) == 6);
    r[1] = r[3]; r[5] = r[7];
    double re = read_double(), im = read_double();
    ParaInterAll[idx] = re + I * im;
    print_double(GreenFunc2_real(r[0], r[2], r[4], r[6], r[3], r[7], ip,
                                myEleIdx, eleCfg, myEleNum, eleProjCnt,
                                myProjCntNew, myBuffer));
    assert(memcmp(originalIdx, myEleIdx, sizeof(originalIdx)) == 0);
    assert(memcmp(originalNum, myEleNum, sizeof(originalNum)) == 0);
  }
  double myEnergy = 0;
  int idx, ri, rj, s, rk, rl, t;
  /* Verbatim serial InterAll loop from calham_real.c. */
#include "interall_real_accumulator_upstream.inc"
  print_double(myEnergy);
  assert(memcmp(originalIdx, myEleIdx, sizeof(originalIdx)) == 0);
  assert(memcmp(originalNum, myEleNum, sizeof(originalNum)) == 0);
  return 0;
}
