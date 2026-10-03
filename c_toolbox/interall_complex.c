/* Optional native complex Green/InterAll oracle; no Cargo dependency. */
#include <assert.h>
#include <complex.h>
#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

int Nsite = 4, Nsite2 = 8, Ne = 2, Nsize = 4, NQPFull = 2;
int NProj = 2, NGutzwillerIdx = 1, NJastrowIdx = 1, FlagRBM = 0;
int NDoublonHolon2siteIdx = 0, NDoublonHolon4siteIdx = 0;
int gutz[4], jast[4][4], *GutzwillerIdx = gutz;
int *JastrowIdx[4], **DoublonHolon2siteIdx, **DoublonHolon4siteIdx;
double complex Proj[2], QPFullWeight[2] = {1, -0.375};
double complex SlaterElm[128], InvM[32], PfM[2];

/* One-process plumbing. Disabled RBM calls must never execute. */
typedef int MPI_Comm;
#define MPI_COMM_SELF 0
#define MPI_DOUBLE_COMPLEX 0
#define MPI_SUM 0
void MPI_Comm_size(MPI_Comm comm, int *size) { (void)comm; *size = 1; }
void MPI_Allreduce(const void *a, void *b, int n, int kind, int op, MPI_Comm comm) {
  (void)a; (void)b; (void)n; (void)kind; (void)op; (void)comm; assert(0);
}
void UpdateRBMCnt(int a, int b, int s, double complex *out,
                  const double complex *old, const int *num) {
  (void)a; (void)b; (void)s; (void)out; (void)old; (void)num; assert(0);
}
double complex RBMRatio(const double complex *a, const double complex *b) {
  (void)a; (void)b; assert(0); return 0;
}
void calculateNewPfMTwo_child_fcmp(const int, const int, const int, const int,
                                 double complex *, const int *, const int,
                                 const int, const int, double complex *, double complex *);
#define inline static inline
#include "interall_complex_upstream.inc"
#undef inline

static double read_double(void) {
  union { uint64_t bits; double value; } v;
  assert(scanf("%" SCNx64, &v.bits) == 1);
  return v.value;
}
static double complex read_complex(void) {
  /* C specifies the same representation as a two-element real array. */
  union { double components[2]; double complex value; } v;
  v.components[0] = read_double(); v.components[1] = read_double();
  return v.value;
}
static void print_complex(double complex value) {
  union { uint64_t bits; double value; } re = {.value = creal(value)};
  union { uint64_t bits; double value; } im = {.value = cimag(value)};
  printf("%016" PRIx64 " %016" PRIx64 "\n", re.bits, im.bits);
}

int main(void) {
  int myEleIdx[4], eleCfg[8], myEleNum[8], eleProjCnt[2], myProjCntNew[2];
  for (int i = 0; i < 4; i++) assert(scanf("%d", &myEleIdx[i]) == 1);
  for (int i = 0; i < 8; i++) assert(scanf("%d", &eleCfg[i]) == 1);
  for (int i = 0; i < 8; i++) assert(scanf("%d", &myEleNum[i]) == 1);
  for (int i = 0; i < 2; i++) assert(scanf("%d", &eleProjCnt[i]) == 1);
  for (int i = 0; i < 2; i++) Proj[i] = read_complex();
  for (int i = 0; i < 128; i++) SlaterElm[i] = read_complex();
  for (int i = 0; i < 2; i++) PfM[i] = read_complex();
  for (int i = 0; i < 32; i++) InvM[i] = read_complex();
  double complex ip = read_complex();
  for (int i = 0; i < 4; i++) {
    JastrowIdx[i] = jast[i];
    for (int j = 0; j < 4; j++) jast[i][j] = i == j ? -1 : 0;
  }
  int NInterAll = 1024, InterAll[1024][8];
  double complex ParaInterAll[1024];
  double complex myBuffer[10], *rbmCnt = NULL, *myRBMCntNew = NULL;
  int originalIdx[4], originalNum[8];
  memcpy(originalIdx, myEleIdx, sizeof(originalIdx));
  memcpy(originalNum, myEleNum, sizeof(originalNum));
  for (int idx = 0; idx < NInterAll; idx++) {
    int *r = InterAll[idx];
    assert(scanf("%d%d%d%d%d%d", r, r+2, r+4, r+6, r+3, r+7) == 6);
    r[1] = r[3]; r[5] = r[7];
    ParaInterAll[idx] = read_complex();
    print_complex(GreenFunc2(r[0], r[2], r[4], r[6], r[3], r[7], ip,
                             myEleIdx, eleCfg, myEleNum, eleProjCnt,
                             myProjCntNew, rbmCnt, myRBMCntNew, myBuffer));
    assert(memcmp(originalIdx, myEleIdx, sizeof(originalIdx)) == 0);
    assert(memcmp(originalNum, myEleNum, sizeof(originalNum)) == 0);
  }
  double complex myEnergy = 0;
  int idx, ri, rj, s, rk, rl, t;
#include "interall_complex_accumulator_upstream.inc"
  print_complex(myEnergy);
  /* Six bounded historical PairHop operators, including duplicates/density.
   * The historical seventh row has site 4 on this four-site lattice; it is
   * excluded from native execution, not claimed as valid C input. */
  int NPairHopping = 6;
  int PairHopping[6][2] = {{0,3}, {3,0}, {0,0}, {0,1}, {1,2}, {0,3}};
  double ParaPairHopping[6] = {0.7, -0.2, -0.125, 0.375, -0.35, 0.0625};
  myEnergy = 0;
#include "pairhop_complex_accumulator_upstream.inc"
  print_complex(myEnergy);
  assert(memcmp(originalIdx, myEleIdx, sizeof(originalIdx)) == 0);
  assert(memcmp(originalNum, myEleNum, sizeof(originalNum)) == 0);
  return 0;
}
