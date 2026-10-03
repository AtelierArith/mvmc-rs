/* Optional native FSZ Green oracle; Cargo never compiles or invokes this file. */
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
double complex SlaterElm[128], InvM[32], PfM[2];

typedef int MPI_Comm;
#define MPI_COMM_SELF 0
#define MPI_DOUBLE_COMPLEX 0
#define MPI_SUM 0
void MPI_Comm_size(MPI_Comm comm, int *size) { (void)comm; *size = 1; }
void MPI_Allreduce(const void *a, void *b, int n, int kind, int op, MPI_Comm comm) {
  (void)a; (void)b; (void)n; (void)kind; (void)op; (void)comm; assert(0);
}
void calculateNewPfMTwo_child_fsz(const int, const int, const int, const int,
                                 double complex *, const int *, const int *,
                                 const int, const int, const int,
                                 double complex *, double complex *);
#define inline static inline
#include "fsz_green_upstream.inc"
#undef inline

static double read_double(void) {
  union { uint64_t bits; double value; } v;
  assert(scanf("%" SCNx64, &v.bits) == 1);
  return v.value;
}
static double complex read_complex(void) {
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
  int inputIdx[4], cfg[8], num[8], cnt[2], spins[4], new_cnt[2];
  for (int i = 0; i < 4; i++) assert(scanf("%d", &inputIdx[i]) == 1);
  for (int i = 0; i < 4; i++) assert(scanf("%d", &spins[i]) == 1);
  for (int i = 0; i < 8; i++) assert(scanf("%d", &cfg[i]) == 1);
  for (int i = 0; i < 8; i++) assert(scanf("%d", &num[i]) == 1);
  for (int i = 0; i < 2; i++) assert(scanf("%d", &cnt[i]) == 1);
  for (int i = 0; i < 2; i++) Proj[i] = read_complex();
  for (int i = 0; i < 128; i++) SlaterElm[i] = read_complex();
  for (int i = 0; i < 2; i++) PfM[i] = read_complex();
  for (int i = 0; i < 32; i++) InvM[i] = read_complex();
  double complex ip = read_complex(), buffer[10];
  for (int i = 0; i < 4; i++) {
    JastrowIdx[i] = jast[i];
    for (int j = 0; j < 4; j++) jast[i][j] = i == j ? -1 : 0;
  }
  int before_idx[4], before_spins[4], before_num[8];
  memcpy(before_idx, inputIdx, sizeof(inputIdx));
  memcpy(before_spins, spins, sizeof(spins));
  memcpy(before_num, num, sizeof(num));
  for (int s = 0; s < 2; s++) for (int t = 0; t < 2; t++)
    for (int ri = 0; ri < 4; ri++) for (int rj = 0; rj < 4; rj++) {
      print_complex(s == t
        ? GreenFunc1_fsz(ri, rj, s, ip, inputIdx, cfg, num, cnt, spins, new_cnt, buffer)
        : GreenFunc1_fsz2(ri, rj, s, t, ip, inputIdx, cfg, num, cnt, spins, new_cnt, buffer));
      assert(memcmp(before_idx, inputIdx, sizeof(inputIdx)) == 0);
      assert(memcmp(before_spins, spins, sizeof(spins)) == 0);
      assert(memcmp(before_num, num, sizeof(num)) == 0);
    }
  int NInterAll = 4098, InterAll[4098][8], operator_index = 0;
  double complex ParaInterAll[4098];
  for (int s = 0; s < 2; s++) for (int t = 0; t < 2; t++)
    for (int u = 0; u < 2; u++) for (int v = 0; v < 2; v++)
      for (int ri = 0; ri < 4; ri++) for (int rj = 0; rj < 4; rj++)
        for (int rk = 0; rk < 4; rk++) for (int rl = 0; rl < 4; rl++) {
          int *row = InterAll[operator_index];
          row[0] = ri; row[1] = s; row[2] = rj; row[3] = t;
          row[4] = rk; row[5] = u; row[6] = rl; row[7] = v;
          ParaInterAll[operator_index] = (operator_index % 7 - 3) / 16.0
            + I * ((operator_index % 11 - 5) / 32.0);
          operator_index++;
          print_complex(s == t && u == v
            ? GreenFunc2_fsz(ri, rj, rk, rl, s, u, ip, inputIdx, cfg, num, cnt, spins, new_cnt, buffer)
            : GreenFunc2_fsz2(ri, rj, rk, rl, s, t, u, v, ip, inputIdx, cfg, num, cnt, spins, new_cnt, buffer));
          assert(memcmp(before_idx, inputIdx, sizeof(inputIdx)) == 0);
          assert(memcmp(before_spins, spins, sizeof(spins)) == 0);
          assert(memcmp(before_num, num, sizeof(num)) == 0);
        }
  for (int duplicate = 0; duplicate < 2; duplicate++) {
    int original = duplicate == 0 ? 73 : 3072;
    memcpy(InterAll[4096 + duplicate], InterAll[original], sizeof(InterAll[0]));
    ParaInterAll[4096 + duplicate] = ParaInterAll[original];
  }
  double complex myEnergy = 0, *myBuffer = buffer;
  int *myEleIdx = inputIdx, *myEleSpn = spins, *myEleNum = num;
  int *eleCfg = cfg, *eleProjCnt = cnt, *myProjCntNew = new_cnt;
  int idx, ri, rj, rk, rl, s, t, u, v;
#include "fsz_interall_accumulator_upstream.inc"
  print_complex(myEnergy);
  assert(memcmp(before_idx, myEleIdx, sizeof(before_idx)) == 0);
  assert(memcmp(before_spins, myEleSpn, sizeof(before_spins)) == 0);
  assert(memcmp(before_num, myEleNum, sizeof(before_num)) == 0);
  return 0;
}
