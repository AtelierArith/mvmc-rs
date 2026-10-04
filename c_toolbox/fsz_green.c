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
#ifdef FSZ_REAL
typedef double GreenScalar;
#define SlaterElm SlaterElm_real
#define InvM InvM_real
#define PfM PfM_real
#else
typedef double complex GreenScalar;
#endif
GreenScalar SlaterElm[128], InvM[32], PfM[2];

typedef int MPI_Comm;
#define MPI_COMM_SELF 0
#define MPI_DOUBLE_COMPLEX 0
#define MPI_DOUBLE 0
#define MPI_SUM 0
void MPI_Comm_size(MPI_Comm comm, int *size) { (void)comm; *size = 1; }
void MPI_Allreduce(const void *a, void *b, int n, int kind, int op, MPI_Comm comm) {
  (void)a; (void)b; (void)n; (void)kind; (void)op; (void)comm; assert(0);
}
#ifdef FSZ_REAL
void calculateNewPfMTwo_child_fsz_real(const int, const int, const int, const int,
                                 double *, const int *, const int *,
                                 const int, const int, const int, double *, double *);
#else
void calculateNewPfMTwo_child_fsz(const int, const int, const int, const int,
                                 double complex *, const int *, const int *,
                                 const int, const int, const int,
                                 double complex *, double complex *);
#endif
#define inline static inline
#ifdef FSZ_REAL
#include "fsz_green_real_upstream.inc"
#define GreenFunc1_fsz GreenFunc1_fsz_real
#define GreenFunc1_fsz2 GreenFunc1_fsz2_real
#define GreenFunc2_fsz GreenFunc2_fsz_real
#define GreenFunc2_fsz2 GreenFunc2_fsz2_real
#else
#include "fsz_green_upstream.inc"
#endif
#undef inline

#if defined(FSZ_HAMILTONIAN) || defined(FSZ_MEASUREMENTS)
static int work_int[32], work_int_used;
static GreenScalar work_green[10];
void RequestWorkSpaceThreadInt(int n) { assert(n <= 32); work_int_used = 0; }
int *GetWorkSpaceThreadInt(int n) {
  int *p = work_int + work_int_used; work_int_used += n; assert(work_int_used <= 32); return p;
}
void ReleaseWorkSpaceThreadInt(void) { work_int_used = 0; }
void StartTimer(int i) { (void)i; }
void StopTimer(int i) { (void)i; }
#ifdef FSZ_REAL
void RequestWorkSpaceThreadDouble(int n) { assert(n <= 10); }
double *GetWorkSpaceThreadDouble(int n) { assert(n <= 10); return work_green; }
void ReleaseWorkSpaceThreadDouble(void) {}
#else
void RequestWorkSpaceThreadComplex(int n) { assert(n <= 10); }
double complex *GetWorkSpaceThreadComplex(int n) { assert(n <= 10); return work_green; }
void ReleaseWorkSpaceThreadComplex(void) {}
#endif
#endif

#ifdef FSZ_HAMILTONIAN
int NCoulombIntra, NCoulombInter, NHundCoupling, NTransfer;
int NPairHopping, NExchangeCoupling, NInterAll;
int CoulombIntra[4], CoulombInter[16][2], HundCoupling[16][2];
int Transfer[65][4], PairHopping[18][2], ExchangeCoupling[18][2], InterAll[4098][8];
double ParaCoulombIntra[4], ParaCoulombInter[16], ParaHundCoupling[16];
double ParaPairHopping[18], ParaExchangeCoupling[18];
double complex ParaTransfer[65], ParaInterAll[4098];
#ifdef FSZ_REAL
#include "fsz_hamiltonian_real_upstream.inc"
#define NativeHamiltonian CalculateHamiltonian_fsz_real
#else
#include "fsz_hamiltonian_upstream.inc"
#define NativeHamiltonian CalculateHamiltonian_fsz
#endif
#endif

#ifdef FSZ_MEASUREMENTS
#ifdef FSZ_REAL
#error CalculateGreenFunc_fsz always uses the complex shadow arrays
#endif
int NCisAjs = 64, NCisAjsCktAltDC = 4098, NCisAjsCktAlt = 8;
int CisAjsIdx[64][4], CisAjsCktAltDCIdx[4098][8];
int CisAjsCktAltIdx[8][2] = {{0,0}, {19,35}, {35,19}, {18,63},
                            {63,0}, {19,19}, {27,35}, {0,0}};
double complex LocalCisAjs[64], PhysCisAjs[64];
double complex PhysCisAjsCktAltDC[4098], PhysCisAjsCktAlt[8];
/* Match upstream global.h; accumulate the real sample weights separately. */
double complex Wc;
void MPI_Comm_rank(MPI_Comm comm, int *rank) { (void)comm; *rank = 0; }
/* The extracted normalizer's multi-rank branch must remain unused. */
void RequestWorkSpaceComplex(int n) { (void)n; assert(0); }
double complex *GetWorkSpaceComplex(int n) { (void)n; assert(0); return NULL; }
void ReleaseWorkSpaceComplex(void) { assert(0); }
void SafeMpiReduce_fcmp(double complex *a, double complex *b, int n, MPI_Comm comm) {
  (void)a; (void)b; (void)n; (void)comm; assert(0);
}
#include "fsz_measurements_upstream.inc"
#include "fsz_measurements_average_upstream.inc"
#endif

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
static GreenScalar read_green(void) {
  double complex value = read_complex();
#ifdef FSZ_REAL
  assert(cimag(value) == 0.0);
  return creal(value);
#else
  return value;
#endif
}

int main(void) {
  int inputIdx[4], cfg[8], num[8], cnt[2], spins[4], new_cnt[2];
  for (int i = 0; i < 4; i++) assert(scanf("%d", &inputIdx[i]) == 1);
  for (int i = 0; i < 4; i++) assert(scanf("%d", &spins[i]) == 1);
  for (int i = 0; i < 8; i++) assert(scanf("%d", &cfg[i]) == 1);
  for (int i = 0; i < 8; i++) assert(scanf("%d", &num[i]) == 1);
  for (int i = 0; i < 2; i++) assert(scanf("%d", &cnt[i]) == 1);
  for (int i = 0; i < 2; i++) Proj[i] = read_complex();
  for (int i = 0; i < 128; i++) SlaterElm[i] = read_green();
  for (int i = 0; i < 2; i++) PfM[i] = read_green();
  for (int i = 0; i < 32; i++) InvM[i] = read_green();
  GreenScalar ip = read_green(), buffer[10];
  for (int i = 0; i < 4; i++) {
    JastrowIdx[i] = jast[i];
    for (int j = 0; j < 4; j++) jast[i][j] = i == j ? -1 : 0;
  }
  int before_idx[4], before_spins[4], before_num[8];
  int before_cfg[8], before_cnt[2];
  GreenScalar before_slater[128], before_pf[2], before_inv[32];
  memcpy(before_idx, inputIdx, sizeof(inputIdx));
  memcpy(before_spins, spins, sizeof(spins));
  memcpy(before_num, num, sizeof(num));
  memcpy(before_cfg, cfg, sizeof(cfg));
  memcpy(before_cnt, cnt, sizeof(cnt));
  memcpy(before_slater, SlaterElm, sizeof(SlaterElm));
  memcpy(before_pf, PfM, sizeof(PfM));
  memcpy(before_inv, InvM, sizeof(InvM));
  for (int s = 0; s < 2; s++) for (int t = 0; t < 2; t++)
    for (int ri = 0; ri < 4; ri++) for (int rj = 0; rj < 4; rj++) {
      print_complex(s == t
        ? GreenFunc1_fsz(ri, rj, s, ip, inputIdx, cfg, num, cnt, spins, new_cnt, buffer)
        : GreenFunc1_fsz2(ri, rj, s, t, ip, inputIdx, cfg, num, cnt, spins, new_cnt, buffer));
      assert(memcmp(before_idx, inputIdx, sizeof(inputIdx)) == 0);
      assert(memcmp(before_spins, spins, sizeof(spins)) == 0);
      assert(memcmp(before_num, num, sizeof(num)) == 0);
    }
  int operator_index = 0;
#ifdef FSZ_HAMILTONIAN
  NInterAll = 4098;
#else
  int NInterAll = 4098, InterAll[4098][8];
  double complex ParaInterAll[4098];
#endif
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
#ifdef FSZ_MEASUREMENTS
    int *r = InterAll[original];
    print_complex(r[1] == r[3] && r[5] == r[7]
      ? GreenFunc2_fsz(r[0], r[2], r[4], r[6], r[1], r[5], ip,
                      inputIdx, cfg, num, cnt, spins, new_cnt, buffer)
      : GreenFunc2_fsz2(r[0], r[2], r[4], r[6], r[1], r[3], r[5], r[7], ip,
                       inputIdx, cfg, num, cnt, spins, new_cnt, buffer));
#endif
  }
  GreenScalar myEnergy = 0, *myBuffer = buffer;
  int *myEleIdx = inputIdx, *myEleSpn = spins, *myEleNum = num;
  int *eleCfg = cfg, *eleProjCnt = cnt, *myProjCntNew = new_cnt;
  int idx, ri, rj, rk, rl, s, t, u, v;
#ifdef FSZ_REAL
#include "fsz_interall_real_accumulator_upstream.inc"
#else
#include "fsz_interall_accumulator_upstream.inc"
#endif
  print_complex(myEnergy);
  assert(memcmp(before_idx, myEleIdx, sizeof(before_idx)) == 0);
  assert(memcmp(before_spins, myEleSpn, sizeof(before_spins)) == 0);
  assert(memcmp(before_num, myEleNum, sizeof(before_num)) == 0);
#ifdef FSZ_HAMILTONIAN
  for (int i = 0; i < 4; i++) {
    CoulombIntra[i] = i; ParaCoulombIntra[i] = (i - 1) / 8.0;
    for (int j = 0; j < 4; j++) {
      int k = i * 4 + j;
      CoulombInter[k][0] = HundCoupling[k][0] = PairHopping[k][0] = ExchangeCoupling[k][0] = i;
      CoulombInter[k][1] = HundCoupling[k][1] = PairHopping[k][1] = ExchangeCoupling[k][1] = j;
      ParaCoulombInter[k] = (k % 5 - 2) / 16.0;
      ParaHundCoupling[k] = (k % 7 - 3) / 32.0;
      ParaPairHopping[k] = (k % 11 - 5) / 16.0;
      ParaExchangeCoupling[k] = (k % 13 - 6) / 32.0;
    }
  }
  int hop = 0;
  for (int s = 0; s < 2; s++) for (int t = 0; t < 2; t++)
    for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) {
      Transfer[hop][0] = i; Transfer[hop][1] = s;
      Transfer[hop][2] = j; Transfer[hop][3] = t;
      ParaTransfer[hop] = (hop % 7 - 3) / 16.0 + I * ((hop % 11 - 5) / 32.0);
      hop++;
    }
  memcpy(Transfer[64], Transfer[19], sizeof(Transfer[0])); ParaTransfer[64] = ParaTransfer[19];
  for (int d = 0; d < 2; d++) {
    int p = d == 0 ? 0 : 7, x = d == 0 ? 0 : 11;
    memcpy(PairHopping[16+d], PairHopping[p], sizeof(PairHopping[0]));
    ParaPairHopping[16+d] = ParaPairHopping[p];
    memcpy(ExchangeCoupling[16+d], ExchangeCoupling[x], sizeof(ExchangeCoupling[0]));
    ParaExchangeCoupling[16+d] = ParaExchangeCoupling[x];
  }
  for (int group = 0; group < 6; group++) {
    NCoulombIntra = group == 0 || group == 5 ? 4 : 0;
    NCoulombInter = NHundCoupling = group == 0 || group == 5 ? 16 : 0;
    NTransfer = group == 1 || group == 5 ? 65 : 0;
    NPairHopping = group == 2 || group == 5 ? 18 : 0;
    NExchangeCoupling = group == 3 || group == 5 ? 18 : 0;
    NInterAll = group == 4 || group == 5 ? 4098 : 0;
    print_complex(NativeHamiltonian(ip, inputIdx, cfg, num, cnt, spins));
    assert(memcmp(before_idx, inputIdx, sizeof(inputIdx)) == 0);
    assert(memcmp(before_spins, spins, sizeof(spins)) == 0);
    assert(memcmp(before_num, num, sizeof(num)) == 0);
  }
#endif
#ifdef FSZ_MEASUREMENTS
  int one = 0;
  for (int a = 0; a < 2; a++) for (int b = 0; b < 2; b++)
    for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) {
      CisAjsIdx[one][0] = i; CisAjsIdx[one][1] = a;
      CisAjsIdx[one][2] = j; CisAjsIdx[one][3] = b;
      one++;
    }
  memcpy(CisAjsCktAltDCIdx, InterAll, sizeof(CisAjsCktAltDCIdx));
  const double weights[3] = {0.375, 1.25, 0.125};
  double total_weight = 0.0;
  for (int call = 0; call < 3; call++) {
    total_weight += weights[call];
    CalculateGreenFunc_fsz(weights[call], ip, inputIdx, cfg, num, spins, cnt);
    for (int i = 0; i < 64; i++) print_complex(PhysCisAjs[i]);
    for (int i = 0; i < 4098; i++) print_complex(PhysCisAjsCktAltDC[i]);
    for (int i = 0; i < 8; i++) print_complex(PhysCisAjsCktAlt[i]);
    /* Validate the native per-thread copies as well as the caller's arrays. */
    assert(memcmp(work_int, before_idx, sizeof(before_idx)) == 0);
    assert(memcmp(work_int + 4, before_spins, sizeof(before_spins)) == 0);
    assert(memcmp(work_int + 8, before_num, sizeof(before_num)) == 0);
    assert(memcmp(before_idx, inputIdx, sizeof(inputIdx)) == 0);
    assert(memcmp(before_spins, spins, sizeof(spins)) == 0);
    assert(memcmp(before_num, num, sizeof(num)) == 0);
    assert(memcmp(before_cfg, cfg, sizeof(cfg)) == 0);
    assert(memcmp(before_cnt, cnt, sizeof(cnt)) == 0);
    assert(memcmp(before_slater, SlaterElm, sizeof(SlaterElm)) == 0);
    assert(memcmp(before_pf, PfM, sizeof(PfM)) == 0);
    assert(memcmp(before_inv, InvM, sizeof(InvM)) == 0);
  }
  assert(total_weight == 1.75);
  Wc = total_weight;
  weightAverageReduce_fcmp(64, PhysCisAjs, MPI_COMM_SELF);
  weightAverageReduce_fcmp(4098, PhysCisAjsCktAltDC, MPI_COMM_SELF);
  weightAverageReduce_fcmp(8, PhysCisAjsCktAlt, MPI_COMM_SELF);
  for (int i = 0; i < 64; i++) print_complex(PhysCisAjs[i]);
  for (int i = 0; i < 4098; i++) print_complex(PhysCisAjsCktAltDC[i]);
  for (int i = 0; i < 8; i++) print_complex(PhysCisAjsCktAlt[i]);
#endif
  return 0;
}
