/* Optional independent runner oracle bridge. Never built or called by Cargo.
 * The included upstream functions retain their notices and SHA provenance.
 * Serial, no RBM, no full-executable/MPI/sampling/SR parity claim. */
#include <assert.h>
#include <complex.h>
#include <math.h>
#include <stdlib.h>

int Nsite, Nsite2, Ne, Nsize, NQPFull, NProj, NGutzwillerIdx, NJastrowIdx;
int NDoublonHolon2siteIdx = 0, NDoublonHolon4siteIdx = 0;
int *GutzwillerIdx, **JastrowIdx, **DoublonHolon2siteIdx, **DoublonHolon4siteIdx;
double complex *Proj, *QPFullWeight;
#ifdef FSZ_REAL
typedef double GreenScalar;
#define SlaterElm SlaterElm_real
#define InvM InvM_real
#define PfM PfM_real
#else
typedef double complex GreenScalar;
#endif
GreenScalar *SlaterElm, *InvM, *PfM;
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
  double *, const int *, const int *, const int, const int, const int, double *, double *);
#else
void calculateNewPfMTwo_child_fsz(const int, const int, const int, const int,
  double complex *, const int *, const int *, const int, const int, const int,
  double complex *, double complex *);
#endif
#define inline static inline
#ifdef FSZ_REAL
#include "fsz_green_real_upstream.inc"
#else
#include "fsz_green_upstream.inc"
#endif
#undef inline
#include "fsz_runner_projection_upstream.inc"

int NCoulombIntra, NCoulombInter, NHundCoupling, NTransfer;
int NPairHopping, NExchangeCoupling, NInterAll;
int *CoulombIntra, **CoulombInter, **HundCoupling, **Transfer;
int **PairHopping, **ExchangeCoupling, **InterAll;
double *ParaCoulombIntra, *ParaCoulombInter, *ParaHundCoupling;
double *ParaPairHopping, *ParaExchangeCoupling;
double complex *ParaTransfer, *ParaInterAll;
static int *work_int, work_int_capacity, work_int_used;
static GreenScalar *work_green;
static int work_green_capacity;
void RequestWorkSpaceThreadInt(int n) {
  if(n > work_int_capacity) {
    work_int = realloc(work_int, sizeof(int) * n); assert(work_int);
    work_int_capacity = n;
  }
  work_int_used = 0;
}
int *GetWorkSpaceThreadInt(int n) {
  int *p = work_int + work_int_used; work_int_used += n;
  assert(work_int_used <= work_int_capacity); return p;
}
void ReleaseWorkSpaceThreadInt(void) { work_int_used = 0; }
void StartTimer(int n) { (void)n; }
void StopTimer(int n) { (void)n; }
static void request_green(int n) {
  if(n > work_green_capacity) {
    work_green = realloc(work_green, sizeof(GreenScalar) * n); assert(work_green);
    work_green_capacity = n;
  }
}
#ifdef FSZ_REAL
void RequestWorkSpaceThreadDouble(int n) { request_green(n); }
double *GetWorkSpaceThreadDouble(int n) { assert(n <= work_green_capacity); return work_green; }
void ReleaseWorkSpaceThreadDouble(void) {}
#include "fsz_hamiltonian_real_upstream.inc"
#define NativeHamiltonian CalculateHamiltonian_fsz_real
#else
void RequestWorkSpaceThreadComplex(int n) { request_green(n); }
double complex *GetWorkSpaceThreadComplex(int n) { assert(n <= work_green_capacity); return work_green; }
void ReleaseWorkSpaceThreadComplex(void) {}
#include "fsz_hamiltonian_upstream.inc"
#define NativeHamiltonian CalculateHamiltonian_fsz
#endif

static int **rows(int **cursor, int n, int width) {
  int **result = n ? malloc(sizeof(int *) * n) : NULL;
  assert(n == 0 || result);
  for(int i = 0; i < n; i++) result[i] = *cursor + width * i;
  *cursor += n * width; return result;
}
static double *real_values(double complex **cursor, int n) {
  double *result = n ? malloc(sizeof(double) * n) : NULL;
  assert(n == 0 || result);
  for(int i = 0; i < n; i++) result[i] = creal((*cursor)[i]);
  *cursor += n; return result;
}
#ifdef FSZ_REAL
static double *real_array(const double complex *source, int n) {
  double *result = malloc(sizeof(double) * n); assert(result);
  for(int i = 0; i < n; i++) {
    assert(cimag(source[i]) == 0.0); result[i] = creal(source[i]);
  }
  return result;
}
#endif

/* dims: site,size,qp,gutz,jast,DH2,DH4,Intra,Inter,Hund,Transfer,Pair,Exchange,InterAll.
 * tables: Gutz[site], Jast[site][site], DH2[type][2*site], DH4[type][4*site],
 *         then the seven interaction index tables.
 * values: projection params, QP weights, seven interaction coefficients.
 * config: idx[size], spin[size], cfg[2*site], num[2*site], counts[NProj].
 * Complex arguments use ordinary interleaved doubles; out avoids C return ABI.
 * The caller keeps all borrowed arrays alive. Native workspace copies isolate
 * proposed moves; the bridge never draws RNG or updates Julia state. */
__attribute__((visibility("default")))
void mvmc_reference_fsz_energy(const int *dims, int *tables, double complex *values,
  double complex *slater, double complex *pf, double complex *inv, int *config,
  const double complex *ip, double *out) {
  Nsite = dims[0]; Nsite2 = 2*Nsite; Nsize = dims[1]; Ne = Nsize/2; NQPFull = dims[2];
  NGutzwillerIdx = dims[3]; NJastrowIdx = dims[4];
  NDoublonHolon2siteIdx = dims[5]; NDoublonHolon4siteIdx = dims[6];
  NProj = NGutzwillerIdx+NJastrowIdx+6*NDoublonHolon2siteIdx+10*NDoublonHolon4siteIdx;
  NCoulombIntra = dims[7]; NCoulombInter = dims[8]; NHundCoupling = dims[9];
  NTransfer = dims[10]; NPairHopping = dims[11]; NExchangeCoupling = dims[12]; NInterAll = dims[13];
  assert(Nsite > 0 && Nsize > 0 && Nsize%2 == 0 && NQPFull > 0);
  int *p = tables;
  GutzwillerIdx = p; p += Nsite; JastrowIdx = rows(&p, Nsite, Nsite);
  DoublonHolon2siteIdx = rows(&p, NDoublonHolon2siteIdx, 2*Nsite);
  DoublonHolon4siteIdx = rows(&p, NDoublonHolon4siteIdx, 4*Nsite);
  CoulombIntra = p; p += NCoulombIntra;
  CoulombInter = rows(&p, NCoulombInter, 2); HundCoupling = rows(&p, NHundCoupling, 2);
  Transfer = rows(&p, NTransfer, 4); PairHopping = rows(&p, NPairHopping, 2);
  ExchangeCoupling = rows(&p, NExchangeCoupling, 2); InterAll = rows(&p, NInterAll, 8);
  double complex *v = values;
  Proj = v; v += NProj; QPFullWeight = v; v += NQPFull;
  ParaCoulombIntra = real_values(&v, NCoulombIntra);
  ParaCoulombInter = real_values(&v, NCoulombInter); ParaHundCoupling = real_values(&v, NHundCoupling);
  ParaTransfer = v; v += NTransfer; ParaPairHopping = real_values(&v, NPairHopping);
  ParaExchangeCoupling = real_values(&v, NExchangeCoupling); ParaInterAll = v;
  /* Independently validate Julia's saved integer projection counters using
   * the actual C MakeProjCnt, including DH type-major parameter offsets. */
  int *expected_counts = NProj ? malloc(sizeof(int)*NProj) : NULL;
  assert(NProj == 0 || expected_counts);
  MakeProjCnt(expected_counts, config+2*Nsize+Nsite2);
  for(int i = 0; i < NProj; i++) assert(expected_counts[i] == config[2*Nsize+2*Nsite2+i]);
  free(expected_counts);
#ifdef FSZ_REAL
  SlaterElm = real_array(slater, NQPFull*Nsite2*Nsite2);
  PfM = real_array(pf, NQPFull); InvM = real_array(inv, NQPFull*Nsize*Nsize);
  assert(cimag(*ip) == 0.0);
  out[0] = NativeHamiltonian(creal(*ip), config, config+2*Nsize,
    config+2*Nsize+Nsite2, config+2*Nsize+2*Nsite2, config+Nsize);
  out[1] = 0.0;
  free(SlaterElm); free(PfM); free(InvM);
#else
  SlaterElm = slater; PfM = pf; InvM = inv;
  double complex energy = NativeHamiltonian(*ip, config, config+2*Nsize,
    config+2*Nsize+Nsite2, config+2*Nsize+2*Nsite2, config+Nsize);
  out[0] = creal(energy); out[1] = cimag(energy);
#endif
  free(JastrowIdx); free(CoulombInter); free(HundCoupling); free(Transfer);
  free(DoublonHolon2siteIdx); free(DoublonHolon4siteIdx);
  free(PairHopping); free(ExchangeCoupling); free(InterAll);
  free(ParaCoulombIntra); free(ParaCoulombInter); free(ParaHundCoupling);
  free(ParaPairHopping); free(ParaExchangeCoupling);
}
