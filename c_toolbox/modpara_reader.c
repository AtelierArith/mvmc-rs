/* Optional #344 probe: runs the unmodified C SetDefaultValuesModPara +
 * GetInfoFromModPara (extracted verbatim into modpara_reader_upstream.inc), the
 * NBlockSize_RBMRatio adjustment and the mVMC 1.4.0 SR-CG/DSROptStepDt
 * normalization of ReadDefFileNInt on one modpara.def. MPI_Abort in that block
 * returns to main and is reported as status 2.
 * Standalone reader-kernel check: no MPI, no full mVMC executable, no sampling.
 * Neither Cargo nor Rust tests compile, invoke or read this probe.
 *
 * Usage: modpara_reader <modpara.def>   (run in a scratch working directory;
 * the C reader runs `mkdir -p output` there).
 *
 * time(NULL) is replaced by the sentinel below so that the RndSeed<0 branch is
 * observable and deterministic. Output lines are prefixed with RESULT.
 */
#include <ctype.h>
#include <setjmp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define TIME_SENTINEL 1700000000
#define time(x) ((time_t)TIME_SENTINEL)

typedef int MPI_Comm; /* readdef.h declares MPI prototypes; no MPI is linked. */
#include "readdef.h"
/* mVMC 1.4.0 readdef.h declares this extern; readdef.c defines it. */
char (*cFileNameListFile)[D_CharTmpReadDef] = NULL;
#include "global.h"
static jmp_buf abort_jump;
static MPI_Comm comm = 0;
static void MPI_Abort(MPI_Comm c, int code) { (void)c; (void)code; longjmp(abort_jump, 1); }
#include "modpara_reader_upstream.inc"

int main(int argc, char **argv) {
  if (argc != 2) return 2;
  int bufInt[ParamIdxInt_End];
  double bufDouble[ParamIdxDouble_End];
  memset(bufInt, 0, sizeof(bufInt));
  memset(bufDouble, 0, sizeof(bufDouble));
  cFileNameListFile = calloc(KWIdxInt_end, D_CharTmpReadDef);
  strcpy(cFileNameListFile[KWModPara], argv[1]);

  SetDefaultValuesModPara(bufInt, bufDouble);
  int status = GetInfoFromModPara(bufInt, bufDouble);
  if (status == 0) {
    ApplyNBlockSizeRBMRatioAdjustment(bufInt);
    DSROptStepDt = bufDouble[IdxSROptStepDt];
    fflush(stdout);
    if (setjmp(abort_jump) == 0) ApplySrCgNormalization(0);
    else status = 2;
  }
  fflush(stdout);
  printf("RESULT status %d\n", status);
  if (status != 0) return 0;
#define I(name, idx) printf("RESULT int %s %d\n", name, bufInt[idx])
#define D(name, idx) printf("RESULT double %s %.17g\n", name, bufDouble[idx])
  I("NVMCCalMode", IdxVMCCalcMode);
  I("NLanczosMode", IdxLanczosMode);
  I("NDataIdxStart", IdxDataIdxStart);
  I("NDataQtySmp", IdxDataQtySmp);
  I("Nsite", IdxNsite);
  I("Ne", IdxNe);
  I("Ncond", IdxNCond);
  I("2Sz", Idx2Sz);
  I("NSPGaussLeg", IdxSPGaussLeg);
  I("NSPStot", IdxSPStot);
  I("NMPTrans", IdxMPTrans);
  I("NSROptItrStep", IdxSROptItrStep);
  I("NSROptItrSmp", IdxSROptItrSmp);
  I("NSROptFixSmp", IdxSROptFixSmp);
  I("NSROptCGMaxIter", IdxSROptCGMaxIter);
  I("NVMCWarmUp", IdxVMCWarmUp);
  I("NVMCInterval", IdxVMCInterval);
  I("NVMCSample", IdxVMCSample);
  I("NExUpdatePath", IdxExUpdatePath);
  I("RndSeed", IdxRndSeed);
  I("NSplitSize", IdxSplitSize);
  I("Nneuron", IdxNneuron);
  I("NneuronCharge", IdxNneuronCharge);
  I("NneuronSpin", IdxNneuronSpin);
  I("NneuronGeneral", IdxNneuronGeneral);
  I("NBlockSize_RBMRatio", IdxNBlockSize_RBMRatio);
  printf("RESULT int NStore %d\n", NStoreO);
  printf("RESULT int NSRCG %d\n", NSRCG);
  printf("RESULT int useDiagScale %d\n", useDiagScale);
  printf("RESULT int NSRCGFallback %d\n", NSRCGFallback);
  printf("RESULT int NSRCGAbortOnFail %d\n", NSRCGAbortOnFail);
  printf("RESULT int RescaleSmat %d\n", RescaleSmat);
  printf("RESULT int reweight %d\n", reweight);
  printf("RESULT int SRFlag %d\n", SRFlag);
  D("DSROptRedCut", IdxSROptRedCut);
  D("DSROptStaDel", IdxSROptStaDel);
  printf("RESULT double DSROptStepDt %.17g\n", DSROptStepDt);
  D("DSROptCGTol", IdxSROptCGTol);
  printf("RESULT str CDataFileHead %s\n", CDataFileHead);
  printf("RESULT str CParaFileHead %s\n", CParaFileHead);
  printf("RESULT int TimeSentinel %d\n", TIME_SENTINEL);
  return 0;
}
