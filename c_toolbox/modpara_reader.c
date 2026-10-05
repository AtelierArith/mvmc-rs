/* Optional #344 probe: runs the unmodified C SetDefaultValuesModPara +
 * GetInfoFromModPara (extracted verbatim into modpara_reader_upstream.inc) and
 * the NBlockSize_RBMRatio adjustment of ReadDefFileNInt on one modpara.def.
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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define TIME_SENTINEL 1700000000
#define time(x) ((time_t)TIME_SENTINEL)

typedef int MPI_Comm; /* readdef.h declares MPI prototypes; no MPI is linked. */
#include "readdef.h"
#include "global.h"
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
  printf("RESULT status %d\n", status);
  if (status != 0) return 0;
  ApplyNBlockSizeRBMRatioAdjustment(bufInt);
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
  D("DSROptRedCut", IdxSROptRedCut);
  D("DSROptStaDel", IdxSROptStaDel);
  D("DSROptStepDt", IdxSROptStepDt);
  D("DSROptCGTol", IdxSROptCGTol);
  printf("RESULT str CDataFileHead %s\n", CDataFileHead);
  printf("RESULT str CParaFileHead %s\n", CParaFileHead);
  printf("RESULT int TimeSentinel %d\n", TIME_SENTINEL);
  return 0;
}
