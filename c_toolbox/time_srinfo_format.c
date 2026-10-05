/* Optional actual-C probe for the run-log files: `_time_` (OutputTime), the SRinfo header
 * (InitFile) and SRinfo row statements, FlushFile and CloseFile. The function bodies in
 * time_srinfo_upstream.inc and the row statements in time_srinfo_rows_*.inc are verbatim
 * C from mVMC-1.3.0. This is a standalone kernel check: no MPI, sampler or SR solver runs,
 * and the counters/solver values are supplied literals. Cargo never builds or reads this file.
 *
 * Usage: time_srinfo_format <out-dir>   (run with TZ=UTC; time() is stubbed to a fixed value)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#define D_FileNameMax 256
char CDataFileHead[D_FileNameMax];
int NVMCCalMode, NDataIdxStart, SRFlag, FlagBinary = 0, NPara = 48, NSROptItrStep;
int NFileFlushInterval = 1;
int Counter[6] = {0, 0, 0, 0, 0, 0};
FILE *FileOut, *FileVar, *FileTime, *FileSRinfo;
/* Fixed clock: 2023-11-14 22:13:20 UTC. */
static time_t fixed_time(time_t *t) {
  if (t) *t = 1700000000;
  return 1700000000;
}
#define time(x) fixed_time(x)
#include "time_srinfo_upstream.inc"

static void row_direct(int nSmat, int optNum, int cutNum, double sDiagMax, double sDiagMin,
                       double rmax, int idx) {
  int smatToParaIdx[1] = {idx}, simax = 0;
#include "time_srinfo_rows_direct.inc"
}
static void row_diag(int nSmat, int optNum, int cutIdx, double eigenMax, double eigenMin,
                     double rmax, int idx) {
  int smatToParaIdx[1] = {idx}, imax = 0;
#include "time_srinfo_rows_diag.inc"
}
static void row_cg(int nSmat, int optNum, int cutNum, double sDiagMax, double sDiagMin,
                   double rmax, int idx, int info) {
  int smatToParaIdx[1] = {idx}, simax = 0;
#include "time_srinfo_rows_cg.inc"
}
static void set_counter(int a, int b, int c, int d, int e, int f) {
  Counter[0] = a; Counter[1] = b; Counter[2] = c; Counter[3] = d; Counter[4] = e; Counter[5] = f;
}
static void enter(const char *base, const char *name) {
  char path[4096];
  snprintf(path, sizeof(path), "%s/%s", base, name);
  mkdir(path, 0777);
  if (chdir(path) != 0) exit(2);
}

int main(int argc, char **argv) {
  if (argc != 2) return 1;
  char base[4096];
  snprintf(base, sizeof(base), "%s", argv[1]);

  /* Case A: ParaOpt, head "custom", NDataIdxStart 7, direct and CG rows, flush interval 2. */
  enter(base, "paraopt");
  strcpy(CDataFileHead, "custom"); NDataIdxStart = 7; NVMCCalMode = 0; SRFlag = 0;
  NFileFlushInterval = 2;
  InitFile("namelist.def", 0);
  for (int step = 0; step < 4; step++) {
    switch (step) {
      case 0: set_counter(0, 0, 0, 0, 0, 0); break;
      case 1: set_counter(1000, 437, 250, 100, 0, 0); break;
      case 2: set_counter(123456789, 61728394, 3, 1, 7, 7); break;
      default: set_counter(2147483647, 1, 0, 5, 9, 4); break;
    }
    OutputTime(step);
    row_direct(40, 4, 4, 3.10304e-2, 0.0, 7.84187e-3, 13);
    row_cg(40, 4, 4, 3.53637e-2, 0.0, -5.5165e-3, 43, 35);
    row_direct(1, 0, 0, 1.0e100, -1.0e-300, 123456.789, 99999);
    row_direct(12345, 100000, 7, -0.0, 2.5, -0.0, -1);
    FlushFile(step, 0);
  }
  set_counter(10, 10, 10, 10, 10, 10);
  NSROptItrStep = 4;
  OutputTime(NSROptItrStep);
  CloseFile(0);

  /* Case B: PhysCal mode writes only the time file; negative index formatted with %03d. */
  enter(base, "physcal");
  strcpy(CDataFileHead, "zvo"); NDataIdxStart = -1; NVMCCalMode = 1;
  InitFile("namelist.def", 0);
  set_counter(0, 0, 0, 0, 0, 0);
  OutputTime(0); FlushFile(0, 0);
  set_counter(20, 5, 0, 0, 0, 0);
  OutputTime(1); FlushFile(0, 0);
  OutputTime(2);
  fclose(FileTime);

  /* Case C: diagonalization-mode header and eigenvalue row. */
  enter(base, "diag");
  strcpy(CDataFileHead, "zvo"); NDataIdxStart = 0; NVMCCalMode = 0; SRFlag = 1;
  InitFile("namelist.def", 0);
  row_diag(3, 1, 2, 4.5, 0.25, -1.0, 4);
  CloseFile(0);
  return 0;
}
