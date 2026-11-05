/* Optional #682 probe: the unmodified C OutputTimerParaOpt and OutputTimerPhysCal
 * (extracted verbatim into c_timer_report_upstream.inc) with every Timer[] slot
 * zero, so the zvo_CalcTimer.dat line set and format are a C expectation.
 * Standalone format check: no MPI, sampling or timing. Neither Cargo nor Rust
 * tests compile, invoke or read this probe.
 *
 * Usage: c_timer_report <paraopt|physcal> <file head>
 */
#include <stdio.h>
#include <string.h>

#define D_FileNameMax 256
static double Timer[1000];
static char CDataFileHead[D_FileNameMax];
#include "c_timer_report_upstream.inc"

int main(int argc, char **argv) {
  if (argc != 3) return 2;
  snprintf(CDataFileHead, sizeof(CDataFileHead), "%s", argv[2]);
  if (strcmp(argv[1], "paraopt") == 0) {
    OutputTimerParaOpt();
  } else if (strcmp(argv[1], "physcal") == 0) {
    OutputTimerPhysCal();
  } else {
    return 3;
  }
  return 0;
}
