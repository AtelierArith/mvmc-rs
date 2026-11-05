/* Check complete RBM section files with the actual C 1.3.0 readers (#667).
 *
 * Reads one RBM section file through the verbatim readdef.c functions in
 * rbm_contracts_upstream.inc (ReadBuffIntCmpFlg, GetInfoRBM_Layer,
 * GetInfoGeneralRBM_Layer, GetInfoRBM_PhysHidden, GetInfoGeneralRBM_PhysHidden)
 * at production sizes, then requires reader status 0, every mapping cell
 * assigned with an index inside the declared width, and every flag present.
 * Prints "ok <width> <complex> <flags>" or "fail ..." and exits nonzero on a
 * failure. Unlike rbm_contracts.c (the 8-site contract oracle), the arrays are
 * sized from the command line. No Rust build or test reads this program.
 *
 *   cc -O0 c_toolbox/rbm_complete_inputs.c -o /tmp/rbm_complete
 *   /tmp/rbm_complete <section 0-8> <Nsite> <neurons> <file>
 *
 * Section order: Charge/Spin/General PhysLayer, Charge/Spin/General
 * HiddenLayer, Charge/Spin/General PhysHidden (C readdef.c:936-1008).
 */
#include <stdio.h>
#include <stdlib.h>
#define D_FileNameMax 256
int Nsite;
#include "rbm_contracts_upstream.inc"

int main(int argc, char **argv) {
  if (argc != 5) return 2;
  int section = atoi(argv[1]), hidden = atoi(argv[3]);
  Nsite = atoi(argv[2]);
  if (section < 0 || section > 8 || Nsite < 1 || hidden < 1) return 2;
  FILE *fp = fopen(argv[4], "r");
  if (!fp) return 2;
  int width = 0, complex = 0;
  if (!ReadBuffIntCmpFlg(fp, &width, &complex)) {
    printf("fail header\n");
    return 1;
  }
  char line[256];
  if (!fgets(line, sizeof(line), fp) || !fgets(line, sizeof(line), fp)) return 2;
  if (width < 1) return 2;
  const int rows_n = section == 2 || section == 8 ? 2 * Nsite : section < 3 ? Nsite
                     : section < 6 ? hidden : Nsite;
  const int cols_n = section < 6 ? 1 : hidden;
  int *flags = malloc(sizeof(int) * 2 * (width + 2));
  int *cells = malloc(sizeof(int) * rows_n * cols_n);
  int **rows = malloc(sizeof(int *) * rows_n);
  for (int i = 0; i < 2 * (width + 2); i++) flags[i] = -7;
  for (int i = 0; i < rows_n * cols_n; i++) cells[i] = -1;
  for (int i = 0; i < rows_n; i++) rows[i] = cells + i * cols_n;
  int count = 0, status;
  /* fidx 0: the flags of this section start at OptFlag[0]. */
  if (section == 2)
    status = GetInfoGeneralRBM_Layer(fp, cells, flags, complex, &count, 0, Nsite, width, "RBM");
  else if (section < 6)
    status = GetInfoRBM_Layer(fp, cells, flags, complex, &count, 0, section < 3 ? Nsite : hidden,
                              width, "RBM");
  else if (section == 8)
    status = GetInfoGeneralRBM_PhysHidden(fp, rows, flags, complex, &count, 0, Nsite, hidden,
                                          width, "RBM");
  else
    status = GetInfoRBM_PhysHidden(fp, rows, flags, complex, &count, 0, Nsite, hidden, width,
                                   "RBM");
  fclose(fp);
  if (status) {
    printf("fail reader status %d\n", status);
    return 1;
  }
  for (int i = 0; i < rows_n * cols_n; i++) {
    if (cells[i] < 0 || cells[i] >= width) {
      printf("fail cell %d index %d (width %d)\n", i, cells[i], width);
      return 1;
    }
  }
  for (int i = 0; i < width; i++) {
    if (flags[2 * i] == -7) {
      printf("fail missing flag %d\n", i);
      return 1;
    }
  }
  printf("ok %d %d %d\n", width, complex, count);
  return 0;
}
