#include <stdio.h>
#include <stdlib.h>
#define D_FileNameMax 256
int Nsite;
#include "rbm_contracts_upstream.inc"

/* Final index assignments and raw flags, with deterministic untouched sentinels.
 * The prefix represents two projection coefficients. No native malloc claim. */
int main(int argc, char **argv) {
  if (argc != 5) return 2;
  int section = atoi(argv[1]), hidden = atoi(argv[3]);
  Nsite = atoi(argv[2]);
  if (section < 0 || section > 8 || Nsite < 1 || Nsite > 8 || hidden < 1 || hidden > 8) return 2;
  FILE *fp = fopen(argv[4], "r");
  if (!fp) return 2;
  int width = 0, complex = 0;
  if (!ReadBuffIntCmpFlg(fp, &width, &complex)) {
    printf("0 %d %d 0 1\n", width, complex);
    fclose(fp);
    return 0;
  }
  char line[256];
  if (!fgets(line, sizeof(line), fp) || !fgets(line, sizeof(line), fp)) return 2;
  if (width < 1 || width > 128) return 2;
  int flags[512] = {3, 0, -2, 0}, count = 0, status;
  int layer[16], matrix[16][8], *rows[16];
  for (int i = 0; i < 16; i++) {
    layer[i] = -1;
    rows[i] = matrix[i];
    for (int j = 0; j < 8; j++) matrix[i][j] = -1;
  }
  if (section == 2)
    status = GetInfoGeneralRBM_Layer(fp, layer, flags, complex, &count, 2, Nsite, width, "RBM");
  else if (section < 6)
    status = GetInfoRBM_Layer(fp, layer, flags, complex, &count, 2, section < 3 ? Nsite : hidden, width, "RBM");
  else if (section == 8)
    status = GetInfoGeneralRBM_PhysHidden(fp, rows, flags, complex, &count, 2, Nsite, hidden, width, "RBM");
  else
    status = GetInfoRBM_PhysHidden(fp, rows, flags, complex, &count, 2, Nsite, hidden, width, "RBM");
  printf("1 %d %d %d %d\n", width, complex, count, status);
  if (!status) {
    if (section < 6) {
      int size = section == 2 ? 2*Nsite : section < 3 ? Nsite : hidden;
      for (int i = 0; i < size; i++) printf("%d ", layer[i]);
    } else {
      for (int i = 0; i < (section == 8 ? 2*Nsite : Nsite); i++)
        for (int j = 0; j < hidden; j++) printf("%d ", matrix[i][j]);
    }
    puts("");
    for (int i = 0; i < 2*(2+width); i++) printf("%d ", flags[i]);
    puts("");
  }
  fclose(fp);
  return 0;
}
