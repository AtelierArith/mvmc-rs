#include <stdio.h>
#include <stdlib.h>
#define D_FileNameMax 256
#include "jastrow_contracts_upstream.inc"

/* Untouched index cells are -1 sentinels; untouched imaginary flags are zero.
 * Neither value claims to reproduce native malloc contents. The prefix models
 * two already-read real Gutzwiller flags at a nonzero Jastrow offset. */
int main(int argc, char **argv) {
  if (argc != 3) return 2;
  int nsite = atoi(argv[1]);
  FILE *fp = fopen(argv[2], "r");
  if (!fp || nsite < 2 || nsite > 8) return 2;
  int width = 0, complex_flag = 0;
  if (!ReadBuffIntCmpFlg(fp, &width, &complex_flag)) {
    printf("0 %d %d 0 1\n", width, complex_flag);
    fclose(fp);
    return 0;
  }
  char line[256];
  if (!fgets(line, sizeof(line), fp) || !fgets(line, sizeof(line), fp)) return 2;
  if (width < 1 || width > 32) return 2;
  int indices[8][8], *idx[8], flags[256] = {3, 0, -2, 0}, count = 0;
  for (int i = 0; i < 8; i++) {
    idx[i] = indices[i];
    for (int j = 0; j < 8; j++) indices[i][j] = -1;
  }
  int status = GetInfoJastrow(fp, idx, flags, complex_flag, &count,
                            2, nsite, width, "Jastrow");
  printf("1 %d %d %d %d\n", width, complex_flag, count, status);
  if (status == 0) {
    for (int i = 0; i < nsite; i++)
      for (int j = 0; j < nsite; j++) printf("%d ", indices[i][j]);
    puts("");
    for (int k = 0; k < 2*(2+width); k++) printf("%d ", flags[k]);
    puts("");
  }
  fclose(fp);
  return 0;
}
