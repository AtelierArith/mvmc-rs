#include <stdio.h>
#include <stdlib.h>
#define D_FileNameMax 256
#include "gutzwiller_contracts_upstream.inc"

/* Storage is zeroed explicitly. Untouched cells are probe sentinels, not
 * evidence of native malloc initialization. Invalid sites are excluded: the
 * upstream reader writes ArrayIdx[i] before validating i. */
int main(int argc, char **argv) {
  if (argc != 3) return 2;
  int nsite = atoi(argv[1]);
  FILE *fp = fopen(argv[2], "r");
  if (!fp || nsite < 1 || nsite > 8) return 2;
  int width = 0, complex_flag = 0;
  if (!ReadBuffIntCmpFlg(fp, &width, &complex_flag)) {
    printf("0 %d %d 0 1\n", width, complex_flag);
    fclose(fp);
    return 0;
  }
  char line[256];
  if (!fgets(line, sizeof(line), fp) || !fgets(line, sizeof(line), fp)) return 2;
  if (width < 1 || width > 32) return 2;
  int indices[8] = {0}, flags[256] = {0}, count = 0;
  int status = GetInfoGutzwiller(fp, indices, flags, complex_flag, &count,
                               nsite, width, "Gutzwiller");
  printf("1 %d %d %d %d\n", width, complex_flag, count, status);
  if (status == 0) {
    for (int k = 0; k < nsite; k++) printf("%d ", indices[k]);
    puts("");
    for (int k = 0; k < 2*width; k++) printf("%d ", flags[k]);
    puts("");
  }
  fclose(fp);
  return 0;
}
