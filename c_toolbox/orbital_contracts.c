#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define D_FileNameMax 256
#include "orbital_contracts_upstream.inc"

int main(int argc, char **argv) {
  if (argc != 4) return 2;
  int parallel = strcmp(argv[1], "P") == 0, nsite = atoi(argv[2]);
  FILE *fp = fopen(argv[3], "r");
  if (!fp) return 2;
  int width = 0, complex_flag = 0;
  char *header = ReadBuffIntCmpFlg(fp, &width, &complex_flag);
  if (!header) { printf("0 %d %d 1\n", width, complex_flag); fclose(fp); return 0; }
  char line[256];
  if (!fgets(line, sizeof(line), fp) || !fgets(line, sizeof(line), fp)) return 2;
  int indices[8][8] = {{0}}, signs[8][8] = {{0}}, *idx[8], *sgn[8];
  int flags[256] = {0}, count = 0;
  for (int i = 0; i < 8; i++) { idx[i] = indices[i]; sgn[i] = signs[i]; }
  int status = parallel
    ? GetInfoOrbitalParallel(fp, idx, flags, sgn, &count, 7, complex_flag, 1, 1, nsite, width, 7, "P")
    : GetInfoOrbitalAntiParallel(fp, idx, flags, sgn, &count, 0, complex_flag, 0, 1, nsite, width, "AP");
  printf("1 %d %d %d\n", width, complex_flag, status);
  if (status == 0) {
    for (int k = 0; k < width; k++) {
      int offset = parallel ? 7 + 2*k : k;
      printf("%d ", flags[2*offset]);
      if (parallel) printf("%d ", flags[2*offset+2]);
    }
    puts("");
  }
  fclose(fp);
  return 0;
}
