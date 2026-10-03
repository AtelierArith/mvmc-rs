/* Optional #180 input audit. Uses the existing extracted authoritative readers.
 * No Cargo build or Rust test depends on this probe. See the #180 evidence doc.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define D_FileNameMax 256
#include "orbital_contracts_upstream.inc"

int main(int argc, char **argv) {
  if (argc != 4) return 2;
  int parallel = strcmp(argv[1], "P") == 0;
  int nsite = atoi(argv[2]);
  if (nsite < 1 || nsite > 32) return 2;
  FILE *fp = fopen(argv[3], "r");
  if (!fp) return 2;
  int width = 0, complex_flag = 0;
  char *header = ReadBuffIntCmpFlg(fp, &width, &complex_flag);
  if (!header) { puts("REJECT header"); fclose(fp); return 0; }
  if (width < 1 || width > 1024) { fclose(fp); return 2; }
  char line[256];
  if (!fgets(line, sizeof(line), fp) || !fgets(line, sizeof(line), fp)) {
    fclose(fp); return 2;
  }
  int indices[64][64] = {{0}}, signs[64][64] = {{0}};
  int *idx[64], *sgn[64], flags[4096] = {0}, count = 0;
  for (int i = 0; i < 2*nsite; i++) { idx[i] = indices[i]; sgn[i] = signs[i]; }
  int status = parallel
    ? GetInfoOrbitalParallel(fp, idx, flags, sgn, &count, 0, complex_flag,
                             1, 1, nsite, width, 0, "P")
    : GetInfoOrbitalAntiParallel(fp, idx, flags, sgn, &count, 0, complex_flag,
                                 0, 1, nsite, width, "AP");
  printf("%s width=%d complex=%d status=%d\n",
         status == 0 ? "ACCEPT" : "REJECT", width, complex_flag, status);
  fclose(fp);
  return 0;
}
