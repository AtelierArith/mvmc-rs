/* Actual C InterAll reader, with bounded comparison storage. */
#include <complex.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define D_FileNameMax 256
int TwoSz;
#include "interall_reader_upstream.inc"

int main(int argc, char **argv) {
  if (argc != 4) return 2;
  int nsite = atoi(argv[1]);
  TwoSz = atoi(argv[2]);
  FILE *fp = fopen(argv[3], "r");
  if (!fp) return 2;
  int width = 0;
  int header_ok = ReadBuffInt(fp, &width) != NULL;
  if (!header_ok || width < 0 || width > 2048) {
    printf("%d %d 1\n", header_ok, width);
    fclose(fp);
    return 0;
  }
  rewind(fp);
  char line[256];
  for (int i = 0; i < 5; i++) (void)fgets(line, sizeof(line), fp);
  /* Extra-row error probes reserve additional storage. Their reader status is
   * evidence of the count check, not of a safe native production allocation. */
  int rows[4096][8] = {{0}}, *indices[4096];
  double complex values[4096] = {0};
  for (int i = 0; i < 4096; i++) indices[i] = rows[i];
  int status = GetInfoInterAll(fp, indices, values, nsite, width, "InterAll");
  printf("%d %d %d\n", header_ok, width, status);
  if (status == 0) {
    for (int i = 0; i < width; i++) {
      for (int j = 0; j < 8; j++) printf("%d ", rows[i][j]);
      union { uint64_t bits; double value; } re = {.value = creal(values[i])};
      union { uint64_t bits; double value; } im = {.value = cimag(values[i])};
      printf("%016" PRIx64 " %016" PRIx64 "\n", re.bits, im.bits);
    }
  }
  fclose(fp);
  return 0;
}
