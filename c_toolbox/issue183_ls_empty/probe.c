#include <stdio.h>
#include <complex.h>
#include "physcal_lanczos.h"

static int check(FILE *file, const char *mode) {
  fflush(file);
  long bytes = ftell(file);
  rewind(file);
  int first = fgetc(file), next = fgetc(file);
  printf("mode=%s bytes=%ld first=%d next=%d\n", mode, bytes, first, next);
  return bytes == 1 && first == 10 && next == EOF ? 0 : 1;
}

int main(void) {
  double q[16] = {0};
  double complex z[16] = {0};
  /* H1=0,H2=1,H22=1,H3=0,H4=4: finite valid analytic alpha. */
  q[3] = q[10] = 1; q[15] = 4;
  for (int i = 0; i < 16; ++i) z[i] = q[i];
  FILE *files[7];
  for (int i = 0; i < 7; ++i) if (!(files[i] = tmpfile())) return 2;
  int real_status = PhysCalLanczos_real(q, NULL, NULL, NULL, 2, 2, 0, 0,
    NULL, NULL, 0, 0, NULL, 2, files[0], files[1], NULL, NULL,
    files[2], files[3], files[4]);
  int failure = check(files[3], "real");
  int cmp_status = PhysCalLanczos_fcmp(z, NULL, NULL, NULL, 2, 2, 0, 0,
    NULL, NULL, 0, 0, NULL, 2, files[5], files[6], NULL, NULL,
    files[2], files[3], files[4]);
  /* Real and complex calls shared one file: second LF must append. */
  fflush(files[3]); rewind(files[3]);
  int first = fgetc(files[3]), second = fgetc(files[3]), end = fgetc(files[3]);
  printf("mode=cmp shared_bytes=2 first=%d second=%d end=%d real_status=%d cmp_status=%d\n",
    first, second, end, real_status, cmp_status);
  failure |= first != 10 || second != 10 || end != EOF || real_status != 0 || cmp_status != 0;
  for (int i = 0; i < 7; ++i) fclose(files[i]);
  return failure;
}
