/* Standalone kernel check for issue #449: the two-hop bilinear form of the authoritative C
 * `calculateNewPfMTwo_child_real` (extern/mVMC-1.3.0/src/mVMC/pfupdate_two_real.c:161-177),
 * evaluated for the inputs of tests/fixtures/pfaffian_cg/two_hop_bilinear.txt.
 *
 * The loops below are the C source lines verbatim (renamed variables only). This is a
 * standalone kernel check, NOT a full vmc.out run. Build WITHOUT fused multiply-add:
 *   gcc -O2 -ffp-contract=off probe.c -o probe        (also: gcc -O2 -o probe, default x86-64)
 * Usage: probe < tests/fixtures/pfaffian_cg/two_hop_bilinear.txt
 * Output: one line per case: `<n> <hex bits of bMa>` with bMa the sequential C reduction.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

static double from_bits(uint64_t b) {
  double d;
  memcpy(&d, &b, sizeof d);
  return d;
}
static uint64_t to_bits(double d) {
  uint64_t b;
  memcpy(&b, &d, sizeof b);
  return b;
}

int main(void) {
  char *line = NULL;
  size_t cap = 0;
  while (getline(&line, &cap, stdin) > 0) {
    if (line[0] == '#' || line[0] == '\n') continue;
    int nsize = atoi(line);
    int count = 1 + nsize * nsize + 2 * nsize;
    double *v = malloc(sizeof(double) * count);
    if (getline(&line, &cap, stdin) <= 0) return 1;
    char *tok = strtok(line, " \n");
    for (int i = 0; i < count; i++) {
      v[i] = from_bits(strtoull(tok, NULL, 16));
      tok = strtok(NULL, " \n");
    }
    const double *invM = v + 1;             /* [nsize*nsize], row msi contiguous in msj */
    const double *vec_a = v + 1 + nsize * nsize;
    const double *vec_b = vec_a + nsize;
    int msi, msj;
    double bMa = 0.0, tmp;
    /* --- extern/mVMC-1.3.0/src/mVMC/pfupdate_two_real.c:167-177 --- */
    for (msi = 0; msi < nsize; msi++) {
      const double *invM_i = invM + msi * nsize;
      tmp = 0.0;
      for (msj = 0; msj < nsize; msj++) {
        tmp += invM_i[msj] * vec_a[msj];
      }
      bMa += vec_b[msi] * tmp;
    }
    /* --- end --- */
    printf("%d %016llx\n", nsize, (unsigned long long)to_bits(bMa));
    free(v);
  }
  return 0;
}
