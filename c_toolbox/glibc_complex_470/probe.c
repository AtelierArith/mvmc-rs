/* Standalone libm kernel check for issue #470: evaluates the C99 complex functions the mVMC RBM
 * code calls (extern/mVMC-1.3.0/src/mVMC/rbm.c: cexp, clog, ccosh, ctanh) on hexadecimal inputs.
 * Not a full vmc.out run. Build and run: see generate.sh.
 *
 * stdin : one `<re bits> <im bits>` per line (16 hex digits each, IEEE-754 binary64)
 * stdout: `<re> <im> <cexp re im> <clog re im> <ccosh re im> <ctanh re im>`, all as 16 hex digits.
 */
#include <complex.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
static void put(double complex z) {
  printf(" %016llx %016llx", (unsigned long long)to_bits(creal(z)),
         (unsigned long long)to_bits(cimag(z)));
}

int main(void) {
  char line[256];
  while (fgets(line, sizeof line, stdin)) {
    unsigned long long a, b;
    if (sscanf(line, "%llx %llx", &a, &b) != 2) continue;
    volatile double re = from_bits(a), im = from_bits(b);
    double complex z = re + im * I;
    /* build the complex from parts exactly (re + im*I would turn inf*0 into NaN) */
    ((double *)&z)[0] = re;
    ((double *)&z)[1] = im;
    printf("%016llx %016llx", a, b);
    put(cexp(z));
    put(clog(z));
    put(ccosh(z));
    put(ctanh(z));
    printf("\n");
  }
  return 0;
}
