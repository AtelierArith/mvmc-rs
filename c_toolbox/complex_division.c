/* Optional native compiler complex quotient probe, including range recovery. */
#include <complex.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
int main(void) {
  union { uint64_t bits[2]; double complex value; } numerator, denominator, result;
  while (scanf("%" SCNx64 " %" SCNx64 " %" SCNx64 " %" SCNx64,
                &numerator.bits[0], &numerator.bits[1],
                &denominator.bits[0], &denominator.bits[1]) == 4) {
    result.value = numerator.value / denominator.value;
    printf("%016" PRIx64 " %016" PRIx64 "\n", result.bits[0], result.bits[1]);
  }
  return 0;
}
