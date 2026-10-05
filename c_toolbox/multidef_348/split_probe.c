/* Standalone kernel probe (issue #348): the MultiDef communicator-split colour of
 * `initMultiDefMode` (extern/mVMC-1.3.0/src/mVMC/vmcmain.c:752-760), with the
 * statements copied verbatim and only wrapped in a loop over (size, nMultiDef, rank).
 * No MPI: this checks the arithmetic only, not a full vmc.out -m run (see
 * generate_c_runs.sh for that).
 *
 * Output: one line per (size, nMultiDef) with 1 <= nMultiDef <= size <= 48:
 *   size nMultiDef group(rank 0) group(rank 1) ... group(rank size-1)
 *
 * Build/run (see generate_split.sh):  gcc -O0 -o split_probe split_probe.c && ./split_probe
 */
#include <stdio.h>

int main(void) {
  int size, nMultiDef, rank;
  for (size = 1; size <= 48; size++) {
    for (nMultiDef = 1; nMultiDef <= size; nMultiDef++) {
      printf("%d %d", size, nMultiDef);
      for (rank = 0; rank < size; rank++) {
        int group1, div, mod, threshold;
        /* ---- verbatim vmcmain.c:752-759 ---- */
        div = size / nMultiDef;
        mod = size % nMultiDef;
        threshold = (div+1)*mod;
        if(rank < threshold) {
          group1 = rank / (div+1);
        } else {
          group1 = mod + (rank-threshold)/div;
        }
        /* ---- end verbatim ---- */
        printf(" %d", group1);
      }
      printf("\n");
    }
  }
  return 0;
}
