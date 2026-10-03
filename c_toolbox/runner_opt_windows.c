/* Optional developer adapter: retain Pauli's history reader and verbatim C
 * aggregation bodies, supplying C's orbital filename/block-selection globals.
 * Never compiled or invoked by Cargo. No Rust numerical values are read. */
#define main history_probe_main
#include "ctest_opt_window.c"
#undef main
#include <errno.h>
#include <limits.h>

static int nonnegative_int(const char *text, int *value) {
    char *end;
    errno = 0;
    long parsed = strtol(text, &end, 10);
    if (errno || *end || end == text || parsed < 0 || parsed > INT_MAX) return 0;
    *value = (int)parsed;
    return 1;
}

int main(int argc, char **argv) {
    if (argc != 6 || !nonnegative_int(argv[3], &iFlgOrbitalGeneral) ||
        !nonnegative_int(argv[4], &iNOrbitalAntiParallel) ||
        !nonnegative_int(argv[5], &iNOrbitalParallel) || iFlgOrbitalGeneral > 1)
        return 2;
    return history_probe_main(3, argv);
}
