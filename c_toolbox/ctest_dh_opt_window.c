/* DH auxiliary filename/split adapter; authoritative avevar bodies unchanged. */
#define main generic_window_main
#include "ctest_opt_window.c"
#undef main
int main(int argc, char **argv) {
    if (argc != 6) return 2;
    iFlgOrbitalGeneral = atoi(argv[3]);
    iNOrbitalAntiParallel = atoi(argv[4]);
    iNOrbitalParallel = atoi(argv[5]);
    if ((iFlgOrbitalGeneral != 0 && iFlgOrbitalGeneral != 1) ||
        iNOrbitalAntiParallel < 0 || iNOrbitalParallel < 0) return 2;
    return generic_window_main(3, argv);
}
