/* Optional fixed-value formatting probe. Templates copied verbatim from
 * mVMC-1.3.0 src/mVMC/vmcmain.c:666-687; no computed numerical oracle.
 * Reproduce: cc -std=c11 physcal_181_format.c -o /tmp/physcal181-format
 * /tmp/physcal181-format ../tests/fixtures/physcal_181/formatting
 * Cargo never builds or invokes this program.
 */
#include <stdio.h>
#include <stdlib.h>
static FILE *open_file(const char *dir, const char *name) {
    char path[4096];
    if (snprintf(path, sizeof(path), "%s/%s", dir, name) >= sizeof(path)) exit(2);
    FILE *file = fopen(path, "w");
    if (!file) exit(3);
    return file;
}
int main(int argc, char **argv) {
    if (argc != 2) return 1;
    FILE *file = open_file(argv[1], "one.dat");
    fprintf(file, "%d %d %d %d % .18e  % .18e \n", 0, 0, 1, 1, 1.5, -2.0);
    fprintf(file, "\n"); fclose(file);
    file = open_file(argv[1], "factored.dat");
    fprintf(file, "% .18e  % .18e ", 3.0, 4.0);
    fprintf(file, "\n"); fclose(file);
    file = open_file(argv[1], "direct.dat");
    fprintf(file, "%d %d %d %d %d %d %d %d % .18e % .18e\n", 0, 0, 1, 1, 2, 0, 3, 1, -5.0, 6.0);
    fprintf(file, "\n"); fclose(file);
    file = open_file(argv[1], "out.dat");
    /* vmcmain.c:647, supplied literal result values, not calculated variance. */
    fprintf(file, "% .18e % .18e  % .18e % .18e %.18e %.18e\n", 1.0, 1.0, 2.0, -1.0, 0.5, 0.25);
    fclose(file);
    file = open_file(argv[1], "var-full.dat");
    /* vmcmain.c:655-657: full declared Para storage, including reserved slots.
     * Gutz2,Jast2,DH2(6),DH4(10),nine RBM slots,Slater2,OptTrans2 = 33.
     * Literal dyadic coefficients; unmapped Gutz/Jast slots 1 and 3 are zero. */
    fprintf(file, "% .18e % .18e 0.0 % .18e % .18e 0.0 ", -3.0, 0.0, 9.0, 0.0);
    for (int i = 0; i < 33; i++) {
        double re = (i == 1 || i == 3) ? 0.0 : (i + 1) * 0.125;
        double im = (i == 1 || i == 3) ? 0.0 : -(i + 1) * 0.0625;
        fprintf(file, "% .18e % .18e 0.0 ", re, im);
    }
    fprintf(file, "\n"); fclose(file);
    return 0;
}
