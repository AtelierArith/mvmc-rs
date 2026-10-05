/*
 * Probe of the compiled C `double complex` expressions used by StdFace_HubbardLocal,
 * StdFace_MagField and StdFace_GeneralJ (extern/mVMC-1.3.0/src/StdFace/src/StdFace_ModelUtil.c).
 *
 * The expressions below are copied verbatim from those functions (same operand order, same mix
 * of `double`, `I` and `double complex`). The probe evaluates them on a grid of operands that
 * includes -0.0 and prints the raw IEEE-754 bit patterns of the real and imaginary parts, so the
 * Rust port can reproduce signed zeros (they appear as "-0.000000000000000" in the output).
 *
 * Build: c_toolbox/stdface/check_complex_expr.sh (uses the same optimisation level as the
 * reference build, -O3 -DNDEBUG, and -O0 for comparison).
 * Output: one line per case: "<expr> <inputs as hex> -> <re hex> <im hex>".
 */
#include <complex.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NOINLINE __attribute__((noinline))

static uint64_t bits(double x) {
  uint64_t u;
  memcpy(&u, &x, sizeof u);
  return u;
}

/* Per-family FNV-1a 64 digest of the exact text lines (set STDFACE_LINES=1 to print them). */
#define MAXFAM 32
static char fam_name[MAXFAM][32];
static uint64_t fam_hash[MAXFAM];
static long fam_count[MAXFAM];
static int nfam = 0;

static void emit(const char *name, int nin, const double *in, double complex v) {
  char line[512];
  int n = snprintf(line, sizeof line, "%s", name);
  for (int i = 0; i < nin; i++)
    n += snprintf(line + n, sizeof line - n, " %016llx", (unsigned long long)bits(in[i]));
  n += snprintf(line + n, sizeof line - n, " -> %016llx %016llx\n",
                (unsigned long long)bits(creal(v)), (unsigned long long)bits(cimag(v)));
  if (getenv("STDFACE_LINES")) {
    fputs(line, stdout);
    return;
  }
  int f = 0;
  while (f < nfam && strcmp(fam_name[f], name) != 0) f++;
  if (f == nfam) {
    strcpy(fam_name[nfam], name);
    fam_hash[nfam] = 1469598103934665603ULL;
    nfam++;
  }
  for (int i = 0; i < n; i++) {
    fam_hash[f] ^= (unsigned char)line[i];
    fam_hash[f] *= 1099511628211ULL;
  }
  fam_count[f]++;
}

/* HubbardLocal */
NOINLINE static double complex hl_im_neg(double Gamma0_y) { return -0.5 * I * Gamma0_y; }
NOINLINE static double complex hl_im_pos(double Gamma0_y) { return 0.5 * I * Gamma0_y; }

/* MagField */
NOINLINE static double complex mf_minus(double Gamma, double Gamma_y, double S, double Sz) {
  return -0.5 * Gamma * sqrt(S * (S + 1.0) - Sz * (Sz + 1.0)) -
         0.5 * I * Gamma_y * sqrt(S * (S + 1.0) - Sz * (Sz + 1.0));
}
NOINLINE static double complex mf_plus(double Gamma, double Gamma_y, double S, double Sz) {
  return -0.5 * Gamma * sqrt(S * (S + 1.0) - Sz * (Sz + 1.0)) +
         0.5 * I * Gamma_y * sqrt(S * (S + 1.0) - Sz * (Sz + 1.0));
}

/* GeneralJ */
NOINLINE static double complex gj_zz(double J22, double Siz, double Sjz) {
  double complex intr0 = J22 * Siz * Sjz;
  return intr0;
}
NOINLINE static double complex gj_pm(const double J[3][3], double Si, double Siz, double Sj,
                                     double Sjz) {
  return 0.25 * (J[0][0] + J[1][1] + I * (J[0][1] - J[1][0])) *
         sqrt(Si * (Si + 1.0) - Siz * (Siz + 1.0)) * sqrt(Sj * (Sj + 1.0) - Sjz * (Sjz + 1.0));
}
NOINLINE static double complex gj_pp(const double J[3][3], double Si, double Siz, double Sj,
                                     double Sjz) {
  return 0.5 * 0.5 * (J[0][0] - J[1][1] - I * (J[0][1] + J[1][0])) *
         sqrt(Si * (Si + 1.0) - Siz * (Siz + 1.0)) * sqrt(Sj * (Sj + 1.0) - Sjz * (Sjz + 1.0));
}
NOINLINE static double complex gj_pz(const double J[3][3], double Si, double Siz, double Sjz) {
  return 0.5 * (J[0][2] - I * J[1][2]) * sqrt(Si * (Si + 1.0) - Siz * (Siz + 1.0)) * Sjz;
}
NOINLINE static double complex gj_zp(const double J[3][3], double Sj, double Siz, double Sjz) {
  return 0.5 * (J[2][0] - I * J[2][1]) * Siz * sqrt(Sj * (Sj + 1.0) - Sjz * (Sjz + 1.0));
}
NOINLINE static double complex conj_of(double complex v) { return conj(v); }

int main(void) {
  /* Operand grid: signed zeros, both signs and a non-trivial magnitude. */
  const double g[] = {-2.0, -0.0, 0.0, 0.5, 3.0};
  const int ng = 5;
  /* spin quantum numbers: S = 1/2 and S = 1 with every Sz */
  const double spins[][2] = {{0.5, 0.5}, {0.5, -0.5}, {1.0, 1.0}, {1.0, 0.0}, {1.0, -1.0}};
  const int nspin = 5;
  volatile double in[4];
  double J[3][3];

  for (int a = 0; a < ng; a++) {
    double in1[1] = {g[a]};
    emit("hl_im_neg", 1, in1, hl_im_neg(g[a]));
    emit("hl_im_pos", 1, in1, hl_im_pos(g[a]));
  }
  for (int a = 0; a < ng; a++)
    for (int b = 0; b < ng; b++)
      for (int s = 0; s < nspin; s++) {
        double in4[4] = {g[a], g[b], spins[s][0], spins[s][1]};
        double complex m = mf_minus(g[a], g[b], spins[s][0], spins[s][1]);
        double complex p = mf_plus(g[a], g[b], spins[s][0], spins[s][1]);
        emit("mf_minus", 4, in4, m);
        emit("mf_plus", 4, in4, p);
        emit("conj_mf_minus", 4, in4, conj_of(m));
      }
  for (int a = 0; a < ng; a++)
    for (int b = 0; b < ng; b++)
      for (int s = 0; s < nspin; s++) {
        double in3[3] = {g[a], g[b], spins[s][0]};
        emit("gj_zz", 3, in3, gj_zz(g[a], g[b], spins[s][1]));
      }
  for (int a = 0; a < ng; a++)
    for (int b = 0; b < ng; b++)
      for (int c = 0; c < ng; c++)
        for (int d = 0; d < ng; d++) {
          memset(J, 0, sizeof J);
          J[0][0] = g[a];
          J[1][1] = g[b];
          J[0][1] = g[c];
          J[1][0] = g[d];
          for (int s = 0; s < nspin; s++)
            for (int t = 0; t < nspin; t++) {
              double inp[8] = {g[a],        g[b],        g[c],        g[d],
                               spins[s][0], spins[s][1], spins[t][0], spins[t][1]};
              double complex pm =
                  gj_pm(J, spins[s][0], spins[s][1], spins[t][0], spins[t][1]);
              emit("gj_pm", 8, inp, pm);
              emit("conj_gj_pm", 8, inp, conj_of(pm));
              double complex pp =
                  gj_pp(J, spins[s][0], spins[s][1], spins[t][0], spins[t][1]);
              emit("gj_pp", 8, inp, pp);
              emit("conj_gj_pp", 8, inp, conj_of(pp));
            }
        }
  for (int a = 0; a < ng; a++)
    for (int b = 0; b < ng; b++)
      for (int s = 0; s < nspin; s++)
        for (int t = 0; t < nspin; t++) {
          memset(J, 0, sizeof J);
          J[0][2] = g[a];
          J[1][2] = g[b];
          J[2][0] = g[a];
          J[2][1] = g[b];
          double inp[6] = {g[a], g[b], spins[s][0], spins[s][1], spins[t][0], spins[t][1]};
          double complex pz = gj_pz(J, spins[s][0], spins[s][1], spins[t][1]);
          emit("gj_pz", 6, inp, pz);
          emit("conj_gj_pz", 6, inp, conj_of(pz));
          double complex zp = gj_zp(J, spins[t][0], spins[s][1], spins[t][1]);
          emit("gj_zp", 6, inp, zp);
          emit("conj_gj_zp", 6, inp, conj_of(zp));
        }
  (void)in;
  for (int f = 0; f < nfam; f++)
    printf("%s %ld %016llx\n", fam_name[f], fam_count[f], (unsigned long long)fam_hash[f]);
  return 0;
}
