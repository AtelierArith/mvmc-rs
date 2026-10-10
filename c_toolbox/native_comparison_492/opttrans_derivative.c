/* Optional standalone probe of mVMC-1.3.0 vmccal.c:639-655.
 * Upstream University of Tokyo (2016), GPL-3.0-or-later.
 * Reproduction: gcc -std=c11 -O3 opttrans_derivative.c -o /tmp/opttrans492;
 * /tmp/opttrans492. No numerical changes to the extracted function.
 */
#include <complex.h>
#include <stdio.h>
static int NQPOptTrans = 2, NQPFix = 2;
static double complex QPFixWeight[] = {0.5+0.25*I, -0.5+0.125*I};
static double complex PfM[] = {1+2*I, 3-I, 2-I, -1+4*I};
void calculateOptTransDiff(double complex *srOptO, const double complex ipAll) {
  int i,j;
  double complex ip;
  double complex *pfM;

  for(i=0;i<NQPOptTrans;++i) {
    ip = 0.0;
    pfM = PfM + i*NQPFix;
    for(j=0;j<NQPFix;++j) {
      ip += QPFixWeight[j] * pfM[j];
    }
    srOptO[i] = ip/ipAll;
  }

  return;
}

int main(void) {
  double complex out[6];
  for(int i=0;i<6;i++) out[i]=7-9*I;
  calculateOptTransDiff(out,1.25-0.75*I);
  for(int i=0;i<6;i++) printf("%.17e %.17e\n",creal(out[i]),cimag(out[i]));
  return 0;
}
