/* GPL-3.0-or-later; optional C reference, never used by Cargo.
 * Verbatim mVMC 1.4.0 vmccal.c calculateOO lines 814-839 (the body is identical to
 * 1.3.0 lines 769-794).
 * Upstream SHA-256 c06d5eef360ee3e442d1843ec218f0beb34fc7813e5c6de46768b4d6999055d6.
 * clang -O0 -ffp-contract=off -Wno-unknown-pragmas c_toolbox/weighted_oo.c -o /tmp/weighted-oo
 */
#include <complex.h>
#include <stdint.h>
#include <inttypes.h>
#include <stdio.h>
#include <string.h>
void calculateOO(double complex *srOptOO, double complex *srOptHO, const double complex *srOptO,
                 const double w, const double complex e, const int srOptSize){
  int i,j;
  double complex tmp;
  #pragma omp parallel for default(shared) private(j,tmp)
  //    private(i,j,tmp,srOptOO)
#pragma loop noalias
  for(j=0;j<2*srOptSize;j++) {
    tmp                            = w * srOptO[j];
    srOptOO[0*(2*srOptSize)+j]    += tmp;      // update O
    srOptOO[1*(2*srOptSize)+j]    += 0.0;      // update 
    srOptHO[j]                    += e * tmp;  // update HO
  }
  
  #pragma omp parallel for default(shared) private(i,j,tmp)
#pragma loop noalias
  for(i=2;i<2*srOptSize;i++) {
    tmp            = w * srOptO[i];
    for(j=0;j<2*srOptSize;j++) {
      srOptOO[i*(2*srOptSize)+j] += w*(srOptO[j])*conj(srOptO[i]); // TBC
      //srOptOO[j+i*(2*srOptSize)] += w*(srOptO[j])*(srOptO[i]); // TBC
    }
  }

  return;
}

static void words(const double complex *v, int n) {
    for (int i=0;i<n;i++) {
        double parts[2]={creal(v[i]),cimag(v[i])};
        for (int j=0;j<2;j++) { uint64_t bits; memcpy(&bits,&parts[j],8); printf("%016" PRIx64 "%c",bits,(i==n-1 && j==1)?'\n':' '); }
    }
}
int main(void) {
    const double weights[]={1e-200,1e200,1.0000000000000002};
    const double values[]={1e200,1e-200,1.2345678901234567};
    for (int c=0;c<3;c++) {
        double complex o[4]={0,0,values[c]+values[c]*I,0};
        double complex oo[16]={0},ho[4]={0};
        calculateOO(oo,ho,o,weights[c],0.0,2);
        double complex w=weights[c]; words(&w,1); words(o,4); words(oo,16);words(ho,4);
    }
}
