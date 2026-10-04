
#include <assert.h>
#include <complex.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
int NMPTrans, APFlag, NQPFix, NQPFull, NSPGaussLeg=1, NSPStot=0;
int NQPOptTrans, NOptTrans, FlagOptTrans;
double complex ParaQPTrans[2], OptTrans[2];
double complex QPFixWeight[4], QPFullWeight[8];
double complex SPGLCos[1], SPGLSin[1], SPGLCosSin[1], SPGLCosCos[1], SPGLSinSin[1];
double scratch[2];
int scratch_offset;
void RequestWorkSpaceDouble(int n) { assert(n == 2); scratch_offset=0; }
double *GetWorkSpaceDouble(int n) { double *p=scratch+scratch_offset; scratch_offset+=n; return p; }
void ReleaseWorkSpaceDouble(void) {}
void GaussLeg(double, double, double *, double *, int);
double LegendrePoly(double, int);
void UpdateQPWeight(void);

#include "projection_qp_upstream.inc"

void print_bits(double complex *v, int n) {
  for(int j=0;j<n;j++) {
    double parts[2]={creal(v[j]),cimag(v[j])};
    for(int k=0;k<2;k++) { uint64_t bits; memcpy(&bits, &parts[k], 8); printf("%016llx ",(unsigned long long)bits); }
  }
  puts("");
}
int main(void) {
  int counts[]={0,1,-1,2,-2};
  for(int c=0;c<5;c++) for(int opt=0;opt<=2;opt+=2) {
    int IdxMPTrans=0, bufInt[]={counts[c]};
    NQPOptTrans=opt?opt:1; NOptTrans=opt; FlagOptTrans=opt>0;
    ParaQPTrans[0]=1.0+0.25*I; ParaQPTrans[1]=-0.5-0.125*I;
    OptTrans[0]=0.75+0.5*I; OptTrans[1]=-0.25+0.75*I;

#include "projection_count_conversion.inc"

    InitQPWeight();
    printf("%d %d %d %d %d %d\n",counts[c],NMPTrans,APFlag,NQPFix,NQPFull,opt);
    print_bits(QPFixWeight,NQPFix); print_bits(QPFullWeight,NQPFull);
    print_bits(SPGLCos,1); print_bits(SPGLSin,1);
  }
  return 0;
}
