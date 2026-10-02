#include <complex.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "SFMT.h"

int NProj, NRBM, FlagRBM, NSlater, NOptTrans;
double complex Proj[37], RBM[27], Slater[13], OptTrans[3];
#include "initial_records_upstream.inc"

void print_parameters(double complex *values, int count) {
  for (int i=0;i<count;i++) {
    double parts[2]={creal(values[i]),cimag(values[i])};
    for(int k=0;k<2;k++) {
      uint64_t bits; memcpy(&bits,&parts[k],8);
      printf("%016llx ",(unsigned long long)bits);
    }
  }
}

int main(int argc, char **argv) {
  if(argc!=6) return 2;
  NProj=atoi(argv[2]); NRBM=atoi(argv[3]); NSlater=atoi(argv[4]); NOptTrans=atoi(argv[5]);
  FlagRBM=NRBM>0;
  for(int i=0;i<NProj;i++) Proj[i]=99.0+99.0*I;
  for(int i=0;i<NRBM;i++) RBM[i]=99.0+99.0*I;
  for(int i=0;i<NSlater;i++) Slater[i]=99.0+99.0*I;
  for(int i=0;i<NOptTrans;i++) OptTrans[i]=99.0+99.0*I;
  init_gen_rand(1);
  ReadInitParameter(argv[1]);
  print_parameters(Proj,NProj); print_parameters(RBM,NRBM);
  print_parameters(Slater,NSlater); print_parameters(OptTrans,NOptTrans); puts("");
  for(int i=0;i<624;i++) printf("%u ",gen_rand32()); puts("");
  return 0;
}
