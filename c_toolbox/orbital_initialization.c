
#include <complex.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "SFMT.h"
#define D_AmpMax 4.0
typedef int MPI_Comm;
int NProj=0, NRBM=0, FlagRBM=0, Nneuron=1, NSlater=13, NPara=13, NOptTrans=0;
int AllComplexFlag, FlagShiftDH2=0, FlagShiftDH4=0, FlagShiftGJ=0;
int NGutzwillerIdx=0, FlagOptTrans=0, OptFlag[140];
double complex Proj[37], RBM[27], Slater[15], OptTrans[1], ParaQPOptTrans[1];
double shiftDH2(void) { return 0.0; }
double shiftDH4(void) { return 0.0; }
void shiftGJ(void) {}
int Nsite=2, Nsite2=4, NQPFull=1, NQPFix=1, NSPGaussLeg=1;
int identity[2]={0,1}, ones[2]={1,1};
int *QPOptTrans[1]={identity}, *QPOptTransSgn[1]={ones};
int *QPTrans[1]={identity}, *QPTransSgn[1]={ones};
int idx_row[2]={0,0}, sgn0[2]={1,-1}, sgn1[2]={-1,1};
int *OrbitalIdx[2]={idx_row,idx_row}, *OrbitalSgn[2]={sgn0,sgn1};
double SPGLCosSin[1]={0.25}, SPGLCosCos[1]={0.75}, SPGLSinSin[1]={0.5};
double complex SlaterElm[16];

#include "orbital_parameter_upstream.inc"

void print_slater(void) {
  for(int i=0;i<NSlater;i++) {
    double parts[2]={creal(Slater[i]),cimag(Slater[i])};
    for(int k=0;k<2;k++) { uint64_t bits; memcpy(&bits,&parts[k],8); printf("%016llx ",(unsigned long long)bits); }
  }
  puts("");
}
int main(int argc, char **argv) {
  if(argc==8) {
    NProj=5; NRBM=27; FlagRBM=1; NSlater=4; AllComplexFlag=1;
    init_gen_rand(11272); ReadInitParameter(argv[2]);
    puts("rbm_layout_loaded"); print_slater(); SyncModifiedParameter(0); print_slater();
    for(int i=0;i<624;i++) printf("%u ",gen_rand32()); puts(""); return 0;
  }
  if(argc==7) {
    NSlater=4; init_gen_rand(11272);
    for(int i=0;i<4;i++) Slater[i]=(65.0+i)/64.0-(65.0+i)/128.0*I;
    puts("layout_sync"); print_slater(); SyncModifiedParameter(0); print_slater();
    for(int i=0;i<624;i++) printf("%u ",gen_rand32()); puts(""); return 0;
  }
  if(argc==6) {
    AllComplexFlag=atoi(argv[2]); NRBM=atoi(argv[3]); FlagRBM=1; NSlater=4;
    int mask=atoi(argv[4]); init_gen_rand(11272);
    for(int i=0;i<NRBM;i++) OptFlag[2*i]=1;
    for(int i=0;i<4;i++) OptFlag[2*(NRBM+i)]=(mask>>i)&1;
    printf("%d %d %d\n",AllComplexFlag,NRBM,mask);
    InitParameter(); print_slater(); SyncModifiedParameter(0); print_slater();
    for(int i=0;i<624;i++) printf("%u ",gen_rand32()); puts(""); return 0;
  }
  if(argc==5) {
    NProj=37; NRBM=27; FlagRBM=1; NSlater=4; AllComplexFlag=1;
    int active=atoi(argv[2]); init_gen_rand(11272);
    for(int i=0;i<27;i++) OptFlag[2*(NProj+i)]=active && i%3!=2;
    for(int i=0;i<4;i++) OptFlag[2*(NProj+NRBM+i)]=active && i%2;
    printf("layout %s\n",active ? "parsed" : "inactive");
    InitParameter(); print_slater(); SyncModifiedParameter(0); print_slater();
    for(int i=0;i<624;i++) printf("%u ",gen_rand32()); puts(""); return 0;
  }
  if(argc==4) {
    NSlater=atoi(argv[3]); AllComplexFlag=atoi(argv[2]); init_gen_rand(11272);
    for(int i=0;i<NSlater;i++) {
      int active=NSlater==4 ? i%2 : i%3!=0;
      OptFlag[2*i]=active; OptFlag[2*i+1]=AllComplexFlag && active;
    }
    printf("%d %d 11272\n",NSlater,AllComplexFlag); InitParameter(); print_slater();
    SyncModifiedParameter(0); print_slater();
    for(int i=0;i<624;i++) printf("%u ",gen_rand32()); puts(""); return 0;
  }
  if(argc==2) {
    if(strcmp(argv[1],"shared")==0) {
      Slater[0]=0.3+0.2*I; Slater[12]=8.0;
      SyncModifiedParameter(0); print_slater(); UpdateSlaterElm_fcmp();
      for(int i=0;i<16;i++) {
        double parts[2]={creal(SlaterElm[i]),cimag(SlaterElm[i])};
        for(int k=0;k<2;k++) { uint64_t bits; memcpy(&bits,&parts[k],8); printf("%016llx ",(unsigned long long)bits); }
      }
      puts(""); return 0;
    }
    init_gen_rand(1); ReadInitParameter(argv[1]); print_slater();
    SyncModifiedParameter(0); print_slater();
    for(int i=0;i<624;i++) printf("%u ",gen_rand32()); puts(""); return 0;
  }
  int seeds[2]={1,11272};
  for(int seed=0;seed<2;seed++) for(int complex_mode=0;complex_mode<2;complex_mode++) for(int fixed_tail=0;fixed_tail<2;fixed_tail++) {
    init_gen_rand(seeds[seed]); AllComplexFlag=complex_mode;
    for(int i=0;i<13;i++) { OptFlag[2*i]=!(fixed_tail && i>=9); OptFlag[2*i+1]=complex_mode && OptFlag[2*i]; }
    InitParameter();
    printf("%d %d %d\n",seeds[seed],complex_mode,fixed_tail);
    print_slater();
    SyncModifiedParameter(0); print_slater();
    for(int i=0;i<624;i++) printf("%u ",gen_rand32()); puts("");
  }
}
