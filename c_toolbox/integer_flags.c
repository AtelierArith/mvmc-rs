/* Complete AP/P reader -> C header normalization -> initialization/filter.
 * Untouched real-mode imaginary flags are supplied deterministic zero storage.
 * SR extraction covers selection only; it does not run an MPI/LAPACK solver. */
#include <complex.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "SFMT.h"
#define D_FileNameMax 256
#include "orbital_contracts_upstream.inc"
int NProj=0,NRBM=0,FlagRBM=0,Nneuron=1,NSlater,NOptTrans=0;
int OptFlag[64]={0},AllComplexFlag;
double complex Proj[1],RBM[1],Slater[15],OptTrans[1],ParaQPOptTrans[1];
#include "integer_flag_init_upstream.inc"

int main(int argc,char **argv) {
  if(argc!=4) return 2;
  FILE *ap=fopen(argv[1],"r"),*p=NULL;
  if(!ap) return 2;
  if(argv[2][0]!='-') { p=fopen(argv[2],"r"); if(!p) return 2; }
  int ap_width=0,p_width=0,ap_complex=0,p_complex=0;
  if(!ReadBuffIntCmpFlg(ap,&ap_width,&ap_complex)) return 2;
  if(p && !ReadBuffIntCmpFlg(p,&p_width,&p_complex)) return 2;
  int rank=0,iComplexFlgGutzwiller=0,iComplexFlgJastrow=0,iComplexFlgDH2=0,iComplexFlgDH4=0;
  int iComplexFlgOrbital=ap_complex+p_complex;
#include "integer_flag_complex_upstream.inc"
  NSlater=ap_width+2*p_width;
  if(NSlater>15) return 2;
  char line[256];
  if(!fgets(line,256,ap) || !fgets(line,256,ap)) return 2;
  int idx_data[4][4]={{0}},sgn_data[4][4]={{0}},*idx[4],*sgn[4],count=0;
  for(int i=0;i<4;i++) { idx[i]=idx_data[i]; sgn[i]=sgn_data[i]; }
  int status=GetInfoOrbitalAntiParallel(ap,idx,OptFlag,sgn,&count,0,iComplexFlgOrbital,p!=NULL,1,2,ap_width,"AP");
  fclose(ap);
  if(p) {
    if(!fgets(line,256,p) || !fgets(line,256,p)) return 2;
    status |= GetInfoOrbitalParallel(p,idx,OptFlag,sgn,&count,ap_width,iComplexFlgOrbital,1,1,2,p_width,ap_width,"P");
    fclose(p);
  }
  printf("%d %d %d %d %d %d %d\n",NSlater,ap_complex,p_complex,AllComplexFlag,iComplexFlgOrbital,count,status);
  if(status) return 2;
  for(int i=0;i<2*NSlater;i++) printf("%d ",OptFlag[i]); puts("");
  init_gen_rand((unsigned)strtoul(argv[3],NULL,10)); InitParameter();
  for(int i=0;i<NSlater;i++) {
    double parts[2]={creal(Slater[i]),cimag(Slater[i])};
    for(int k=0;k<2;k++) { uint64_t bits; memcpy(&bits,&parts[k],8); printf("%016llx ",(unsigned long long)bits); }
  }
  puts("");
  for(int i=0;i<624;i++) printf("%u ",gen_rand32()); puts("");
  int nPara=NSlater,pi,si,nSmat,optNum=0,cutNum=0,smatToParaIdx[30];
  double r[30],sDiag,diagCutThreshold=0;
  for(int i=0;i<30;i++) r[i]=1.0;
#include "integer_flag_sr_upstream.inc"
  for(int i=0;i<nSmat;i++) printf("%d ",smatToParaIdx[i]); puts("");
  return 0;
}
