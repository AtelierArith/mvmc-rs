/* Standalone plumbing for actual C General readers and FSZ kernels.
 * The probe initializes untouched matrix diagonals to zero explicitly;
 * it does not establish upstream malloc/diagonal initialization behavior. */
#include <complex.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define D_FileNameMax 256
#include "orbital_contracts_upstream.inc"
#include "general_reader_upstream.inc"

int Nsite, Nsite2, NSlater, Nsize=4, NQPFull=2, NQPFix=2;
int NSPGaussLeg=1, NMPTrans=2, NQPOptTrans=1;
int **OrbitalIdx, **OrbitalSgn, **QPTrans, **QPTransSgn;
int **QPOptTrans, **QPOptTransSgn;
double complex *Slater, *SlaterElm, *PfM, *InvM, *QPFullWeight;
static int int_work[4096], int_offset;
static double complex complex_work[4096];
void RequestWorkSpaceInt(int n) { if(n>4096) abort(); int_offset=0; }
void RequestWorkSpaceComplex(int n) { if(n>4096) abort(); }
int *GetWorkSpaceInt(int n) { int *p=int_work+int_offset; int_offset+=n; return p; }
double complex *GetWorkSpaceComplex(int n) { (void)n; return complex_work; }
void ReleaseWorkSpaceInt(void) {}
void ReleaseWorkSpaceComplex(void) {}
#include "general_fsz_upstream.inc"

static void print_complex(const double complex *array, int n) {
  for(int k=0;k<n;k++) {
    double real=creal(array[k]), imag=cimag(array[k]); uint64_t r,i;
    memcpy(&r,&real,8); memcpy(&i,&imag,8);
    printf("%016llx %016llx ",(unsigned long long)r,(unsigned long long)i);
  }
  puts("");
}

int main(int argc,char **argv) {
  if(argc!=4) return 2;
  Nsite=atoi(argv[1]); Nsite2=2*Nsite;
  int anti=atoi(argv[2]);
  if(Nsite<1 || Nsite>8) return 2;
  FILE *fp=fopen(argv[3],"r"); if(!fp) return 2;
  int complex_flag=0;
  char *header=ReadBuffIntCmpFlg(fp,&NSlater,&complex_flag);
  if(!header) { printf("0 %d %d 1\n",NSlater,complex_flag); fclose(fp); return 0; }
  char line[256];
  if(!fgets(line,256,fp) || !fgets(line,256,fp)) return 2;
  int idx_buffer[16][16]={{0}},sgn_buffer[16][16]={{0}},*idx[16],*sgn[16];
  int flags[256]={0},count=0;
  for(int i=0;i<16;i++) { idx[i]=idx_buffer[i]; sgn[i]=sgn_buffer[i]; }
  int status=GetInfoOrbitalGeneral(fp,idx,flags,sgn,&count,0,complex_flag,1,anti,Nsite,NSlater,"General");
  printf("1 %d %d %d\n",NSlater,complex_flag,status); fclose(fp);
  if(status) return 0;
  for(int i=0;i<Nsite2;i++) for(int j=0;j<Nsite2;j++) printf("%d ",idx[i][j]); puts("");
  for(int i=0;i<Nsite2;i++) for(int j=0;j<Nsite2;j++) printf("%d ",sgn[i][j]); puts("");
  for(int k=0;k<NSlater;k++) printf("%d ",flags[2*k]); puts("");
  OrbitalIdx=idx; OrbitalSgn=sgn;
  int map_buffer[2][8],sign_buffer[2][8],*maps[2],*signs[2],*optmap[1],*optsign[1];
  for(int qp=0;qp<2;qp++) {
    maps[qp]=map_buffer[qp]; signs[qp]=sign_buffer[qp];
    for(int i=0;i<Nsite;i++) {
      maps[qp][i]=(i+qp)%Nsite;
      signs[qp][i]=(anti && qp && i%2==0)?-1:1;
    }
  }
  optmap[0]=maps[0]; optsign[0]=signs[0];
  QPTrans=maps; QPTransSgn=signs; QPOptTrans=optmap; QPOptTransSgn=optsign;
  double complex coeff[128],elm[512],pf[2],inverse[32],weight[2],result[256];
  if(NSlater>128) return 2;
  for(int k=0;k<NSlater;k++) coeff[k]=(k+1)*0.125+(complex_flag?((k%3)-1)*0.0625:0.0)*I;
  for(int qp=0;qp<2;qp++) {
    pf[qp]=(qp+1)*0.5+(qp?0.25:-0.125)*I;
    weight[qp]=qp?-0.375:1.0;
    for(int i=0;i<4;i++) for(int j=0;j<4;j++) {
      int sign=i<j?1:i>j?-1:0,lo=i<j?i:j,hi=i<j?j:i;
      inverse[16*qp+4*i+j]=sign*((lo+hi+qp+1)*0.125+(hi-lo+qp)*0.0625*I);
    }
  }
  double complex ip=1.5+0.5*I;
  Slater=coeff; SlaterElm=elm; PfM=pf; InvM=inverse; QPFullWeight=weight;
  int electrons[4]={0,1,0,1},spins[4]={0,0,1,1};
  UpdateSlaterElm_fsz(); SlaterElmDiff_fsz(result,ip,electrons,spins);
  print_complex(coeff,NSlater); print_complex(pf,2); print_complex(inverse,32);
  print_complex(&ip,1); print_complex(elm,2*Nsite2*Nsite2); print_complex(result,2*NSlater);
  return 0;
}
