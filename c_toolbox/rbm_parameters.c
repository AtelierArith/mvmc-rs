#include <complex.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "SFMT.h"
#define D_FileNameMax 256
int Nsite;
int NProj=2, NRBM=0, FlagRBM=0, Nneuron=1, NSlater=4, NOptTrans=0;
int AllComplexFlag=0, OptFlag[4096];
double complex Proj[2], RBM[1024], Slater[4], OptTrans[1], ParaQPOptTrans[1];
#include "rbm_contracts_upstream.inc"
#include "rbm_parameters_upstream.inc"

int main(int argc, char **argv) {
  if (argc != 14 && argc != 15) return 2;
  Nsite=atoi(argv[1]);
  int hidden=atoi(argv[2]);
  Nneuron=atoi(argv[3]);
  unsigned seed=(unsigned)strtoul(argv[4], NULL, 10);
  int widths[9]={0};
  OptFlag[0]=3; OptFlag[2]=-2;
  for (int section=0;section<9;section++) {
    if (strcmp(argv[5+section], "-")==0) continue;
    FILE *fp=fopen(argv[5+section],"r");
    if (!fp) return 2;
    int complex_flag=0;
    if (!ReadBuffIntCmpFlg(fp,&widths[section],&complex_flag)) return 2;
    char line[256];
    if (!fgets(line,sizeof(line),fp) || !fgets(line,sizeof(line),fp)) return 2;
    if (widths[section]<1 || NRBM+widths[section]>1024 || Nsite<1 || Nsite>8 || hidden<1 || hidden>8) return 2;
    int layer[16]={0}, values[16][8]={{0}}, *rows[16], count=0, status;
    for (int i=0;i<16;i++) rows[i]=values[i];
    if (section==2) status=GetInfoGeneralRBM_Layer(fp,layer,OptFlag,complex_flag,&count,NProj+NRBM,Nsite,widths[section],"RBM");
    else if (section<6) status=GetInfoRBM_Layer(fp,layer,OptFlag,complex_flag,&count,NProj+NRBM,section<3 ? Nsite : hidden,widths[section],"RBM");
    else if (section==8) status=GetInfoGeneralRBM_PhysHidden(fp,rows,OptFlag,complex_flag,&count,NProj+NRBM,Nsite,hidden,widths[section],"RBM");
    else status=GetInfoRBM_PhysHidden(fp,rows,OptFlag,complex_flag,&count,NProj+NRBM,Nsite,hidden,widths[section],"RBM");
    fclose(fp);
    if (status) return 2;
    NRBM+=widths[section];
  }
  FlagRBM=NRBM>0;
  int flags[4]={1,2,-1,3};
  for(int i=0;i<4;i++) OptFlag[2*(NProj+NRBM+i)]=flags[i];
  init_gen_rand(seed);
  InitParameter();
  if (argc==15) ReadInitParameter(argv[14]);
  for(int i=0;i<2*(NProj+NRBM+NSlater);i++) printf("%d ",OptFlag[i]);
  puts("");
  for(int i=0;i<NProj+NRBM+NSlater;i++) {
    double complex value=i<NProj ? Proj[i] : i<NProj+NRBM ? RBM[i-NProj] : Slater[i-NProj-NRBM];
    double parts[2]={creal(value),cimag(value)};
    for(int k=0;k<2;k++) { uint64_t bits; memcpy(&bits,&parts[k],8); printf("%016llx ",(unsigned long long)bits); }
  }
  puts("");
  for(int i=0;i<624;i++) printf("%u ",gen_rand32());
  puts("");
  return 0;
}
