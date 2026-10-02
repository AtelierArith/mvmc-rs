/* Actual DH readers and SetFlagShift. Unwritten imaginary cells are zeroed.
 * No full C executable, gauge arithmetic, sampling or MPI is exercised. */
#include <stdio.h>
#include <stdlib.h>
#define D_FileNameMax 256
#include "orbital_contracts_upstream.inc"
#include "dh_flag_reader_upstream.inc"
int NGutzwillerIdx,NJastrowIdx,NDoublonHolon2siteIdx,NDoublonHolon4siteIdx;
int OptFlag[64]={0},FlagShiftGJ,FlagShiftDH2,FlagShiftDH4;
#include "flag_shift_upstream.inc"

int main(int argc,char **argv) {
  if(argc==4) {
    int family=atoi(argv[1]),complex=atoi(argv[2]),width=family==2?6:10;
    FILE *fp=fopen(argv[3],"r"); if(!fp) return 2;
    int values[12]={0},*indices[1]={values},count=0;
    int status=family==2?GetInfoDH2(fp,indices,OptFlag,complex,&count,2,3,1,"DH2"):
                        GetInfoDH4(fp,indices,OptFlag,complex,&count,2,3,1,"DH4");
    fclose(fp);
    printf("%d %d\n",status,count);
    for(int i=0;i<2*(width+2);i++) printf("%d ",OptFlag[i]); puts("");
    for(int i=0;i<3*family;i++) printf("%d ",values[i]); puts("");
    return status;
  }
  if(argc!=1) return 2;
  if(scanf("%d %d %d %d",&NGutzwillerIdx,&NJastrowIdx,
      &NDoublonHolon2siteIdx,&NDoublonHolon4siteIdx)!=4) return 2;
  int n=NGutzwillerIdx+NJastrowIdx+6*NDoublonHolon2siteIdx+10*NDoublonHolon4siteIdx;
  if(n>32) return 2;
  for(int i=0;i<2*n;i++) if(scanf("%d",&OptFlag[i])!=1) return 2;
  SetFlagShift(); printf("%d %d %d\n",FlagShiftGJ,FlagShiftDH2,FlagShiftDH4);
  return 0;
}
