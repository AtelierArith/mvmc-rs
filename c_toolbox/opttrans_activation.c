#include <stdio.h>
#define D_FileNameMax 256
#define D_CharTmpReadDef 256
#include "reader_audit_upstream.inc"
int main(void) {
 for(int enabled=0;enabled<=1;enabled++) {
  int map0[2]={0},map1[2]={0},sgn0[2]={0},sgn1[2]={0};
  int *maps[2]={map0,map1},*signs[2]={sgn0,sgn1};
  int flags[20],count=0; double weights[2]={-7,-7};for(int i=0;i<20;i++) flags[i]=-7;
  FILE *fp=tmpfile(); fputs("0 0.5\n1 0.75\n0 0 0 -1\n0 1 1 1\n1 0 1 1\n1 1 0 -1\n",fp);rewind(fp);
  int status=GetInfoOptTrans(fp,maps,weights,flags,signs,enabled,&count,3,0,2,2,"probe");fclose(fp);
  printf("enabled=%d status=%d opt_count=%d weights=%g,%g defined_flag_writes=",enabled,status,count,weights[0],weights[1]);
  for(int i=0;i<20;i++) if(flags[i]!=-7) printf("%d:%d,",i,flags[i]);puts("");
 }
}
