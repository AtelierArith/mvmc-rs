
#include <assert.h>
#include <ctype.h>
#include <complex.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
typedef int MPI_Comm;
#define D_FileNameMax 256
#include "readdef.h"

#include "orbital_readdef_upstream.inc"

int main(int argc, char **argv) {
  if (argc == 2) {
    FILE *fp=tmpfile(); assert(fp); fputs("0 0\n",fp); rewind(fp);
    int flags[4]={0}, count=0;
    assert(GetInfoOptOrbitalParalell(fp,flags,1,&count,0)==1);
    assert(count==2 && flags[0]==0 && flags[1]==1 && flags[2]==0 && flags[3]==1);
    fclose(fp); puts("complex P inactive real flags: 0 1 0 1"); return 0;
  }
  assert(argc == 3);
  char names[KWIdxInt_end][D_CharTmpReadDef], defname[D_CharTmpReadDef];
  int iKWidx;
  assert(GetFileName(argv[1], names) == 0);
  cFileNameListFile = names;

#include "orbital_keyword_loop.inc"

      printf("%s ", cKWListOfFileNameList[iKWidx]);
  }
  puts("");
  int bufInt[ParamIdxInt_End]={0}, iNOrbitalAntiParallel=0, iNOrbitalParallel=0;
  int iFlgOrbitalAntiParallel=0, iFlgOrbitalParallel=0, iOrbitalComplex=0;
  char *cerr; FILE *fp;

#include "orbital_keyword_loop.inc"

      if (iKWidx!=KWOrbital && iKWidx!=KWOrbitalAntiParallel && iKWidx!=KWOrbitalParallel) continue;
      fp=fopen(defname,"r"); assert(fp);
      switch(iKWidx) {

#include "orbital_ap_headers.inc"

      }
      assert(cerr); fclose(fp);
  }
  int width=bufInt[IdxNOrbit], ap=iNOrbitalAntiParallel, p=iNOrbitalParallel;
  printf("%d %d\n",ap,width);
  int idx_data[4][4]={{0}}, sign_data[4][4]={{0}}, *idx[4], *sign[4], flags[26]={0};
  for(int i=0;i<4;i++) {idx[i]=idx_data[i];sign[i]=sign_data[i];}
  int count=0, info=0, boundary=atoi(argv[2]); char buf[256];
  fp=fopen(names[KWOrbital][0]?names[KWOrbital]:names[KWOrbitalAntiParallel], "r");
  for(int i=0;i<5;i++) assert(fgets(buf,sizeof(buf),fp));
  info=GetInfoOrbitalAntiParallel(fp,idx,flags,sign,&count,0,0,1,boundary,2,ap,"AP"); fclose(fp);
  fp=fopen(names[KWOrbitalParallel],"r");
  for(int i=0;i<5;i++) assert(fgets(buf,sizeof(buf),fp));
  info+=GetInfoOrbitalParallel(fp,idx,flags,sign,&count,ap,0,1,boundary,2,p,ap,"P"); fclose(fp);
  assert(info==0 && count==width);
  for(int i=0;i<4;i++) for(int j=0;j<4;j++) printf("%d ",idx[i][j]); puts("");
  for(int i=0;i<4;i++) for(int j=0;j<4;j++) printf("%d ",sign[i][j]); puts("");
  for(int i=0;i<2*width;i++) printf("%d ",flags[i]); puts("");
}
