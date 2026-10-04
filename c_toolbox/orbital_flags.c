#include <stdio.h>
#define D_FileNameMax 256
#define D_CharTmpReadDef 256
#include "reader_audit_upstream.inc"
int main(void) {
 int ap[4]={0},p[4]={0},count=0; FILE *fp=tmpfile();
 fputs("1 0\n0 1\n",fp);rewind(fp);GetInfoOpt(fp,ap,1,&count,0);fclose(fp);
 printf("AP row-order flags: %d %d %d %d\n",ap[0],ap[1],ap[2],ap[3]);
 fp=tmpfile();fputs("0 1\n",fp);rewind(fp);count=0;GetInfoOptOrbitalParalell(fp,p,2,&count,0);fclose(fp);
 printf("P aggregate-complex=2 flags: %d %d %d %d\n",p[0],p[1],p[2],p[3]);
 return 0;
}
