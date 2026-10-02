#include <stdio.h>
#define D_FileNameMax 256
#define D_CharTmpReadDef 256
#include "reader_audit_upstream.inc"
int main(void) {
 FILE *fp=tmpfile(); int width=0,complex_mode=0;
 fputs("===\nNChargeRBM_PhysLayerIdx 97\nComplexType 0\n===\n===\n",fp); rewind(fp);
 int ok=ReadBuffIntCmpFlg(fp,&width,&complex_mode)!=NULL;
 printf("header_ok=%d declared=%d complex=%d\n",ok,width,complex_mode);
 fclose(fp); fp=tmpfile(); fputs("0 0\n1 2\n2 0\n",fp);
 for(int i=0;i<width;i++) fprintf(fp,"%d 1\n",i); rewind(fp);
 int mappings[3]={0}, flags[194]={0},count=0;
 int result=GetInfoRBM_Layer(fp,mappings,flags,complex_mode,&count,0,3,width,"complete_sparse_charge");
 printf("reader_status=%d opt_count=%d mapped=%d,%d,%d\n",result,count,mappings[0],mappings[1],mappings[2]);
 fclose(fp); return result;
}
