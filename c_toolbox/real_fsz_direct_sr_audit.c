/* Optional standalone original-C direct SR audit; never a Cargo dependency.
 * Original files are included unchanged. See real_fsz_direct_sr_audit.md. */
#include <complex.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#define _STCOPT_HEADER
#define _STCOPT_DPOSV
typedef int MPI_Comm;
#define MPI_INT 0
static int MPI_Comm_rank(int c,int *r) { (void)c; *r=0; return 0; }
static int MPI_Comm_size(int c,int *s) { (void)c; *s=1; return 0; }
static int MPI_Bcast(void *p,int n,int t,int r,int c) {
    (void)p;(void)n;(void)t;(void)r;(void)c; return 0;
}
static void StartTimer(int t) { (void)t; }
static void StopTimer(int t) { (void)t; }
int NPara,SROptSize,AllComplexFlag=0;
int *OptFlag;
double DSROptStaDel,DSROptStepDt,DSROptRedCut;
double complex *SROptOO,*SROptHO,*Para;
double *SROptOO_real;
FILE *FileSRinfo;
int NProj,NGutzwillerIdx,NJastrowIdx,NSlater;
int NDoublonHolon2siteIdx=0,NDoublonHolon4siteIdx=0;
int FlagShiftGJ,FlagShiftDH2,FlagShiftDH4,FlagOptTrans=0,NOptTrans=0;
double complex *Proj,*Slater,*OptTrans;
#define D_AmpMax 4.0
void shiftGJ(void);
double shiftDH2(void);
double shiftDH4(void);
#include "real_fsz_sync_upstream.inc"
void stcOptInit(double *,double *,int,const int *);
int stcOptMain(double *,int,const int *,MPI_Comm);
extern void dposv_(char *,int *,int *,double *,int *,double *,int *,int *);
static void vector(const char *name,const double *a,int n) {
    printf("%s %d",name,n);
    for(int i=0;i<n;i++) printf(" %.17g",a[i]);
    puts("");
}
static void observed_dposv(char *u,int *n,int *nr,double *a,int *ld,
                           double *b,int *ldb,int *info) {
    vector("matrix",a,(*n)*(*n)); vector("rhs",b,*n);
    dposv_(u,n,nr,a,ld,b,ldb,info);
    printf("lapack_info %d\n",*info); vector("increment",b,*n);
}
#define M_DPOSV observed_dposv
#define stcOptMain reviewed_original_main
#include "stcopt_dposv.c"
#undef stcOptMain
int stcOptMain(double *g,int n,const int *mapping,MPI_Comm c) {
    printf("active %d",n);
    for(int i=0;i<n;i++) printf(" %d",mapping[i]);
    puts("");
    return reviewed_original_main(g,n,mapping,c);
}
#include "stcopt.c"
static double complex pair(FILE *f) {
    double re,im;
    if(fscanf(f,"%lf %lf",&re,&im)!=2 || !isfinite(re) || !isfinite(im)) exit(3);
    return re+I*im;
}
int main(int argc,char **argv) {
    if(argc!=3) return 2;
    FILE *f=fopen(argv[1],"r"); if(!f) return 2;
    int unknown=atoi(argv[2]);
    if(fscanf(f,"%d %lf %lf %lf",&NPara,&DSROptStaDel,&DSROptStepDt,&DSROptRedCut)!=4 || NPara<=0) return 3;
    if(fscanf(f,"%d %d %d",&NGutzwillerIdx,&NJastrowIdx,&NSlater)!=3) return 3;
    NProj=NGutzwillerIdx+NJastrowIdx;
    if(NProj+NSlater!=NPara || NGutzwillerIdx<0 || NJastrowIdx<0 || NSlater<=0) return 3;
    SROptSize=NPara+1; int s=SROptSize;
    OptFlag=calloc(2*NPara,sizeof(int)); Para=calloc(NPara,sizeof(*Para));
    Proj=Para; Slater=Para+NProj;
    SROptOO_real=calloc(s*(s+2),sizeof(double));
    SROptOO=calloc(2*s*(2*s+2),sizeof(*SROptOO)); SROptHO=SROptOO+4*s*s;
    for(int i=0;i<2*NPara;i++) {
        int mask,value;
        if(fscanf(f,"%d %d",&mask,&value)!=2 || (mask!=0 && mask!=1)) return 3;
        OptFlag[i]=mask?value:unknown;
    }
    for(int i=0;i<NPara;i++) Para[i]=pair(f);
    for(int i=0;i<s*(s+2);i++) {
        double complex z=pair(f); if(cimag(z)!=0) return 3;
        SROptOO_real[i]=creal(z);
    }
    /* Separate Julia arrays -> the original C contiguous OO/HO/O allocation.
     * StochasticOpt's original complete real-shadow copy includes this HO tail. */
    for(int i=0;i<s;i++) {
        double complex z=pair(f); if(cimag(z)!=0) return 3;
        SROptOO_real[s*s+i]=creal(z);
    }
    int ch; while((ch=fgetc(f))!=EOF) if(ch!=' ' && ch!='\n' && ch!='\t' && ch!='\r') return 3;
    fclose(f); FileSRinfo=tmpfile(); if(!FileSRinfo) return 2;
    SetFlagShift();
    int info=StochasticOpt(0); printf("status %d\n",info);
    printf("parameters_before_sync %d",NPara);
    for(int i=0;i<NPara;i++) printf(" %.17g %.17g",creal(Para[i]),cimag(Para[i]));
    puts("");
    if(info==0) SyncModifiedParameter(0);
    printf("parameters %d",NPara);
    for(int i=0;i<NPara;i++) printf(" %.17g %.17g",creal(Para[i]),cimag(Para[i]));
    puts(""); rewind(FileSRinfo); char row[1024];
    if(!fgets(row,sizeof(row),FileSRinfo)) return 3;
    printf("sr_info %s",row);
    return info?4:0;
}
