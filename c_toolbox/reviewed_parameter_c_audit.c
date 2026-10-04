/* Optional declared-slot C oracle. See reviewed_parameter_c_audit.md.
 * Reuses actual C parameter/overlay/sync and SFMT functions; no Cargo calls. */
#include <complex.h>
#include <math.h>
#include <stdint.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "SFMT.c"
typedef int MPI_Comm;
#define MPI_COMM_WORLD 0
#define D_FileNameMax 4096
#define D_AmpMax 4.0
static void MPI_Comm_rank(int comm, int *rank) { (void)comm; *rank=0; }
static void MPI_Abort(int comm, int code) { (void)comm; exit(code); }
#include "readdef.h"
int NProj, NRBM, NSlater, NOptTrans, NPara, FlagRBM, AllComplexFlag, Nsite;
int NGutzwillerIdx, NJastrowIdx, NDoublonHolon2siteIdx, NDoublonHolon4siteIdx;
int NChargeRBM_PhysLayerIdx, NSpinRBM_PhysLayerIdx, NGeneralRBM_PhysLayerIdx;
int NChargeRBM_HiddenLayerIdx, NSpinRBM_HiddenLayerIdx, NGeneralRBM_HiddenLayerIdx;
int NChargeRBM_PhysHiddenIdx, NSpinRBM_PhysHiddenIdx, NGeneralRBM_PhysHiddenIdx;
int NRBM_PhysLayerIdx, NRBM_HiddenLayerIdx, Nneuron;
int iFlgOrbitalGeneral, iNOrbitalAntiParallel, iNOrbitalParallel, NOrbitalIdx;
int FlagShiftDH2, FlagShiftDH4, FlagShiftGJ, FlagOptTrans, OptFlag[8192];
double complex Proj[4096], RBM[4096], Slater[4096], OptTrans[4096], ParaQPOptTrans[4096];
char filenames[KWIdxInt_end][D_CharTmpReadDef];
static int ReadDefFileError(char *path) { fprintf(stderr,"cannot read %s\n",path); return 1; }
double shiftDH2(void);
double shiftDH4(void);
void shiftGJ(void);
static unsigned long draw_count;
static double observed_real2(void) { ++draw_count; return genrand_real2(); }
#define genrand_real2 observed_real2
#include "reviewed_parameter_c_upstream.inc"
#undef genrand_real2
static int mapped[4096];
static FILE *table;
static int sequence;
static int acquisition_group, acquisition_groups;

static void checkpoint(const char *dir, const char *stage, int draws) {
    ++sequence;
    int index=0;
    double complex *arrays[]={Proj,RBM,Slater,OptTrans};
    int lengths[]={NProj,NRBM,NSlater,NOptTrans};
    for(int b=0;b<4;b++) for(int i=0;i<lengths[b];i++,index++)
        fprintf(table,"%d\t%s\t%d\t%d\t%.18e\t%.18e\t%d\t%d\n",
            sequence,stage,index,mapped[index],creal(arrays[b][i]),cimag(arrays[b][i]),OptFlag[2*index],OptFlag[2*index+1]);
    char path[8192];
    char grouped_stage[128];
    if(acquisition_groups>1) {
        snprintf(grouped_stage,sizeof(grouped_stage),"group-%d-%s",acquisition_group,stage);
        stage=grouped_stage;
    }
    snprintf(path,sizeof(path),"%s/%s-state.txt",dir,stage);
    FILE *state=fopen(path,"w"); if(!state) exit(10);
    for(int i=0;i<N;i++) for(int j=0;j<4;j++) fprintf(state,"%u ",sfmt[i].u[j]);
    fprintf(state,"\n%d\n%lu\n",idx,draw_count); fclose(state);
    snprintf(path,sizeof(path),"%s/%s-parameters.txt",dir,stage);
    FILE *file=fopen(path,"w"); if(!file) exit(10);
    double complex *blocks[]={Proj,RBM,Slater,OptTrans};
    int sizes[]={NProj,NRBM,NSlater,NOptTrans};
    for(int b=0;b<4;b++) for(int i=0;i<sizes[b];i++)
        fprintf(file,"% .18e % .18e\n",creal(blocks[b][i]),cimag(blocks[b][i]));
    fclose(file);
    snprintf(path,sizeof(path),"%s/%s-next624.txt",dir,stage);
    file=fopen(path,"w"); if(!file) exit(10);
    w128_t saved[N]; memcpy(saved,sfmt,sizeof(saved)); int saved_idx=idx;
    for(int i=0;i<624;i++) fprintf(file,"%u ",gen_rand32());
    fputc('\n',file); fclose(file);
    memcpy(sfmt,saved,sizeof(saved)); idx=saved_idx;
    snprintf(path,sizeof(path),"%s/%s-draw-count.txt",dir,stage);
    file=fopen(path,"w"); if(!file) exit(10);
    fprintf(file,"%lu\n",draw_count); fclose(file);
}

int main(int argc,char **argv) {
    if(argc!=6) return 2;
    acquisition_groups=atoi(argv[5]);
    if(acquisition_groups<1 || acquisition_groups>100) return 2;
    FILE *spec=fopen(argv[1],"r"); if(!spec) return 3;
    int *widths[]={&NGutzwillerIdx,&NJastrowIdx,&NDoublonHolon2siteIdx,&NDoublonHolon4siteIdx,
        &NChargeRBM_PhysLayerIdx,&NSpinRBM_PhysLayerIdx,&NGeneralRBM_PhysLayerIdx,
        &NChargeRBM_HiddenLayerIdx,&NSpinRBM_HiddenLayerIdx,&NGeneralRBM_HiddenLayerIdx,
        &NChargeRBM_PhysHiddenIdx,&NSpinRBM_PhysHiddenIdx,&NGeneralRBM_PhysHiddenIdx,&NSlater,&NOptTrans};
    for(int i=0;i<15;i++) if(fscanf(spec,"%d",widths[i])!=1 || *widths[i]<0) return 4;
    int mode;
    if(fscanf(spec,"%d%d%d%d",&AllComplexFlag,&Nneuron,&Nsite,&mode)!=4 || Nsite<1 || (mode!=0 && mode!=1)) return 4;
    NProj=NGutzwillerIdx+NJastrowIdx+6*NDoublonHolon2siteIdx+10*NDoublonHolon4siteIdx;
    NRBM_PhysLayerIdx=NChargeRBM_PhysLayerIdx+NSpinRBM_PhysLayerIdx+NGeneralRBM_PhysLayerIdx;
    NRBM_HiddenLayerIdx=NChargeRBM_HiddenLayerIdx+NSpinRBM_HiddenLayerIdx+NGeneralRBM_HiddenLayerIdx;
    NRBM=NRBM_PhysLayerIdx+NRBM_HiddenLayerIdx+NChargeRBM_PhysHiddenIdx+NSpinRBM_PhysHiddenIdx+NGeneralRBM_PhysHiddenIdx;
    NPara=NProj+NRBM+NSlater+NOptTrans;
    if(NPara>4096 || NSlater<1) return 4;
    NOrbitalIdx=NSlater; FlagRBM=NRBM>0; FlagOptTrans=NOptTrans>0;
    for(int i=0;i<2*NPara;i++) OptFlag[i]=INT_MIN;
    int offset=0, count=0;
    for(int section=0;section<14;section++) {
        int width=*widths[section]*(section==2?6:section==3?10:1);
        int complex_flag=0;
        if(fscanf(spec,"%d",&complex_flag)!=1 || (complex_flag!=0 && complex_flag!=1)) return 4;
        FILE *records=tmpfile(); if(!records) return 4;
        for(int i=0;i<width;i++) {
            int raw;
            if(fscanf(spec,"%d",&raw)!=1) return 4;
            fprintf(records,"%d %d\n",i,raw);
        }
        rewind(records);
        if(GetInfoOpt(records,OptFlag,complex_flag,&count,offset)!=width) return 4;
        fclose(records); offset+=width;
    }
    for(int i=0;i<NPara;i++) if(fscanf(spec,"%d",mapped+i)!=1 || (mapped[i]!=0 && mapped[i]!=1)) return 4;
    for(int i=0;i<NOptTrans;i++) {
        double weight;
        if(fscanf(spec,"%lf",&weight)!=1) return 4;
        ParaQPOptTrans[i]=weight;
    }
    char optpath[4096];
    if(fscanf(spec,"%4095s",optpath)!=1) return 4;
    if(NOptTrans>0) {
        FILE *opt=fopen(optpath,"r"); if(!opt) return 4;
        char header[8192];
        for(int i=0;i<5;i++) if(!fgets(header,sizeof(header),opt)) return 4;
        int **indices=calloc(NOptTrans,sizeof(*indices));
        int **signs=calloc(NOptTrans,sizeof(*signs));
        double *weights=calloc(NOptTrans,sizeof(*weights));
        if(!indices || !signs || !weights) return 4;
        for(int i=0;i<NOptTrans;i++) {
            indices[i]=calloc(Nsite,sizeof(**indices)); signs[i]=calloc(Nsite,sizeof(**signs));
            if(!indices[i] || !signs[i]) return 4;
        }
        if(GetInfoOptTrans(opt,indices,weights,OptFlag,signs,FlagOptTrans,&count,
            NProj+NSlater,0,Nsite,NOptTrans,optpath)!=0) return 4;
        fclose(opt);
        for(int i=0;i<NOptTrans;i++) {
            if(weights[i]!=creal(ParaQPOptTrans[i])) return 4;
            free(indices[i]); free(signs[i]);
        }
        free(indices); free(signs); free(weights);
    }
    cFileNameListFile=filenames;
    int key; char path[4096];
    while(fscanf(spec,"%d %4095s",&key,path)==2) {
        if(key<KWInGutzwiller || key>=KWIdxInt_end) return 4;
        if(strlen(path)>=D_CharTmpReadDef) return 4;
        strcpy(filenames[key],path);
    }
    fclose(spec);
    char maskpath[8192]; snprintf(maskpath,sizeof(maskpath),"%s/flags-written.txt",argv[3]);
    FILE *mask=fopen(maskpath,"w"); if(!mask) return 10;
    for(int i=0;i<2*NPara;i++) {
        fprintf(mask,"%d ",OptFlag[i]!=INT_MIN);
        if(OptFlag[i]==INT_MIN) OptFlag[i]=0; /* Explicit adapter sentinel, not native C value. */
    }
    fputc('\n',mask); fclose(mask);
    char tablepath[8192]; snprintf(tablepath,sizeof(tablepath),"%s/parameter-audit.tsv",argv[3]);
    table=fopen(tablepath,"w"); if(!table) return 10;
    if(mode==0) SetFlagShift();
    for(acquisition_group=1;acquisition_group<=acquisition_groups;acquisition_group++) {
    init_gen_rand((unsigned)strtoul(argv[4],NULL,10));
    draw_count=0;
    InitParameter();
    int draws=0;
    for(int i=0;i<NRBM;i++) if(OptFlag[2*(NProj+i)]>0) draws+=AllComplexFlag ? 2 : 1;
    for(int i=0;i<NSlater;i++) if(OptFlag[2*(NProj+NRBM+i)]>0) draws+=AllComplexFlag ? 2 : 1;
    checkpoint(argv[3],"initialized",draws);
    if(strcmp(argv[2],"-")!=0) ReadInitParameter(argv[2]); 
    if(ReadInputParameters(NULL,0)!=0) return 11; checkpoint(argv[3],"overlaid",draws);
    SyncModifiedParameter(0); checkpoint(argv[3],"synchronized",draws);
    }
    fclose(table);
    return 0;
}
