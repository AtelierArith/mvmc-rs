/* Optional serial kernel-stage oracle; not a full C executable/MPI trace.
 * The companion Julia command extracts verbatim authoritative functions.
 * Dimensions/flags and keyword paths are supplied explicitly, not read by
 * C's definition parser. Unwritten flag cells are deliberately zero-filled.
 */
#include <complex.h>
#include <math.h>
#include <stdint.h>
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
int NProj, NRBM, NSlater, NOptTrans, NPara, FlagRBM, AllComplexFlag;
int NGutzwillerIdx, NJastrowIdx, NDoublonHolon2siteIdx, NDoublonHolon4siteIdx;
int NChargeRBM_PhysLayerIdx, NSpinRBM_PhysLayerIdx, NGeneralRBM_PhysLayerIdx;
int NChargeRBM_HiddenLayerIdx, NSpinRBM_HiddenLayerIdx, NGeneralRBM_HiddenLayerIdx;
int NChargeRBM_PhysHiddenIdx, NSpinRBM_PhysHiddenIdx, NGeneralRBM_PhysHiddenIdx;
int NRBM_PhysLayerIdx, NRBM_HiddenLayerIdx, Nneuron;
int iFlgOrbitalGeneral, iNOrbitalAntiParallel, iNOrbitalParallel, NOrbitalIdx;
int FlagShiftDH2, FlagShiftDH4, FlagShiftGJ, FlagOptTrans, OptFlag[8192];
double complex Proj[4096], RBM[4096], Slater[4096], OptTrans[4096], ParaQPOptTrans[4096];
char filenames[KWIdxInt_end][D_CharTmpReadDef];
static double shiftDH2(void) { abort(); }
static double shiftDH4(void) { abort(); }
static void shiftGJ(void) { abort(); }
static int ReadDefFileError(char *path) { fprintf(stderr,"cannot read %s\n",path); return 1; }
#include "physcal_181_parameters_upstream.inc"

static void checkpoint(const char *dir, const char *stage, int draws) {
    char path[8192];
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
    fprintf(file,"%d\n",draws); fclose(file);
}

int main(int argc,char **argv) {
    if(argc!=4) return 2;
    FILE *spec=fopen(argv[1],"r"); if(!spec) return 3;
    int *widths[]={&NGutzwillerIdx,&NJastrowIdx,&NDoublonHolon2siteIdx,&NDoublonHolon4siteIdx,
        &NChargeRBM_PhysLayerIdx,&NSpinRBM_PhysLayerIdx,&NGeneralRBM_PhysLayerIdx,
        &NChargeRBM_HiddenLayerIdx,&NSpinRBM_HiddenLayerIdx,&NGeneralRBM_HiddenLayerIdx,
        &NChargeRBM_PhysHiddenIdx,&NSpinRBM_PhysHiddenIdx,&NGeneralRBM_PhysHiddenIdx,&NSlater,&NOptTrans};
    for(int i=0;i<15;i++) if(fscanf(spec,"%d",widths[i])!=1 || *widths[i]<0) return 4;
    if(fscanf(spec,"%d%d",&AllComplexFlag,&Nneuron)!=2) return 4;
    NProj=NGutzwillerIdx+NJastrowIdx+6*NDoublonHolon2siteIdx+10*NDoublonHolon4siteIdx;
    NRBM_PhysLayerIdx=NChargeRBM_PhysLayerIdx+NSpinRBM_PhysLayerIdx+NGeneralRBM_PhysLayerIdx;
    NRBM_HiddenLayerIdx=NChargeRBM_HiddenLayerIdx+NSpinRBM_HiddenLayerIdx+NGeneralRBM_HiddenLayerIdx;
    NRBM=NRBM_PhysLayerIdx+NRBM_HiddenLayerIdx+NChargeRBM_PhysHiddenIdx+NSpinRBM_PhysHiddenIdx+NGeneralRBM_PhysHiddenIdx;
    NPara=NProj+NRBM+NSlater+NOptTrans;
    if(NPara>4096 || NSlater<1) return 4;
    NOrbitalIdx=NSlater; FlagRBM=NRBM>0; FlagOptTrans=NOptTrans>0;
    for(int i=0;i<2*NPara;i++) if(fscanf(spec,"%d",OptFlag+i)!=1) return 4;
    for(int i=0;i<NOptTrans;i++) {
        double weight;
        if(fscanf(spec,"%lf",&weight)!=1) return 4;
        ParaQPOptTrans[i]=weight;
    }
    cFileNameListFile=filenames;
    int key; char path[4096];
    while(fscanf(spec,"%d %4095s",&key,path)==2) {
        if(key<KWInGutzwiller || key>=KWIdxInt_end) return 4;
        if(strlen(path)>=D_CharTmpReadDef) return 4;
        strcpy(filenames[key],path);
    }
    fclose(spec);
    init_gen_rand(1);
    InitParameter();
    int draws=0;
    for(int i=0;i<NRBM;i++) if(OptFlag[2*(NProj+i)]>0) draws+=AllComplexFlag ? 2 : 1;
    for(int i=0;i<NSlater;i++) if(OptFlag[2*(NProj+NRBM+i)]>0) draws+=AllComplexFlag ? 2 : 1;
    checkpoint(argv[3],"initialized",draws);
    ReadInitParameter(argv[2]); checkpoint(argv[3],"fixed",draws);
    ReadInputParameters(NULL,0); checkpoint(argv[3],"overlaid",draws);
    SyncModifiedParameter(0); checkpoint(argv[3],"synchronized",draws);
    return 0;
}
