/*
 * Optional source-extracted C caller-order control, not a native kernel/MPI
 * or SFMT oracle. shared-initializer.inc carries upstream GPL/origin/license.
 * All numeric validation codes here are explicitly synthetic fixtures.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
typedef int MPI_Comm;
#define MPI_INT 1
#define MPI_MAX 2
#define MPI_COMM_WORLD 0
static const int Nsize=2,Nsite=2,Nsite2=4,Ne=1;
static int LocSpn[2]={0,0};
static int calls,words,mode;
static uint32_t gen_rand32(void) { ++words;return 0; }
static double genrand_real2(void) { ++words;return 0.0; }
static void MakeProjCnt(int *p,const int *n) { p[0]=n[0]+n[2]; }
static int MPI_Comm_size(MPI_Comm c,int *n) { (void)c;*n=mode==3?2:1;return 0; }
static int MPI_Comm_rank(MPI_Comm c,int *n) { (void)c;*n=0;return 0; }
static int CalculateMAll_fcmp(const int *idx,int begin,int end) {
    (void)idx;(void)begin;(void)end;++calls;
    int info=mode==1?(calls==1?2:0):mode==2?-3:
        mode==4?(calls==101?0:1):mode==5?1:0;
    printf("complex %d %d %d\n",calls,info,words);return info;
}
static int MPI_Allreduce(const int *local,int *result,int count,int type,int op,MPI_Comm c) {
    (void)c;if(count!=1||type!=MPI_INT||op!=MPI_MAX) exit(70);
    const int peer=calls==1?4:0;*result=*local>peer?*local:peer;
    printf("integer-max %d %d %d %d\n",calls,*local,peer,*result);return 0;
}
static int MPI_Abort(MPI_Comm c,int status) {
    (void)c;printf("exhausted %d %d %d\n",calls,words,status);exit(42);
}
#include "shared-initializer.inc"
int main(int argc,char **argv) {
    if(argc!=2)return 64;
    const char *names[]={"success","retry","negative","peer-retry","call101-success","exhaustion"};
    for(mode=0;mode<6;mode++)if(strcmp(argv[1],names[mode])==0)break;
    if(mode==6)return 64;
    int idx[2],cfg[4],num[4],proj[1];
    const int status=makeInitialSample(idx,cfg,num,proj,0,1,MPI_COMM_WORLD);
    printf("returned %d %d %d\n",status,calls,words);
    /* Distinct real setup after shared success, its synthetic INFO ignored. */
    printf("real-setup -7\n");
    printf("continued-after-real-setup\n");
    return 0;
}
