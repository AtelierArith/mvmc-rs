/* Explicit optional call-chain probe, not full C sampling/CG validation.
 * Fill otherwise-unwritten columns with TWO declared values. Neither fill is
 * an expectation for native malloc contents. Never invoked by Cargo/Rust.
 */
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include "splitloop.c"
static int NSRCG = 1;
static void unused_dgemm(char *a, char *b, int *c, int *d, int *e, double *f,
                         double *g, int *h, double *i, int *j, double *k,
                         double *l, int *m) {
  (void)a;(void)b;(void)c;(void)d;(void)e;(void)f;(void)g;
  (void)h;(void)i;(void)j;(void)k;(void)l;(void)m; abort();
}
#define M_DGEMM unused_dgemm
#include "mpi_issue179_store_upstream.inc"

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int rank, world;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world);
  if(argc!=2 || atoi(argv[1])<1) MPI_Abort(MPI_COMM_WORLD, 2);
  int width=atoi(argv[1]), group=rank/width;
  MPI_Comm chain;
  MPI_Comm_split(MPI_COMM_WORLD, group, rank, &chain);
  int local, size, start, end;
  MPI_Comm_rank(chain, &local); MPI_Comm_size(chain, &size);
  SplitLoop(&start, &end, 5, local, size);
  for(int fill=0;fill<=17;fill+=17) {
    double SROptO_Store_real[10], SROptO_real[2], oo[4], ho[2]={0};
    const int SROptSize=2;
    for(int i=0;i<10;i++) SROptO_Store_real[i]=fill;
    double measured=0;
    for(int sample=start;sample<end;sample++) {
      double sqrtw=1;
      SROptO_real[0]=1; SROptO_real[1]=100*group+sample+1;
      measured+=SROptO_real[1];
      for(int int_i=0;int_i<SROptSize;int_i++) {
        /* vmccal.c:241 actual absolute-column assignment, unchanged. */
        SROptO_Store_real[int_i+sample*SROptSize]  = sqrtw*SROptO_real[int_i];
      }
    }
    /* vmccal.c:314 passes BASE store, not store+sampleStart*SROptSize. */
    calculateOO_Store_real(oo,ho,SROptO_Store_real,1,0,SROptSize,end-start);
    double actual, assigned;
    MPI_Allreduce(&oo[1],&actual,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
    MPI_Allreduce(&measured,&assigned,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
    printf("STORE world=%d rank=%d group=%d local=%d size=%d range=%d..%d explicit_unwritten_fill=%d local_prefix_sum=%.17g local_assigned_sum=%.17g global_prefix_sum=%.17g global_assigned_sum=%.17g\n",
           world,rank,group,local,size,start,end,fill,oo[1],measured,actual,assigned);
  }
  MPI_Comm_free(&chain); MPI_Finalize(); return 0;
}
