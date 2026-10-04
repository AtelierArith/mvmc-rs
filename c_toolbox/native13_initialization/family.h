#ifndef NATIVE13_INIT_FAMILY_H
#define NATIVE13_INIT_FAMILY_H
#include <stdint.h>
#include <complex.h>
/* Each interval records the actual original reader, not global AllComplexFlag. */
enum Native13Reader { NATIVE13_GETINFOOPT=1, NATIVE13_PARALLEL=2 };
typedef struct {const char *name; int start, length, complex_flag, reader;} Native13Segment;
typedef struct {
    int npara,nproj,nrbm,nslater,nopt,nqpopt,nprojbf;
    int ap_count,parallel_count,orbital_general,flag_rbm;
    const double *qpopt;
    const Native13Segment *segments; int segment_count;
} Native13Layout;
int native13_family_checkpoint(const char *,int,int,int,int,int,
    const double complex *,const int *,const Native13Layout *);
int native13_family_masks(const Native13Layout *,uint8_t *,int);
int native13_snapshot(uint32_t [624],int *,uint64_t *,uint32_t [624]);
#endif
