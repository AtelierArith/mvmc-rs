#ifndef NATIVE13_INIT_OBSERVER_H
#define NATIVE13_INIT_OBSERVER_H
#include <stdint.h>
#include <complex.h>
int native13_snapshot(uint32_t raw[624], int *cursor, uint64_t *count,
                      uint32_t next[624]);
int native13_checkpoint(const char *stage, int rank, int group, int seed,
                        int all_complex, int npara, const double complex *para,
                        const int *flags, int nproj, int nrbm, int nslater,
                        int nopt, int nqpopt, const double *qpopt,
                        int ngutz, int njast, int complex_gutz, int complex_jast,
                        int complex_orbital, int simple_orbital);
#endif
