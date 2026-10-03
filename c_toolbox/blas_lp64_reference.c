/* Optional Julia ILP64 -> system OpenBLAS LP64 ABI adapter for #200.
 * Numerical routines are the linked OpenBLAS routines, unchanged. Cargo must
 * never build or load this oracle. See docs/APPLE_SILICON_PARITY.md. */
#include <stdint.h>
#include <limits.h>
#include <stdlib.h>

extern void openblas_set_num_threads(int);
extern char *openblas_get_config(void);
extern char *openblas_get_corename(void);
extern void dpotrf_(const char *, const int *, double *, const int *, int *);
extern void dpotrs_(const char *, const int *, const int *, const double *,
                    const int *, double *, const int *, int *);
extern void dgemv_(const char *, const int *, const int *, const double *,
                   const double *, const int *, const double *, const int *,
                   const double *, double *, const int *);
extern void dsyrk_(const char *, const char *, const int *, const int *,
                   const double *, const double *, const int *, const double *,
                   double *, const int *);
extern void dgemm_(const char *, const char *, const int *, const int *, const int *,
                   const double *, const double *, const int *, const double *,
                   const int *, const double *, double *, const int *);

static int lp64(const int64_t *value) {
    if (*value < INT_MIN || *value > INT_MAX) abort();
    return (int)*value;
}
void mvmc_reference_blas_init(void) { openblas_set_num_threads(1); }
const char *mvmc_reference_blas_config(void) { return openblas_get_config(); }
const char *mvmc_reference_blas_core(void) { return openblas_get_corename(); }

void mvmc_reference_dpotrf(const char *u, const int64_t *n, double *a,
                           const int64_t *lda, int64_t *info) {
    int nn = lp64(n), ld = lp64(lda), status;
    dpotrf_(u, &nn, a, &ld, &status);
    *info = status;
}
void mvmc_reference_dpotrs(const char *u, const int64_t *n, const int64_t *nrhs,
                           const double *a, const int64_t *lda, double *b,
                           const int64_t *ldb, int64_t *info) {
    int nn = lp64(n), nr = lp64(nrhs), la = lp64(lda), lb = lp64(ldb), status;
    dpotrs_(u, &nn, &nr, a, &la, b, &lb, &status);
    *info = status;
}
void mvmc_reference_dgemv(const char *t, const int64_t *m, const int64_t *n,
                          const double *alpha, const double *a, const int64_t *lda,
                          const double *x, const int64_t *incx, const double *beta,
                          double *y, const int64_t *incy) {
    int mm = lp64(m), nn = lp64(n), ld = lp64(lda), ix = lp64(incx), iy = lp64(incy);
    dgemv_(t, &mm, &nn, alpha, a, &ld, x, &ix, beta, y, &iy);
}
void mvmc_reference_dsyrk(const char *u, const char *t, const int64_t *n,
                          const int64_t *k, const double *alpha, const double *a,
                          const int64_t *lda, const double *beta, double *c,
                          const int64_t *ldc) {
    int nn = lp64(n), kk = lp64(k), la = lp64(lda), lc = lp64(ldc);
    dsyrk_(u, t, &nn, &kk, alpha, a, &la, beta, c, &lc);
}
void mvmc_reference_dgemm(const char *ta, const char *tb, const int64_t *m,
                          const int64_t *n, const int64_t *k, const double *alpha,
                          const double *a, const int64_t *lda, const double *b,
                          const int64_t *ldb, const double *beta, double *c,
                          const int64_t *ldc) {
    int mm = lp64(m), nn = lp64(n), kk = lp64(k), la = lp64(lda), lb = lp64(ldb), lc = lp64(ldc);
    dgemm_(ta, tb, &mm, &nn, &kk, alpha, a, &la, b, &lb, beta, c, &lc);
}

extern void dtrtri_(const char *, const char *, const int *, double *, const int *, int *);
extern void ztrtri_(const char *, const char *, const int *, void *, const int *, int *);
extern void dtrmm_(const char *, const char *, const char *, const char *, const int *, const int *, const double *, const double *, const int *, double *, const int *);
extern void ztrmm_(const char *, const char *, const char *, const char *, const int *, const int *, const void *, const void *, const int *, void *, const int *);
extern void zgemv_(const char *, const int *, const int *, const void *, const void *, const int *, const void *, const int *, const void *, void *, const int *);
extern void zgeru_(const int *, const int *, const void *, const void *, const int *, const void *, const int *, void *, const int *);
extern void zscal_(const int *, const void *, void *, const int *);
extern void zaxpy_(const int *, const void *, const void *, const int *, void *, const int *);
void mvmc_reference_dtrtri(const char *u, const char *d, const int64_t *n, double *a, const int64_t *lda, int64_t *info) {
    int nn=lp64(n), ld=lp64(lda), status;
    dtrtri_(u,d,&nn,a,&ld,&status); *info=status;
}
void mvmc_reference_dtrmm(const char *s, const char *u, const char *t, const char *d, const int64_t *m, const int64_t *n, const double *alpha, const double *a, const int64_t *lda, double *b, const int64_t *ldb) {
    int mm=lp64(m),nn=lp64(n),la=lp64(lda),lb=lp64(ldb);
    dtrmm_(s,u,t,d,&mm,&nn,alpha,a,&la,b,&lb);
}
void mvmc_reference_ztrtri(const char *u, const char *d, const int64_t *n, void *a, const int64_t *lda, int64_t *info) {
    int nn=lp64(n), ld=lp64(lda), status;
    ztrtri_(u,d,&nn,a,&ld,&status); *info=status;
}
void mvmc_reference_ztrmm(const char *s, const char *u, const char *t, const char *d, const int64_t *m, const int64_t *n, const void *alpha, const void *a, const int64_t *lda, void *b, const int64_t *ldb) {
    int mm=lp64(m),nn=lp64(n),la=lp64(lda),lb=lp64(ldb);
    ztrmm_(s,u,t,d,&mm,&nn,alpha,a,&la,b,&lb);
}
void mvmc_reference_zgemv(const char *t, const int64_t *m, const int64_t *n, const void *alpha, const void *a, const int64_t *lda, const void *x, const int64_t *incx, const void *beta, void *y, const int64_t *incy) {
    int mm=lp64(m),nn=lp64(n),ld=lp64(lda),ix=lp64(incx),iy=lp64(incy);
    zgemv_(t,&mm,&nn,alpha,a,&ld,x,&ix,beta,y,&iy);
}
void mvmc_reference_zgeru(const int64_t *m, const int64_t *n, const void *alpha, const void *x, const int64_t *incx, const void *y, const int64_t *incy, void *a, const int64_t *lda) {
    int mm=lp64(m),nn=lp64(n),ix=lp64(incx),iy=lp64(incy),ld=lp64(lda);
    zgeru_(&mm,&nn,alpha,x,&ix,y,&iy,a,&ld);
}
void mvmc_reference_zscal(const int64_t *n, const void *alpha, void *x, const int64_t *incx) {
    int nn=lp64(n),ix=lp64(incx); zscal_(&nn,alpha,x,&ix);
}
void mvmc_reference_zaxpy(const int64_t *n, const void *alpha, const void *x, const int64_t *incx, void *y, const int64_t *incy) {
    int nn=lp64(n),ix=lp64(incx),iy=lp64(incy); zaxpy_(&nn,alpha,x,&ix,y,&iy);
}
