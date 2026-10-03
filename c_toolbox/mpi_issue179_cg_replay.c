/* Optional fixed-Julia-operand MPI replay, NOT full C sampling parity.
 * Numerical Main/operator/dot includes remain byte-exact GPL upstream extracts.
 * See mpi_issue179_cg_replay.md. Never invoked by Cargo or Rust runtime. */
#include <inttypes.h>
#include <math.h>
#include <mpi.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef MVMC_SRCG_REAL
#define USE_IMAG 0
#else
#define USE_IMAG 1
#endif
static int NVMCSample, NSROptCGMaxIter, dimension, rank_id;
static double DSROptCGTol, DSROptStaDel, Wc;
static double *workspace, *current_search, *current_product;
static FILE *evidence;
static unsigned operator_count, check_count;

static void vector(const char *key, const double *data, int n) {
    fprintf(evidence, "%s", key);
    for (int i = 0; i < n; ++i) fprintf(evidence, " %.17g", data[i]);
    fputc('\n', evidence);
}
static void operator_vector(const char *field, const double *data) {
    char key[96];
    snprintf(key, sizeof(key), "n:operator-%06u-%s", operator_count - 1, field);
    vector(key, data, dimension);
}
static int observed_bcast(void *data, int n, MPI_Datatype type, int root, MPI_Comm comm) {
    int rc = PMPI_Bcast(data, n, type, root, comm);
    if (rc != MPI_SUCCESS || n != dimension || type != MPI_DOUBLE || root != 0)
        MPI_Abort(comm, 10);
    current_search = data;
    ++operator_count;
    operator_vector("root-search", data);
    return rc;
}
static int observed_allreduce(const void *send, void *recv, int n,
                              MPI_Datatype type, MPI_Op op, MPI_Comm comm) {
    if (n != dimension || type != MPI_DOUBLE || op != MPI_SUM) MPI_Abort(comm, 11);
    operator_vector("local-product", send);
    int rc = PMPI_Allreduce(send, recv, n, type, op, comm);
    if (rc != MPI_SUCCESS) MPI_Abort(comm, 12);
    current_product = recv;
    operator_vector("global-product", recv);
    return rc;
}
/* The authoritative SafeMpi function supplies its original chunking/domain. */
#define _mpi_use
#define MPI_Allreduce observed_allreduce
#include "safempi.c"
#undef MPI_Allreduce

static void timer_stop(int timer) {
    if (timer == 53) {
        if (!current_search || !current_product) MPI_Abort(MPI_COMM_WORLD, 13);
        operator_vector("corrected-product", current_product);
    }
}
static int observed_fprintf(FILE *stream, const char *format, ...) {
    va_list args, copy;
    va_start(args, format);
    va_copy(copy, args);
    /* Original debug boundary exposes the ACTUAL local delta and threshold.
     * No numerical variable/branch is replaced or reconstructed here. */
    if (strcmp(format, "delta = %lg, cg_thresh = %lg\n") == 0) {
        double delta = va_arg(copy, double), threshold = va_arg(copy, double);
        double *local = workspace + 4 * dimension +
            (USE_IMAG + 1) * NVMCSample * (dimension + 1);
        char key[96];
        fprintf(evidence, "n:check-%06u-delta %.17g\n", check_count, delta);
        fprintf(evidence, "n:check-%06u-threshold %.17g\n", check_count, threshold);
        snprintf(key, sizeof(key), "n:check-%06u-solution", check_count);
        vector(key, workspace, dimension);
        snprintf(key, sizeof(key), "n:check-%06u-residual", check_count);
        vector(key, local + 3 * dimension, dimension);
        snprintf(key, sizeof(key), "n:check-%06u-direction", check_count);
        vector(key, local + 2 * dimension, dimension);
        ++check_count;
    }
    va_end(copy);
    int rc = vfprintf(stream, format, args);
    va_end(args);
    return rc;
}
extern void dgemv_(const char *, const int *, const int *, const double *,
                   const double *, const int *, const double *, const int *,
                   const double *, double *, const int *);
#define M_DGEMV dgemv_
#define inline static
#include "ctest_cg_dot_upstream.inc"
#undef inline
int fn_operate_by_S(int, double *, double *, double *, MPI_Comm);
#define MPI_Bcast observed_bcast
#define StartTimer(timer) ((void)(timer))
#define StopTimer(timer) timer_stop(timer)
#define _DEBUG_STCOPT_CG
#define fprintf observed_fprintf
#include "ctest_cg_main_upstream.inc"
#undef fprintf
#undef MPI_Bcast

static void read_values(FILE *input, double *data, size_t count) {
    for (size_t i = 0; i < count; ++i)
        if (fscanf(input, "%la", data + i) != 1 || !isfinite(data[i]))
            MPI_Abort(MPI_COMM_WORLD, 14);
}
int main(int argc, char **argv) {
    int provided, world;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_id);
    MPI_Comm_size(MPI_COMM_WORLD, &world);
    if (argc != 3 || world != 4 || provided < MPI_THREAD_FUNNELED)
        MPI_Abort(MPI_COMM_WORLD, 2);
    char path[4096];
    snprintf(path, sizeof(path), "%s/operand-rank-%d.txt", argv[1], rank_id);
    FILE *input = fopen(path, "r");
    int imag, recorded_rank, recorded_world, width;
    if (!input || fscanf(input, "%d %d %d %d %d %d %d %la %la %la",
            &dimension, &NVMCSample, &imag, &recorded_rank, &recorded_world,
            &width, &NSROptCGMaxIter, &Wc, &DSROptStaDel, &DSROptCGTol) != 10 ||
        dimension != (USE_IMAG ? 20 : 10) || NVMCSample != 3 || imag != USE_IMAG ||
        recorded_rank != rank_id || recorded_world != world || width != 1 ||
        Wc != 12 || DSROptStaDel != 1e-5 || DSROptCGTol != 1e-10 || NSROptCGMaxIter != 0)
        MPI_Abort(MPI_COMM_WORLD, 3);
    size_t count = (size_t)8 * dimension +
        (size_t)(USE_IMAG + 1) * NVMCSample * (dimension + 1);
    workspace = calloc(count, sizeof(*workspace));
    if (!workspace) MPI_Abort(MPI_COMM_WORLD, 4);
    double *gradient = workspace + dimension, *diag = workspace + 2 * dimension;
    double *mean = workspace + 3 * dimension, *real = workspace + 4 * dimension;
    double *imaginary = real + NVMCSample * dimension;
    read_values(input, mean, dimension); read_values(input, diag, dimension);
    read_values(input, real, (size_t)dimension * NVMCSample);
    read_values(input, imaginary, (size_t)USE_IMAG * dimension * NVMCSample);
    read_values(input, gradient, dimension);
    char trailing[2];
    if (fscanf(input, "%1s", trailing) != EOF) MPI_Abort(MPI_COMM_WORLD, 5);
    fclose(input);
    snprintf(path, sizeof(path), "%s/c-rank-%d.txt", argv[2], rank_id);
    evidence = fopen(path, "w");
    if (!evidence) MPI_Abort(MPI_COMM_WORLD, 6);
    fprintf(evidence, "d:rank %d\nd:world %d\nd:width 1\nd:dimension %d\nd:samples 3\nd:imag %d\n",
            rank_id, world, dimension, USE_IMAG);
    fprintf(evidence, "n:weight %.17g\nn:shift %.17g\nn:tolerance %.17g\n", Wc, DSROptStaDel, DSROptCGTol);
    vector("n:mean", mean, dimension); vector("n:diagonal", diag, dimension);
    vector("n:gradient", gradient, dimension);
    vector("n:real-samples", real, dimension * NVMCSample);
    if (USE_IMAG) vector("n:imag-samples", imaginary, dimension * NVMCSample);
    int iterations = fn_StochasticOptCG_Main(dimension, workspace, MPI_COMM_WORLD);
    double *local = imaginary + USE_IMAG * NVMCSample * dimension + (USE_IMAG + 1) * NVMCSample;
    fprintf(evidence, "d:iterations %d\nd:operator-count %u\nd:check-count %u\n", iterations, operator_count, check_count);
    vector("n:solution", workspace, dimension);
    vector("n:residual", local + 3 * dimension, dimension);
    vector("n:direction", local + 2 * dimension, dimension);
    if (fclose(evidence)) MPI_Abort(MPI_COMM_WORLD, 7);
    free(workspace);
    MPI_Finalize();
    return 0;
}
