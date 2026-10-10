/* Untimed diagnostic only: observe actual threads entering libgomp teams.
 * LD_PRELOAD alongside world_observer.so. Never use these runs for timings.
 * GOMP_parallel ABI: GCC libgomp, GOMP_4.0 (the reference executable's import).
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <mpi.h>
#include <omp.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>

typedef void (*parallel_fn)(void (*)(void *), void *, unsigned, unsigned);
struct observed_call { void (*fn)(void *); void *data; };
static atomic_ulong worker_bits;
static atomic_ulong parallel_calls;

static void observed_worker(void *arg) {
    struct observed_call *call = arg;
    unsigned worker = (unsigned)omp_get_thread_num();
    if (worker >= sizeof(unsigned long) * 8) abort();
    atomic_fetch_or_explicit(&worker_bits, 1UL << worker, memory_order_relaxed);
    call->fn(call->data);
}

void GOMP_parallel(void (*fn)(void *), void *data, unsigned threads, unsigned flags) {
    /* Resolve per call to avoid a racy first-call cache in nested regions. */
    parallel_fn original = (parallel_fn)dlsym(RTLD_NEXT, "GOMP_parallel");
    if (!original) abort();
    struct observed_call call = {fn, data};
    atomic_fetch_add_explicit(&parallel_calls, 1, memory_order_relaxed);
    original(observed_worker, &call, threads, flags);
}

int MPI_Finalize(void) {
    int rank;
    PMPI_Comm_rank(MPI_COMM_WORLD, &rank);
    unsigned long bits = atomic_load_explicit(&worker_bits, memory_order_relaxed);
    unsigned workers = 0;
    while (bits) { workers += bits & 1UL; bits >>= 1; }
    printf("CEXECUTION %d %lu %u\n", rank,
           atomic_load_explicit(&parallel_calls, memory_order_relaxed), workers);
    fflush(stdout);
    return PMPI_Finalize();
}
