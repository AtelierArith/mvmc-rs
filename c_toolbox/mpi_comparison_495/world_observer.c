/* Optional observer of the unmodified C reference's MPI initialization.
 * MPI profiling API; no RNG access, extra collective, or numerical change.
 * vmcmain.c calls MPI_Init before StartTimer(0). */
#include <mpi.h>
#include <omp.h>
#include <stdio.h>
extern int openblas_get_num_threads(void);

int MPI_Init(int *argc, char ***argv) {
    int rc = PMPI_Init(argc, argv);
    if (rc == MPI_SUCCESS) {
        int rank, size;
        PMPI_Comm_rank(MPI_COMM_WORLD, &rank);
        PMPI_Comm_size(MPI_COMM_WORLD, &size);
        printf("WORLD %d %d\nTHREADS %d %d\nBLAS_THREADS %d %d\n",
               rank, size, rank, omp_get_max_threads(), rank,
               openblas_get_num_threads());
        fflush(stdout);
    }
    return rc;
}
