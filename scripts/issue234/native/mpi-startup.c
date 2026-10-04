/* Process-manager health only. No mVMC numerical/reference code or fixtures. */
#include <mpi.h>
#include <stdio.h>
#include <string.h>

int main(int argc, char **argv) {
    if (argc != 2 || (strcmp(argv[1], "2") && strcmp(argv[1], "4"))) return 2;
    const int expected = argv[1][0] - '0';
    int provided, rank, size, sum;
    if (MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided) != MPI_SUCCESS)
        return 3;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Allreduce(&rank, &sum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    const int ok = size == expected && rank >= 0 && rank < size &&
        provided >= MPI_THREAD_FUNNELED && sum == expected * (expected - 1) / 2;
    printf("MPI_STARTUP rank=%d size=%d required=%d provided=%d rank_sum=%d ok=%d\n",
           rank, size, MPI_THREAD_FUNNELED, provided, sum, ok);
    fflush(stdout);
    const int finalized = MPI_Finalize();
    return ok && finalized == MPI_SUCCESS ? 0 : 4;
}
