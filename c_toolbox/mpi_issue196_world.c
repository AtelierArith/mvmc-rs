/* Independent MPI API diagnostic, not linked to Rust and not invoked by Cargo.
 * Reproduction and source hash: scripts/verify_mpi_issue196_runtime.sh.
 * This is original diagnostic code, not extracted numerical reference code.
 */
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank = -1, size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int expected = argc == 2 ? atoi(argv[1]) : 0;
    if ((expected != 2 && expected != 4) || size != expected) {
        fprintf(stderr, "invalid world: rank=%d size=%d expected=%d PMI_RANK=%s\n",
                rank, size, expected, getenv("PMI_RANK") ? getenv("PMI_RANK") : "unset");
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    int contribution = rank + 1, total = 0;
    MPI_Allreduce(&contribution, &total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    int root_payload = rank == size - 1 ? 196 : -1;
    MPI_Bcast(&root_payload, 1, MPI_INT, size - 1, MPI_COMM_WORLD);
    if (total != size * (size + 1) / 2 || root_payload != 196) {
        MPI_Abort(MPI_COMM_WORLD, 2);
        return 2;
    }
    printf("WORLD %d %d %d %d\n", rank, size, total, root_payload);
    fflush(stdout);
    MPI_Finalize();
    return 0;
}
