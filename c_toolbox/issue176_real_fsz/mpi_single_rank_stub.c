/* Single-rank MPI shim for hosts without an MPI runtime (optional developer tool).
 * Only the symbols reached by the probe (initialization, communicator queries and the
 * size>1 reduction, which never executes with one rank) are defined. It changes no
 * numerical routine; the probe calls the unmodified mVMC kernels. */
#include <mpi.h>

int MPI_Init(int *argc, char ***argv) { (void)argc; (void)argv; return MPI_SUCCESS; }
int MPI_Finalize(void) { return MPI_SUCCESS; }
int MPI_Comm_size(MPI_Comm comm, int *size) { (void)comm; *size = 1; return MPI_SUCCESS; }
int MPI_Comm_rank(MPI_Comm comm, int *rank) { (void)comm; *rank = 0; return MPI_SUCCESS; }
int MPI_Allreduce(const void *s, void *r, int n, MPI_Datatype t, MPI_Op o, MPI_Comm c) {
  (void)s; (void)r; (void)n; (void)t; (void)o; (void)c;
  return MPI_ERR_OTHER; /* unreachable with one rank */
}
