# CPU execution ownership and bounded configuration (#583)

This policy keeps the production Rayon path and its existing work gates. Shared
Engine-team inner dispatch remains an explicit experiment; a held team is not
a universal speed or CPU-efficiency improvement. The migration does not select
a global automatic scheduler.

## Owners and budgets

- A backend owns its Engine resource; the process stage registry shares one
  backend per kind. Independently constructed mvmc CPU Engines participate in
  the same optional additional-worker reservation budget. Caller-provided
  external Engines remain their caller's responsibility.
- `MVMC_RS_TENFERRO_WORKER_BUDGET=E` bounds the sum of reserved **additional**
  mvmc Engine workers. A `T`-participant Engine needs `T-1` reservations because
  its executing caller is already counted. Insufficient capacity selects a
  serial Engine immediately; it does not wait while holding a walker collective.
  Capacity returns after the Engine is destroyed. Settings are captured before
  first construction; later changes fail instead of silently reconfiguring it.
- For a stock CLI rank with `I` inner participants, count its driver in `I`
  (inside the hoisted pool when `I>1`, otherwise one caller). The experimental
  Linux MPI placement option requires an explicit `E` and `I+E <= N`, where `N`
  is the logical CPU assignment per rank. This conservative reservation bound
  counts the idle Rayon workers as well as the Engine; do not subtract a pool
  simply because it is expected to be idle in one phase.
- Optimization uses one independent OS caller per walker. All must remain
  runnable for ThreadComm collectives: never queue them in a smaller blocking
  pool. With `W` callers, inner=1 and BLAS=1, account for `W+E` compute participants.
  With a separate inner pool, account for that pool as well. The single-rank CLI
  `I+E` formula must not be applied to a multiwalker library caller.
- PhysCal's walker Rayon pool is independently owned and capped by walker count;
  it may have fewer workers than walkers because it has no optimizer collective.
  Count its configured workers, inner pools, Engine reservations and callers
  separately when composing library execution.
- Set BLAS/OMP threads explicitly to 1 for these configurations. MPI progress,
  the blocked outer driver, runtime helper threads and external pools are not
  Engine reservations. Record their observed OS-thread activity separately.
  Managed reservation counts are not total OS-thread counts or simultaneous use.

## Placement and allocation

The operator supplies the allocation ceiling, accounting for both scheduler
affinity and cgroup CPU quota. Logical CPU count is not a physical-core count
or a quota. Never divide host CPUs by world size: ranks may already be pinned,
have overlapping masks, or occupy different nodes.

`MVMC_RS_MPI_CPUS_PER_RANK=N` collectively matches rank requests to allowed CPU
ids within each shared-memory node. It rejects infeasible requests before I/O,
restricts the initializing caller and every existing mvmc inner worker, and
lets future workers inherit those masks. Partial application failures restore
successful workers/ranks; an error is agreed before further collectives. Explicit
guard restoration reports OS failures; normal drop/unwind attempts restoration.

The API does not infer cgroup hierarchy quotas or pin unrelated existing threads.
Use the launcher's initial per-rank affinity to constrain MPI/runtime threads
created before CLI preflight too. Choose `N` and the number of ranks so the
total allocation fits the actual job quota. Set all controls before startup.
The unchanged default does not promise an automatic cross-pool or quota cap.

## Selection rules

1. Retain Rayon unless a matched E2E measurement supports the experimental shared
   path for the actual workload and CPU budget. L32 improvements do not establish
   an L64/L128 default. Distinguish wall-time objectives from aggregate CPU cost.
2. A busy shared Engine executes fixed blocks on the caller; an unavailable team
   uses Rayon. An already active bounded saved-sample team keeps ownership, and
   nested blocks execute inline. These avoid new nested dispatch rather than
   pretending all pools have been removed.
3. Use a serial Engine when caller/walker concurrency consumes the CPU allocation.
   It retains the caller's numerical work without adding a waiting reservation.
4. Keep the solve-scoped lease across SR-CG products. Recreating a team per product
   and static subtraction caps were measured and rejected for regressions; they
   are not adopted alternatives.

This is an explicit, bounded operator configuration contract. It does not claim
automatic quota discovery, zero waiting cost, universal speedup or the resolution
of the separate CPU/GPU performance issues #586/#587.
