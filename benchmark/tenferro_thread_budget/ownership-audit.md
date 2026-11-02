# CPU execution ownership audit (#583, #585)

Source inspected: `35269150deebb39adcd2a79ef37290c0fc49120b`.
This is an implementation audit; it introduces no timing measurements or new policy.

## Current ownership

| Resource | Owner and lifetime | Bound and limitation |
|---|---|---|
| Shared tenferro CPU stage | Process-wide `stage_backend::SHARED` registry; first initialization is retained | One registered backend per kind; independent direct stage instances are outside this registry |
| CPU Engine budget | `sr_backend::cpu_engine_settings` at construction | Explicit `MVMC_RS_TENFERRO_THREADS`, otherwise `RAYON_NUM_THREADS`, otherwise `available_parallelism`; no subtraction for other pools or ranks |
| Shared scoped team | `ScopedCpuTeam`, enabled by `MVMC_RS_TENFERRO_TEAM_LIFETIME=stage` | First lease creates the Engine; final lease replaces it with a serial Engine; concurrent leases retain one team |
| Stage access | `StageHandle` holds the shared stage mutex and its own optional team lease | Serializes stage access; replacing the stage object does not lose the handle's acquired lease |
| SR-CG solve | `retain_cpu_team` returns a separate team lease | Retains the Engine across products without holding the stage mutex across walker/MPI collectives |
| Shared inner dispatch experiment | Existing process Engine, selected by `MVMC_EXPERIMENT_SHARED_CPU_TEAM=1` | Busy/reservation failure uses the sequential caller; unavailable shared Engine uses the Rayon path; it does not allocate a global CPU budget |
| BLAS and MPI processes | Outside the shared stage registry | BLAS threads and overlapping rank CPU masks must be accounted for separately |

The explicit tenferro setting is independent of the Rayon setting. Engine settings
are captured at construction; changing environment variables after a shared stage
has initialized is not a reconfiguration API. A serial Engine executes on its
caller, which still consumes a CPU when active.

## What the experiments establish

The [multi-process report](multi-process.md),
[solve-scoped lifetime report](solve-lease-experiment.md),
[shared-team report](shared-team-20261102/README.md), and
[multi-walker report](walkers-20261102/README.md) record separate ownership,
placement and waiting experiments. Read each report's timing scope and limitations
before comparing numbers. In particular, sampled CPU-active thread IDs do not
prove simultaneous execution, and overlapping per-thread gaps are not process
idle time.

A static cap that subtracts configured inner workers was rejected after standalone
SR regressions: configured pools need not be active together. Recreating the Engine
for every product was rejected after a large SR-CG regression. The solve lease
addresses repeated construction within a solve; it does not coordinate all other
runnable work. See [the cap experiment](hoisted-cap-experiment.md) and
[the rejected per-acquisition experiment](scoped-engine-experiment.md).

## Remaining acceptance work

A complete resource policy must explicitly handle independently opened Engines,
concurrent walker callers, an already initialized shared stage, BLAS, rank CPU
affinity and allocation quotas. Dividing by world size is insufficient when ranks
are already pinned or span multiple nodes. Taking every rank's full affinity is
also insufficient when masks overlap.

Before selecting a default, validate shared/disjoint/overlapping masks, nested
execution, failed reservation, unwind and cleanup, and collective progress. Compare
Opt, PhysCal and standalone SR with matched inputs and explicit worker/BLAS
settings. Preserve independent walker RNG ownership and C-authoritative numerical
contracts. The current options and measurements do not complete #583 or #585.
