# 12. Accelerated and GPU backends

[Contents](README.md) · Previous: [11. Compatibility and differences](11-compatibility.md) · Next: [Appendix: how the citations were checked](appendix-checks.md)

This chapter describes the optional accelerated implementations in `mvmc-rs`: the tenferro CPU stages that the normal `mvmc` binary can select, and the device-resident GPU Opt/PhysCal runners of the standalone `gpu/mvmc-gpu-cuda` workspace. It covers how to build and select them, what has been validated and to what bounds, and their known limits. The design records are [docs/design/gpu-readiness.md](../design/gpu-readiness.md) (tenferro stages, issue #417), [docs/design/gpu-resident-opt.md](../design/gpu-resident-opt.md) and [docs/design/gpu-resident-physcal.md](../design/gpu-resident-physcal.md) (resident GPU runners); the validation policy is the section "Accelerated-backend validation (#424)" of [docs/NUMERICAL_COMPARISONS.md](../NUMERICAL_COMPARISONS.md).

The legacy C-order/SFMT GPU route (`MVMC_RS_SR_BACKEND`/`MVMC_RS_MEASURE_PF_BACKEND`/`MVMC_RS_MEASURE_BACKEND=cuda[:N]`, `MVMC_RS_SAMPLER_BACKEND`, the `gpu-cuda` feature, the CUDA provider and `mvmc-cuda` as a CPU-driver wrapper) was removed in #643. Its measurements remain as historical records in the dated `benchmark/*/results/` directories.

## 12.1 Policy

- **The CPU C-order path is the default and the oracle.** Every accelerated implementation is an *additional* implementation. It never replaces, reorders or reroutes the CPU path, and no default build, default test or fixture depends on a GPU or on the tenferro GPU crates.
- **No silent fallback.** An unsupported stage, dtype or input returns an error (`StageError::Unsupported`); the program never substitutes the CPU result.
- **Selected explicitly.** Nothing is chosen automatically; the default is `c-order`.
- **Two numerical contracts.** The tenferro CPU stages keep the C-order RNG draw order and count and differ only in floating-point summation order, within explicit bounds ([12.5](#125-numerical-guarantees-and-what-was-validated)). The resident GPU runners follow the Julia GPU contract (cuRAND XORWOW walker ensemble, as authorized in `AGENTS.md`); they are validated against Julia GPU fixtures, not against C/SFMT trajectories.

## 12.2 What exists

| Component | Where | What it does | Used by a normal `mvmc` run? |
|-----------|-------|--------------|------------------------------|
| Unified stage backend | `crates/mvmc-core/src/stage_backend.rs`, `sr_backend.rs` (#421, #437) | one `StageBackend` object composing `SrStages` (Gram, S/g assembly, Cholesky solve, CG product), `PfaffianStages` and `MeasurementStages` | yes, through `MVMC_RS_SR_BACKEND`, `MVMC_RS_MEASURE_PF_BACKEND`, `MVMC_RS_MEASURE_BACKEND` |
| SR through tenferro | `TenferroSr` in `sr_backend.rs` (#421) | the SR stages as tenferro `dot_general`/`cholesky`/`triangular_solve` on the CPU (`cpu-faer`) | yes: `MVMC_RS_SR_BACKEND=tenferro` |
| Measurement stage B through tenferro | `TenferroMeasurement` in `measurement_backend.rs` (#518) | local energy ratio tables and Slater derivative as tensor contractions (real, non-FSZ) | yes: `MVMC_RS_MEASURE_BACKEND=tenferro` |
| Batched Pfaffian and inverse | `crates/mvmc-gpu` (#423) | CPU variants, tenferro-native and `ExtensionOp` batched Pfaffian/inverse of `[n,n,NQP,B]` planes | no: library and harness |
| Validation harness | `crates/mvmc-core/src/accel_validation.rs` (#424) | teacher-forced replay, decision-flip detector, repeatability, benchmark metadata | test tool |
| Sample-batched measurement | `crates/mvmc-core/src/measurement_batch.rs` (#422) | CPU reordering of independent per-sample work; `MVMC_RS_MEASURE_BATCH` | yes, always (output independent of the batch size) |
| Multi-walker runner | `crates/mvmc-core/src/multichain.rs` (#425, #435) | `W` independent chains in one process: fixed-parameter PhysCal and full optimization with C-compatible reductions | library API (no CLI flag) |
| Resident GPU Opt | `gpu/mvmc-gpu-cuda/src/opt.rs`, `opt/model.rs` (#560, #574) | `run_gpu_para_opt_from_namelist`: walker ensemble, sampling, observables and SR resident on the device; per-family support: [`docs/design/gpu-resident-model-support.md`](../design/gpu-resident-model-support.md) (#572) | no: standalone workspace |
| Resident GPU PhysCal | `gpu/mvmc-gpu-cuda/src/opt/physcal.rs` (#636) | `run_gpu_phys_cal_from_namelist`; `mvmc-cuda resident-physcal` | no: standalone workspace |

## 12.3 Building and running

### Default build (CPU only)

Nothing changes: `cargo build --release -p mvmc-cli`. The accelerated code that needs no GPU (the `mvmc-gpu` crate, the tenferro CPU stages, the harness) is part of the normal workspace and its tests run in normal CI.

For the optional tenferro CPU stage, `MVMC_RS_TENFERRO_THREADS` sets the thread count for each tenferro CPU Engine independently of the mVMC inner Rayon pool (`MVMC_RS_INNER_THREADS`) and other Rayon users. Set a positive integer; `1` uses a serial tenferro Engine. If it is unset, the Engine uses `RAYON_NUM_THREADS` if set, or the process's available parallelism otherwise. An invalid or zero selected value is an error. The normal shared stage has one Engine per process; directly creating multiple stages or using multiple MPI processes creates additional Engines. Set the per-process budgets and BLAS threads deliberately, for example `MVMC_RS_TENFERRO_THREADS=1 OPENBLAS_NUM_THREADS=1`, to limit contention. This setting alone does not coordinate or cap all pools, walkers, or MPI ranks.

The experimental `MVMC_RS_TENFERRO_WORKER_BUDGET=N` limits additional workers held by mvmc-rs CPU Engine factories within one process. A `T`-participant Engine reserves `T - 1` workers; if capacity is unavailable it uses a serial Engine without waiting. Capacity returns after the last owning Engine resource is destroyed, including scoped team replacement. `0` permits only serial Engines. Set this before the first CPU Engine is created: changing, setting or unsetting the captured budget later is an error. Unset leaves the previous allocation policy in effect. This does not include walker callers, inner Rayon, BLAS, externally supplied Engines or other MPI ranks; account for their CPU use separately. Acquisition order can affect parallel/serial selection, so this is a validation option, not an established default performance recommendation.

For experiments with a multi-threaded tenferro CPU Engine, `MVMC_RS_TENFERRO_SPIN_POLLS` sets threadteam's wait-spin poll count to a nonnegative integer. If unset, the upstream default is used; `0` skips those spin polls but still yields before sleeping. It is validated even when the Engine is serial. This option affects team waiting and does not limit the number of threads or provide a process-wide budget; choose it only after measuring the workload.

The experimental `MVMC_RS_TENFERRO_TEAM_LIFETIME=stage` setting releases the shared CPU stage's tenferro team after its last active stage or SR-CG solve ends and reacquires it for later use. An SR-CG solve retains the team across its local products, while releasing the shared stage lock between products and MPI collectives. The default, `held`, retains the Engine for the lifetime of the shared stage. Directly constructed stages are unaffected. A measurement handle can live across many samples, so `stage` does not necessarily release the team after every sample. Recreating the Engine discards its local plans and scratch, and repeated acquisitions can cost more; measure the target workload. Thread and wait settings are fixed when the shared stage is created. This setting does not coordinate other Engines, Rayon pools, walkers or MPI ranks and is not a global CPU limit. Set it before the first shared CPU stage is opened; unsupported values fail during backend preflight.

```bash
cargo nextest run -p mvmc-gpu --cargo-profile test-fast          # batched Pfaffian CPU variants and harness
cargo nextest run -p mvmc-core --cargo-profile test-fast \
  -E 'binary(multichain) | binary(multiwalker_sr_435) | binary(stage_backend_437) | binary(sr_backend_421)'
```

### Experimental Linux MPI CPU placement

The [ownership evaluation and policy](../../benchmark/tenferro_thread_budget/ownership-evaluation-20261102/README.md) records the resource accounting and measurements. For a single hoisted CLI rank, count inner participants (including its driver) plus additional Engine reservations. Multiwalker library calls must also count their independent callers; with inner=1, `W` optimization walkers need `W + E` compute participants for an additional-worker budget `E`. Keep every optimizer caller runnable for collectives. Set BLAS=1 and account separately for runtime helpers, external pools, physical-core allocation and quota. Retain Rayon/default work gates; shared-team dispatch remains an experiment for a measured workload/allocation.

`MVMC_RS_MPI_CPUS_PER_RANK=N` assigns each rank a disjoint set of `N` logical CPUs within its node, respecting its initial affinity. Use a Linux MPI build and a multi-rank launch, and set an explicit `MVMC_RS_TENFERRO_WORKER_BUDGET`. The sum of configured inner participants and this additional Engine worker budget must fit `N`. For example:

```bash
cargo build --release -p mvmc-cli --features mpi
MVMC_RS_MPI_CPUS_PER_RANK=2 MVMC_RS_INNER_THREADS=1 \
MVMC_RS_TENFERRO_WORKER_BUDGET=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
mpirun -n 4 target/release/mvmc namelist.def
```

All ranks must use the same placement setting. Impossible assignments or inconsistent controls stop collectively before input/output. Affinity is applied to the MPI calling thread and every existing mvmc inner Rayon worker; future threads inherit their creator's mask. Existing MPI/BLAS threads are unaffected. This option does not inspect cgroup CPU quotas, distinguish SMT siblings, or cap externally created threads. Set BLAS threads separately. The library guard restores the original caller and inner-worker affinities on drop; explicit `restore()` reports restoration errors. No performance benefit is established by this option alone.

### The standalone GPU workspace

The CUDA dependency tree (`tenferro-gpu` with `cuda`, cudarc and the vendored `gpu/tenferro-ext` raw-execution helper) lives in the standalone workspace `gpu/mvmc-gpu-cuda` (own `[workspace]`, own `Cargo.lock`, excluded from the root workspace). Keeping it out of the root lock file is deliberate: an optional `tenferro-gpu` dependency added about 2000 lines to the lock file and to the audit surface of every default build. The workspace uses `mvmc-core` only for input parsing, validation and parameter I/O.

- Build **from `gpu/mvmc-gpu-cuda`** (`cargo build --release --manifest-path gpu/mvmc-gpu-cuda/Cargo.toml ...`).
- **`mvmc-cuda`** (`gpu/mvmc-gpu-cuda/src/bin/mvmc-cuda.rs`) runs only the resident subcommand `mvmc-cuda resident-physcal NAMELIST PARAMS NEW_OUTPUT_DIR`, which writes `energy.dat`, `green_one.dat`, `green_two.dat`, `green_two_ex.dat` and `sampling.txt`. Any other invocation is an error; the CPU runner is `mvmc`.
- The resident optimizer is the library call `mvmc_gpu_cuda::opt::run_gpu_para_opt_from_namelist`; the workers `examples/resident_opt_worker.rs` and `examples/resident_physcal_worker.rs` drive the two runners for benchmarks ([gpu-resident-opt.md](../design/gpu-resident-opt.md), [gpu-resident-physcal.md](../design/gpu-resident-physcal.md)).
- Requirements for a CUDA run: a CUDA driver (`libcuda`) and the CUDA libraries NVRTC, cuBLAS, cuSOLVER and cuRAND at run time (loaded with `dlopen`; no toolkit is needed to *build*). The standalone workspace sets `debug = 0` because linking tenferro can exhaust a hosted runner's disk.

### The CUDA gate (native or docker)

`scripts/run_cuda_gate.sh [native|docker] [extra cargo-test args]` builds the standalone workspace and runs its CUDA tests with `--include-ignored`: the resident Opt/PhysCal library tests (Julia fixtures, 20-step repeatability, transfer and synchronization contracts), the resident interval sampler tests (`resident_interval`) and the interval kernel gates (`interval_gate`, `interval_parity`). The two-device context test and the throughput benchmarks are not part of the gate.

```bash
scripts/run_cuda_gate.sh native                          # needs driver + CUDA libraries on LD_LIBRARY_PATH
scripts/run_cuda_gate.sh docker                          # NVIDIA CUDA toolkit image, --gpus all
MVMC_RS_CUDA_IMAGE=my/cuda:tag scripts/run_cuda_gate.sh docker
```

- `docker` is the default when `docker` exists. The default image is `tenferro-benchmark-cuda:full-verify-20260822`; override with `MVMC_RS_CUDA_IMAGE`. The host `~/.rustup` and `~/.cargo` are mounted so the image needs no Rust toolchain; the script installs OpenBLAS/LAPACK inside the container (as root) and then drops to your user id so `target/` files keep host ownership. Cargo targets go to `gpu/mvmc-gpu-cuda/target`.
- **Fail-closed.** `nvidia-smi -L` must list a device before any test runs; no device is a hard failure, never a pass.
- **Log.** The test log is written to `gpu/mvmc-gpu-cuda/results/cuda-gate.log` (override with `MVMC_RS_CUDA_GATE_OUT`; the directory is git-ignored).
- **CI.** The optional-gates workflow input `cuda_gate` dispatches the job `cuda-gate` on a self-hosted runner (`self-hosted, linux, x64, gpu`; placeholder labels), which calls `scripts/run_cuda_gate.sh native`. Hosted runners have no GPU, so default CI never runs it ([OPTIONAL_GATES_DISPATCH.md](../OPTIONAL_GATES_DISPATCH.md)).

### End-to-end GPU benchmark

`scripts/benchmark_gpu.sh` (`scripts/benchmark/run.py gpu`) times the resident Rust Opt/PhysCal workers against Julia GPU on one NVIDIA GPU for the shared L128 inputs; `--smoke` runs a short configuration and `--dry-run` prints the plan. See `scripts/benchmark/README.md`.

## 12.4 Selecting a backend

`MVMC_RS_SR_BACKEND` ([8.5](08-running.md#85-environment-variables)) chooses the backend that production uses for the SR stages; `MVMC_RS_MEASURE_PF_BACKEND` and `MVMC_RS_MEASURE_BACKEND` choose the measurement stage A Pfaffian source and the stage B consumer. The selectors are parsed by `parse_stage_backend`, `parse_measurement_pfaffian` and `parse_measurement_consumer` (`crates/mvmc-core/src/stage_backend.rs`, `crates/mvmc-core/src/measurement_batch.rs`):

| Value | Backend | Notes |
|-------|---------|-------|
| unset, empty, `c`, `c-order`, `corder`, `default` | C-order CPU (`COrderSr` + `COrderPfaffian` + `COrderMeasurement`) | the default and the parity oracle; byte-identical to the former inline code |
| `tenferro`, `tenferro-cpu` | tenferro eager ops on the CPU (`cpu-faer`); the Pfaffian slot is a prepared tenferro program of the PfaPack extension operation (`TenferroPfaffian`, #613) | feasibility path: 1.1-4x **slower** than OpenBLAS/LAPACK on one thread (faer GEMM instead of SYRK, tensor creation and read-back per stage) |

`MVMC_RS_MEASURE_PF_BACKEND` additionally accepts `calc-m-all` (its default, the per-sample C-order kernels); `MVMC_RS_MEASURE_BACKEND` additionally accepts `direct`/`inline` (the inline C-order calls).

Rules:

- **An invalid value is an error, never a fallback.** The CLI validates every selector once at startup, before any file is read or written: an invalid value, including the removed `cuda`/`cuda:N` (`MVMC_RS_SR_BACKEND="cuda" is not one of c-order, tenferro`), or an unavailable backend prints `error: <VARIABLE>: ...` and exits with the usage-error status `2`, as other option errors do ([8.1](08-running.md#81-the-mvmc-command)); no output directory is created. Under MPI every rank takes part in an agreement, so all ranks stop together. Library callers that reach production code without this preflight (`selected_stage_backend`, `acquire`) panic with the same message.
- **What it routes.** `MVMC_RS_SR_BACKEND` routes the SR stages only: the Gram product (`finalize_oo_store_real` and the complex finalizer, `observables.rs`), S/g assembly and the Cholesky solve (`sr.rs`), and the CG product (`sr_cg.rs`). The sampler always uses the C-order kernels.
- **Non-default backends are built once per process** and shared (a tenferro runtime is expensive); the constant CG operand is cached under an explicit version counter.
- **(observed)** with a default-feature release build (Linux x86_64) on `benchmark/hubbard_chain/inputs/hubbard_chain_L16`, `--nsteps 3 --nsmp 2`: `tenferro` writes the same set of files as `c-order` (`zqp_opt.dat`, `zqp_*_opt.dat`, `zvo_out.dat`, `zvo_var.dat`, `zvo_SRinfo.dat`, `zvo_time_001.dat`); the first two steps of `zvo_out.dat` are identical, the third differs in the last digits and `zqp_opt.dat` differs by at most `5.7e-14` absolute; two `tenferro` runs are byte-identical. The output is **not byte-identical** to the default, which is why it is opt-in and checked against bounds; this short run is an illustration, not a validation.
- **The state is process-wide.** There is no `mvmc` command-line option for the backend; tests use `set_stage_backend_override`.

### Using the backend from Rust

```rust
use mvmc_core::stage_backend::{open_stage_backend, StageBackendKind};

let mut backend = open_stage_backend(StageBackendKind::TenferroCpu)?;
println!("{} via {}", backend.label(), backend.provider());
```

`StageBackend::c_order()` is the oracle; `with_pfaffian` composes a different Pfaffian implementation with a given SR implementation. The harness opens its backend through the same `open_stage_backend` call that production uses ("validated means deployed", design 14.1).

## 12.5 Numerical guarantees and what was validated

### Contract (tenferro CPU stages)

1. The C-order CPU path is bit-for-bit what it was before any accelerated code existed; its fixtures and 20-step repeatability tests run only against it.
2. An accelerated stage is compared with the C-order oracle with **explicit absolute and relative bounds** justified by the operation, the problem scale and the conditioning ([11.4](11-compatibility.md#114-numerical-comparison-policy)). Computed floating-point values are never compared bitwise, and bounds are derived from the first measured divergence, not tuned upward.
3. **Exact** (independently of the backend): RNG initialization and state, draw order and count, the update-type and candidate draws, indices, flags, and the discrete decision protocol.
4. A Metropolis decision may differ only through a *located* numerical flip: the decision margin `|w - u|` must be below the demonstrated weight error. A flip with a larger margin, or any difference in RNG draw count on an identical control path, is a **defect**. After a legitimate flip the trajectory may diverge; comparison then reverts to same-implementation repeatability.
5. Same backend, same input, same seed: reproducible to the stated tolerance (bitwise only where verified).
6. Statistical checks supplement, and never replace, these.

The resident GPU runners follow the Julia GPU contract instead: initialization, proposal, decision and update order of the Julia GPU implementation with its cuRAND XORWOW streams, validated against independent Julia fixtures with explicit bounds (`gpu/mvmc-gpu-cuda/tests/fixtures/opt_julia.md`, `physcal_julia.md`), plus same-implementation 20-step repeatability and transfer/synchronization counters.

### The validation harness

`mvmc_core::accel_validation` (`crates/mvmc-core/src/accel_validation.rs`) implements the checks. It drives a `StageBackend` on a synthetic pairing-orbital system that exercises exactly the linear-algebra stages (it is a validation fixture, not a physics model):

- **Teacher-forced replay** (`replay`). The oracle drives a Metropolis trajectory; at each step the backend under test evaluates the same candidate, so deviations never compound. The report gives, for `pf`, `invM`, the weight `(pf_new/pf_old)^2`, the O store, `S` and `g`, the entries compared, the maximum absolute and relative deviation and the number outside the bound `|a-b| <= abs + rel*max(|a|,|b|)`. Defaults (`Tolerances`): `abs = 1e-12`, `rel = 1e-10`, justified by `n = 6`, O(1) entries, Pfaffian condition below about 1e3 and at most a few hundred summed terms (errors of O(10) eps with a margin of about 1e4 that still catches a layout, sign or dtype defect).
- **Decision recording and flip detection.** Each proposal records oracle and backend weights, the draw, both decisions, the margin and the weight error. Each proposal draws the electron, the empty site and the acceptance draw (always three draws, also when rejected), so the draw sequence and final RNG state are independent of the backend; a test checks that a backend returning wrong weights does not change RNG consumption.
- **Repeatability** (`repeatability`): the same backend twice for 20 steps; discrete state exact, computed fields within bounds.
- **Benchmark metadata** (`BenchMetadata`): revision, CPU model and threads, OS, rustc, tenferro version, provider, thread settings, dtype, batch, warm-ups, iterations, whether tensor creation and read-back are timed. `bench_stages` reports the median of 30 iterations after 5 warm-ups and `None` (not 0) for an unsupported stage. Use the same thread count on both sides of a comparison.

A stage a backend does not implement is listed `Unsupported`/not compared, never replaced by the CPU result. The tenferro CPU variant runs in normal CI.

### Bounds and observed agreement

| Stage | Bound (derived, not tuned) | Observed | Where |
|-------|----------------------------|----------|-------|
| Batched Pfaffian, CPU variants (normal CI) | pfapack-identical, or `1e-12` (Pfaffian) and `1e-11` (inverse) for tensor-native and `ExtensionOp` at `n = 16` | pfapack and `ExtensionOp` bit-identical; tensor-native inverse relative error 7e-18..1.4e-15 (real) | design 11.3, 11.6 |
| tenferro SR vs C order (Gram, CG product) | `2 gamma_k sum abs(terms)` per entry, `gamma_k = k eps/(1-k eps)` (`k` = samples, `+3` complex, `n + samples` for CG) | 1e-15..1.5e-13 relative | `NUMERICAL_COMPARISONS.md` row "tenferro SR backend vs C order"; design 12.3 |
| S and g assembly | `2 eps` relative (same IEEE operations as C) | exactly equal | same |
| Cholesky solve | `2 * 8 n eps kappa(S)` plus residual `8 n eps norm(S) norm(x)` | within bound | same |
| Direct-SR short runs | `abs 1e-10 / rel 1e-8` | within bound | same |
| tenferro measurement stage B vs `COrderMeasurement` | `abs 1e-13 / rel 1e-13` per sample; `abs 1e-11 / rel 1e-8` end to end | 6.8e-16 relative (local energy), 8.9e-16 absolute (Slater derivative) | design 14.5, `measurement_tenferro_518.rs` |

CG is **not** compared step by step: the sampled `S` is ill-conditioned (#358), the squared residual cancels to 1e-16, and a one-ulp change moves the C-order solution by 1e-4..7e-4 at `max_iterations = n`; the policy is repeatability. The complex Gram keeps C's sequential sample order on the default path; the tenferro path is validated by the pure reassociation bound (design section 12.2, "Complex Gram sample order").

## 12.6 What the measurements say

- **Where the time goes on the CPU** (Hubbard chain L=16/32/64, one thread, direct SR): the batched Pfaffian/inverse is 58-61 % of wall time, the sampler's rank-one updates 16-30 %, local energy plus Slater derivative 6-10 %, and the SR matrices only 0.2-0.3 % (0.6-1.5 % with the O store and Gram). SR dominates only when `NPara` reaches the thousands (design 3).
- **tenferro CPU stages** are a feasibility path, 1.1-4x slower than OpenBLAS/LAPACK on one thread (design 12.3).
- **Resident GPU runners**: the L128 optimization and PhysCal timings against Julia GPU on an RTX 3060 are recorded in [gpu-resident-opt.md](../design/gpu-resident-opt.md), [gpu-resident-physcal.md](../design/gpu-resident-physcal.md) and the reports of `scripts/benchmark_gpu.sh`.

## 12.7 Multi-walker runs

A single Markov chain cannot profit from a GPU (one attempted hop costs 2-13 microseconds on one core; a device round trip costs tens of microseconds), so the unit of batching is the independent *walker*. `crates/mvmc-core/src/multichain.rs` runs `W` walkers in one process. There is **no command-line flag**; the API is `run_phys_cal_multichain(&MultiChainConfig)` for fixed-parameter sampling and measurement and `run_para_opt_multichain(&ParaOptMultiChainConfig)` for the whole optimization.

- **Seeds are the C group seeds.** Walker `w` uses `RndSeed + group_base + w` (`walker_seed`, `crates/mvmc-core/src/multichain.rs:88`), the seed of C group `group1` (`init_gen_rand(RndSeed + group1)`, [4.6](04-theory-sampling.md#46-parallelism-inside-the-sampler)). `W` walkers on one process are the chains of `W` groups of a grouped C run, each with its own host SFMT stream; a time-based seed (`RndSeed < 0`) is resolved once for all walkers. Each walker is byte-for-byte the serial run with seed offset `group_base + w`, and `W = 1` equals the plain serial run.
- **Draw order and count are untouched.** Tests compare for every walker the complete final SFMT state (624 words and position), the consumed word count, the recorded `(weight, draw)` sequence and the whole sampling/observable state against the serial run; results do not depend on the worker pool size.
- **Threads.** PhysCal: one rayon pool over walkers (`threads`, default `min(W, cores)`), each walker single-threaded, BLAS pinned to one thread. Optimization: one OS thread per walker, because the cross-walker reductions need all walkers running. With more walker threads than OpenBLAS's `MAX_THREADS`, OpenBLAS prints `precompiled NUM_THREADS exceeded, adding auxiliary array for thread metadata`; this is a warning, not an error. 320 Hubbard L128 walkers (OpenBLAS 0.3.26 `MAX_THREADS=64`, 36 cores) completed three times with identical per-walker SFMT states, configurations and outputs, at about 24 GB peak memory (#642) **(observed)**.
- **Optimization with C-compatible reductions.** `run_para_opt_multichain` runs the unchanged optimizer against an in-process `ThreadReducer` (`crates/mvmc-core/src/multichain.rs:433`) that stands in for the MPI communicator: `HO`, `OO`, the energy and weights are summed over walkers, every walker solves the same SR system, `SROptO` stays walker-local, exactly the collective sequence of an ungrouped MPI run with `NSplitSize = 1`. The sum is a rank-order left fold, `((v0 + v1) + v2) + ...`, independent of thread scheduling; `MPI_Allreduce` leaves the order to the MPI library, so Rust and C can differ by last-bit roundoff for `W > 2`. The bound used against C is `|a - b| <= 1e-13 + 1e-12 |b|`.
- **Validated** (`crates/mvmc-core/tests/multiwalker_sr_435.rs`, no MPI, C or Julia needed): every ungrouped C cell of the #179 matrix with 2 and 4 ranks and width 1 (real, complex, FSZ, OptTrans; direct and CG) per walker for the counters, saved configurations and SFMT state exactly, and the reduced operands within the bound; two walkers equal the two groups of the 4-rank `NSplitSize = 2` cells; `W = 1` is byte-identical to the serial optimization; a four-step 4-walker run is bitwise repeatable (beyond step 1 the ill-conditioned solve amplifies roundoff, #358, so trajectories are checked for repeatability, not forced onto C); the same holds with the tenferro SR backend.
- **Output files** are written by walker 0 only, the output root, as in the MPI run.
- **Not covered.** `NSplitSize > 1` (the QP/sample split of one chain over ranks) is not a walker; MPI (`mvmc` with the `mpi` feature remains the way to scale out across processes, [8.4](08-running.md#84-mpi-and-grouped-execution)).
- **Scaling record.** On the Heisenberg-chain fixture the CPU walker runner efficiency `T(1)/T(W)` was 0.94 at `W = 2` and 0.56 at `W = 8`, on a host loaded by other jobs (load average 58-81 on 36 threads), so these are upper bounds on the cost of the runner, not a property of it (design 10.6).
- **Decision margins** are recorded observationally (`DecisionSummary`: proposals, accepted, smallest margin `|w - u|`, near-flip count). The smallest margin on the Heisenberg fixture was 1.5e-7 to 3.2e-7, three orders of magnitude above a `1e-10` weight perturbation, so a backend within the harness bounds cannot flip a decision on that input.

## 12.8 Known limits

- **tenferro stages** cover real and complex SR and the real, non-FSZ measurement stage B (no RBM, no two-body Green terms); tenferro has no skew-symmetric factorization upstream, so the tenferro measurement Pfaffian source is the pure-Rust PfaPack kernel registered as a tenferro extension operation (`mvmc_core::tenferro_pfaffian`): the tensor program, its preparation and dispatch are tenferro, the factorization is the C-order PfaPack sequence (#613). Other upstream gaps recorded by the design (no determinism contract, no argmax, `Tensor` not `Clone`, `Module` not `Send`): design 4.7 and 11.5.
- **Resident GPU runners** accept real normal orbitals with finite real parameters, Gutzwiller/Jastrow, DH2/DH4 and real RBM factors, local spins with `NExUpdatePath` 0-2, direct or CG SR (`NSRCG`; PhysCal ignores it), real OptTrans weights (an `OptTrans` definition activates them; Opt optimizes them), serial execution and `2 Ne <= 1024` (PhysCal also `NLanczosMode = 0`); anything else is rejected before any device work (`gpu/mvmc-gpu-cuda/src/opt/model.rs`). They do not reproduce C/SFMT trajectories by design.
- **RTX 3060 FP64.** The recorded GPU measurements come from a card that runs FP64 at 1/64 of FP32; a data-center GPU is expected to shift the ratios.
- **No GPU in CI.** The CUDA gate is optional on a self-hosted runner; the GPU numbers in the design documents come from local runs.

## 12.9 Reproducing the numbers

```bash
cargo nextest run -p mvmc-gpu --cargo-profile test-fast         # CPU batched Pfaffian variants, harness
cargo run --release -p mvmc-core --example sr_backend_bench     # SR stages, C order vs tenferro CPU
scripts/run_cuda_gate.sh docker                                 # resident GPU Opt/PhysCal and interval tests
bash scripts/benchmark_gpu.sh --smoke                           # resident Rust vs Julia GPU, short run
```

Record the metadata block with every result (device model, driver, CUDA library versions, tenferro version, host CPU and load, thread settings, git revision); GPU results without it are not comparable, and the reference environment for numerical comparison remains Linux x86_64 ([11.4](11-compatibility.md#114-numerical-comparison-policy)).
