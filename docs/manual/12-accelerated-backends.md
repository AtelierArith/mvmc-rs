# 12. Accelerated and GPU backends

[Contents](README.md) · Previous: [11. Compatibility and differences](11-compatibility.md) · Next: [Appendix: how the citations were checked](appendix-checks.md)

This chapter describes the optional accelerated implementations of the tensor-shaped parts of `mvmc-rs` (tenferro on the CPU, CUDA through the standalone `gpu/mvmc-gpu-cuda` workspace), how to build and select them, what has been validated and to what bounds, and what is known not to work well yet. The design record is [docs/design/gpu-readiness.md](../design/gpu-readiness.md) (issue #417, sections 10-15); the validation policy is the section "Accelerated-backend validation (#424)" of [docs/NUMERICAL_COMPARISONS.md](../NUMERICAL_COMPARISONS.md); the CUDA gate dispatch is in [docs/OPTIONAL_GATES_DISPATCH.md](../OPTIONAL_GATES_DISPATCH.md). Numbers quoted here are copied from those documents and were measured by their authors on the hardware named there (Linux x86_64, 2x NVIDIA GeForce RTX 3060, driver 580.178.04, CUDA 12.9 toolkit, tenferro 0.7.1, a 36-thread shared host); they were **not re-measured** for this manual.

## 12.1 Policy: what the accelerated paths are, and are not

- **The CPU C-order path is the default and the oracle.** Every accelerated implementation is an *additional* implementation behind a stage-level backend trait. It never replaces, reorders or reroutes the CPU path, and no default build, default test or fixture depends on a GPU or on the tenferro GPU crates (design 5.1, 5.3).
- **Correctness first.** The direction decision recorded on issue #417 is *option B*: the device-resident sampler (PR #445) is kept as a validated foundation, but effort now goes to large-scale stochastic reconfiguration (where the GPU already wins) and to CPU optimization; sampler-side GPU work is parked and is to be re-evaluated on FP64-strong GPUs (A100/H100) with data from the benchmark suite of issue #450 ([12.9](#129-the-benchmark-and-validation-suite-450)). Throughput work that touches production routing (#452) is on hold until correctness-first work allows it.
- **No silent fallback.** An unsupported stage, dtype, device or build returns an error (`StageError::Unsupported`, `BackendError`); the program never substitutes the CPU result.
- **RNG and Metropolis decisions are host-side and exact.** SFMT draws, candidate generation, acceptance decisions, projection counters and file output stay on the host in every backend. An accelerated backend may change the *numerical values* of weights and matrices within tolerance; it must not change draw order or count ([12.5](#125-numerical-guarantees-and-what-was-validated)).
- **Selected explicitly.** Nothing is chosen automatically; the default is `c-order`.

## 12.2 What exists

| Component | Where | What it does | Wired into a normal `mvmc` run? |
|-----------|-------|--------------|---------------------------------|
| Backend selection and device report | `crates/mvmc-core/src/backend.rs` (`BackendKind`, `DeviceReport`, `cuda_gate_decision`; feature `gpu-cuda`, #420) | CPU always available; `Cuda(n)` only with the feature and a registered provider | no (library API) |
| Unified stage backend | `crates/mvmc-core/src/stage_backend.rs`, `sr_backend.rs` (#421, #437) | one `StageBackend` object composing `SrStages` (Gram, S/g assembly, Cholesky solve, CG product, composite `direct_step`/`cg_step`) and `PfaffianStages` | **SR stages only**, through `MVMC_RS_SR_BACKEND` |
| SR through tenferro | `TenferroSr` in `sr_backend.rs` (#421) | the SR stages as tenferro `dot_general`/`cholesky`/`triangular_solve`, on the CPU (`cpu-faer`) or on CUDA | yes: `MVMC_RS_SR_BACKEND=tenferro`, and `cuda[:N]` from a program that registers the provider |
| Batched Pfaffian and inverse | `crates/mvmc-gpu` (CPU variants, tenferro-native, `ExtensionOp`) and `gpu/mvmc-gpu-cuda` (CUDA kernel), #423 | Pfaffian and inverse of `[n,n,NQP,B]` skew-symmetric planes, `f64` and `Complex64`, per-plane status | no: exercised by the harness and the gates; the sampler and measurement still call `calc_m_all_*` |
| Validation harness | `crates/mvmc-core/src/accel_validation.rs` (#424) | teacher-forced replay, decision-flip detector, repeatability, benchmark metadata | test and gate tool |
| Sample-batched measurement | `crates/mvmc-core/src/measurement_batch.rs` (#422) | CPU reordering of independent per-sample work; `MVMC_RS_MEASURE_BATCH` | yes, always (CPU; output independent of the batch size) |
| Multi-walker runner | `crates/mvmc-core/src/multichain.rs` (#425, #435) | `W` independent chains in one process: fixed-parameter PhysCal and full optimization with C-compatible reductions | library API (no CLI flag) |
| Device-resident sampler | `crates/mvmc-core/src/device_sampler.rs` and `gpu/mvmc-gpu-cuda/src/device_sampler.rs` (#434) | lock-step multi-walker real-mode sampler, state resident on the GPU | no: validated foundation, parked ([12.8](#128-known-limits)) |
| Pinned/async transfer helper | `gpu/mvmc-gpu-cuda/src/transfer.rs` (#432) | pinned memory pool, asynchronous copies, event ordering | used by the device sampler and the device-resident SR |
| Device-resident SR | `gpu/mvmc-gpu-cuda/src/sr_device.rs`, `stages.rs` (#447) | Gram, S/g, Cholesky and the CG loop on the device (real parameters) | **no**: opt-in stage backend, production call sites are not routed (#452, on hold) |

## 12.3 Building and running

### Default build (CPU only)

Nothing changes: `cargo build --release -p mvmc-cli`. The accelerated code that needs no GPU (the `mvmc-gpu` crate, tenferro on the CPU, the host sampler service, the harness) is part of the normal workspace and its tests run in normal CI.

```bash
cargo nextest run -p mvmc-gpu --cargo-profile test-fast          # batched Pfaffian CPU variants and harness
cargo nextest run -p mvmc-core --cargo-profile test-fast \
  -E 'binary(multichain) | binary(multiwalker_sr_435) | binary(stage_backend_437) | binary(sr_backend_421) | binary(device_sampler)'
```

### The `gpu-cuda` feature and the standalone workspace

`mvmc-core` and `mvmc-cli` have a feature `gpu-cuda` (`crates/mvmc-core/Cargo.toml`, forwarded by `crates/mvmc-cli/Cargo.toml`). It adds **no dependency** and leaves `Cargo.lock` unchanged: it only enables the `CudaProvider` registry of `mvmc_core::backend`. `cargo check -p mvmc-cli --features gpu-cuda` therefore needs no CUDA toolkit, and **does not link CUDA**.

The CUDA dependency tree (`tenferro-gpu` with `cuda`, cudarc, a patched `lru`) lives in the standalone workspace `gpu/mvmc-gpu-cuda` (own `[workspace]`, own `Cargo.lock`, excluded from the root workspace; design 10.1, 10.2). The decision to keep it out of the root lock file is deliberate: an optional `tenferro-gpu` dependency added about 2000 lines to the lock file and to the audit surface of every default build. Consequences for users:

- GPU work is built **from `gpu/mvmc-gpu-cuda`** (`cd gpu/mvmc-gpu-cuda && cargo ...`), not with `cargo build --features gpu-cuda` at the root.
- The provider is registered by calling `mvmc_gpu_cuda::install()` (`gpu/mvmc-gpu-cuda/src/lib.rs:69`) in the program that wants CUDA. The stock `mvmc` binary does not call it, so `MVMC_RS_SR_BACKEND=cuda` is rejected there with a usage error ([12.4](#124-selecting-a-backend)). **The CUDA-capable command is `mvmc-cuda`**: a thin binary of the standalone workspace (`gpu/mvmc-gpu-cuda/src/bin/mvmc-cuda.rs`) that calls `install()` and then runs the unchanged `mvmc` driver (`mvmc_cli::run_cli`, so it takes exactly the options of [8.1](08-running.md#81-the-mvmc-command); it does not forward the `mpi` feature). Build and run it from `gpu/mvmc-gpu-cuda`: `cargo build --release --bin mvmc-cuda`, then `MVMC_RS_SR_BACKEND=cuda target/release/mvmc-cuda namelist.def`; in docker use the image and mounts of `scripts/run_cuda_gate.sh`.
- Requirements for a CUDA run: a CUDA driver (`libcuda`), and the CUDA toolkit libraries cuBLAS, cuSOLVER and NVRTC at run time (they are loaded with `dlopen`; no toolkit is needed to *build*). The tenferro survey on issue #417 gives CUDA 12.6.2 as the floor (12.8 for full capability), and the measurements used a 12.9 toolkit. The standalone workspace sets `debug = 0` because linking tenferro can exhaust a hosted runner's disk.

### The CUDA gate (native or docker)

`scripts/run_cuda_gate.sh [native|docker] [extra cargo-test args]` builds and runs the ignored gate tests of `gpu/mvmc-gpu-cuda/tests/` (`cuda_gate`, `pfaffian_gate`, `transfer_gate`, `sampler_gate`, `sr_device_gate`) with `MVMC_RS_CUDA_GATE=1`:

```bash
scripts/run_cuda_gate.sh native                          # needs driver + CUDA toolkit libraries on this host
MVMC_RS_CUDA_GATE=1 scripts/run_cuda_gate.sh docker      # NVIDIA CUDA toolkit image, --gpus all
MVMC_RS_CUDA_IMAGE=my/cuda:tag scripts/run_cuda_gate.sh docker
```

- `docker` is the default when `docker` exists. The default image is `tenferro-benchmark-cuda:full-verify-20260822` (CUDA 12.9.2); override with `MVMC_RS_CUDA_IMAGE`. The host `~/.rustup` and `~/.cargo` are mounted so the image needs no Rust toolchain; the script installs OpenBLAS/LAPACK inside the container (as root) and then drops to your user id so `target/` files keep host ownership. Cargo targets go to `gpu/mvmc-gpu-cuda/target`.
- **Fail-closed.** `MVMC_RS_CUDA_GATE=1` (or `require`) and no device is a hard failure; with the variable unset and no device the tests print `cuda-gate: ExplicitSkip: skipped, no device (...)`, which is *not* a pass (`cuda_gate_decision`, `crates/mvmc-core/src/backend.rs:232`). The script always sets the variable to `1`.
- **Reports.** Results are written to `gpu/mvmc-gpu-cuda/results/` (`cuda-gate.md`, `cuda-validation.md`, `cuda-roundtrip.md`, `cuda-transfer.md`; override with `MVMC_RS_CUDA_GATE_OUT`, `..._VALIDATION_OUT`, `..._ROUNDTRIP_OUT`, `..._TRANSFER_OUT`; the directory is git-ignored). Each report carries the device, compute capability, memory, driver (NVML), CUDA driver API, NVRTC, cuBLAS, cuSOLVER and tenferro versions (`DeviceReport`; an entry that cannot be queried prints `unavailable`).
- **CI.** The optional-gates workflow input `cuda_gate` dispatches the job `cuda-gate` on a self-hosted runner (`self-hosted, linux, x64, gpu`; placeholder labels). It is outside the bounded-family ledger: not selected is NotRun, no device fails the job. Hosted runners have no GPU, so default CI never runs it ([OPTIONAL_GATES_DISPATCH.md](../OPTIONAL_GATES_DISPATCH.md), "CUDA gate (#420)").

### Benchmarks of the individual components

Each script takes `native` or `docker` and writes a CSV with a metadata block (host load average, device report):

| Script | Component | Output |
|--------|-----------|--------|
| `scripts/run_pfaffian_bench.sh` | batched Pfaffian/inverse (#423) | `benchmark/gpu_pfaffian/results/pfaffian_batched.csv` |
| `scripts/run_device_sampler_bench.sh` | device-resident sampler against the CPU multi-chain runner (#434); `MVMC_RS_SAMPLER_CORES=0-3` restricts the container to four host cores (`docker --cpuset-cpus`) | `benchmark/gpu_device_sampler/results/` |
| `scripts/run_sr_device_bench.sh` | device-resident SR step (#447) | `benchmark/gpu_sr_device/results/` |

Run them on a quiet host: both the CPU rows and the host side of the device rows are sensitive to other load, and the checked-in numbers were taken on a shared host (design 11.4, 13.3, 15.3).

## 12.4 Selecting a backend

`MVMC_RS_SR_BACKEND` ([8.5](08-running.md#85-environment-variables)) chooses the backend that production uses for the SR stages. It is validated at startup by `validate_selected_stage_backend` (`crates/mvmc-core/src/stage_backend.rs:423`) and read by `selected_stage_backend` (`crates/mvmc-core/src/stage_backend.rs:411`) and parsed by `parse_stage_backend` (`crates/mvmc-core/src/stage_backend.rs:370`):

| Value | Backend | Notes |
|-------|---------|-------|
| unset, empty, `c`, `c-order`, `corder`, `default` | C-order CPU (`COrderSr` + `COrderPfaffian`) | the default and the parity oracle; byte-identical to the former inline code |
| `tenferro`, `tenferro-cpu` | tenferro eager ops on the CPU (`cpu-faer`), `TenferroSr`; the Pfaffian slot is `Unsupported` | feasibility path: 1.1-4x **slower** than OpenBLAS/LAPACK on one thread (faer GEMM instead of SYRK, tensor upload/download per stage) |
| `cuda`, `cuda:N` | tenferro CUDA on device `N` (default 0): `TenferroSr` on the device plus the batched CUDA Pfaffian kernel | needs the `mvmc-cuda` binary (feature `gpu-cuda` and a registered provider, `mvmc_gpu_cuda::install()`) |

Rules:

- **An invalid value is an error, never a fallback.** The CLI validates the selector once at startup, before any file is read or written (`stage_backend::validate_selected_stage_backend`): an invalid value (`MVMC_RS_SR_BACKEND is not one of c-order, tenferro, cuda[:N]`) or an unavailable backend (no `gpu-cuda` feature, no registered provider, device ordinal out of range, missing CUDA runtime library) prints `error: MVMC_RS_SR_BACKEND: ...` and exits with the usage-error status `2`, as other option errors do ([8.1](08-running.md#81-the-mvmc-command)); no output directory is created. Under MPI every rank takes part in an agreement, so all ranks stop together. The same preflight covers `MVMC_RS_MEASURE_PF_BACKEND` (the measurement Pfaffian source, [8.5](08-running.md#85-environment-variables)) with the same contract. Library callers that reach production code without this preflight (`selected_stage_backend`, `acquire`) still panic with the same message. Before issue #464 the CLI panicked here with exit status 101.
- **What it routes.** The SR stages only: the Gram product (`finalize_oo_store_real` and the complex finalizer, `observables.rs`), S/g assembly and the Cholesky solve (`sr.rs`), and the CG product (`sr_cg.rs`). The sampler and the measurement Pfaffian (`calc_m_all_*`) always use the C-order kernels.
- **Non-default backends are built once per process** and shared (a tenferro runtime or a CUDA context is expensive); the constant CG operand is cached under an explicit version counter.
- **The device-resident SR step is not reachable through the variable.** `cuda_resident_stage_backend` (`gpu/mvmc-gpu-cuda/src/stages.rs:198`) builds the opt-in backend whose `direct_step`/`cg_step` run on the device, but production `sr.rs` and `sr_cg.rs` still assemble S/g from the host `OO` array, so selecting it in a real optimizer run is the follow-up #452 (on hold).
- **Which binary.** `MVMC_RS_SR_BACKEND=tenferro mvmc ...` works in the stock binary; `cuda[:N]` needs `mvmc-cuda` ([12.3](#the-gpu-cuda-feature-and-the-standalone-workspace)). **(observed)** with a default-feature release build (Linux x86_64) on `benchmark/hubbard_chain/inputs/hubbard_chain_L16`, `--nsteps 3 --nsmp 2`: `tenferro` writes the same set of files as `c-order` (`zqp_opt.dat`, `zqp_*_opt.dat`, `zvo_out.dat`, `zvo_var.dat`, `zvo_SRinfo.dat`, `zvo_time_001.dat`); the first two steps of `zvo_out.dat` are identical, the third differs in the last digits and `zqp_opt.dat` differs by at most `5.7e-14` absolute; two `tenferro` runs are byte-identical. `mvmc-cuda` built and run in the docker image of the gate on an RTX 3060 (driver 580.178.04, CUDA 12.9.2) with `MVMC_RS_SR_BACKEND=cuda` on the same input: all three steps ran, `zvo_out.dat` agrees with `c-order` to a maximum relative `2.4e-15` and `zqp_opt.dat` to `8.0e-14` absolute (`tenferro` in the same container: `8.6e-15` and `7.1e-15`), well inside the SR bounds of [12.5](#125-numerical-guarantees-and-what-was-validated); `cuda:9` on that two-GPU host exits `2` with `CUDA device 9 requested but 2 device(s) found`. Natively on a host without the CUDA runtime libraries `mvmc-cuda` exits `2` with `backend unavailable: Unable to dynamically load the "cudart" shared library`. The output of `tenferro` and `cuda` is **not byte-identical** to the default, which is why they are opt-in and checked against bounds; these short runs are illustrations, not a validation.
- **The state is process-wide.** There is no `mvmc` command-line option for the backend; tests use `set_stage_backend_override`.

### Using the backend from Rust

```rust
use mvmc_core::stage_backend::{open_stage_backend, StageBackendKind};

mvmc_gpu_cuda::install();                       // register the CUDA provider
let mut backend = open_stage_backend(StageBackendKind::Cuda(0))?;
println!("{} via {}", backend.label(), backend.provider());
```

`StageBackend::c_order()` is the oracle; `with_pfaffian` composes a different Pfaffian implementation with a given SR implementation. The harness and the gates open their backend through the same `open_stage_backend` call that production uses ("validated means deployed", design 14.1).

## 12.5 Numerical guarantees and what was validated

### Contract

1. The C-order CPU path is bit-for-bit what it was before any accelerated code existed; its fixtures and 20-step repeatability tests run only against it.
2. An accelerated stage is compared with the C-order oracle with **explicit absolute and relative bounds** justified by the operation, the problem scale and the conditioning ([11.4](11-compatibility.md#114-numerical-comparison-policy)). Computed floating-point values are never compared bitwise, and bounds are derived from the first measured divergence, not tuned upward.
3. **Exact** (independently of the backend): RNG initialization and state, draw order and count, the update-type and candidate draws, indices, flags, and the discrete decision protocol.
4. A Metropolis decision may differ only through a *located* numerical flip: the decision margin `|w - u|` must be below the demonstrated weight error. A flip with a larger margin, or any difference in RNG draw count on an identical control path, is a **defect**. After a legitimate flip the trajectory may diverge; comparison then reverts to same-implementation repeatability.
5. Same backend, same input, same seed, same device and library versions: reproducible to the stated tolerance (bitwise only where verified; not assumed for CUDA).
6. Statistical checks supplement, and never replace, these.

### The validation harness

`mvmc_core::accel_validation` (`crates/mvmc-core/src/accel_validation.rs`) implements the checks. It drives a `StageBackend` on a synthetic pairing-orbital system that exercises exactly the linear-algebra stages (it is a validation fixture, not a physics model):

- **Teacher-forced replay** (`replay`, `crates/mvmc-core/src/accel_validation.rs:382`). The oracle drives a Metropolis trajectory; at each step the backend under test evaluates the same candidate, so deviations never compound. The report gives, for `pf`, `invM`, the weight `(pf_new/pf_old)^2`, the O store, `S` and `g`, the entries compared, the maximum absolute and relative deviation and the number outside the bound `|a-b| <= abs + rel*max(|a|,|b|)`. Defaults (`Tolerances`, `accel_validation.rs:76`): `abs = 1e-12`, `rel = 1e-10`, justified by `n = 6`, O(1) entries, Pfaffian condition below about 1e3 and at most a few hundred summed terms (errors of O(10) eps with a margin of about 1e4 that still catches a layout, sign or dtype defect).
- **Decision recording and flip detection.** Each proposal records oracle and backend weights, the draw, both decisions, the margin and the weight error. Each proposal draws the electron, the empty site and the acceptance draw (always three draws, also when rejected), so the draw sequence and final RNG state are independent of the backend; a test checks that a backend returning wrong weights does not change RNG consumption.
- **Repeatability** (`repeatability`, `accel_validation.rs:578`): the same backend twice for 20 steps; discrete state exact, computed fields within bounds.
- **Benchmark metadata** (`BenchMetadata`, `accel_validation.rs:635`): revision, CPU model and threads, GPU and driver/CUDA/cuBLAS/cuSOLVER versions, OS, rustc, tenferro version, provider, thread settings, dtype, batch, warm-ups, iterations, whether upload and download are timed. `bench_stages` reports the median of 30 iterations after 5 warm-ups and `None` (not 0) for an unsupported stage. Use the same thread count on both sides of a comparison.

A stage a backend does not implement is listed `Unsupported`/not compared, never replaced by the CPU result. The tenferro CPU variant runs in normal CI; the CUDA variant is part of the gate.

### Bounds and observed agreement

| Stage | Bound (derived, not tuned) | Observed | Where |
|-------|----------------------------|----------|-------|
| Batched Pfaffian/inverse, CUDA vs `CpuPfapack` | `16 n eps cond_F(A)` on the worst of the inverse and Pfaffian relative errors (forward-error scale of a backward-stable inverse); independent checks `max abs(A A^-1 - I)`, skew defect of `A^-1`, `Pf^2 = det A` (4x) | real Pfaffian bit-identical to pfapack at every size; observed/allowed at most 1.4e-2 (c64, `n = 2`), below 1e-3 for `n >= 16`; max inverse relative error 7.7e-15 (f64, `n = 128`) and 3.3e-13 (c64, `n = 128`) | design 11.3, `pfaffian_gate.rs` |
| Batched Pfaffian, CPU variants (normal CI) | pfapack-identical, or `1e-12` (Pfaffian) and `1e-11` (inverse) for tensor-native and `ExtensionOp` at `n = 16` | pfapack and `ExtensionOp` bit-identical; tensor-native inverse relative error 7e-18..1.4e-15 (real) | design 11.3, 11.6 |
| Zero pivots and non-finite planes | exact status agreement, neighbouring planes unaffected | CPU and GPU statuses agree exactly | design 11.3 |
| tenferro SR vs C order (Gram, CG product) | `2 gamma_k sum abs(terms)` per entry, `gamma_k = k eps/(1-k eps)` (`k` = samples, `+3` complex, `n + samples` for CG) | 1e-15..1.5e-13 relative; CUDA at most 6.5e-15 | `NUMERICAL_COMPARISONS.md` row "tenferro SR backend vs C order"; design 12.3 |
| S and g assembly | `2 eps` relative (same IEEE operations as C) | exactly equal | same |
| Cholesky solve | `2 * 8 n eps kappa(S)` plus residual `8 n eps norm(S) norm(x)` | within bound | same |
| Direct-SR short runs | `abs 1e-10 / rel 1e-8` | within bound | same |
| Device-resident Gram | per entry `2 k eps (abs(O) abs(O)^T)_ij` | worst observed/bound 6.2e-2 | design 15.2, `sr_device_gate.rs` |
| Device-resident direct solution | `4 n eps kappa(S)` relative, residual `1e-9` | relative error 4.7e-14 at `n = 129`, 2.4e-13 at `n = 400` (bound from a host estimate of kappa), residuals 5e-15..9e-14 | design 15.2 |
| Device-resident CG product | `4 (k + m) eps scale_i` | worst observed/bound 5.8e-4 | design 15.2 |
| Device sampler vs CPU sampler | weights within `1e-12 + 1e-10 w`; decision sequence, configuration and final RNG state exact | 0 flips, 0 defects on all three gated cases; max relative weight difference 5.9e-9 (Hubbard L=32), 5.9e-12 (L=16), 3.6e-13 (Heisenberg); smallest margin 1.9e-4 | design 13.2, `sampler_gate.rs` |
| Device sampler on the host service (CI) | bitwise: statistics, every `(weight, draw)`, all 624 SFMT words, saved configurations, for `W = 1` and `W = 3` | exactly equal | `crates/mvmc-core/tests/device_sampler.rs` |

CG is **not** compared step by step: the sampled `S` is ill-conditioned (#358), the squared residual cancels to 1e-16, and a one-ulp change moves the C-order solution by 1e-4..7e-4 at `max_iterations = n`. The device CG agrees with the host to `2.9e-15` on a well-conditioned problem (equal iteration counts) and to `1.8e-16..2e-15` after 1-8 iterations on an ill-conditioned one but differs by `6.7e-4` after 119 iterations; this is the amplification the C-order Rust path shows against instrumented C, not a defect of either side, and the policy is repeatability (the device trajectory is bitwise repeatable and finite). The complex Gram keeps C's sequential sample order on the default path; the tenferro path is validated by the pure reassociation bound (design section 12.2, "Complex Gram sample order").

Limits of this validation, stated by the design documents: the SR operands of the device gate are synthetic (real-store structure, controlled conditioning), not a sampled run of the optimizer; there is one GPU model (RTX 3060); no accelerated path has been checked inside a full production optimization run against C.

## 12.6 What the measurements say

All from the design document (RTX 3060, FP64 at 1/64 of FP32, shared host; see the caveats there). They are a feasibility record, not a speed-up claim for `mvmc`.

- **Where the time goes on the CPU** (Hubbard chain L=16/32/64, one thread, direct SR): the batched Pfaffian/inverse is 58-61 % of wall time, the sampler's rank-one updates 16-30 %, local energy plus Slater derivative 6-10 %, and the SR matrices only 0.2-0.3 % (0.6-1.5 % with the O store and Gram). SR dominates only when `NPara` reaches the thousands.
- **Batched Pfaffian** (data resident on the device, `B >= 8`): 5.5-14x (f64 `n = 16`), 9-24x (`n = 32`), 11-19x (`n = 64`), about 9x (`n = 128`) faster than one pfapack thread; against all 36 threads 1.5-11x. With upload and download in every call it is slower than 36 threads almost everywhere: do not offload plane by plane through host memory.
- **SR on the device** (design 15.3, 15.4): CG with `NPara * samples >= 3e6` is 8.5-11x faster than all 36 cores with the matrix resident (5.7-8.1x end to end, 15-22x against one thread; 0.38 s against 2.75 s for 36 cores at 10^4 x 10^4, 50 iterations). Direct SR is 4.4-5.5x faster than one core end to end for `NPara >= 3000` but only 1.1-1.8x faster than 36 cores and 0.6x (slower) at 1000 x 10000. Below `NPara * samples ~ 1e6` the device is no faster than the CPU.
- **Device sampler** (design 13.4): does not beat the multi-core CPU runner on this machine. With all 36 cores the CPU runner is 1.15x (L=128, `W=1`) to 11.4x (L=16, `W=64`) faster; with 4 host cores the device wins only at L=128 (1.14-1.37x for `W >= 8`). A round costs 31-70 microseconds against 3-43 microseconds per CPU hop, and the Pfaffian is only 32-45 % of the sampler, so even an infinitely fast device cannot give more than about 1.5-1.8x.
- **Transfers** (design 10.7): the PCIe link and pinned memory are not the bottleneck; tenferro 0.7.1 reaches only 0.5-1.7 GB/s up and 1-5 GB/s down against 7-13 GB/s for plain cudarc copies. The in-repo helper (`transfer.rs`) bypasses tenferro's `upload_tensor`/`download_tensor` for hot transfers.

## 12.7 Multi-walker runs

A single Markov chain cannot profit from a GPU (one attempted hop costs 2-13 microseconds on one core; a device round trip costs tens of microseconds), so the unit of batching is the independent *walker*. `crates/mvmc-core/src/multichain.rs` runs `W` walkers in one process. There is **no command-line flag**; the API is `run_phys_cal_multichain(&MultiChainConfig)` for fixed-parameter sampling and measurement and `run_para_opt_multichain(&ParaOptMultiChainConfig)` for the whole optimization.

- **Seeds are the C group seeds.** Walker `w` uses `RndSeed + group_base + w` (`walker_seed`, `crates/mvmc-core/src/multichain.rs:88`), the seed of C group `group1` (`init_gen_rand(RndSeed + group1)`, [4.6](04-theory-sampling.md#46-parallelism-inside-the-sampler)). `W` walkers on one process are the chains of `W` groups of a grouped C run, each with its own host SFMT stream; a time-based seed (`RndSeed < 0`) is resolved once for all walkers. Each walker is byte-for-byte the serial run with seed offset `group_base + w`, and `W = 1` equals the plain serial run.
- **Draw order and count are untouched.** Tests compare for every walker the complete final SFMT state (624 words and position), the consumed word count, the recorded `(weight, draw)` sequence and the whole sampling/observable state against the serial run; results do not depend on the worker pool size.
- **Threads.** PhysCal: one rayon pool over walkers (`threads`, default `min(W, cores)`), each walker single-threaded, BLAS pinned to one thread. Optimization: one OS thread per walker, because the cross-walker reductions need all walkers running.
- **Optimization with C-compatible reductions.** `run_para_opt_multichain` runs the unchanged optimizer against an in-process `ThreadReducer` (`crates/mvmc-core/src/multichain.rs:433`) that stands in for the MPI communicator: `HO`, `OO`, the energy and weights are summed over walkers, every walker solves the same SR system, `SROptO` stays walker-local, exactly the collective sequence of an ungrouped MPI run with `NSplitSize = 1`. The sum is a rank-order left fold, `((v0 + v1) + v2) + ...`, independent of thread scheduling; `MPI_Allreduce` leaves the order to the MPI library, so Rust and C can differ by last-bit roundoff for `W > 2`. The bound used against C is `|a - b| <= 1e-13 + 1e-12 |b|`.
- **Validated** (`crates/mvmc-core/tests/multiwalker_sr_435.rs`, no MPI, C or Julia needed): every ungrouped C cell of the #179 matrix with 2 and 4 ranks and width 1 (real, complex, FSZ, OptTrans; direct and CG) per walker for the counters, saved configurations and SFMT state exactly, and the reduced operands within the bound; two walkers equal the two groups of the 4-rank `NSplitSize = 2` cells; `W = 1` is byte-identical to the serial optimization; a four-step 4-walker run is bitwise repeatable (beyond step 1 the ill-conditioned solve amplifies roundoff, #358, so trajectories are checked for repeatability, not forced onto C); the same holds with the tenferro SR backend.
- **Output files** are written by walker 0 only, the output root, as in the MPI run.
- **Not covered.** `NSplitSize > 1` (the QP/sample split of one chain over ranks) is not a walker; MPI (`mvmc` with the `mpi` feature remains the way to scale out across processes, [8.4](08-running.md#84-mpi-and-grouped-execution)).
- **Scaling record.** On the Heisenberg-chain fixture the CPU walker runner efficiency `T(1)/T(W)` was 0.94 at `W = 2` and 0.56 at `W = 8`, on a host loaded by other jobs (load average 58-81 on 36 threads), so these are upper bounds on the cost of the runner, not a property of it (design 10.6).
- **Decision margins** are recorded observationally (`DecisionSummary`: proposals, accepted, smallest margin `|w - u|`, near-flip count). The smallest margin on the Heisenberg fixture was 1.5e-7 to 3.2e-7, three orders of magnitude above a `1e-10` weight perturbation, so a backend within the harness bounds cannot flip a decision on that input.

## 12.8 Known limits

- **RTX 3060 FP64.** The measured card runs FP64 at 1/64 of FP32. All device-versus-CPU ratios here come from it; GEMM-bound stages (direct SR Gram and Cholesky, FP64 GEMM at `n >= 1024`) are at parity with a 36-core host. A data-center GPU (FP64 at 1/2 of FP32) is expected to shift the ratios; this has **not** been measured. The A100 evaluation through the suite of #450 exists for that reason ([12.9](#129-the-benchmark-and-validation-suite-450)).
- **tenferro transfer path.** tenferro 0.7.1 copies host-device data at 0.5-1.7 GB/s (up) and 1-5 GB/s (down) because `upload_tensor`/`download_tensor` add device synchronization, CubeCL staging and redundant host copies (87 % of an upload at 256 MB). The upstream issue is **tensor4all/tenferro-rs#2009**; the request text is drafted in `docs/design/tenferro-transfer-request-draft.md`. The in-repo pinned/async helper is the workaround for the device sampler and the resident SR path, and large plane uploads should be avoided altogether by building planes on the device. Other upstream gaps recorded by the design (no Pfaffian or skew factorization, no determinism contract, no argmax, `Tensor` not `Clone`, `Module` not `Send`): design 4.7 and 11.5.
- **Parked sampler-side GPU work.** The device-resident sampler (#434) is implemented, validated and kept, but per direction B the further work (moving the projection factors and `log_proj_ratio` to the device, kernel tuning with shared-memory LTL^T, persistent kernels or CUDA graphs, batched local energy #426) is deferred. It covers real-mode normal sampling with hopping and exchange only: complex and FSZ modes, the measurement stage and MPI QP splitting are not covered. The host's `inv_m_real` is stale after a device run (the device holds the truth).
- **Production routing is on hold (#452).** Neither the batched Pfaffian nor the device-resident SR step is called by a normal `mvmc` run. Only the SR stages are routed (`MVMC_RS_SR_BACKEND`), and `cuda` is reachable only from a program that registers the provider.
- **Device-resident SR** covers real parameters only (direct Cholesky and CG); complex parameters need the complex Gram and Hermitian Cholesky, which are not implemented. CG over several ranks needs the cross-rank reduction that the resident loop does not do, so it is for single-process runs. Memory is `8 (n^2 + n_active^2 + n samples)` bytes for direct SR (2.4 GB at 10^4 x 10^4) and `8 n samples` for CG.
- **Batched Pfaffian engine limits.** `n <= 1024`, `n` even, `f64` or `Complex64`; planes are processed in chunks of at most 3 GiB of device buffers. The CPU walker scaling above and the benchmark tables are single-run numbers on shared hosts (about +-30 % between campaigns).
- **Reproducibility** on a GPU is "same device, same library versions, to tolerance" until a deterministic reduction order is available upstream.
- **No GPU in CI.** The CUDA variant of every check is an optional gate on a self-hosted runner; the numbers in this chapter come from local runs on one machine.

## 12.9 The benchmark and validation suite (#450)

Issue #450 is implemented ([PR #462](https://github.com/AtelierArith/mvmc-rs/pull/462)). It is a portable, function-level suite that first checks the **numerics** of every GPU, tenferro and CPU variant against the C-order oracle and only then records timings, so that the parked sampler-side work can be re-evaluated on an FP64-strong GPU (A100) that the maintainer runs on a separate server. The authoritative reference is `benchmark/function_suite/README.md`; the harness is the `function_suite` example of `gpu/mvmc-gpu-cuda`; a reference report from an RTX 3060 is `benchmark/function_suite/results/rtx3060-reference.md`.

### Running it

```bash
git submodule update --init --recursive
scripts/bench/run_all.sh --native --quick --out bench-out          # about 10 minutes, smoke run
scripts/bench/run_all.sh --native --full --gpu 0 --out bench-out   # about 2 hours on one GPU
scripts/bench/run_all.sh --docker --full --gpu 0 --out bench-out   # nvidia/cuda image instead of a host toolkit
scripts/bench/run_all.sh --native --preflight-only                 # only the checks
```

- **Requirements.** An NVIDIA driver (`nvidia-smi` works), Rust 1.96 or newer, python3, and either (native) a CUDA toolkit 12.6 or newer with NVRTC, cuBLAS and cuSOLVER plus the OpenBLAS development files (`libopenblas.so`), or (docker) docker with the NVIDIA container toolkit; the image is `nvidia/cuda:12.9.1-devel-ubuntu24.04` unless `MVMC_RS_CUDA_IMAGE` is set, OpenBLAS is installed inside it and the host `~/.rustup` and `~/.cargo` are mounted.
- **Preflight.** The script stops with a message before building if the driver, the selected `--gpu`, rustc, the toolkit version, the CUDA libraries or OpenBLAS are missing. Without `--native`/`--docker` it picks native when a toolkit is found, else docker.
- **Steps.** Release build, input generation for the Hubbard chain (L = 16..256 in `--full`) with the Rust StdFace port (`mvmc --dry-run`), then the families `pfaffian`, `sr` (plus a separate one-core CPU pass), `sr_resident`, `sampler` and `transfers`. A failing family does not stop the others. The `--full` sampler timing grid is capped at `W * L^2 <= 2.1e6` (`MVMC_BENCH_WORK_CAP` raises it); skipped points are explicit `SKIPPED` rows.
- **Output.** `DIR/results-<host>-<date>.tar.gz` with one CSV per family (`csv/`), `report.md`, raw logs (`logs/`) and `metadata.txt` (GPU model, compute capability, FP64 peak if in the script's table, driver, CUDA, tenferro and cudarc versions, CPU, OS, rustc, git revision, thread environment). The CSV schema is `family,function,variant,dtype,params,reps,median_s,min_s,max_s,dev_metric,dev_value,dev_bound,dev_ratio,verdict,note`.
- **The device-resident SR hook (#447).** `run_all.sh` runs `gpu/mvmc-gpu-cuda/examples/bench_sr_resident.rs` if that program exists and otherwise records one `NotAvailable` row; it never invents numbers. The existing `bench_sr_device` example is a separate program that does not write this schema yet, so the family is `NotAvailable` today.

### What the verdicts mean

The **numerical verdict is the primary result** and the process exits non-zero on any `FAIL` or `ERROR`; timings (median, min, max after warm-up) are secondary reference columns, and the report's "GPU-ize?" recommendation is never `YES` for a function with a failing check.

| Verdict | Meaning |
|---------|---------|
| `PASS` | the deviation from the oracle is within the stated bound (`dev_ratio` = observed/bound <= 1), or an exact check (RNG state, draw count, configuration, status codes, bitwise copy) holds |
| `FAIL` | outside the bound, or an exact check failed |
| `ERROR` | an unexpected run-time error; counts as a failure |
| `ORACLE` | the reference row itself (pfapack, `COrderSr` with OpenBLAS, the CPU sampler) |
| `INFO` | timing only |
| `NotAvailable` | the function or hook does not exist in this build; no numbers |
| `SKIPPED` | not run (memory or work cap); the note says why |
| `KNOWN-ISSUE` | a report label, not a CSV value: a `FAIL` that matches a tracked defect ([#465](https://github.com/AtelierArith/mvmc-rs/issues/465) resident inverse of a sampler walker, [#466](https://github.com/AtelierArith/mvmc-rs/issues/466) tenferro-native c64 Pfaffian at n = 128). It still counts as `FAIL`; an untracked failure is labelled `NEW` |

Bounds per family (the derivations are in the source headers of `gpu/mvmc-gpu-cuda/examples/function_suite/*.rs`; they are not tuned to a device and must not be loosened to make a run pass):

- **Pfaffian and inverse:** per plane, `max(inverse rel, Pfaffian rel) <= 16 n eps cond(A)` with `cond = ||A||_F ||A^-1||_F` of the actual plane, plus the independent invariants `A inv = I`, skew symmetry of `inv` and `Pf^2 = det`, and identical per-plane status codes (zero pivot, NaN).
- **SR stages:** relative max-norm against `COrderSr`, with bounds `4 k eps` (Gram), `8 eps` (S/g), `8 n eps kappa` (Cholesky solve, Gershgorin kappa), `4 (k + n) eps` (CG product) and `2 K kappa 4 (k + n) eps` (a K-iteration CG solve).
- **Sampler:** the hard gate is exact: RNG state and draw count, electron configuration, and the free-run decision sequence bit-identical to the CPU, with zero draw mismatches and zero unexplained decision flips ("defects") from the #424 teacher. A free-run divergence passes only when the teacher located decision flips, as the numerical policy allows. The weight deviation of a teacher-forced run is additionally checked per walker against `2 * 16 * (n + s) * eps * kappa / sqrt(w_min)`.
- **Transfers:** a bitwise round trip (a copy has no arithmetic).

The sampler weight bound rests on stated assumptions, not on a proof: the constant 16 is the one of the Pfaffian bound; `s`, the number of updates between recomputes, is taken as the worst case in which every proposal of the recompute window is accepted; `kappa` is the largest plane condition number sampled only at the start and at the end of each walker's run (not along the trajectory); and `w_min`, the smallest nonzero reference weight, stands in for the cancellation factor of the ratio's dot product (it also contains the projection factors). The bound is therefore loose, 600 to 20000 times above the deviation observed on the RTX 3060 (observed/allowed 5e-5 to 1.7e-3 for L = 16..256); tightening it needs per-proposal deviations exposed by the teacher.

### Sending results back and comparing machines

Send the single archive `bench-out/results-<host>-<date>.tar.gz` (a few hundred kilobytes). A non-zero exit status means a `FAIL` or `ERROR` was recorded; the archive is written either way and should be sent either way. To merge archives, with only the standard library:

```bash
uv run --no-project scripts/bench/analyze.py results-a100.tar.gz results-rtx3060.tar.gz --out comparison.md
```

Each argument is one machine (an archive or an unpacked directory). The report lists, in order, (1) numerical validation: verdict counts per machine, the failures with their KNOWN-ISSUE or NEW label, and the worst deviation/bound per function and variant across machines; (2) timing as a reference: speedup tables and break-even sizes (batch size, NPara, walker count); (3) the "GPU-ize?" recommendation per function, gated by numerics; (4) `NotAvailable` and `SKIPPED` rows; then the metadata of each machine.

Timings of the sampler depend on the host cores as much as on the GPU (its host side runs one thread per walker), so compare against the CPU multichain row of the same machine; CPU rows are sensitive to other load, so run on a quiet host.

## 12.10 Reproducing the numbers

```bash
cargo nextest run -p mvmc-gpu --cargo-profile test-fast                         # CPU variants, harness
MVMC_RS_CUDA_GATE=1 scripts/run_cuda_gate.sh docker                             # every CUDA gate
scripts/run_pfaffian_bench.sh docker                                            # design 11.7
scripts/run_device_sampler_bench.sh docker                                      # design 13.5
scripts/run_sr_device_bench.sh docker                                           # design 15.5
cargo run --release -p mvmc-core --example sr_backend_bench                     # SR stages, C order vs tenferro CPU
```

Record the metadata block with every result (device model, driver, CUDA/cuBLAS/cuSOLVER versions, tenferro version, host CPU and load, thread settings, git revision); GPU results without it are not comparable, and the reference environment for numerical comparison remains Linux x86_64 ([11.4](11-compatibility.md#114-numerical-comparison-policy)).
