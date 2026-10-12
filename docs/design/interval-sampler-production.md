# Device-resident interval sampler: production wiring (issue #547)

Status: design and parity analysis. The device mechanism and its throughput are
measured on the A100 (branch `gpu/device-interval-sampler-547`); the production
wiring and the RNG-contract correction below are **not implemented yet**. This
document fixes the architecture and the open parity questions before the code
lands, in the spirit of `docs/design/gpu-readiness.md` (which this extends,
section 5.5).

## 1. What exists today

`gpu/mvmc-gpu-cuda` has a raw-CUDA interval sampler:

* `interval_kernels.cu::k_interval_hop` - one CUDA block per walker, a whole
  interval per launch, driven by a `u32` tape; Gutzwiller/Jastrow delta,
  rank-one ratio, Metropolis, rank-two inverse update, and a per-walker
  accepted-window.
* `interval_rebuild.cu::interval_assemble` / `interval_commit` around the
  existing `pfinv_f64`, batched over every walker that requests a rebuild
  (device recomputation).
* `interval.rs::IntervalSampler` - prepare / upload_tape / run / rebuild /
  download, on tenferro's captured stream through the vendored
  `tenferro_ext::raw_exec`.

Measured (A100, L=128, `nsite=128`, `nqp=8`):

| case | result |
|---|---|
| walker sweep, no rebuild (`256` hops/walker) | `W=1` 84 us/hop -> `W=320` **0.78 us/hop (1.28 M hops/s)** |
| full run `W=320`, `2688` hops/walker, 2277 device rebuilds, 9 rounds | **0.805 s (1.07 M hops/s)**, finite Pfaffians |

A single block is latency-bound (84 us/hop, worse than the per-hop path), so the
gain requires walker batching.

## 2. Two problems to solve before production

### 2.1 Parity reference: the Julia GPU tape, not the C draw order

The GPU path's parity target is the **Julia GPU implementation**
(`extern/Julia-mVMC/MVMCOptimizers.jl/ext`), not C/SFMT. `gpu_sample!`
(`cuda_state.jl:269`) runs `cached_sampling_kernel!` / `cooperative_sampling_kernel!`
(`cuda_sampling_update.jl`), one block per walker (`b = blockIdx().x`), on a
device `UInt32` tape `[5, batch, steps]` (`[word, walker, step]`, word-fastest):

```julia
draws = tape === nothing ? CUDA.rand(UInt32, 5, batch, steps) : tape   # cuda_state.jl:282
```

Per step (`cuda_sampling_update.jl:278-389`):

* `tape[1]`: source electron, `a = tape[1] % nelec + 1 + spin*nelec`;
* `tape[2]`: spin, `tape[2] < 0x80000000 ? 0 : 1`;
* `tape[3]`: destination. With `empty_destinations` (the default for
  `nsite >= 64`, `JULIA_MVMC_GPU_PROPOSALS=empty`) it is the rank over empty
  sites (`empty_hopping_destination`, `:196`); otherwise a raw `% nsite` with
  rejection, which is the non-default `self-loop` proposal;
* `tape[4]`: exchange threshold, `tape[4]/2^32 < exchange_fraction`
  (`exchange_probability`: `NExUpdatePath==0 -> 0.0`, `==1 -> 0.5`, `==2 -> 1.0`);
* `tape[5]`: acceptance uniform, `u = tape[5]/2^32`, accept when
  `log(u) < logratio` with `logratio = 2*(d_corr + log|ip_new| - log|ip_old|)`.

The rebuild trigger is `accepted_since_refactor[b] > nsite` (or a non-rank-one
candidate) - the same rule the interval kernel uses (`window > nsite`).

So the interval kernel's structured tape is the `NExUpdatePath == 0` case of this
contract (`empty` destination, no exchange). What is missing is only the unused
`tape[4]` exchange word and the exact `[5, walker, step]` layout. For a
**shared-tape parity test** that feeds the same device tape to Julia and Rust,
the interval tape adopts the Julia layout (5 words, `[word, walker, step]`) and
takes `exchange_fraction` (0 for the benchmark; reject `> 0` until the exchange
branch is implemented).

The earlier draft of this section claimed the tape had to reproduce C's
`makeCandidate_hopping` draw order. That is **wrong for the GPU path**: the
rejection-sampled destination and the raw SFMT word stream describe the CPU/C
and lock-step parity contract (`vmcmake.c:515`, `SFMT.h:107`). The Julia GPU
sampler deliberately uses a device-RNG structured tape and is the reference
here. C stays authoritative for the CPU path and for the physics (ratios,
projections, signs, operation order).

The RNG stream itself: Julia draws with the device RNG (`CUDA.rand`, seeded by
`CUDA.seed!(seed)`); exact equality with CUDA.jl's stream is not claimed
(`docs/design/gpu-xorwow.md`). Parity is therefore established by feeding an
**identical tape** to both implementations - `gpu_sample!(s; tape=...)` accepts
one - which isolates the sampler semantics from the RNG stream. The XORWOW
device generator remains the production word source, with a separate,
already-documented contract.

### 2.2 Acquisition model: walker batching needs a coordinator

The optimizer calls the sampler inside the per-walker threads of
`run_para_opt_multichain` (`multichain.rs:582`): each of the `W` threads calls
`run_para_opt_from_namelist_observed`, which acquires the sampler itself
(`run.rs:1036`) and calls `sample_real` per SR step (`run.rs:1070`,
`run.rs:1339`). Three obstacles:

1. `acquire_sampler_kind` locks the process-wide shared sampler for the whole
   run (`sampler_backend.rs:354`), so `W` walkers serialize on the mutex and no
   batching can happen.
2. `check_serial_reducer` rejects `W>1` (`sampler_backend.rs:233`); the CUDA
   sampler is intentionally single-chain today.
3. `SamplerStages::sample_real` sees one `&mut state`; a batched launch needs
   all `W` states resident on the device at once.

The W-block launch must therefore be a single owning coordinator, not `W`
independent launches (kernels on one stream serialize; `W` single-block
launches would run at 84 us/hop each).

## 3. Proposed production architecture

### 3.1 Seam

Add a group entry point to `SamplerStages` (default: call `sample_real`, so
C-order and the existing `W=1` CUDA path are unchanged):

```rust
// Landed in `sampler_backend.rs` (commit on the #547 branch).
fn group_capable(&self) -> bool { false }   // lock-step and C-order: false

/// One walker of a batched sampling round: every walker of the run calls this
/// exactly once per sample series and rendezvous through `reducer`. The walker
/// index is `reducer.rank()`, the walker count `reducer.world_size()`.
fn sample_real_group(
    &mut self,
    data: &ExpertModeData,
    state: &mut VmcOptimizationState,
    rng: &mut Sfmt19937Rng,
    timing: &mut dyn SamplerTiming,
    reducer: &dyn Reducer,
) -> Result<SampleStats, StageError> {
    self.sample_real(data, state, rng, timing, reducer)
}
```

The two call sites (`run.rs`) use `sample_real_group`; the default keeps every
existing backend byte-identical. `Reducer` already exposes `rank()`,
`world_size()` and `barrier()` (`reducer.rs:119`, `:104`, `:33`), and
`ThreadReducer` implements all three, so no walker index has to be threaded
through the optimizer signatures.

The group backend does not hold the walker `state`/`rng` across the barrier;
the coordinator owns **host staging slots** (plain `Vec`s). Each walker copies
its own state into its slot, rendezvous, one elected walker (rank 0) does the
upload/launch/download for all slots, rendezvous, each walker copies its slot
back into its state. `state` and `rng` stay thread-local (no aliasing).

### 3.2 Acquisition and two implementation constraints

For a group-capable backend, `acquire_sampler_kind` must return a **per-walker
handle** backed by an `Arc<GroupCoordinator>` instead of one `MutexGuard` for
the whole run, and `check_serial_reducer` is relaxed to accept a group-capable
backend with `reduction_size == 1`.

**Constraint 1 - no mutex around the rendezvous.** The per-walker handle must
not be a `MutexGuard` held for the whole call: if walker 0 held a lock across
`reducer.barrier()` while walker 1 blocked on the lock, the barrier would never
complete. The coordinator therefore guards only the device critical section
(rank 0's transfer + launch), uses one lock per staging slot (each walker writes
only its own slot), and does the rendezvous outside every lock.

**Constraint 2 - the device backend is thread-affine.** `CudaBackend` /
`with_backend_session` require `&mut` on a single owner, and the tenferro
context is captured per thread. Only rank 0 touches the device, and rank 0 is
always the same walker thread across every SR step, so the group keeps a
**thread-local device backend** on the rank-0 thread while the resident
`DeviceBuffer`s (which are `Send`) live in the `Arc`ed coordinator. The other
walkers only touch host slots.

### 3.3 Step flow (one SR step, `W` walkers)

1. Each walker generates its raw SFMT word chunk from its own `rng` (its own
   stream; seeds stay `RndSeed + group_base + w`, `gpu-readiness.md` 5.5).
2. Each walker packs `ele`, `inv`, `pf`, `aw`, `logip` and its tape into its
   staging slot (host memory).
3. `reducer.barrier()`.
4. Rank 0 uploads all slots (one contiguous transfer), launches
   `k_interval_hop` with `W` blocks plus any rebuilds, runs the batched rebuild
   for the walkers that asked, and downloads all slots.
5. `reducer.barrier()`.
6. Each walker unpacks its slot into `state`; the caller runs the host
   measurement (`accumulate_observables`) and SR reduction as today.

### 3.4 Transfer budget at L=128

Per SR step, `W=320`, `n=256` (`2*Nsite`), `nqp=8`:

* `inv`: `320 * 8 * 256^2 * 8 B = 1.34 GB` per direction. That is too much for
  a per-step round trip.

The rebuild can refresh `inv` on the device (already implemented), so the host
only needs the **configuration** (`ele`, `320 * 256 * 4 B = 328 KB`) and the
Pfaffians `pf` (`320 * 8 * 8 B = 20 KB`) in and out - the inverse stays
resident and is only re-derived on the device. The host measurement
(`accumulate_observables`) does need the inverse of the samples it measures, but
that is a separate, sample-batched stage (`measurement_batch.rs`), not the
sampler round trip. This keeps the per-step transfer small; the exact split
(sampler state vs. measurement input) is the main thing to pin down when
implementing 3.3.

### 3.5 `W=1` is a hard error

A single-block launch is latency-bound and slower than the CPU (84 us/hop).
The group backend must document and reject `W=1` (or route it to C order), like
`check_serial_reducer` does today.

### 3.6 The real work is the sampler loop, not the launch

`sample_real` is not a thin wrapper: the production path is
`vmc_make_sample_real_staged` (`crates/mvmc-core/src/sampling/driver.rs:128`),
which runs `init_loc_spn`, a typed preflight, the real Slater sync, then either
restores the burn snapshot or runs `make_initial_sample_normal_with_info` (which
**draws RNG**), `calc_m_all_real`, `sampling_log_ip_real`, and the
`nOutStep`/`nInStep` warm-up + measure loops that save configurations, advance
the counters, update the burn flag and return `SampleStats`. A device sampler
must reproduce all of that for the rest of the optimizer and the host
measurement to stay unchanged.

There is precedent: the CUDA lock-step path already reimplements this loop in
`run_lockstep_real_one_inline`
(`crates/mvmc-core/src/device_sampler.rs:1446`) behind the `SamplerService`
trait, drawing the candidate on the host each hop and calling the device only
for the Pfaffian stage. The interval group backend is the same kind of
reimplementation, with two changes:

* draw each walker's proposals into a **device tape** for a whole interval
  instead of per hop (the Julia GPU contract of 2.1), and
* run `k_interval_hop` once per interval for all `W` walkers (plus the batched
  device rebuild), reading the saved configurations back at the measure points.

So the remaining work is: (a) `IntervalSampler` per-walker upload/download of
`ele`/`occ`/`inv`/`pf`/`aw`/`logip` and the tape; (b) a group backend that
mirrors `run_lockstep_real_one_inline`'s loop but drives `IntervalSampler`
through the 3.3 staging/rendezvous protocol; (c) the acquisition change of 3.2
(per-walker handles + relaxed `check_serial_reducer`); (d) the state extraction
from `VmcOptimizationState` (`electron_config`, `slater_matrix.inv_m_real`,
`slater_matrix.pf_m_real`, the projection tables). This is the bulk of #547 and
is where the group seam of 3.1 hooks in.

## 4. Measurement plan (E2E, L=128)

After 2.1, 2.2 and 3:

* Deterministic parity: feed the **same device tape** to Julia's
  `gpu_sample!(s; tape=...)` and the interval sampler on identical initial
  states and compare the configuration, Pfaffian, counts and acceptance per
  step and per walker (not just the energy). This isolates the sampler from the
  RNG stream. Supplement with a Rust-only repeatability test (same tape, same
  result).
* `benchmark/host_gpu_comparison/run_all.sh` at L=128 vs. the Julia GPU path
  (`extern/Julia-mVMC/MVMCOptimizers.jl/ext/cuda_*`).
* Expected: sampling ~0.8 s (measured) + host measurement ~1.7 s + SR ~0.2 s
  ~ 2.7 s, against the Julia GPU Opt 3.07 s / PhysCal 0.30 s of #544.

## 5. Open questions

1. `NExUpdatePath` of the benchmark inputs (drives `exchange_fraction`:
   `0 -> 0.0`, `1 -> 0.5`, `2 -> 1.0`). The benchmark Hubbard inputs are all
   `NExUpdatePath=0`, so the interval sampler implements `exchange_fraction == 0`
   only and rejects `> 0`, matching the Julia `empty`/`hopping` default.
2. Whether `pf` alone (plus device-side inverse) is enough for the host
   measurement of the sampled configurations, or whether the inverse must also
   come back for the measured samples (section 3.4).
3. Whether the interval kernel should also implement the non-default
   `self-loop` (raw `% nsite`) proposal and the exchange branch, or keep the
   `empty`/hopping contract and reject the rest.
