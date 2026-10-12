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

### 2.1 RNG contract: the tape must be a raw SFMT word stream

The current tape is **structured**, four `u32` words per hop:
`w0 = mi` (`w0 % ne`), `w1` (spin `= w1 >> 31`), `w2` = destination rank over
empty sites, `w3` = accept uniform (`w3 / 2^32`). The gate and bench generate it
randomly, so they validate the *mechanism*, not C parity.

C's draw order does not match this. For the normal (non-FSZ) real mode, one
hopping proposal (`extern/mVMC-1.3.0/src/mVMC/vmcmake.c`,
`vmcmake_real.c:138-202`, `makeCandidate_hopping` at `vmcmake.c:515`):

1. `updateType = getUpdateType(NExUpdatePath)` - for `path==0` (hopping only) no
   draw; for `path>=1` one `genrand_real2()` (`vmcmake.c:606-627`).
2. `mi = gen_rand32() % Ne` - one word (`SFMT.c:410`).
3. `s = (genrand_real2() < 0.5)` - one word (`SFMT.h:107`, `to_real2`).
4. destination: `do { rj = gen_rand32() % Nsite; ... } while (eleCfg[rj+s*Nsite]
   != -1 || LocSpn[rj]==1)` - **rejection sampling**, a variable number of
   words (at least one), not a one-word rank over empty sites.
5. `if (!rejectFlag) { ... if (w > genrand_real2()) accept }` - one word,
   skipped only when the destination loop exhausted `icnt_max` (all sites
   occupied; impossible while `Ne < Nsite`).

Both `gen_rand32()` and `genrand_real2()` consume exactly one 32-bit word
(`to_real2(x) = x * 2^-32`), so the whole hop is a raw SFMT word stream. The
interval tape must therefore be the **raw `u32` stream**, and the kernel must
reproduce steps 1-5 with a per-walker variable cursor. The structured tape is
wrong and must be replaced (or the kernel must be handed the raw words and
decode them itself). This is a parity blocker, not a performance detail: the
current structured tape changes both the draw count and, through the destination
distribution, the trajectory.

Consequences for the kernel:

* Add per-walker `word_off[w]`; each hop consumes words of whatever type the C
  algorithm asks for at that point.
* Reproduce `getUpdateType(NExUpdatePath)` (read `NExUpdatePath` from the
  input; the benchmark Hubbard inputs use a fixed value).
* Reproduce the `while (LocSpn[ri]==1)` redraw of `mi`/`s` (no-op in normal
  mode, but keep it for generality or reject unsupported inputs).
* Bound the raw buffer generously and handle exhaustion by host refill (the
  destination loop is geometric; expected attempts at half filling are
  `Nsite/(Nsite-Ne)`).

Validation: against the lock-step host-draw CUDA sampler
(`device_sampler.rs`) on a frozen fixture - same seed, same number of hops,
compare the final SFMT state, word count, accepted configuration and Pfaffian
for every walker, not just the energy. This is the deterministic reference
parity of `AGENTS.md`, not a statistical energy check.

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
fn group_capable(&self) -> bool { false }

/// One walker of a batched sampling round. `walker` is the device slot
/// (`0..world_size`); every walker of the run must call this exactly once per
/// round and rendezvous through `reducer` (the `ThreadReducer` barrier).
fn group_sample_real(
    &mut self,
    walker: usize,
    data: &ExpertModeData,
    state: &mut VmcOptimizationState,
    rng: &mut Sfmt19937Rng,
    timing: &mut dyn SamplerTiming,
    reducer: &dyn Reducer,
) -> Result<SampleStats, StageError> {
    let _ = walker;
    self.sample_real(data, state, rng, timing, reducer)
}
```

The group backend does not hold the walker `state`/`rng` across the barrier;
the coordinator owns **host staging slots** (plain `Vec`s). Each walker copies
its own state into its slot, rendezvous, one elected walker (rank 0) does the
upload/launch/download for all slots, rendezvous, each walker copies its slot
back into its state. `state` and `rng` stay thread-local (no aliasing).

### 3.2 Acquisition

For a group-capable backend, `acquire_sampler_kind` must return a **per-walker
handle** backed by an `Arc<GroupCoordinator>` instead of one `MutexGuard` for
the whole run. The coordinator guards only the device critical section (rank 0's
transfer + launch); the rendezvous uses `Reducer::barrier()` (`reducer.rs:33`),
which `ThreadReducer` already implements. `check_serial_reducer` is relaxed to
accept a group-capable backend with `reduction_size == 1`.

Run routing: `run_para_opt_multichain` passes the walker index `w` to the
optimizer; the call sites switch to `group_sample_real(w, ...)` when the backend
is group-capable, else stay on `sample_real`.

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

## 4. Measurement plan (E2E, L=128)

After 2.1, 2.2 and 3:

* Deterministic parity: interval vs. lock-step on a frozen fixture at fixed `W`
  and hops (final SFMT state, word count, configuration, Pfaffian per walker).
* `benchmark/host_gpu_comparison/run_all.sh` at L=128 vs. the Julia GPU path
  (`extern/Julia-mVMC/MVMCOptimizers.jl/ext/cuda_*`).
* Expected: sampling ~0.8 s (measured) + host measurement ~1.7 s + SR ~0.2 s
  ~ 2.7 s, against the Julia GPU Opt 3.07 s / PhysCal 0.30 s of #544.

## 5. Open questions

1. `NExUpdatePath` of the benchmark inputs (drives the `getUpdateType` draw and
   whether exchange/exchange-hopping must be modelled). The normal-mode sampler
   here supports hopping only; other paths must be rejected, not silently
   approximated.
2. Whether `pf` alone (plus device-side inverse) is enough for the host
   measurement of the sampled configurations, or whether the inverse must also
   come back for the measured samples (section 3.4).
3. Buffer sizing / host refill for the geometric destination loop.
