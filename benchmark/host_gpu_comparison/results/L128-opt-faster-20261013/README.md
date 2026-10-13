# L128 GPU Opt: Rust faster than Julia, 2026-10-13

Rust's resident GPU Opt public API takes **4.851715 s**, versus **5.137973 s**
for Julia GPU: **5.57% less time** (Rust/Julia = 0.944286). Every measured Rust
warm call is faster than every measured Julia warm call. The preceding Rust
implementation took [7.687900 s](../L128-opt-resident-20261013/README.md).
Related to #550 and #560.

## Matched workload and timing

Half-filled Hubbard chain, **L=128, 320 walkers, 20 SR steps, FP64, seed 79**,
128 warmup and 128 sampling proposals per walker per step. Both implementations
use 8 QP planes, 772 parameters and 860,160 proposals. The input files are the
same as the preceding comparison; their hashes are in [environment.json](environment.json).
Both processes run directly on this Linux host, on **RTX 3060 GPU 1** with one
host/BLAS thread. No container, concurrent test, build or benchmark was used.

| Public API | Warm median | Warm min–max | First call |
| --- | ---: | ---: | ---: |
| Rust resident GPU | **4.851715 s** | 4.850720–4.866548 s | 14.746787 s |
| Julia GPU | 5.137973 s | 5.133839–5.154681 s | 55.124394 s |

| Warm repetition | Rust (s) | Julia (s) |
| --- | ---: | ---: |
| 1 | 4.853179027 | 5.154680842 |
| 2 | 4.866548422 | 5.134940316 |
| 3 | 4.851427116 | 5.137973162 |
| 4 | 4.850720373 | 5.133839196 |
| 5 | 4.851715147 | 5.143356748 |

Each process remains alive for one first call followed by five warm calls. The
languages alternate, reversing order on even repetitions. Each timed public API
call constructs a fresh run: parsing/model setup, device allocation and uploads,
configuration initialization, warmup, all sampling/measurement/SR steps and final
energy/parameter downloads are included. Rust run-state destruction is included.
Julia returns its device state; explicit Julia GC and diagnostic configuration
downloads occur outside its timer. Both finish GPU execution before stopping the
timer. Receipt serialization and process startup are outside; startup measurements
are in [timings.json](timings.json). First-call times are single observations,
including kernel/library startup, rather than warm medians.

## Implementation

For matrices of side 128 or larger, the resident sampler queues four accepted
rank-two changes as device vectors. Proposal ratios include all pending changes,
including their separate electron-slot unit-vector terms. Materialization uses
the existing factorization output allocation as an alternate inverse buffer;
each return and rebuild boundary drains the queue and publishes into the usual
inverse storage. Four warps per walker balance QP cooperation and occupancy.

Reconstruction uses paired skew elimination with up to four pending elimination
updates in shared memory, followed by an inverse that grows two columns at a
time. Failed factors, nonfinite inverses and strongly cancelling projected
overlaps retry the conventional factorizer on device before publication. This
follows Julia's numerical strategy. Small matrices retain their established
operation order; the legacy host-coordinated interval entry remains separate.

The remaining supported-input restrictions and API are documented in
[GPU-resident Opt](../../../../docs/design/gpu-resident-opt.md). In particular,
singular-QP cofactor fallback remains under #560. This result concerns the resident
Opt API; it does not establish a PhysCal speedup or redirect the C-order CLI.

## Numerical and transfer verification

All six calls per language pass comparison of **every one of the 20 energies
and 772 final parameters**, using the unchanged `atol=rtol=1e-10` bound:

| Quantity | Maximum Rust–Julia absolute difference |
| --- | ---: |
| Total energy history | 2.75577e-10 |
| Final parameters | 1.47888e-11 |

Same-implementation repeatability passes at `atol=rtol=1e-12`; observed differences
are zero for both languages. Final energy per site is 0.7795418080426808 in Rust
and 0.7795418080426915 in Julia. Julia's configurations and acceptance counts also
repeat exactly. Rust's public API returns energies/parameters, so these receipts
do not establish cross-language final configuration identity.

The new FMA and warp reductions change floating-point rounding. Independent CPU
factorizations check inverse residuals and Pfaffians for dimensions 128, 130,
192, 256 and 512, partial queues, partial QP groups, forced rebuild boundaries,
cancellation fallback and failed-publication behavior. Supplied-tape tests check
exact proposal replay, while the independent Julia RNG fixture checks generated
words and reseeding. Maximum raw inverse residual in the 29-accepted-move queue
test is 2.265e-11; maximum relative inverse error is 1.783e-12. Bounds scale with
dimension, conditioning and accumulated updates; the full Opt comparison above
retains the prior fixed tolerance.

| Rust scope | H2D calls / bytes | D2H calls / bytes | Synchronizations | Raw scopes |
| --- | ---: | ---: | ---: | ---: |
| Entire public call, all six repetitions | 27 / 27,621,476 | 4 / 6,344 | 11 | 7 |
| Warmup + all 20 Opt steps | **0 / 0** | **0 / 0** | **1** | **1** |

The latter is also asserted in the separately run L128/320-walker diagnostic
([transfers.log](transfers.log)). These are explicit `RawExec` counters, not a
physical PCIe trace; provider internals and kernel argument traffic are excluded.

## Source and validation

- Measured Rust commit: **`051f8983ebbeb47a5edc4a9f770fc70d5a159916`**.
- Release worker SHA-256:
  `1c6792abe3847f7db3d17601ae797d7014dab6e18673ea00d7880e40bb744246`.
- Julia source: **`47452f9061ec3de049d891a7aa7fd9cfeac228a4`**,
  Julia 1.13.1, CUDA.jl 6.4.2. The GPU lock SHA-256 is
  `86c69ea3a0c04c2b6af6f48a3f36ec2a08fa2bc564ac4845f5799e53244fc573`.
- RTX 3060 UUID `GPU-ed896702-ca5c-93f6-2537-06fffdac4a91`, driver 580.178.04.
  Actual loaded CUDA/BLAS library paths, submodule pins, input and receipt hashes
  are recorded in [environment.json](environment.json).
- Full CPU workspace: **1,611 passed, 63 skipped**, `test-fast`.
- Related CUDA lane: **26 passed**, two manual diagnostics/benchmarks excluded;
  the full L128 transfer diagnostic was then run separately and passed.
- Root/CUDA clippy and fmt passed. Root doc tests passed (none defined).

Full numerical receipts, timing samples, verification output and raw validation
logs accompany this report. Original per-call receipts remain in
`/tmp/mvmc-550-e2e-verified/{rust-output,julia-output}`.

## Reproduce

Build the pinned Rust source:

```sh
cargo build --release --locked --manifest-path gpu/mvmc-gpu-cuda/Cargo.toml \
  --example resident_opt_worker
```

Prepare the identical input and pinned Julia source described in the
[earlier report](../L128-opt-20261012/README.md). A fresh Julia checkout needs
`git submodule update --init --recursive`, plus `Pkg.instantiate()` and
`Pkg.build()` as described in its README; SFMT's native library must exist before
measurement. Preserve the recorded GPU lock. The measured reference checkout was
clean, including its submodules, and already built.

```sh
uv run --no-project python \
  benchmark/host_gpu_comparison/results/L128-opt-faster-20261013/mvmc-opt-gpu-comparison.py \
  /absolute/resident_opt_worker /absolute/Julia-mVMC /absolute/input/namelist.def \
  /absolute/new-results
uv run --no-project python \
  benchmark/host_gpu_comparison/results/L128-opt-faster-20261013/verify_receipts.py \
  /absolute/new-results
```

The harness retains the prior timing protocol and additionally records loaded
libraries after each first call, outside timing. It selects this host's GPU and
CUDA artifact paths explicitly; adjust and record them on another host.
