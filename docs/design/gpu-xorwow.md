# CUDA sampling XORWOW (issue #530)

The CUDA sampler can generate sampling words on the GPU with native XORWOW kernels,
without loading cuRAND. CPU sampling and the default CUDA lockstep verification mode
continue to use SFMT with the original draw order and state.

```sh
MVMC_RS_SAMPLER_BACKEND=cuda:1 MVMC_RS_CUDA_RNG=xorwow cargo run --manifest-path gpu/mvmc-gpu-cuda/Cargo.toml --bin mvmc-cuda -- ...
```

Set `MVMC_RS_CUDA_RNG=sfmt` (or omit it) for C/SFMT deterministic parity. With XORWOW,
`MVMC_RS_CUDA_RNG_SEED` can supply an explicit unsigned 64-bit seed. Otherwise the
first sampling series consumes exactly two words from the run's initialized SFMT
state (low word followed by high word) to seed XORWOW once. Its state persists across
later sampling series. Parameter initialization continues to use SFMT. An XORWOW
trajectory is a separate RNG contract and is not expected to reproduce the SFMT
trajectory. The backend label records the selected generator.

The production stage is cached per CUDA device. Its RNG mode and optional seed
are captured when that device's stage is first opened, so set the environment
before opening the backend; changing it later does not reconfigure that stage.
Acquiring a sampler handle starts one complete Opt or PhysCal run and resets its
XORWOW stream. The same handle survives all SR steps and sample series, preserving
within-run continuity. A second public run with the same seed starts a fresh
stream even when it reuses the compiled CUDA service.

Both sources use the production inline single-chain service, including the existing
C timer, optional sampler profiling and optional CUDA-graph service path. RNG
selection changes the word source; it does not replace those execution paths.

The XORWOW word recurrence and seed initialization match NVIDIA
`curand_init(seed, 0, 0)` followed by `curand(state)` integer draws. This does not
claim equality with CUDA.jl's host cuRAND library output order: the host generator
interleaves multiple internal states. `DeviceXorwow` lane `i` uses seed `seed+i`
(with wrapping addition), not NVIDIA's `2^67` subsequence skipahead. Integer draws
are mapped using C's biased modulo or exact `word / 2^32` in `[0,1)`, preserving
sampling conversion and draw-count semantics. cuRAND uniform floating conversion
is intentionally not used.

`DeviceXorwow::generate` leaves its state and output on the device; the state and
output capacity are allocated once. The current production sampler still runs
proposal and acceptance control on the host, so `GpuSamplingRng` downloads one
4096-word batch on refill. GPU generation alone does not eliminate the sampler's
other proposal transfers or guarantee a speedup. Fully device-resident proposal
control remains follow-up work.

## Verification

CPU `XorwowRng` replays exactly the same integer words and conversions. Optional
CUDA gates check multi-lane words and final state, repeated generation with varying
batch sizes, nonconsuming previews across refill boundaries, and a Hubbard L16
20-sample run. That sampler gate compares exact proposal/draw/configuration events
and future words with CPU XORWOW; its existing teacher mechanism diagnoses any
numerical acceptance divergence using the existing absolute/relative weight bounds.
An additional two-run, 20-step public optimization gate checks stream continuity,
future words, saved configurations, counters and seeded same-process repeatability.
Its CPU XORWOW word replay checks continuity; independent algorithm expectations
come from the NVIDIA-header fixtures, not from a Rust-generated trajectory.
Normal CPU tests read independent NVIDIA-header fixtures and require no NVIDIA
headers, C compiler, Julia, or GPU. See `c_toolbox/xorwow/README.md` for optional
fixture regeneration and provenance.

```sh
cargo nextest run -p mvmc-core --locked -E 'test(sampling::rng) or test(candidate) or test(trace)'
LD_LIBRARY_PATH=/path/to/cuda/lib MVMC_RS_XORWOW_DEVICE=1 cargo nextest run --manifest-path gpu/mvmc-gpu-cuda/Cargo.toml --locked --test xorwow_gate --run-ignored all --no-fail-fast --retries 0
LD_LIBRARY_PATH=/path/to/cuda/lib MVMC_RS_XORWOW_DEVICE=1 cargo run --manifest-path gpu/mvmc-gpu-cuda/Cargo.toml --locked --profile test --example bench_xorwow
```

The benchmark warms compilation/allocation out of scope, reports synchronized
resident generation separately from repeated explicit word downloads, and makes
no end-to-end optimization or PhysCal speedup claim.

## Linux RTX 3060 measurements (2026-10-12)

GPU 1, NVIDIA GeForce RTX 3060, NVRTC 12.8.93, Linux x86_64, Rust 1.99.0;
standalone Cargo `test` profile (opt-level 2, debug assertions enabled). CPU
baselines generate raw words serially with the same build profile; no BLAS is
involved in this RNG benchmark. GPU 0 was allocated to separate Julia work.
Times include launch/queue/synchronization for generation and synchronous
pageable transfers into a reused host buffer for downloads. Compilation,
allocation, and five generation warmups are excluded. Results are local
microbenchmarks, not end-to-end VMC timings.
These measurements were collected before rebasing the implementation onto the
later inline-service milestones (prototype commit `dd402cdc`); they are historical
RNG measurements rather than new measurements of the current production sampler.

| Lanes × words/lane | Resident generation | Download | CPU XORWOW | CPU SFMT |
|---|---:|---:|---:|---:|
| 1 × 4096 | 50.850 µs | 7.519 µs | 25.861 µs | 31.870 µs |
| 256 × 4096 | 81.080 µs | 405.673 µs | 6543.662 µs | 8011.495 µs |
| 1024 × 4096 | 293.024 µs | 1377.329 µs | 26124.556 µs | 31975.039 µs |

Repetitions are 200, 100, and 30 respectively. Single-stream GPU generation
is slower than the serial CPU generators here, so selecting XORWOW in the
current single-chain sampler is not a performance recommendation. Parallel
resident output reaches roughly 13–14 billion words/s, but downloads dominate
those batches. Moving proposal consumption onto the device remains the useful
performance direction.
