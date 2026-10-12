# L128 Opt: Rust GPU versus Julia GPU, 2026-10-12

The public Rust GPU optimizer takes **47.8588 s**, versus **5.12493 s** for
the optimized Julia GPU optimizer: **Rust takes 9.3384 times as long**.
These are medians of five complete 20-step optimizations after one first-call
warmup, on the same RTX 3060. Related to #550; this establishes the production
baseline before the Rust interval/multiwalker performance changes.

## Workload and timing boundary

- Half-filled Hubbard chain, L = 128, real FP64, 20 SR steps, 320 retained
  configurations per step, seed 79, one host/BLAS thread.
- Both runners read the same namelist and model files. `NVMCWarmUp=1`,
  `NVMCInterval=1`; Julia receives the corresponding 128-proposal warmup and
  interval. The final SR averaging window is one step.
- Rust: `run_para_opt_from_namelist`, with sampler, measurement, measurement
  Pfaffian and SR backends all set to `cuda`. One sequential C/SFMT chain.
- Julia: `run_gpu_para_opt_from_namelist`, 320 independent XORWOW walkers.
- Each measured call includes parsing, initialization, sampling, measurement,
  SR updates and the returned result. Rust writes its ordinary output files
  inside this boundary; Julia returns the energy/parameter arrays. Receipt
  serialization is outside the timer. Rust's C stage timer is enabled.
- Each worker stays alive for all six calls. The first call is recorded
  separately; the next five calls alternate between languages, with the order
  reversed on even repetitions. GPU work is synchronized before stopping the
  Julia timer. No other local benchmark, test or build ran concurrently.

Equal retained counts do **not** mean matching Markov trajectories, effective
sample sizes or convergence. The proposal budgets also differ: 821,760 for
Rust versus 860,160 for Julia. This is a comparison of the existing public
implementations, including their different chain structures and output
lifecycles, rather than a same-trajectory kernel comparison.

## Results

| Implementation | Warm median, 20 steps | Warm min–max | Mean time per step from median |
| --- | ---: | ---: | ---: |
| Rust GPU | 47.858827 s | 47.738789–48.063633 s | 2.392941 s |
| Julia GPU | 5.124929 s | 5.120737–5.139374 s | 0.256246 s |

| Repetition | Rust GPU (s) | Julia GPU (s) |
| --- | ---: | ---: |
| 1 | 47.952186203 | 5.139374207 |
| 2 | 48.063632809 | 5.122108566 |
| 3 | 47.858827265 | 5.120736923 |
| 4 | 47.738788764 | 5.124929273 |
| 5 | 47.847555228 | 5.126625978 |

The first public call took 55.461386 s in Rust and 55.313756 s in Julia.
Time from process launch to readiness was separately 0.006190 s and 6.812934 s,
respectively (Julia imports occur there). These first-call values include
CUDA initialization and, for Julia, first-call compilation. They are single
observations and are not the warm optimization comparison above. Approximate
launch-to-first-result totals are 55.468 s and 62.127 s.

### Rust stage breakdown

Independent medians of the five warm stage timers:

| Stage | Seconds | Share of public-call median |
| --- | ---: | ---: |
| Sampling (`VMCMakeSample [3]`) | 40.50339 | 84.6% |
| Observables (`VMCMainCal [4]`) | 7.01587 | 14.7% |
| SR (`StochasticOpt [5]`) | 0.13373 | 0.28% |

Within sampling, proposal Pfaffian evaluation (`CalculateNewPfM2 [61]`)
takes 31.59648 s, and Pfaffian/inverse rebuilds (`[34]`) take 6.40626 s.
Within observables, `CalculateMAll [40]` takes 5.41526 s. These nested timers
must not be added to their parent timers. Reducing proposal-level host/device
round trips and batching walker work is the priority indicated by this
profile; accelerating SR alone cannot close the measured gap.

## Numerical observations

All six runs of each implementation completed successfully with finite
energies. Each implementation reproduced its own outputs. Julia's final
configurations and acceptance counters matched exactly across repetitions;
all energy and parameter entries were checked at `atol=rtol=1e-12`, with
observed maximum differences zero. Rust's full output files are identical
across repetitions; hashes are recorded as artifact provenance.

Final energy/site is 0.7671166509232439 in Rust and 0.7795418080426915 in
Julia. These are **not** a cross-language parity pass: the C/SFMT sequential
chain and the XORWOW walker ensemble use different proposal streams. No
numerical-equivalence or equal-statistical-accuracy claim is made from this
timing experiment.

## Provenance and reproduction

- Rust production source: `48e04e8980350fc2e9e38204deb20b27a4fc8f36`
  (PR #549 head). Release build, unchanged production sources; only the
  archived timing example was added for measurement. The Rust optimization
  work in progress is excluded from this binary.
- Julia source: `47452f9061ec3de049d891a7aa7fd9cfeac228a4`, PR #87, now
  merged upstream. Julia 1.13.1; production optimized GPU Opt runner.
- Linux x86_64, Intel Xeon E5-2699 v3, GPU 1: RTX 3060 12 GiB,
  UUID `GPU-ed896702-ca5c-93f6-2537-06fffdac4a91`, driver 580.178.04.
- Rust host BLAS: OpenBLAS 0.3.26. Loaded CUDA libraries: cuBLAS 12.8.4.1,
  cuSOLVER 11.7.3.90, NVRTC 12.8.93. Julia BLAS configuration is in `julia.log`.
- Exact timings: `timings.json`; aggregates: `summary.json`; raw stage timers:
  `rust-timer-0.txt` through `rust-timer-5.txt`. `environment.json`, library,
  lock, input, worker and raw-output hash files record additional provenance.

To reproduce, use isolated checkouts of these source revisions, initialize
their reference dependencies, and copy `opt_timing_worker.rs` into the Rust
checkout's `gpu/mvmc-gpu-cuda/examples/`. Build it with:

```sh
cargo build --release --locked --manifest-path gpu/mvmc-gpu-cuda/Cargo.toml \
  --example opt_timing_worker
```

Copy `benchmark/hubbard_chain/inputs/hubbard_chain_L128` to a writable input
directory and replace `modpara.def` and `namelist.def` with the archived
versions. The archived Python/Julia worker pair is the exact script used:

```sh
uv run --no-project python mvmc-opt-gpu-comparison.py \
  /absolute/path/to/opt_timing_worker \
  /absolute/path/to/Julia-mVMC \
  /absolute/path/to/input/namelist.def \
  /absolute/path/to/new-results
```

The Python script records this host's GPU ordinal and CUDA artifact library
paths explicitly; adjust those for another host and record the changes.
Julia uses its `test/gpu` project and recorded lock. The two scripts must
remain next to each other. Full original numerical receipts remain locally
under `/tmp/mvmc-opt-gpu-comparison/{rust-output,julia-output}`; their hashes
are archived here. The earlier A100 report in `../20261012T091610Z` used
different hardware and sources and should not be mixed with these results.
