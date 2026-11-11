# Matched resident CUDA API measurements

`../run_gpu_e2e.py` drives the same `pfaffian_e2e_worker.rs` in two checkouts.
Build that example in each checkout using the same locked CUDA workspace,
release profile, Rust toolchain, CUDA libraries and separate Cargo target
volumes. Copy the worker into the baseline checkout without changing its
library sources. The script checks that the worker and Cargo lock agree.

Example (from the repository root, after both binaries have been built):

```sh
uv run --no-project python benchmark/pfaffian_faer/run_gpu_e2e.py \
  --before /tmp/mvmc731-baseline --after "$PWD" \
  --before-volume mvmc-731-before-cuda-target \
  --after-volume mvmc-731-cuda-target \
  --input gpu/mvmc-gpu-cuda/tests/fixtures/complex/hubbard_complex_L128/namelist.def \
  --output benchmark/pfaffian_faer/e2e/NEW-COHORT \
  --mode opt --image mvmc731-cuda-toolchain:local
```

The local image is CUDA 12.9.2 plus OpenBLAS 0.3.26 and LAPACK 3.12.0;
each cohort records its exact image ID. It is not a published image.
The baseline is main `fae8eb1428b5a5c8810dcecd19e2fe6e7aaa6677`.
The device is physical GPU 1, an RTX 3060, and the host thread is pinned to
CPU 17. Both revisions run serially on that same device with BLAS/OpenMP,
Rayon and mVMC inner thread counts fixed to one. The host is shared;
recorded host load is descriptive, not a claim of exclusive CPU isolation.

Opt uses 320 walkers and 20 optimization steps. PhysCal uses 320 walkers
and one saved-sample window. Both use seed 79, warmup 128 and interval 128.
There is one complete warmup call per revision, then three measured calls,
with before/after order alternating. Timing includes the fresh public API
call, initialization, sampling, final downloads and internal teardown;
worker startup, first-call JIT warmup and post-call text serialization are
excluded. Raw transfer counters cover explicit tenferro-ext raw-execution
operations; library-internal communication is not counted.

For real inputs use `benchmark/cpu_scaling_l128/inputs/{opt,phys}/namelist.def`.
For complex PhysCal copy the complex fixture's definition files into a fresh
directory and change only `NVMCCalMode` in `modpara.def` from 0 to 1; input
hashes are recorded. No optimized-parameter output from either revision is
fed into the other revision. PhysCal starts from the same seeded input state.

Each cohort retains all real/imaginary outputs, observed cross-revision errors,
same-implementation repeatability checks, timings and traffic counters.
Cross-revision error reports are observations, not standalone parity acceptance:
they must be read with the independent Pfaffian/inverse residual, supplied-tape,
RNG, Julia GPU stage and repeatability gates. No numerical tolerance in the
Rust suite is changed by this utility.

`smoke-complex-opt-20261111` and `full-complex-opt-20261111` predate the
original-input preservation fix in `4fef0a87`. They are preliminary evidence,
not final-source measurements. Cohorts named `final-*` use the fixed kernel;
the first records the source snapshot separately because the fix was committed
while its measurement was already running.

## Fixed-kernel complex Opt result

`final-complex-opt-20261112` measures the kernel in `4fef0a87` with the
settings above. Median time falls from 73.0904 s to 63.0455 s: 1.1593x
throughput, or 13.74% less elapsed time. Individual measured values are
73.0581/73.0904/73.1690 s before and 63.0455/62.9816/63.2192 s after.
The complete calls warm the JIT before these measurements.

Every repetition has the same explicit raw traffic: 28 uploads totaling
63,067,392 bytes, four downloads totaling 3,256 bytes, 11 synchronizations
and seven scopes. Both implementations pass their 20-step same-seed
repeatability checks. Cross-revision maximum absolute differences are
7.25e-13 in energies, 2.48e-13 in real parameters and 1.96e-13 in imaginary
parameters. The first parameter entries exceeding 64 epsilon at unit scale
are recorded in `numerical.json`; these observations accompany the kernel
residual and stage gates, rather than excusing an RNG discrepancy.

The matching `final-complex-phys-20261112` cohort gives 5.8456 s to
5.1913 s (1.1260x) for one saved-sample window. Its explicit traffic is
27 uploads/21,648,912 bytes, three downloads/392 bytes, 11 synchronizations
and eight scopes on every repetition. Maximum absolute cross-revision energy
error is 1.85e-13; all Green-function component errors are below 1.88e-14.

## Real-path regression under investigation

These `final-*` names identify the post-overflow-fix measurement cohorts, not
a declaration that adoption validation is finished. The same source gives:

| Input | Before median | After median | Before/after |
| --- | ---: | ---: | ---: |
| Real Opt, 20 steps | 4.9551 s | 5.3252 s | 0.9305x |
| Real PhysCal, one window | 0.5637 s | 0.5960 s | 0.9457x |

Both real cohorts preserve traffic and pass repeatability. They retain the
established large-real arithmetic, but the new panel branch was compiled into
the same resident fallback entry. A separate small-real entry is being tested
to remove that potential resource interaction. The table records a regression,
not a successful real-path speedup or a resolved explanation. Final adoption
requires measuring and explaining the effect; later cohorts must retain these
results rather than overwrite them.

Two subsequent controlled experiments do not cure the real regression.
`separated-real-*` uses `../separate-entry-experiment.patch`; the real Opt
ratio is 0.9336x and PhysCal is 0.9490x. `noinline-real-*` adds only
`__noinline__` to `resident_block_pfaffian` relative to the combined source;
the respective ratios are 0.9292x and 0.9426x. These production changes were
reverted after measurement. All outputs and raw traffic remain reproducible.
The resource diagnostics do not establish the cause of the regression.

`reference-proposal-real-*` changes only the real standalone wrapper to call
the original unblocked Pfaffian. The full-rebuild panel dispatch is retained.
Real Opt then measures 4.9498 s before and 4.9628 s after (0.9974x), and
PhysCal 0.5588 s before and 0.5620 s after (0.9942x). This controlled comparison
removes the earlier regression at this workload's measured resolution; it
does not claim a real L128 speedup. Outputs and explicit traffic still agree.
The production real standalone wrapper therefore retains the established loop.
`selected-complex-*-20261112` repeats complex timing after this selection:

| Input | Before median | After median | Before/after |
| --- | ---: | ---: | ---: |
| Complex Opt, 20 steps | 73.0071 s | 63.0738 s | 1.1575x |
| Complex PhysCal, one window | 5.8701 s | 5.2273 s | 1.1230x |

The 132-test final CUDA gate passes at `d71edcee`, including the expanded
real n=64/126 supplied-tape rebuild cases. Repeated outputs and transfer
counters pass; Opt retains 28 uploads / 63,067,392 bytes, four downloads /
3,256 bytes and 11 explicit synchronizations, while PhysCal retains 27 uploads /
21,648,912 bytes, three downloads / 392 bytes and 11 synchronizations.
These counters cover explicit raw-execution calls, as defined in metadata.

## Final CPU cohorts

`cpu-{real,complex}-{opt,phys}-20261112` compares baseline `fae8eb14` with
CPU source `da97a821` (immutable provider pin `a3dd7b45`). Later report/test-only
commits do not change the measured numerical implementation. Both release
workers use Rust 1.99.0 in the same cached development image, OpenBLAS 0.3.26,
CPU 17, one BLAS/OMP/inner worker, seed 79 and separate target volumes. Source,
workspace/lock and input hashes, exact image ID and compiler output are recorded.
There were no edits, commits or builds in the measured checkout during the
cohorts. This is a shared host with one L3 domain; load and raw repetitions
are retained, without claiming exclusive-host or strict provider-campaign results.

Prepare bounded inputs without editing existing fixtures:

```sh
uv run --no-project python benchmark/pfaffian_faer/prepare_cpu_inputs.py /tmp/cpu-pf-inputs
```

The generator copies the established L128 real scaling and complex Hubbard
inputs, changes only `NVMCSample` to 32 and selects `NVMCCalMode=0/1` for
Opt/PhysCal. Opt's public `RunConfig` sets 20 steps and retains all 20 history
points (`nsmp=20`); PhysCal retains the input's one measurement iteration.
Real uses eight QP planes, complex sixteen. API timing includes parsing,
initialization, sampling, SR and normal output IO; it excludes worker startup,
post-call serialization and destruction of the returned observed state.
The worker removes only its own fresh API output directory after serialization.

Copy the identical `crates/mvmc-cli/examples/pfaffian_e2e_worker.rs` into both
checkouts and build with `cargo build --release -p mvmc-cli --example
pfaffian_e2e_worker --locked`. Use the development container and separate named
cache volumes, as in `docs/DEV_CONTAINER.md`; the script expects their standard
`target/linux-x86_64/release/examples` layout. Then run, substituting the image
and volume names of those builds:

```sh
uv run --no-project python benchmark/pfaffian_faer/run_cpu_e2e.py \
  --before /path/to/baseline --after "$PWD" \
  --before-volume before-cpu-cache --after-volume after-cpu-cache \
  --input /tmp/cpu-pf-inputs/cpu-complex-opt/namelist.def \
  --output /tmp/cpu-complex-opt-results --mode opt --dtype cmp \
  --steps 20 --reps 3 --cpu 17 --image YOUR_DEVELOPMENT_IMAGE
```

Change `--dtype real` / input for real; use `--mode phys` and its measurement
input for PhysCal. The output directory must be new. Repetition zero warms
caches; subsequent repetitions alternate order. Same-implementation repeats
use a separate `1e-14 + 1e-13*max(|a|,|b|)` floating bound and exact discrete
comparisons. Across implementations, all recorded RNG words/count/index,
configurations and acceptance counters match for all four cohorts. Thus no
acceptance-trajectory divergence needs a tolerance exemption in these runs.

The first recorded floating differences occur in step zero: maximum
unit-scaled errors over its energy/moment/parameter section are 4.03e-13 real
and 1.14e-13 complex. Panel reduction and triangular inversion have different
operation order; the inverse residual tests bound the kernel independently.
Later optimizer roundoff propagates into parameters and Pfaffian products;
the final large-magnitude Pfaffians differ by up to 5.77e-12 relative (real)
and 7.35e-11 (complex). These trajectory observations are not standalone
Pfaffian accuracy bounds. Full raw step histories and section errors remain
in `numerical.json` and `before/after/rep-*.txt`; no cross-provider tolerance
was added or historical reference expectation replaced.

Committed CPU cohorts store those raw text files in `raw-results.tar.gz` to
keep generated repetitions out of the code diff. Run `tar -xzf raw-results.tar.gz`
inside a cohort to restore `before/after/rep-*.txt`. The archive was verified
against every original file before packaging; `raw-results.sha256.json` retains
the individual checksums. Fresh benchmark runs still write ordinary text files.
