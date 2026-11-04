# Validation for resident GPU PhysCal (#636)

Linux x86_64 / RTX 3060, Rust 1.99.0. Latest runtime source `03b33998`. It restores C fixed-file → `In*.def` priority;
the no-`In*` L128 numerical/sampling contract is unchanged from `7955a882`. See metadata/archives for
exact sources, binary/input hashes, actual library paths and all outputs.

| Check | Result |
|---|---|
| CUDA library tests, including resident Opt and 20-window PhysCal | 20 passed |
| Latest focused PhysCal checks (including C priority and 20 ×128-hop windows) | 5 passed |
| Unique CUDA library cases covered across both runs | 22 passed |
| CLI preflight + legacy sampler/measurement/SR integration gates | 10 passed |
| Benchmark orchestration | 7 passed |
| Standalone all-target Clippy with warnings denied | Passed |
| Standalone formatting and diff checks | Passed |
| Standalone documentation test discovery | Passed; 0 doctests |
| New GPU entrypoint `--smoke --workloads phys` | Complete Rust/Julia groups |
| Full warmed API cohort | All 12 timing rows retained, all groups finite |
| Full returned values vs independent Julia | 4 × 1,025 values within existing bounds |
| Same-implementation repetition checks | 4,100 values each; maximum difference 0 |
| New CLI execution/export | Successful; 1,025 means agree with Julia |
| Nsight kernel/API correlation | 33 kernels = 29 driver + 4 runtime launches |
| Historical trace analyzer regression | All 72,110 legacy kernels matched |

The first broader legacy integration run lacked the existing cuTENSOR library
on its search path: 8 passed and 2 failed at backend initialization. Adding
`/tmp/mvmc-570-cuda-env/lib/python3.12/site-packages/cutensor/lib` and rerunning
the complete group produced 10 passes. The failed log is retained alongside the
final log; no code, expectation or tolerance was changed to bypass it. The new
resident API comparison itself requires the CUDA 12 libraries but does not use
this generic tensor-contraction backend.

The smoke cohort uses 16 walkers and one measured repetition. Its times are
not substituted for the full 320-walker performance table. The CLI's 8.54 s is
one process-wall verification with setup/NVRTC/output/exit included, not a warm
API observation. Root-workspace CI provides the final Linux/macOS CPU checks;
CUDA numerical gates here were run on Linux/NVIDIA hardware.
The final-source full cohort repeats all 12 timing rows and compares all 4,100
resident values with both independent Julia and the earlier Rust cohort; the
previous Rust cohort has observed numerical difference zero. Input bytes match,
there are no `In*` entries, and the independent initialization/parameter fixture
passes unchanged. The original performance/Nsight archives remain untouched.

The first new overlay test used an incomplete four-coefficient Gutzwiller file
and correctly failed input validation. Completing the fixture to its declared
width produced five focused passes; both logs are retained. Latest all-target
Clippy also passes without warnings.

## Re-run checks

Set `LD_LIBRARY_PATH` for the installed CUDA 12 libraries and, for the broader
legacy tensor gates, cuTENSOR. The actual paths and versions used are retained
in the evidence. Keep target directories separate from other checkouts.

```bash
export CARGO_TARGET_DIR=/tmp/mvmc-636-verify-target
export CUDA_VISIBLE_DEVICES=0 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export MVMC_RS_CUDA_GATE=1 RAYON_NUM_THREADS=1 MVMC_RS_INNER_THREADS=1
cargo nextest run --release --locked --manifest-path gpu/mvmc-gpu-cuda/Cargo.toml \
  --lib --run-ignored all --test-threads 1 --no-fail-fast --retries 0
cargo nextest run --release --locked --manifest-path gpu/mvmc-gpu-cuda/Cargo.toml \
  --bin mvmc-cuda --test sampler_production_gate --test measurement_gate \
  --test sr_routing_gate --run-ignored all --test-threads 1 --no-fail-fast --retries 0
cargo clippy --release --locked --manifest-path gpu/mvmc-gpu-cuda/Cargo.toml \
  --all-targets -- -D warnings
cargo test --release --locked --manifest-path gpu/mvmc-gpu-cuda/Cargo.toml --doc
cargo fmt --manifest-path gpu/mvmc-gpu-cuda/Cargo.toml --all --check
uv run --no-project python scripts/test_benchmark_entrypoints.py
bash scripts/benchmark_gpu.sh --smoke --workloads phys --device 0
uv run --no-project python benchmark/cpu_scaling_l128/verify_results.py \
  benchmark/gpu_physcal_resident/results/20261103
```

Library logs contain the independent Julia fixture residuals and exact supplied
tape/configuration checks. Broad integration logs independently check that the
ordinary single-SFMT CUDA path retains C-order behavior. Ordinary Rust tests do
not compile, read, or invoke `c_toolbox` or a Julia/C oracle.

## Optional trace reproduction

Use Nsight Systems 2022.4.2.50 (recorded package/tool hashes), the release
`resident_physcal_worker`, and fresh output paths. Set `MVMC_PHYSCAL_PROFILE_REP=1`
and send `0`, `1`, `quit` on stdin. Capture with:

```bash
nsys profile --sample=none --cpuctxsw=none --trace=cuda,osrt \
  --capture-range=cudaProfilerApi --capture-range-end=stop --output=trace \
  taskset -c 17 resident_physcal_worker NAMELIST PARAMS FRESH_OUTPUT
QdstrmImporter -i trace.qdstrm -o trace.nsys-rep
nsys export --type sqlite --output trace.sqlite trace.nsys-rep
uv run --no-project python benchmark/gpu_physcal_l128/analyze_trace.py \
  trace.sqlite summary.json --scope 'One warmed resident PhysCal API.'
```

The raw capture's before/after executable SHA-256 agrees. The driver profiler
requires a current context; the optional helper retains/restores the primary
context at the capture edges. Those context operations may appear in the trace.
Instrumented durations are excluded from the warmed API table. The profiler
and Cargo executables/caches are not included in archives.
