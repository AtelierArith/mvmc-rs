# Native C versus Rust measurement (2026-10-10)

The maintained #492 commands are `scripts/bench_native_c.py` for timing and
output comparisons, `scripts/capture_native_c.py` for the full native SR/RNG
boundary, and `scripts/audit_native_c_system.py` for conditioning/residuals.
The older commands below reproduce the historical pre-fix measurements;
their FSZ initialization and RBM semantics limitations are annotated in the
historical report.

See [the #492 results and numerical audit](results/issue492-linux-x86_64.md).

```sh
cargo build --release -p mvmc-cli
uv run --no-project python scripts/bench_native_c.py \
  --rust /home/vscode/.cache/mvmc/target/linux-x86_64/release/mvmc \
  --c /home/vscode/.cache/mvmc/c-benchmark-independent-20261010/build/src/mVMC/vmc.out \
  --out /home/vscode/.cache/mvmc/native-comparison-new \
  --workloads opt_hubbard_L16,opt_hubbard_L32,opt_hubbard_L64,phys_hubbard_L32 \
  --threads 1 4 --reps 3
```

This retains independent inputs, logs, numerical output, environment, binary
and input hashes, raw internal/wall timers and per-file comparison bounds.
Rep 0 is warm-up. Exit status 1 preserves numerical failures rather than
changing tolerances. Use a separate `--observe --reps 1 --threads 4` run to
record actual Rust worker entries; instrumentation adds overhead and its
times must not be mixed with the primary timings.

For a short numerical prefix use `--steps 1`; sample/data overrides are
explicit in the environment record. Rust `--initial-def none` matches C's
absence of automatic neighboring parameter-file loading. For full boundary
capture, build `c_toolbox/native_comparison_492/build_probe.sh` and the Rust
`native_c_diagnostics` example, then run `capture_native_c.py --c PROBE
--rust EXAMPLE --out NEW_DIRECTORY`. Audit each emitted C-system JSON against
the corresponding Rust diagnostics through `uv run
scripts/audit_native_c_system.py C-system.json Rust-diagnostics.json`.

## Historical timing round before #492

Fresh measurements of the authoritative, unmodified `extern/mVMC-1.3.0`
executable and the current Rust CLI are in `results/`. Both executables run
inside the same Linux x86_64 Dev Container, sequentially and interleaved by
workload. This reuses the inputs and workload overrides in
`scripts/bench_cpu_round.py`; no reference source or Rust numerical code is
changed.

Each workload has one discarded warm-up pair and three measured Rust/C pairs.
BLAS/BLIS/MKL threads are fixed to one. Rust uses its default release CLI build;
C uses CMake Release, one MPICH rank, `USE_GEMMT=ON` and
`PFAFFIAN_BLOCKED=OFF`. C links system OpenBLAS and the upstream downloaded
static BLIS artifact; Rust links system OpenBLAS. Compiler flags, binary hashes,
source revisions and the dirty working-tree state are in the environment file.

The primary comparison uses the `[0] All` internal timer. This includes
initialization and computation but excludes the C MPI launcher/startup outside
that timer. The historical CSV column `sec2_total` actually contains timer
`[0]`, not `[2]`. `wall_s` additionally includes process launch and shutdown;
C is launched with `mpirun -np 1`, while Rust is a single-process CLI. Do not
interpret a short-run wall-time ratio as a kernel speedup. Both executables
have their internal timing instrumentation enabled.

These runs establish elapsed-time measurements for matching input contracts,
not a new verification of every RNG draw or numerical trajectory. Long
optimization trajectories may differ after numerical acceptance divergence;
these timings alone cannot establish or explain numerical parity. Existing
reference fixtures remain independent of this optional benchmark.

## Reproduction

Inside the Dev Container, from the repository root:

```sh
cargo build --locked --release -p mvmc-cli
bash c_toolbox/physcal_native/build.sh /home/vscode/.cache/mvmc/c-benchmark-independent-20261010
uv run --no-project python /tmp/run_c_comparison.py \
  --rust "$CARGO_TARGET_DIR/release/mvmc" \
  --c-dir /home/vscode/.cache/mvmc/c-benchmark-independent-20261010 \
  --reps 3 --inner 1 --tag linux-20261010-current \
  --out benchmark/c_comparison/results/linux-x86_64-20261010-1t.csv

uv run --no-project python /tmp/run_c_comparison.py \
  --rust "$CARGO_TARGET_DIR/release/mvmc" \
  --c-dir /home/vscode/.cache/mvmc/c-benchmark-independent-20261010 \
  --reps 3 --inner 4 \
  --workloads opt_hubbard_L16,opt_hubbard_L32,opt_hubbard_L64,phys_hubbard_L32 \
  --tag linux-20261010-current \
  --out benchmark/c_comparison/results/linux-x86_64-20261010-4t.csv
```

Create `/tmp/run_c_comparison.py` using the adapter below. It imports the existing
harness and replaces only Docker-per-run C execution with execution inside the
current container. It uses `subprocess.run(check=True)` so a failed C process
cannot become a timing sample. It uses `perf_counter` for wall time on both sides.
The adapter is kept here with the measurement record rather than installed as a
new supported harness.

```python
import importlib.util
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

root = Path('/workspaces/mvmc-rs')
spec = importlib.util.spec_from_file_location('bench_cpu_round', root / 'scripts/bench_cpu_round.py')
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)

def native_c(c_dir, image, dest, flags, para, threads):
    shutil.rmtree(dest / 'output', ignore_errors=True)
    for f in dest.glob('zvo_*'):
        f.unlink()
    env = dict(os.environ, OMP_NUM_THREADS=str(threads), OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', BLIS_NUM_THREADS='1')
    cmd = ['/opt/mpich/bin/mpirun', '-np', '1', str(c_dir / 'build/src/mVMC/vmc.out'), *flags, 'namelist.def']
    if para:
        cmd.append(para)
    start = time.perf_counter()
    with (dest / 'c_run.log').open('w') as log:
        subprocess.run(cmd, cwd=dest, env=env, check=True, stdout=log, stderr=subprocess.STDOUT)
    return time.perf_counter() - start, b.parse_timer(dest / 'zvo_CalcTimer.dat')

b.run_c = native_c
Path('/tmp/claude-1000').mkdir(exist_ok=True)
os.environ.update(OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', BLIS_NUM_THREADS='1', OMP_NUM_THREADS='1')
b.main()
```

Temporary inputs are prepared beneath `/tmp/claude-1000` by the existing harness
and removed when it finishes. The vendored sources and benchmark inputs are
preserved. These measurements do not change or merge the user's existing MPI
benchmark work.
