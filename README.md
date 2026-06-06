# Julia-mVMC → Rust port

This Cargo workspace is the Rust port of
[Julia-mVMC](../extern/Julia-mVMC) — a Julia port of the C-mVMC
many-variable Variational Monte Carlo solver.

## Status

**Phase 5 complete.** All four upstream models run end-to-end with
full bit-parity against the Julia gold standard:

| Gate | Tolerance | Result |
|---|---|---|
| 10-step `zvo_out.dat` vs. C reference (all 4 models) | 1e-8 | ✅ |
| 50-step regression vs. Julia reference (all 4 models) | 1e-8 | ✅ |

See [`docs/PORTING_PLAN.md`](docs/PORTING_PLAN.md) for the full phased
plan, the module-by-module mapping, the bit-parity strategy, and the
open / answered clarifications.

## Crate layout

| Crate | License | Upstream |
|---|---|---|
| [`crates/sfmt19937`](crates/sfmt19937) | BSD-3-Clause | `SFMT.jl/src` |
| [`crates/pfapack`](crates/pfapack) | BSD-3-Clause + MPL-2.0 | `PfaPack.jl/src/{pfaffian,ltl_decomposition,utu2}.jl` |
| [`crates/mvmc-expert-parsers`](crates/mvmc-expert-parsers) | GPL-3.0-or-later | `MVMCExpertModeParsers.jl/src` |
| [`crates/mvmc-core`](crates/mvmc-core) | GPL-3.0-or-later | `MVMCOptimizers.jl/src` |
| [`crates/mvmc-cli`](crates/mvmc-cli) | GPL-3.0-or-later | `examples/*.jl` + CLI binary |
| [`xtask`](xtask) | GPL-3.0-or-later | build / regression driver |

## Quick start

```bash
cd rust          # workspace root
cargo check --workspace
cargo test  --workspace
```

## CLI binary

The `mvmc` binary accepts any Expert-mode `namelist.def` and runs
`NSROptItrStep` SR optimisation steps:

```bash
cargo run -p mvmc-cli -- <namelist.def> [options]

Options:
  --nsteps <N>      SR steps (overrides NSROptItrStep in modpara.def)
  --out-dir <DIR>   Output directory (default: namelist parent / output/)
  --seed <N>        RNG seed (overrides RndSeed in modpara.def)

Environment:
  MVMC_NSTEPS       Same as --nsteps (CLI flag takes precedence)
```

## Example scripts

Four ready-to-run examples mirror the Julia `examples/*.jl` scripts.
Input files are read from `extern/Julia-mVMC/examples/inputs/`; output
files are written to `rust/output/<model>/`.

| Rust example | Julia counterpart | Model |
|---|---|---|
| `heisenberg_chain_real` | `heisenberg_chain_real.jl` | 16-site Heisenberg chain (real) |
| `heisenberg_chain_cmp`  | `heisenberg_chain_cmp.jl`  | 16-site Heisenberg chain (complex) |
| `heisenberg_chain_fsz`  | `heisenberg_chain_fsz.jl`  | 16-site Heisenberg chain (fsz) |
| `hubbard_chain`         | `hubbard_chain.jl`         | Hubbard chain (real) |

```bash
# 50 SR steps (default)
cargo run --example heisenberg_chain_real

# Override step count (same env var as the Julia examples)
JULIA_MVMC_EXAMPLE_STEPS=10 cargo run --example hubbard_chain

# Override output root (default: rust/output/<model>/)
MVMC_OUT_DIR=/tmp/my-run cargo run --example heisenberg_chain_cmp
```

Output files written under `rust/output/<model>/`:

```
rust/output/
├── heisenberg_chain_real/
│   ├── zvo_out.dat    # energy per SR step (6 columns)
│   ├── zvo_var.dat    # variational parameter evolution
│   └── zqp_opt.dat    # final optimised parameters
├── heisenberg_chain_cmp/
├── heisenberg_chain_fsz/
└── hubbard_chain/
```

The `rust/output/` directory is listed in `.gitignore` and is never
committed.

### Numerical accuracy vs. Julia gold standard (10 steps)

| Model | Julia energy/site | Rust energy/site | Δ |
|---|---|---|---|
| heisenberg_chain_real | −0.29540502896265 | −0.29540502896336 | 7e-12 |
| heisenberg_chain_cmp  | −0.13101146733898 | −0.13101146733902 | 4e-12 |
| heisenberg_chain_fsz  | −0.19029132348179 | −0.19029132348172 | 7e-12 |
| hubbard_chain         | −0.64842160163507 | −0.64842160163507 | 4e-12 |

All differences are well within the 50-step regression tolerance of 1e-8.

## Speed benchmark vs. Julia-mVMC

Use `xtask bench-julia` to compare the Rust port against the Julia
implementation in `../extern/Julia-mVMC` on the same four example models.
The task builds the Rust examples in release mode once, runs matched Rust /
Julia workloads with the same SR step count, and writes per-repetition timings
to CSV.

```bash
# Quick single-model smoke benchmark
cargo run -p xtask -- bench-julia \
  --model heisenberg_chain_real \
  --steps 10 \
  --reps 3 \
  --warmups 1

# Full comparison across all example models
cargo run -p xtask -- bench-julia --steps 50 --reps 5 --warmups 1 --threads 1
```

Options:

```text
--steps <N>          SR steps per run (default: 10)
--reps <N>           measured repetitions (default: 3)
--warmups <N>        warmup repetitions, excluded from timing (default: 1)
--threads <N>        pin BLAS / OpenMP / Julia threads on both sides
--blas-backend       build mvmc-cli with `--features blas-backend`
--model <NAME>       benchmark one model; repeatable
--julia-root <DIR>   Julia-mVMC checkout (default: ../extern/Julia-mVMC)
--csv <PATH>         CSV output (default: target/bench/julia_vs_rust.csv)
--keep-output        keep per-run zvo_out.dat / zqp_opt.dat files
```

The Julia side is timed inside one Julia process per model after optional
warmup, so Julia startup and first-call JIT cost are excluded from measured
repetitions. The Rust side runs prebuilt release example binaries directly
(no Cargo timing). The summary reports median / min times and prints
`speedup = julia / rust`, so values above `1.0x` mean Rust was faster.

For an apples-to-apples comparison, pass `--threads N` (e.g. `--threads 1`
for single-threaded BLAS, or `--threads <#physical cores>`). The runner
propagates `OPENBLAS_NUM_THREADS` / `OMP_NUM_THREADS` / `MKL_NUM_THREADS`
/ `BLIS_NUM_THREADS` / `VECLIB_MAXIMUM_THREADS` / `JULIA_NUM_THREADS` /
`RAYON_NUM_THREADS` to both processes, and Julia additionally calls
`LinearAlgebra.BLAS.set_num_threads(N)`. The summary also prints
`|ΔE| = |E_rust - E_julia|` per site so divergent setups (e.g. mismatched
`namelist.def`) are caught immediately.

## pfapack BLAS backend

`crates/pfapack` ships two interchangeable kernel implementations selected
at compile time:

| Feature | Default | Rank-2 update (`pfaffian_ltl!`) | `dsktf2` rank-2 | trtri / trmm | scal | Bit parity vs Julia |
|---|---|---|---|---|---|---|
| (none) | yes | scalar | scalar (upper-triangular only) | scalar | scalar | bit-deterministic, full buffer match |
| `blas-backend` | opt-in | `dger` / `zgeru` ×2 | scalar (preserves lower triangle) | `dtrtri` / `dtrmm` (`ztrtri` / `ztrmm`) | `dscal` / `zscal` | within ≤1e-13 |

Why a hybrid in BLAS mode? Julia's `julia_dsktf2!` itself replaces
`BLAS.ger!` with an upper-triangle-only scalar loop because the LTL
output contract preserves the strict lower triangle. We mirror that
choice byte-for-byte so the `tests/fixtures/dump_pfapack_reference/`
golden diff passes in both backends. The dense Pfaffian rank-2 update,
the unit-upper trtri, and the trmm Mᵀ · A pass are the actual hot path
for `vmc_sampling` and are dispatched to BLAS.

Build & run examples with BLAS on. On macOS, install Homebrew OpenBLAS
first because the BLAS backend expects the Fortran BLAS/LAPACK symbols
from OpenBLAS rather than Accelerate:

```bash
brew install openblas
```

`openblas` is keg-only on macOS; the Rust build scripts add the Homebrew
library path automatically when `--features blas-backend` is enabled. On
Linux, install the system OpenBLAS/LAPACK package, for example
`libopenblas-dev` on Debian/Ubuntu.

```bash
cargo run --release -p mvmc-cli --features blas-backend --example heisenberg_chain_real

# Benchmark in the same configuration:
cargo run -p xtask -- bench-julia --steps 50 --reps 3 --warmups 1 --threads 1 --blas-backend
```

## Out of scope for v0.1 (mirrors Julia-mVMC v0.1)

- BackFlow correlation factor.
- MPI parallelisation.
- Full Lanczos (only step-0 comparison verified upstream).
- The `cimpl_utu2inv!` ccall path and `fimpl_zsktf2_/_dsktf2_` Fortran
  wrappers from `PfaPack.jl` — the optimizer's hot path uses the
  pure-Julia routines, so we port those directly and drop the FFI
  surface entirely (no `gfortran` / `g++` required).

## License

The workspace as a whole ships under **GPL-3.0-or-later** because
`mvmc-core` and `mvmc-expert-parsers` translate from GPL-3.0-or-later
upstream sources. Two subcomponents retain their original non-GPL
licenses and may be vendored independently of the engine:

- `sfmt19937` — BSD-3-Clause (matches `SFMT.jl`).
- `pfapack` — BSD-3-Clause root with per-file MPL-2.0 headers on the
  files derived from `xrq-phys/Pfaffine` (`src/utu2.rs`). Matches the
  upstream `PfaPack.jl` THIRD\_PARTY\_LICENSES split.
