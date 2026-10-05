# Julia-mVMC → Rust port

This Cargo workspace is the Rust port of [Julia-mVMC](extern/Julia-mVMC) — a Julia port of the C-mVMC many-variable Variational Monte Carlo solver.

## Status

**Phase 5 complete.** All four upstream models run end-to-end with full bit-parity against the Julia gold standard:

| Gate | Tolerance | Result |
|---|---|---|
| 10-step `zvo_out.dat` vs. C reference (all 4 models) | 1e-8 | match |
| 50-step regression vs. Julia reference (all 4 models) | 1e-8 | match |

See [`docs/PORTING_PLAN.md`](docs/PORTING_PLAN.md) for the full phased plan, the module-by-module mapping, the bit-parity strategy, and the open / answered clarifications.

## User manual

A user manual that starts from the theory (variational Monte Carlo, the Pfaffian pair-product wave function with correlation factors and quantum-number projection, Metropolis sampling, stochastic reconfiguration, Green functions and the Lanczos correction) and maps every key equation to the C reference function and the Rust function that implements it is in [`docs/manual/`](docs/manual/README.md). It also documents the input files (with the supported/rejected matrix), the `mvmc` command line, serial/MPI/grouped (`NSplitSize`) execution, environment variables, every output file and which rank writes it, a worked tutorial and the differences from C and Julia.

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
cargo check --workspace
cargo nextest run --workspace --cargo-profile test-fast
cargo test --workspace --doc
```

For long numerical regressions, use `cargo nextest run --workspace --cargo-profile test-fast`. See [`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md) for build profiles, compiler caching, debugging overrides and measurements.

## CLI binary

The `mvmc` binary accepts any Expert-mode `namelist.def` and runs `NSROptItrStep` SR optimisation steps:

```bash
cargo run -p mvmc-cli -- <namelist.def> [options]

Options:
  --nsteps <N>      SR steps (overrides NSROptItrStep in modpara.def)
  --nsmp <N>        Final averaging window (overrides NSROptItrSmp)
  --out-dir <DIR>   Output directory (default: namelist parent / output/)
  --seed <N>        RNG seed (overrides RndSeed in modpara.def)
  --mode <MODE>     Sanity label real/cmp/fsz (the mode comes from the inputs)
  --initial-def <auto|none|PATH>  Starting parameter file (default: auto)
  -o, --opt-trans   Enable the C OptTrans mode
  --physcal <PATH>  Fixed-parameter PhysCal (requires NVMCCalMode=1)
  --physcal-trace <NEW_DIR>  Nonconsuming serial PhysCal diagnostics

Environment:
  MVMC_NSTEPS       Same as --nsteps (CLI flag takes precedence)
```

`NVMCCalMode` in `modpara.def` selects the calculation (0: optimisation, 1: PhysCal) and must agree with `--physcal`. Serial, MPI (`--features mpi`) and grouped (`NSplitSize`) execution, the timer/diagnostic and inner-thread environment variables (`MVMC_C_TIMER`, `MVMC_*_DIAG`, `MVMC_RS_INNER_THREADS`, `MVMC_RS_INNER_THRESHOLD`) and every output file (`zvo_out.dat`, `zvo_var.dat`, `zqp_*_opt.dat`, `zvo_SRinfo.dat`, `zvo_cisajs*`, `zvo_ls_*`, `zvo_CalcTimer*.dat`) are described in the [user manual](docs/manual/README.md) ([running](docs/manual/08-running.md), [outputs](docs/manual/09-output-files.md)).

## PhysCal

In addition to SR optimisation, the `mvmc` binary can run fixed-parameter physical calculations (PhysCal) from an existing parameter file:

```bash
cargo run -p mvmc-cli -- <namelist.def> --physcal <zqp_opt.dat> [options]
```

`modpara.def` must set `NVMCCalMode 1` for `--physcal` (and `--physcal` is required when it does). `--mode real|cmp|fsz` is only a sanity label: the real, complex or FSZ path is determined by the input declarations. For nonconsuming serial diagnostics, add `--physcal-trace <NEW_DIR>`.

Reference comparisons are in `crates/mvmc-core/tests/physcal_issue181.rs` and `crates/mvmc-cli/tests/physcal_reference.rs`. The single-step Lanczos correction (`NLanczosMode = 1, 2`, written to `zvo_ls_out_*`, `zvo_ls_qqqq_*` and, for mode 2, the `zvo_ls_cisajs*` files) is implemented for the sz-conserved path with `NSplitSize = 1`, without `InterAll` or spin-changing `Trans` terms (issues #31 and #32); see the [manual](docs/manual/06-theory-observables-lanczos.md) for the formulas and [the supported-input matrix](docs/manual/07-input-files.md#75-supported-and-rejected-inputs) for the rejected combinations.

### PhysCal verification

Fixed-parameter PhysCal was verified against the committed C-mVMC reference fixtures in `extern/Julia-mVMC/test/integration/reference/*/physcal_ref/`. The table below shows the maximum absolute difference between Rust and Julia outputs across the three Green files produced by a single sample (`NDataQtySmp=1`):

| Model | Mode | `zvo_cisajs` | `zvo_cisajscktalt` | `zvo_cisajscktaltex` |
|---|---|---|---|---|
| heisenberg_chain_real | real | 0.0     | 1.4e-17 | 0.0     |
| heisenberg_chain_cmp  | cmp  | 0.0     | 0.0     | 0.0     |
| heisenberg_chain_fsz  | fsz  | 8.7e-19 | 2.8e-17 | 2.8e-17 |
| hubbard_chain_real    | real | 0.0     | 2.8e-17 | 0.0     |

Both implementations also agree with the C reference within the Julia integration-test tolerances (1e-10 for one-body, 1e-9 for direct/factored two-body). The Rust-vs-Julia differences are at machine-epsilon level and do not indicate a defect in either implementation.

### PhysCal benchmark

Use `xtask bench-physcal` to compare Rust (`mvmc-cli --physcal`) and Julia (`run_phys_cal_from_namelist`) on the same fixed-parameter PhysCal fixtures:

```bash
# All four fixtures, 3 measured reps, 1 warmup, single-threaded
cargo run -p xtask -- bench-physcal --reps 3 --warmups 1 --threads 1

# Single model
cargo run -p xtask -- bench-physcal --model hubbard_chain_real --reps 3 --warmups 1
```

Results are written to `target/bench/physcal_chain.csv`. The summary prints `speedup = julia / rust`, so values above `1.0x` mean Rust was faster. The fixtures use `NVMCSample=100` and `NDataQtySmp=1`; larger production PhysCal runs can be benchmarked by preparing a custom namelist with a larger sample count.

## Example scripts

Four ready-to-run examples mirror the Julia `examples/*.jl` scripts. Input files are read from `extern/Julia-mVMC/examples/inputs/`; output files are written to `output/<model>/`.

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

# Override output root (default: output/<model>/)
MVMC_OUT_DIR=/tmp/my-run cargo run --example heisenberg_chain_cmp
```

Output files are written under `output/<model>/`:

```
output/
├── heisenberg_chain_real/
│   ├── zvo_out.dat    # energy per SR step (6 columns)
│   ├── zvo_var.dat    # variational parameter evolution
│   └── zqp_opt.dat    # final optimised parameters
├── heisenberg_chain_cmp/
├── heisenberg_chain_fsz/
└── hubbard_chain/
```

The `output/` directory is listed in `.gitignore` and is never committed.

### Numerical accuracy vs. Julia gold standard (10 steps)

| Model | Julia energy/site | Rust energy/site | Δ |
|---|---|---|---|
| heisenberg_chain_real | −0.29540502896307 | −0.29540502896248 | 5.9e-13 |
| heisenberg_chain_cmp  | −0.13101146733921 | −0.13101146733921 | 0       |
| heisenberg_chain_fsz  | −0.19029132348171 | −0.19029132348177 | 5.8e-14 |
| hubbard_chain         | −0.64842160163507 | −0.64842160163507 | 0       |

All differences are well within the 50-step regression tolerance of 1e-8.

### Numerical reference policy

The C implementation in `extern/mVMC-1.3.0/` is treated as the authoritative source for numerical behavior: parameter layout, initialization and draw order, arithmetic order, signs, and floating-point results. Julia-mVMC is used as a secondary reference and test aid. Where the two references differ, Rust follows the C contract, and cross-language comparisons use explicit absolute and relative tolerances justified by the first divergence, operation order, problem scale, and conditioning. Exact-bit gates compare same-implementation runs; Rust-to-C or Rust-to-Julia comparisons never claim bitwise equality.

## Speed benchmark vs. Julia-mVMC

Use `xtask bench-julia` to compare the Rust port against the Julia implementation in `extern/Julia-mVMC` on the same four example models. The task builds the Rust examples in release mode once, runs matched Rust / Julia workloads with the same SR step count, and writes per-repetition timings to CSV.

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
--model <NAME>       benchmark one model; repeatable
--julia-root <DIR>   Julia-mVMC checkout (default: extern/Julia-mVMC)
--csv <PATH>         CSV output (default: target/bench/julia_vs_rust.csv)
--keep-output        keep per-run zvo_out.dat / zqp_opt.dat files
```

The Julia side is timed inside one Julia process per model after optional warmup, so Julia startup and first-call JIT cost are excluded from measured repetitions. The Rust side runs prebuilt release example binaries directly (no Cargo timing). The summary reports median / min times and prints `speedup = julia / rust`, so values above `1.0x` mean Rust was faster.

For an apples-to-apples comparison, pass `--threads N` (e.g. `--threads 1` for single-threaded BLAS, or `--threads <#physical cores>`). The runner propagates `OPENBLAS_NUM_THREADS` / `OMP_NUM_THREADS` / `MKL_NUM_THREADS` / `BLIS_NUM_THREADS` / `VECLIB_MAXIMUM_THREADS` / `JULIA_NUM_THREADS` / `RAYON_NUM_THREADS` to both processes, and Julia additionally calls `LinearAlgebra.BLAS.set_num_threads(N)`. The summary also prints `|ΔE| = |E_rust - E_julia|` per site so divergent setups (e.g. mismatched `namelist.def`) are caught immediately.

### Hubbard-chain report benchmark

`bench-julia` only covers the four small `examples` models. Use `bench-hubbard` for the larger Hubbard-chain workloads used by the C-vs-Julia performance reports (see `docs/reference/c-to-julia/performance/2026-06-17-julia-mvmc-hubbard-locenergy-slater-calham1-optimization-record.md`). It runs `L=16/24/32`, half filling, `U=4.0`, `t=1.0`, `NSPGaussLeg=8`, `NSPStot=0`, `NSplitSize=1`, `NStore=1`, `NSRCG=0` at `R=1` and one thread:

```bash
cargo run -p xtask -- bench-hubbard --steps 300 --reps 3 --warmups 1 --threads 1
```

The committed Expert inputs live in `benchmark/hubbard_chain/inputs/` with their `StdFace.def` and provenance (see `benchmark/hubbard_chain/README.md`), so no C/StdFace rebuild is needed. Unlike `bench-julia`, this task invokes the Rust `mvmc-cli` binary and matches the two sides on the internal `run_para_opt_from_namelist` wall clock (Julia JIT and process startup are excluded). It writes per-repetition CSV (`target/bench/hubbard_chain.csv`) and a Markdown report (`target/bench/hubbard_chain_report.md`) with median timings, `speedup = julia / rust`, and full-precision `|ΔE|` per size.

## pfapack BLAS backend

`crates/pfapack` ships two interchangeable kernel implementations selected at compile time:

| Feature | Default | Rank-2 update (`pfaffian_ltl!`) | `dsktf2` rank-2 | trtri / trmm | scal | Bit parity vs Julia |
|---|---|---|---|---|---|---|
| (none) | standalone pfapack only | scalar | scalar (upper-triangular only) | scalar | scalar | deterministic scalar reference; may differ from Julia |
| `blas-backend` | mvmc-core / CLI | `dger` / `zgeru` ×2 | scalar (preserves lower triangle) | `dtrtri` / `dtrmm` (`ztrtri` / `ztrmm`) | `dscal` / `zscal` | exact-bit gates against Julia 1.13.1 |

Why a hybrid in BLAS mode? Julia's `julia_dsktf2!` itself replaces `BLAS.ger!` with an upper-triangle-only scalar loop because the LTL output contract preserves the strict lower triangle. We mirror that operation order so the `tests/fixtures/pfapack/` golden diff passes in both backends. The dense Pfaffian rank-2 update, the unit-upper trtri, and the trmm Mᵀ · A pass are the actual hot path for `vmc_sampling` and are dispatched to BLAS.

The optimizer always enables the BLAS/LAPACK backend, including small matrices. Scalar shortcuts can change the inputs to SR-CG by a few ulps and significantly change its truncated solution. Standalone pfapack retains its scalar reference backend. On macOS, install Homebrew OpenBLAS first for the standard optimizer and SR-CG BLAS/LAPACK kernels:

```bash
brew install openblas
```

`openblas` is keg-only on macOS; the Rust build scripts add the Homebrew library path automatically for the optimizer. On Linux, install the system OpenBLAS/LAPACK package, for example `libopenblas-dev` on Debian/Ubuntu.

The ordinary complex optimizer follows Julia's `julia_zsktf2_turbo!` operation order. FSZ follows its separate ordinary LTL and direct-division inverse path; on macOS that reference path uses Accelerate. The Rust implementation selects those numerical kernels without calling a native PfaPack wrapper.

Standard SR-CG (`NSRCG=1`) has exact-bit Julia 1.13.1 gates for the real, complex, FSZ Heisenberg and real Hubbard examples: runs of 1, 2, 3, and 50 optimizer steps check final parameters, energy, every saved configuration, the next full SFMT block, and the complete SRinfo output. The fixed-input gate also checks every CG iterate through both the 20th and 40th residual refreshes. Run the reference checks with:

```bash
julia +1.13.1 --project=extern/Julia-mVMC scripts/check_sr_cg_fixed_parity.jl
julia +1.13.1 --project=extern/Julia-mVMC scripts/check_sr_cg_fixed_parity.jl --sampled
julia +1.13.1 --project=extern/Julia-mVMC scripts/check_sr_cg_runner_parity.jl --case=real
julia +1.13.1 --project=extern/Julia-mVMC scripts/check_sr_cg_runner_parity.jl --case=cmp
julia +1.13.1 --project=extern/Julia-mVMC scripts/check_sr_cg_runner_parity.jl --case=fsz
julia +1.13.1 --project=extern/Julia-mVMC scripts/check_sr_cg_runner_parity.jl --case=hubbard
```

Direct SR (`NSRCG=0`) checks the same models with `NStore=0` and `NStore=1`. Real stored Gram construction follows Julia's SYRK dispatch and upper-triangle copy; complex construction preserves its sequential sample sum. Projection ratios and Metropolis probabilities use Julia's Float64 exponential. The direct solver's sampled matrices, gradients, factors, and solutions also have exact-bit fixtures. See `tests/fixtures/sr_direct/README.md` for settings and the `scripts/check_sr_direct_runner_parity.jl` reference commands.

Pure `OrbitalGeneral` uses combined spin-site coordinates in its cached `2*Nsite` index/sign matrices, shared by FSZ Slater updates and derivatives. Explicit General and equivalent AP/P inputs have exact-bit matrix/derivative checks and same-seed 50-step direct-SR/CG gates. See [the General fixtures](tests/fixtures/orbital_general/README.md) for inputs and reference commands. Real FSZ execution remains tracked by issue #43.

```bash
cargo run --release -p mvmc-cli --example heisenberg_chain_real

# Benchmark in the same configuration:
cargo run -p xtask -- bench-julia --steps 50 --reps 3 --warmups 1 --threads 1
```

## Out of scope

The authoritative list of rejected inputs is [the supported/rejected matrix in the manual](docs/manual/07-input-files.md#75-supported-and-rejected-inputs) (`crates/mvmc-core/src/validation.rs`).

- BackFlow correlation factor and `SpinJastrow` (rejected), `NSRCG >= 2`, `useDiagScale`, `RescaleSmat`.
- Lanczos corrections for FSZ/general orbitals, with `InterAll` or spin-changing `Trans`, or with `NSplitSize > 1` (rejected). The single-step Lanczos correction itself (`NLanczosMode` 1/2) is implemented.
- The `cimpl_utu2inv!` ccall path and `fimpl_zsktf2_/_dsktf2_` Fortran wrappers from `PfaPack.jl` — the optimizer's hot path uses the pure-Julia routines, so we port those directly and drop the FFI surface entirely (no `gfortran` / `g++` required).

## License

The workspace as a whole ships under **GPL-3.0-or-later** because `mvmc-core` and `mvmc-expert-parsers` translate from GPL-3.0-or-later upstream sources. Two subcomponents retain their original non-GPL licenses and may be vendored independently of the engine:

- `sfmt19937` — BSD-3-Clause (matches `SFMT.jl`).
- `pfapack` — BSD-3-Clause root with per-file MPL-2.0 headers on the files derived from `xrq-phys/Pfaffine` (`src/utu2.rs`). Matches the upstream `PfaPack.jl` THIRD_PARTY_LICENSES split.
