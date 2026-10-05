# mvmc-rs User Manual

This manual describes the Rust port of the many-variable variational Monte Carlo
package mVMC (`mvmc-rs`). It starts from the theory and maps every key equation
to the C function that defines the reference behaviour and to the Rust function
that implements it. The later chapters are usage references: input files,
command line, parallel execution, output files, a worked tutorial and the
differences from the C and Julia implementations.

The manual is written in plain GitHub-rendered Markdown. Equations use
`$...$` and `$$...$$` (rendered by GitHub). A Japanese translation will follow.

## Contents

| # | Chapter | Content |
|---|---------|---------|
| 1 | [Overview and installation](01-overview-install.md) | What the workspace contains, authority of the C code, requirements, Cargo features (`blas-backend`, `simd-backend`, `mpi`), build and smoke test |
| 2 | [Theory I: VMC and the Hamiltonian](02-theory-vmc-hamiltonian.md) | Variational Monte Carlo estimator, local energy, supported Hamiltonian terms, one- and two-body Green function ratios |
| 3 | [Theory II: the variational wave function](03-theory-wavefunction.md) | Pfaffian pair-product part (normal, parallel, general/FSZ orbitals; real and complex), Gutzwiller, Jastrow, doublon-holon, RBM, quantum-number projection, parameter layout and synchronization |
| 4 | [Theory III: Markov-chain sampling](04-theory-sampling.md) | Configurations, initial sample, hopping/exchange/local-spin-flip updates, acceptance, Pfaffian and inverse updates, burn-in |
| 5 | [Theory IV: stochastic reconfiguration](05-theory-sr.md) | Log-derivatives $O_k$, the $S$ matrix and force, cutoff and diagonal shift, direct solver and CG, parameter update, final averaging |
| 6 | [Theory V: physical quantities and Lanczos](06-theory-observables-lanczos.md) | One-/two-body Green functions, weighted averages, single-step Lanczos energy and Green functions (`NLanczosMode` 1/2) |
| 7 | [Input files](07-input-files.md) | `namelist.def`, `modpara.def`, Expert-mode definition files, initial parameter files; what Rust supports and rejects |
| 8 | [Running](08-running.md) | The `mvmc` CLI, serial/MPI/grouped (`NSplitSize`) execution, environment variables |
| 9 | [Output files](09-output-files.md) | Every file Rust writes, its columns and which rank writes it |
| 10 | [Tutorial](10-tutorial.md) | A complete Hubbard-chain optimization followed by a physical-quantity calculation, with real output |
| 11 | [Compatibility and differences](11-compatibility.md) | Differences from C and Julia, numerical comparison policy, open observations |
| A | [Appendix: how the citations were checked](appendix-checks.md) | Citation checker, provenance of the observed output, what was not verified |

## How to read the "Implementation" boxes

Each key equation is followed by a box of this form:

> **Implementation**
> - C: `Function` — `extern/mVMC-1.3.0/src/mVMC/file.c:LINE`
> - Rust: `function` — `crates/<crate>/src/file.rs:LINE`
> - Parity: operation-order or contract remarks that matter for numerical comparison.

Conventions:

- **C is authoritative** for numerical behaviour (parameter layout, draw order,
  operation order, signs, file formats). Julia-mVMC is the design reference for
  the Rust public API and runner structure. See [AGENTS.md](../../AGENTS.md) and
  [chapter 11](11-compatibility.md).
- File and line numbers were checked against
  - the C reference `extern/mVMC-1.3.0` (git submodule, commit `d73d06bd`), and
  - this repository at commit `37348eac` (`main`, "Add public RBM flag refresh and manual orbital mode (#336)").

  Line numbers drift as the code changes; the symbol names are the stable key.
  [`appendix-checks.md`](appendix-checks.md) describes how the citations were
  verified and lists what was *not* verified.
- Statements marked **(unverified)** were taken from documentation or reading
  code but not exercised in this session (for example MPI runs and macOS
  behaviour). Statements marked **(observed)** were reproduced with a
  release build of `mvmc` while writing this manual.
- Rust APIs are under active development and backward compatibility is not a
  goal (see [AGENTS.md](../../AGENTS.md)); anything here may change.

## Sources used

- C manual: `extern/mVMC-1.3.0/doc/en/source/{algorithm,expert,output,standard,start,tutorial}.rst`
  (the Japanese manual is `doc/ja`).
- C implementation: `extern/mVMC-1.3.0/src/mVMC/`.
- Julia documentation (structure only): `extern/Julia-mVMC/docs/src/en/*.md`.
- Rust workspace: `crates/`.
