# Canonical thirteen-model prefix observations (#180)

Independent Julia 1.13.1 observations of canonical ctest inputs at their
original seeds and 1/2/3/50 steps, including Hubbard real and Heisenberg FSZ
`initial.def` overlays. Generated externally in
`/tmp/mvmc-ctest-oracles-cslots.2101Nf` by the toolbox observer; no Rust output
was used. See `provenance.txt`, per-prefix `inputs.sha256`, and
`model-settings.txt` for versions, BLAS, source hashes, seeds, solver settings,
and overlay order. Source observer hash identifies the generation revision;
settings metadata was added afterwards from those same canonical inputs.

These are mixed C-contract/Julia references, not full native C executable
results. FSZ local energies use the separately validated native C bridge;
Slater coefficient retention follows the validated C shim. The C-declared
parameter output stream is separate from Julia's mapped-row output. See
[toolbox provenance](../../../c_toolbox/ctest_orbital_contracts.md) for source
boundaries and reproduction commands. Ordinary Rust tests never invoke C,
Julia or toolbox programs.

New `general_rbm_cmp` observations preserve canonical seed 12395, initial.def,
direct NSRCG=0/NStore=1. `general_rbm_cmp_cg` uses the same input/seed and
explicitly records only NSRCG=1/NStore=0 solver overrides. C spin-major
physical/coupling tables and the validated counter shim apply to both.
Per-prefix `generation-provenance.txt` records the later observer revision.
The existing GeneralRBM goldens remain separate and are also checked; they
are never used as another model's expectation.

`zqp_c_window_opt.dat` is produced by actual C `avevar.c` aggregation/formatting
of independent chronological Julia/C-contract histories, not Julia's
final-parameter-only output. The standalone input is
`c-window-declared-input.txt`; source, adapter, input and executable hashes
are in `c-window-provenance.txt`. Reproduce with the optional
[C window probe](../../../c_toolbox/ctest_opt_window.md). No Rust-generated
value enters this expectation. The production writer must match these
fixtures; their presence alone does not establish a passing Rust gate.

Original/effective optimization steps and averaging windows are recorded
separately: every short prefix overrides both counts to the same 1/2/3/50,
without a min-with-canonical-window clamp. Native long checks retain original
counts. The C-window first integer records the effective sample window.

The ordinary `ctest_window_fixtures` test audits these standalone finite-domain
inputs and C outputs without external runtimes. It is an aggregation/fixture
policy/production-writer check, not model runner execution. It includes
mixed optimization flags, unmapped declared Slater slots, exact one-window
literal zeros and finite nonnegative multi-window deviations. Bounds, rounding/variance
propagation and window-1 behavior are documented in the C probe notes.
Actual native C and Julia backend information is recorded in
`native-reference-environment.txt`; C energy/aggregation have no BLAS runtime.
The Julia reference checkout was clean at the post-generation source audit.
`post-generation-source-audit.sha256` additionally pins initialization,
sampling, synchronization and numerical source files; verify from the repo
root with `sha256sum -c` on that manifest. Both Julia direct and CG solvers
are in the already-recorded `stochastic_opt.jl`, not separate CG files.
