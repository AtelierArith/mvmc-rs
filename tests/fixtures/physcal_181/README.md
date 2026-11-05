# Issue #181 PhysCal inputs and C probes

The PhysCal numerical references are the native C 1.4.0 runs in
`../native_c_physcal_181/`. This directory keeps what tests still read:

- Inputs: each model directory (and `two-samples/<model>/`, `third-hubbard/`,
  `slater-s104/`, `unnormalized-reserved-slater/`) holds `inputs/` and, for
  PhysCal, the fixed parameter record `zqp_opt.dat`. The native C runs, the
  MultiDef, varbin and grouped-NSplit generators, and many runtime tests read
  these inputs.
- Standalone C probes, rerun against `extern/mVMC-1.4.0` in #695 with
  unchanged data (each directory's `provenance.txt` records the 1.4.0 source
  hash): `duplicate-reader/`, `all-terms-reader/`, `energy-only/`,
  `formatting/`, `native-c-weighted-green/` and the `native-c-stages/`
  parameter stages of the DH/RBM/OptTrans models
  (`c_toolbox/generate_physcal_181_*.jl`, `c_toolbox/physcal_181_format.c`).
- `weighted-average/` holds the synthetic inputs of the weighted-green C probe.

The Julia 1.13.1 trajectories, measurement records and handoff notes that used
to live here (sample, RNG and accumulator snapshots, `expected/` outputs,
optimization-flag dumps, the macOS measurement diagnostic) were removed with
the other Julia CPU references in #695. `scripts/generate_physcal_181.jl` is
kept only because the parameter-stage probe reuses its C definition-record
enumeration. Rust tests read the checked-in files only; they never run C or
Julia.
