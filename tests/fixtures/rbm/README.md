# Canonical RBM production parity fixtures

These fixtures use the unmodified Julia-mVMC checkout at `8bb1b9e8ae47b1512c00b321be05664ddcac0fd1` (numerical/parser sources at `c2ea432785bc14364a3cd5e9eef44db464289cc9`), Julia **1.13.1**, `extern/Julia-mVMC/Manifest-v1.13.toml`, and OpenBLAS **0.3.30 ILP64**, one thread. They cover parsing, initialization/loading, kernels, sampling, Green ratios, SR and public library/CLI execution against the original source.

Regenerate and verify from the repository root:

```sh
julia +1.13.1 --project=extern/Julia-mVMC scripts/check_rbm_parser_parity.jl --write
julia +1.13.1 --project=extern/Julia-mVMC scripts/check_rbm_parser_parity.jl
cargo test -p mvmc-expert-parsers --test rbm
cargo test -p mvmc-expert-parsers rbm_phase_sine_cosine_bits
cargo test -p mvmc-core parsed_and_programmatic_rbm_terms
```

The source script also runs Julia's `test_parameter_init_complexflag_rbm.jl` and `test_read_input_parameters_rbm_layout.jl` unit contracts.

`parser.txt` contains **194** original extended-parser results across all nine Charge/Spin/General physical, hidden and physical-hidden sections. Each block records success, declared/inferred width, ComplexType, last mapping line, exact errors, sorted optimization flags and every mapping's coordinates/index/value bits. Cases cover absent and short headers, invalid header integers, arbitrary header labels, permissive coordinates, malformed/ignored row widths, missing flags, duplicate flags, the repeated-first-coordinate delimiter in two-column files, tied indices, comments and failed atomic mappings. Maximum signed indices retain Julia's wrapping count inference.

`initial.txt` contains **98** initialization comparisons: 14 namelists times seven flag/mode/neuron variants. Each block contains the nine mapped widths, final component flags, projection/RBM/Slater row-value bits and the next **624 UInt32 SFMT words**, seeded with **11272**. Namelists cover all section orders, missing/inactive/truncated flags, sparse and tied indices, empty and failed replacements, mixed empty families, cross-section optimization indices and DH2/DH4 offsets. The source initializes RBM before Slater, draws once per active indexed slot (including unmapped gaps), and scatters the same value to tied mappings. Missing RBM flags are inactive; missing Slater flags permit initialization. RBM ComplexType controls listed imaginary optimization flags but is excluded from the initializer's AllComplexFlag calculation. Real RBM normalization sums all four neuron settings and falls back to a divisor of one for a nonpositive sum.

`phase.txt` contains **4,146** exact Float64 sine/cosine comparisons, including signed zero, neighboring quadrant boundaries and phases from **2,048** SFMT draws with seed **20251002**, for both signs. It verifies Julia's arithmetic through `[-2π, 2π]`, used by complex RBM initialization. No floating-point tolerance or RNG tolerance is used in these fixtures.

## Input loaders and production kernels

```sh
julia +1.13.1 --project=extern/Julia-mVMC scripts/check_rbm_production_kernels.jl --write
julia +1.13.1 --project=extern/Julia-mVMC scripts/check_rbm_production_kernels.jl
cargo test -p mvmc-core --test rbm_production
cargo test -p mvmc-core rbm_
```

The production-kernel script also runs the canonical `test_unit_vmc_sampling_rbm.jl` tests. `production/loading.txt` compares four models (all nine sections, tied indices, mixed empty sections, and combined DH2/DH4/RBM) through initialization, optional full `initial.def`, InRBM overlays, and parameter synchronization. It records every row's exact component bits and the next **624 SFMT UInt32 words** after overlays, with seed **11272**. The full optimized record contains six diagnostics and projection/RBM/Slater triples; loaders validate every token before changing data. The optional indexed InRBM files retain Julia's permissive handling of count headers, duplicate indices, invalid floats, missing files and unknown section suffixes.

`production/updates.txt` compares the same four models before and after the original indexed SR update for every projection/RBM/Slater slot. It records count boundaries, exact row bits, and dense RBM values, including zero-valued unmapped gaps. Shared indices update every corresponding row; the last row supplies a dense coefficient when programmatic mappings disagree.

`production/kernels.txt` records **320** model/occupation combinations (five models times all 64 spin occupations of three sites), with full RBM counters, interleaved real/imaginary derivatives, and log weights. Each combination also records all **18** site/spin hops: **5,760** incremental counters and log ratios. Counters, derivatives, log weights and ratios now compare by exact Float64 bits, including signed zero. RNG comparisons remain exact. The parameter widths follow mapped indices; counter widths infer hidden neurons from coordinates and neuron settings. The source log-weight helper uses the configured neuron widths, preserving its separate behavior.

## Sampling, Green functions and SR

```sh
julia +1.13.1 --project=extern/Julia-mVMC scripts/check_rbm_math_parity.jl
julia +1.13.1 --project=extern/Julia-mVMC scripts/check_two_body_green_parity.jl --rbm
julia +1.13.1 --project=extern/Julia-mVMC scripts/check_fsz_green_parity.jl --rbm
for model in rbm_real rbm_cmp rbm_general_cmp rbm_dh24_cmp rbm_fsz; do
    julia +1.13.1 --project=extern/Julia-mVMC scripts/check_sr_direct_runner_parity.jl --case=$model --store=0
    julia +1.13.1 --project=extern/Julia-mVMC scripts/check_sr_direct_runner_parity.jl --case=$model --store=1
    julia +1.13.1 --project=extern/Julia-mVMC scripts/check_sr_cg_runner_parity.jl --case=$model
done
julia +1.13.1 --project=extern/Julia-mVMC scripts/check_sr_direct_runner_parity.jl --case=rbm_reference_cmp --store=1
julia +1.13.1 --project=extern/Julia-mVMC scripts/check_sr_cg_runner_parity.jl --case=rbm_reference_cmp
cargo test -p mvmc-core --lib rbm_
cargo test -p mvmc-core --test rbm_production
cargo test -p mvmc-core --test two_body_green
cargo test -p mvmc-cli --test runtime_contract rbm_
```

Append `--write` to a source command to regenerate its fixtures. Source scripts only add observation hooks to copies of the optimizer; sampling and numerical kernels are unchanged, and no vendored source is modified.

`production/math.txt` checks Julia Base complex exp/log/log1p/tanh for **2,616** inputs: large real parts, phase reduction through 1e300, neighboring quadrant boundaries, signed zeros, subnormals, infinities/NaNs accepted by all four operations, and 2,048 seeded inputs. Rust ports the Julia 1.13.1 arithmetic and compares all **10,464** complex results bit for bit. The parser sine/cosine checks remain exact.

`production/green_normal.txt` and `production/green_fsz.txt` compare exhaustive four-site one-/two-body operators, including coincident sites, spin changes and real/complex Slater tables, against the original RBM-aware Green functions. Independent Fock-space checks use the existing explicit 2e-12 absolute tolerance; Rust/source comparisons use exact component bits. Source verification passes **4,109** normal and **24,979** FSZ assertions.

The five `run_rbm_*` inputs exercise all nine RBM sections with nonzero indexed overlays, real and complex normal modes, General-only mappings, combined DH2/DH4/RBM, and complex FSZ. Normal models retain the canonical Hubbard sample=100/warmup=10 settings; FSZ retains the canonical PairHop fixture sample=2000/warmup=10 settings with PairHop omitted. Complex normal inputs set Orbital ComplexType explicitly: RBM ComplexType alone does not select the source's AllComplexFlag.

The `sr_direct/rbm_*` and `sr_cg/rbm_*` fixtures cover **68** prefix runs at 1/2/3/50 requested steps: five models with direct NStore=0/1 and CG, plus the unmodified ten-site `general_rbm_cmp` reference inputs with direct NStore=1 and CG. The reference case uses its original seed **12395** and complete `initial.def`; other models use seed **1**. Prefix comparisons include exact parameter/energy bits, all saved configuration arrays, packed burn-in state, proposal/acceptance counters, the next **624 SFMT UInt32 words**, per-step output and final parameter files. The first direct prefix also compares sampled SR OO/HO and stored derivatives, followed by matrix/gradient/Cholesky/solution and Gram-contraction checks. The GeneralRBM reference coverage here is deterministic through 50 steps; the separate 1,500-step statistical ctest acceptance is part of the model-selectable harness work.

The runtime intentionally preserves source behavior:

- Normal complex sampling updates RBM counters on accepted moves, rebuilds them after full inverse refresh, and includes the real log-ratio in acceptance. Real normal and FSZ sampling omit RBM acceptance weights in the canonical source.
- Normal main calculation places RBM derivatives between projection and Slater. FSZ omits RBM derivatives and places Slater directly after projection while still reserving RBM-sized parameter/state buffers.
- Nonfinite local energy is rejected using `isfinite(real(e)+imag(e))`, before energy/SR accumulation. Julia merges local SR arrays into cleared global arrays; the matching addition preserves its signed-zero behavior in saved derivatives.
- The FSZ direct runs with 50 requested steps return status **1** at zero-based step **13**, after all 2,000 energies become nonfinite. Both NStore variants compare the actual failure state/RNG, existing per-step files and absent final parameter files. FSZ CG continues through empty-weight steps and matches the full 50-step source result. Failure is recorded in `step-50-status.txt`; it is not presented as a successful 50-step direct optimization.
- Full input/optimized-record loaders accept projection/RBM/Slater triples. The original history and writer retain Gutzwiller/Jastrow/orbital mapping rows only, omitting RBM and DH coefficients from their snapshots and emitting no separate RBM/DH optimized files. These omissions are reproduced; parameter fixtures independently capture all RBM rows.

The public runner accepts the nine RBM sections and optional InRBM overlays. Library/CLI tests compare source output with production inputs, and existing real-FSZ/normal-InterAll/MPI rejection gates remain separately scoped.
