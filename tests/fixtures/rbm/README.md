# Canonical RBM parser, initialization and production-kernel fixtures

These fixtures use the unmodified Julia-mVMC checkout at `8bb1b9e8ae47b1512c00b321be05664ddcac0fd1` (numerical/parser sources at `c2ea432785bc14364a3cd5e9eef44db464289cc9`), Julia **1.13.1**, `extern/Julia-mVMC/Manifest-v1.13.toml`, and OpenBLAS **0.3.30 ILP64**, one thread. They do not establish production RBM sampling/SR support; issue #26 remains open.

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

`production/kernels.txt` records **320** model/occupation combinations (five models times all 64 spin occupations of three sites), with full RBM counters, interleaved real/imaginary derivatives, and log weights. Each combination also records all **18** site/spin hops: **5,760** incremental counters and log ratios. Counter components compare exactly; transcendental kernel results use a fixed absolute tolerance of **5e-14**. RNG comparisons remain exact. The parameter widths follow mapped indices; counter widths infer hidden neurons from coordinates and neuron settings. The source log-weight helper uses the configured neuron widths, preserving its separate behavior.

These changes establish loader, kernel and indexed-update parity. The public runner still rejects RBM until sampling, Green functions, main-calculation derivative placement and complete SR trajectories have been connected and verified.
