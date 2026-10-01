# Canonical RBM parser and initialization fixtures

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
