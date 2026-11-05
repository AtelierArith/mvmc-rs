# RBM C contracts and inputs

The authoritative implementation is `extern/mVMC-1.3.0`. `c_reader_contracts.txt`
contains 1,307 actual-reader cases (766 accepted / 541 rejected), covering all
nine sections, positive declared widths, complete geometry and ordered signed
flags. A declared width of 97 is retained even when only indices 0 and 2 are
mapped. The historical width-97 definitions below have incomplete mappings and
flags; C rejects them. Duplicate coordinates can leave cells untouched; index
`-1` and imaginary flag `0` in the reader probe are explicit sentinels, not
evidence of native malloc initialization. Rust bounds checks also cover unsafe
indices that are deliberately excluded from native execution.

`c_parameters.txt` contains 84 actual `InitParameter`/`ReadInitParameter` and
native SFMT checks: all nine individual sections, combined sections, absent
sections and the 97-slot block, seeds 1/11272, real initialization with total
neuron divisors 0/-7/10, and initial records including distinct successive
records and an empty file. Full declared projection/RBM/Slater coefficient
bits and the following 624 RNG words are retained. The zero/negative divisor
cases exercise actual C arithmetic; Julia's divisor-one repair is not used.
This oracle does not establish complex `cexp` initialization or normalization.

`c_counters.txt` contains 4,994 actual `MakeRBMCnt`/`UpdateRBMCnt` cases across
66 completely assigned tables, at 1/3/6 sites and 1/2/4 neurons per family.
Cases cover complete reordered mappings, shared and unused declared slots,
all nine families, a 97-slot prefix, binary and cancellation-sensitive complex
coefficients, no-op moves and legal hops in both directions/spins. The generator
also verifies C's in-place update against its separate-output result. All
counter components are recorded as exact binary64 bits; coefficients are
supplied to C as bits. Total neuron count here equals the family-count sum.
This scope excludes FSZ updates, transcendental weight/derivative functions,
MPI and full Monte Carlo execution.

```sh
uv run --no-project python scripts/check_rbm_contracts_c_parity.py
uv run --no-project python scripts/check_rbm_parameters_c_parity.py
uv run --no-project python scripts/check_rbm_counters_c_parity.py
cargo nextest run -p mvmc-expert-parsers --test c_rbm_contracts
cargo nextest run -p mvmc-core --test c_rbm_parameters --test c_rbm_counters
```

Append `--write` only to regenerate both source excerpts and expectations.
The optional native probes preserve actual function bodies and source SHA-256
in `c_toolbox/`; they use Apple clang 17, `-O0 -ffp-contract=off`, with
`-DMEXP=19937` for the SFMT initializer. Ordinary Rust tests consume only
checked-in fixtures and do not compile, invoke or read the toolbox.

The Julia 1.13.1 CPU outputs that used to live here (parser, layout, initialization, kernel, Green-function and SR expectations) were removed with the other Julia CPU references in #695; C 1.4.0 is the only CPU reference.

