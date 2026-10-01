# Direct SR reference fixtures

Authoritative source: `extern/Julia-mVMC`, commit
`c2ea432785bc14364a3cd5e9eef44db464289cc9`, Julia 1.13.1 with
`Manifest-v1.13.toml`. The verification environment is Intel macOS 15.8.1,
Julia CPU `skylake`, one BLAS thread. Julia uses OpenBLAS 0.3.30 ILP64
Haswell kernels; Rust uses OpenBLAS 0.3.34 LP64 Haswell kernels. The FSZ
inverse uses the Accelerate provider described in `../sr_cg/README.md`.

The real, complex, FSZ Heisenberg and real Hubbard cases each have
independent seed-1 prefix runs of 1, 2, 3, and 50 steps, with both
`NSRCG=0, NStore=0` (`*_runner`) and `NSRCG=0, NStore=1`
(`*_store_runner`). Hubbard additionally covers prefixes 4 through 10.
Each checks final parameter and energy bits, every saved configuration,
and the next full 624-word SFMT block. Store modes are compared against
their corresponding Julia runs, since their reduction algorithms differ.

`fixed-input.txt` records the first normalized overlap and gradient
buffers, active-component mapping, stabilized SR matrix, gradient,
Cholesky factor, and solved increment. `gram.txt` additionally records
the first raw sample store and source-finalized Gram matrix.
`small_gram.txt` covers the 12 combinations around Julia's generic/SYRK
dispatch boundary and preserves the extra buffer slots.

The reference runner and solver are copied with read-only observation
hooks; original numerical kernels and RNG calls remain unchanged.
Regenerate or verify with `scripts/check_sr_direct_runner_parity.jl`
(`--case=real|cmp|fsz|hubbard`, `--store=0|1`, optional `--steps=1,2,...`),
using `julia +1.13.1 --project=extern/Julia-mVMC`. `--write` regenerates.
Use `scripts/check_sr_gram_small_parity.jl` for the small Gram fixture.

These fixtures establish exact parity for these inputs and this environment;
other architectures and BLAS providers require their own source checks.
