# Standard SR-CG reference fixtures

These fixtures use the unmodified numerical implementation at
`extern/Julia-mVMC`, commit `c2ea432785bc14364a3cd5e9eef44db464289cc9`,
with Julia 1.13.1 and `Manifest-v1.13.toml`.

The generation and verification environment is Intel macOS 15.8.1 (`x86_64`,
Julia CPU name `skylake`), with one BLAS thread:

- Julia: OpenBLAS 0.3.30, ILP64, Haswell kernels.
- Rust: Homebrew OpenBLAS 0.3.34, LP64, Haswell kernels.
- FSZ inverse: the reference uses macOS Accelerate through its native
  inverse implementation. Rust performs the inverse algorithm itself and
  selects Accelerate's triangular inverse and matrix multiplication kernels.

`real.txt` and `complex.txt` check fixed sampled operators and all CG
solution, residual, and direction vectors for iteration limits 1 through
41, including residual refreshes at iterations 20 and 40.
`sampled_complex.txt` repeats this check with inputs from the complex
optimizer's first sampled step.

The `real_runner`, `cmp_runner`, `fsz_runner`, and `hubbard_runner` directories contain
independent optimizer runs of one, two, three, and fifty steps. Every run starts
with seed 1 and `NSRCG=1`, and checks parameter and energy bits, all saved
electron configurations, the next 624 SFMT words, and the complete
`SRinfo` output. The RNG block is read after the run; reference capture
does not copy or advance the process-global Julia SFMT state mid-run.

Run the corresponding `scripts/check_sr_cg_*_parity.jl` scripts with
`julia +1.13.1 --project=extern/Julia-mVMC`. Pass `--write` to regenerate.
An observation hook added to a copy of the Julia runner captures results;
all numerical routines remain authoritative Julia code.

These checks establish exact parity for the recorded environment and
inputs. Other architectures and BLAS implementations require their own
source comparisons; these fixtures do not establish universal bit parity.

## Historical Julia-order status (#358)

The ordinary real Rust Pfaffian path now follows C's operation order. For the
real, hubbard, opttrans, dh2/dh4/dh24, pairhop and rbm real CG/direct families
these Julia-order trajectories are historical: only configurations, RNG, step-1
energy and SR S-diagonal columns are still checked against them. Parameter
trajectories are not a portable reference (finite-iteration CG amplifies
last-bit differences; see `docs/NUMERICAL_COMPARISONS.md`). Native C step-1
operands are in `../c_order_sr_operands/`.
