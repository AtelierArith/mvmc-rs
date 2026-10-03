# Native macOS FSZ measurement first-divergence diagnostic (#181/#185)

These are **diagnostic observations, not golden expectations or a portable
numerical acceptance budget**. Cargo does not read this directory or run the
optional Julia script. No Rust-produced expected values, parameter replacement,
sampler rerun, tolerance update or algorithm change is part of this artifact.

## Actual source and capture

- Native macOS ARM64 all-features job
  [111188179757](https://github.com/AtelierArith/mvmc-rs/actions/runs/37117907937/job/111188179757),
  source `0347bdf8d4b4478f74494140cc2a54d875ab225e`, completed failure
  2026-10-03 11:03:42 UTC. Actual saved configurations, counters, draw count and
  non-consuming next624 passed before each measurement. This historical job
  is not current PR208 native-macOS acceptance evidence.
- Complete downloaded job log SHA256:
  `b8c987f93ac233eeb5c53547d6b4f5c236e64611b2cedcbd280b1cda335ee613`.
  Its optional `FSZ181 ... scope=private-measurement-replay` rows are scratch
  replay from the actual saved state, **not** production per-addition observation.
- `mac-inverses.txt` contains all 400 captured 6x6 complex inverses: frame,
  walker, then 36 real/imag pairs in the original flat buffer order. SHA256:
  `7811e7f6d90b64267f956f93f09956220bf4516789e541ef548c3434be32b546`.
  Values were extracted without arithmetic from `qp=0 inverse=` rows. They are
  measured Rust diagnostic **inputs**, never reference expectations.
- The Mac job installed Homebrew OpenBLAS0.3.34 and configured
  `OPENBLAS_CORETYPE=NEOVERSEN1`, threads1. These are recorded job settings,
  not a claim to have observed a particular BLAS call during every operation.
- Reference measurement replay uses the unchanged shared Julia revision
  `8bb1b9e8ae47b1512c00b321be05664ddcac0fd1`, Julia1.13.1, Linux x86_64,
  actual ILP64 OpenBLAS and BLAS threads1. It parses the existing independent
  `two-samples/heisenberg_chain_fsz` input, fixed records and saved configurations.
  It does not sample. This is **not native macOS Julia execution**.

## C operator, branch and sign identity

Authoritative mVMC1.3.0 revision
`d73d06bd529d3b2573f38eb5817c4a5f52971006`:

- `src/mVMC/pfupdate_two_fsz.c` SHA256
  `a5bc91a5478789f9c460203bddf699e9a99fd1fad8d9587c311d2b8449fc9872`.
  `calculateNewPfMTwo_child_fsz` gathers destination rows into vec_a/vec_b,
  computes four dot products and the nested bMa sum, then evaluates
  `inv_ab*vec_ba + inv_ab*bMa + p_a*q_b - p_b*q_a` in that order and
  multiplies by PfM. Source lines111-187 were reviewed directly.
- `src/mVMC/locgrn_fsz.c` SHA256
  `01766dc6aefe5228ab6d01c9f6d882f228e3e2537873ad25a2dae1e4e370249c`.
  `GreenFunc2_fsz` moves ml first, then mj, calls the two-electron update with
  `(ml,t,mj,s)`, multiplies by projection/IPnew, restores the configuration and
  returns `conj(z/ip)`. No sign or reassociation repair is assumed.

The first failing raw direct **complex entry13** is
`<c†_(0,up) c_(1,up) c†_(1,down) c_(0,down)>`, not flattened numeric field13.
For its nonzero general branch, site0-up/site1-down are empty and
site1-up/site0-down occupied. There are no coincident-spin-site reductions.
Set ma=ml (electron at site0-down), mb=mj (site1-up); move ma to site1-down
then mb to site0-up. Thus vec_a uses destination spin-site7, vec_b uses
destination spin-site0, and both are gathered against the twice-moved state.
The script repeats both projection-count updates and asserts projection ratio1.
It also asserts the single QP full weight is1; PF equals IP for this input.
Measurement weight is1 per saved walker, total200. These are model-specific
checked conditions, not arbitrary assumptions for other inputs or Lanczos modes.

## First numerical cause and bounded interpretation

The same captured Mac inverse fed into the unchanged Julia Green kernel
reproduces frame0 direct13 accumulated imaginary value
`-0.07331601999933586`. The actual Mac failure was `-0.07331601999933571`
(difference about1.5e-16); Linux/independent Julia gives
`-0.07331602002844077`. PF/IP are identical across all 400 captured matrices.
The first captured inverse difference is not asserted to be the first internal
BLAS operation, and a tiny diagonal difference is not blamed for the failure.

First nonzero Green difference: walker2. Four ratio terms contain approximately
`-12.922-2.762i`, `209.034-2184.805i`, `-195.722+2187.566i`, and
`-0.18334+0.000047i`: cancellation factor21336.21. Worst local difference
for this entry: walker22, magnitude5.82e-12, cancellation factor50513.34.
Its Mac inverse normwise backward error is4.92e-17, residualInf1.28e-13,
conditionInf2599.43. The 256-bit off-diagonal-only input perturbation changes
the ratio by9.49e-12. Thus the amplified four-term subtraction, not a diagonal
1e-19 observation alone or Monte Carlo noise, explains this captured failure.

The diagnostic evaluates arithmetic centers at256 bits. Complex products use
component absolute-product scales and gamma2; addition radii propagate through
the exact C dot/nested-sum/four-term graph. The tail uses actual PfM/IP operands:
PfM multiplication, single-QP weight1 and projection1 multiplication, then
separate operand-dependent quotient balls for macOS ARM compiler-rt exponent
scaling/FMA/norm-squared division and Julia's ordinary-range Smith r/t path.
Finite normal-range intermediate centers, overflow limits, nonzero denominator
intervals, nonzero Smith component products and excluded extreme-range branches
are guarded. Binary denominator scaling and conjugation preserve the radius.
This replaces the earlier unsupported fixed24u terminal operation count.

The inverse perturbation term is still the **observed Mac-Linux difference**.
Residuals/conditioning are reported separately; they have not been converted
into an independently admissible future-backend inverse perturbation. Therefore
the pairwise propagated result (about2.7744e-9 for this captured sum) is strictly
posthoc diagnostic evidence. It is **not** a universal1.9e-9/2.8e-9 tolerance,
not a residual-based portable acceptance budget, and not adopted by Rust tests.
A future budget requires independently justified inverse admissibility and
all relevant quantity/branch scopes, not this single entry or frame1's worst
Exchange term. Exact RNG/configuration checks remain prerequisites.

## Optional reproduction and hashes

```sh
julia +1.13.1 --project=extern/Julia-mVMC \
  scripts/diagnose_physcal181_fsz_roundoff.jl
```

The script also accepts an explicit captured-inverse input path as its sole
argument. It requires the existing Julia1.13 manifest and reference dependencies,
not C compilation. Normal Cargo tests require neither that runtime nor these
observations.

Final owner handle18608 terminal0; source hashes before/after matched. All 200
frame0 walkers were processed, including exact zero branches; every nonzero
branch's observed pairwise error satisfied its propagated diagnostic bound.
The log prints the terms and per-walker condition/residual information and the
ordered accumulated sum. Earlier portable-path handle64215 failed only a
diagnostic type guard comparing the QP-weight struct with a vector; corrected
guard uses `qp_full_weight`. That failed run is not a PASS.

- `scripts/diagnose_physcal181_fsz_roundoff.jl` SHA256:
  `fec889059519bbe0610528d6aaa3d052cbc826ad079b6a452c7e6f8bd154e297`.
- `frame0-direct13-replay.txt` (complete83-line final log) SHA256:
  `c416a9f3676129d22758d97833454f1fc0e19cb6afb8888ecd40fbc507c3a998`.
- Reviewed quotient sources: core `c_complex.rs` SHA256
  `a01be8efe7e9276109742bb10013b1276504ac501e1305d21d506c32d89b7ae3`;
  core `c_complex_gnu.rs` SHA256
  `928a0fc7cde1b507af1efa99407b908fe899631307ef134d45c304ddcb93b448`;
  PfaPack `julia_complex.rs` SHA256
  `6029c2daae9c6247b01d114b7fa6a6c6c76c65ed277269259775ec03930f093d`.

No production code, historical golden, numerical tolerance or shared reference
was changed. This artifact alone does not close #181/#185 or establish current
Mac CI acceptance, full native-C sampling, MPI measurement or every Green entry.
