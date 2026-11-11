# Shared provider factorization prototype

The released faer Pfaffian API discards its paired elimination factors. Computing
another LU (or another paired skew reduction) for the inverse duplicates O(n^3)
work. The checked-in patch extends the **provider itself** with
`pfaffian_factor`, `pfaffian_inverse`, and `pfaffian_inverse_scratch`, retaining
paired L/D factors during the original reduction. Inverse evaluation then uses
faer triangular inversion and triangular GEMM. Input is preserved and complex
transposes do not conjugate. Error behavior of the original standalone API is
unchanged; its implementation is specialized without factor-output overhead.

The patch applies to tensor4all/faer `144621059bca3f0348c4ddb64d68428185528daf`,
the original benchmark's provider revision. It has **not been submitted upstream,
published, or used by the main workspace**. The existing user checkout is never
modified by the reproduction script. Its SHA-256 is
`64d3301db637cda2ea3cd29454c1850600754a1ceba5ea4e0c8dbbb565fb7d8b`.

```sh
bash benchmark/pfaffian_faer/provider-prototype/run.sh
# Optional local source (still cloned into an isolated temporary directory):
MVMC_PFAFFIAN_FAER_SOURCE=/path/to/t4a-faer \
  bash benchmark/pfaffian_faer/provider-prototype/run.sh
```

The probe first checks random matrices at n=2/6/24/26/64/130 against the
independent factor identity `A[perm,perm] = L D L^T` (absolute bound 1e-11)
and a scale-normalized inverse residual (bound 1e-12). It then validates the
timing inputs' inverse residuals before benchmarking. The main crate's normal
tests and dependency graph do not require this prototype.

`shared-factor-probe.csv` is one 250 ms adaptive timing sample per row on the
same Xeon E5-2699 v3/OpenBLAS 0.3.26 host, BLAS/OpenMP threads=1, rustc 1.99.1,
release opt=3/codegen-units=1/thin-LTO. The dependency lock is included. Compared
with current BLAS-enabled PfaPack full rebuilds, the prototype gains about
1.10–1.51x for real n=64–512 and 1.28–2.94x for complex n=32–512. Small sizes
still regress. This is preliminary kernel evidence, not E2E evidence or a claim
that a provider release already contains these APIs.

The isolated `run.sh` reproduction passed too; its output is retained in
`shared-factor-reproduction.csv`. Five subsequent samples are in
`shared-factor-samples.csv`. Ratios of mean timings are 1.09/1.26/1.48/1.49x
for real n=64/128/256/512 and 1.28/1.81/2.46/2.74/2.96x for complex
n=32/64/128/256/512. The probe now additionally checks empty/odd/nonsquare,
zero pivots, nonfinite input, underflow, inverse overflow, lower-triangle
authority and padded input/output windows. Those checks passed in a subsequent
run. Team execution and broad scale/conditioning tests are still pending.

To adopt it, the provider API needs full error/shape/strided/team/conditioning
tests and an approved upstream or maintained-dependency integration path. Then
the main workspace can replace the duplicate-factor experiment with a reviewed
pinned provider version and complete C/GPU/RNG/repeatability and E2E gates.
