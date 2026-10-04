# Mixed independent measurement and native C normalization evidence

Raw sums and Wc are independently captured by Julia1.13.1/Manifest-v1.13
for both frames of all nine `two-samples/` fixtures. The optional generator
`c_toolbox/generate_physcal_181_weighted_green.jl` feeds these inputs to the
actual extracted C `average.c:weightAverageReduce_fcmp` body. Serial MPI
adapters return rank0/size1; unexpected multi-rank workspace calls abort.
No Rust values are used to generate expected data.

The resulting `model/frame-i/{one,factored,direct}.txt` contain native C
normalized arrays. `comparison.txt` records each raw input hash and its
first C–Julia differing component, not merely a blanket tolerance result.
All 54 actual-frame/family comparisons have `first_difference=none`.
Input reference records retain seed, sample stages, revision, Julia/BLAS
versions, threads and their own source hashes; this C scalar kernel uses
no BLAS/RNG. Extraction boundaries, driver/source hashes and compiler flags
are in `provenance.txt`.

The non-real synthetic weight deliberately probes the complex reciprocal.
The first divergence is already in `1/(2.5+0.125im)`: C real part
0.399002493765586 versus Julia0.39900249376558605, delta
-5.551115123125783e-17; imaginary parts agree. The first Green component
then differs by -2.7755575615628914e-17; maximum output difference is
1.1102230246251565e-16. See `reciprocal-first-divergence.txt`.
This is a well-conditioned reciprocal and ordered multiplication, not a
trajectory or algorithm discrepancy. Tests preserve the existing explicit
1e-12 absolute/1e-10 relative comparison policy; no tolerance is increased.

C accumulation authority: `calgrn.c:95–110` adds each sample's `w*one`,
`w*direct`, and `w*one[first]*conj(one[second])`, in requested order.
`average.c:209–217,276–300` subsequently computes one complex reciprocal
of Wc and multiplies the ordered contiguous Green arrays. Julia's raw
checkpoint is before `weight_average_we!`/`weight_average_green_func!`;
normal and FSZ measurements therefore have independent pre-normalization
references, not values reconstructed from Rust's normalized outputs.

Normal Rust tests read these fixtures only. The actual runner's recording
reducer compares raw energy and already-normalized Green arrays to independent
Julia and native C results. The same two-sample test now installs the public
read-only Green observer and compares actual serial pre-division weight and
all three ordered Green arrays to `two-samples/*/accumulated-{0,1}` for all
nine models. This is actual runner observation, not a second sampler or
fixture-fed averaging replay; it does not claim globally reduced MPI raw
arrays or full native C sampling. The observer's separate normal/FSZ identity
tests verify that observation preserves outputs, configuration and RNG.

## Fresh bounded serial runner verification

On committed source77ac9976, parent fresh run
`a9efb626-781d-4783-b229-69d9dce6dd57` terminal0:1PASS,43 unselected,0.792s.
Exact command:

```sh
cargo nextest run --locked --cargo-profile test-fast -p mvmc-core --test physcal_issue181 -E 'test(two_sample_runners_match_independent_saved_states_rng_and_ordered_outputs)' --no-fail-fast --retries 0
```

Parent confirmed the actual public read-only serial raw observer and all nine
models/two frames in this ONE named test. Test SHA256
`fc39e4901470b13650bf31e11cead4c821730d1c32b4affeb789c4a5e8798b60`;
run.rs SHA256 `af5204754499721f75667ae20df8e802ef8c220f24a0a515a159d92131563a6e`.
Working-file hashes and `git show 77ac9976:` hashes were independently checked
equal during this documentation handoff. Only these selected production/test
files were matched to committed hashes. The shared checkout includes dirty
harness/support files, including support/ctest_provenance.rs; no complete
compiler-input closure was captured or verified. No production or fixture
changes here, no fresh C/Julia acquisition, no numerical-bound change.
Historical raw-observer and separate identity runs retain their own versions;
this fresh selection does not rerun those separate identity tests, every #181
case, globally reduced MPI raw arrays, native macOS or full native C sampling.

This satisfies independent C normalization-contract verification, not full
native C composed-input sampling. The latter is stronger optional evidence,
not a blanket prerequisite for every #181 fixture cell. Required parity
claims remain mixed Julia numerical/trajectory references plus actual C
parameter, reader, formatting and averaging kernels, with supported C input
and operation-order authority stated for each cell.

Regenerate explicitly, never from Cargo:

```sh
julia +1.13.1 --project=extern/Julia-mVMC c_toolbox/generate_physcal_181_weighted_green.jl
```
