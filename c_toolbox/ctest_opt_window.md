# Independent C optimization-window output oracle (#180)

`ctest_opt_window.c` supplies storage and a text-history parser to verbatim
C `avevar.c` bodies. It does not run the C sampler, SR, MPI or BLAS. History
values come from the separately recorded canonical Julia/C-contract observer,
never Rust. Results are explicitly mixed references, not full C execution.
Cargo builds/tests never compile, invoke or read these programs.

Authoritative source: `extern/mVMC-1.3.0/src/mVMC/avevar.c`, SHA-256
`509a573944a5eabde93864346902674faaff6c23d0c88f89867263f8372dd51a`.
`ctest_opt_window_upstream.inc` retains lines 1–27 (copyright/license) and
34–260 (complete `CalcAveVar`, `WriteHeader`, `Child_OutputOptData`,
`StoreOptData`, `OutputOptData` bodies). Only include directives and the
outer include guard are omitted. No numerical bodies are translated or
edited. Verify this extraction before reproducing:

```sh
diff -u <(sed -n '1,27p;34,260p' extern/mVMC-1.3.0/src/mVMC/avevar.c) \
  c_toolbox/ctest_opt_window_upstream.inc
ctest_window_build=$(mktemp -d /tmp/mvmc-ctest-window.XXXXXX)
cc -O0 -ffp-contract=off c_toolbox/ctest_opt_window.c -lm \
  -o "$ctest_window_build/probe"
# Generate independent prefix history first with ctest_prefix_oracle.jl.
julia +1.13.1 --project=extern/Julia-mVMC \
  c_toolbox/ctest_aggregate_windows.jl "$ctest_oracle_stage" "$ctest_window_build/probe"
```

Compiler environment: native Linux x86_64, Ubuntu GCC
13.3.0-6ubuntu2~24.04.1, `-O0 -ffp-contract=off -lm`. Upstream `sprintf`
produces compiler size warnings; the adapter bounds the output head with
64 bytes reserved for all suffixes. No warnings are fixed in extracted C.

The adapter accepts an optional `iFlgOrbitalGeneral iNOrbitalAntiParallel
iNOrbitalParallel` triple after `INPUT OUTPUT_HEAD`; omitting it leaves the
upstream globals zero (plain Slater `_orbital_opt.dat`). The window-compared
DH2/DH4/DH24/RBM/OptTrans FSZ cases record `1 36 15` so the verbatim
`OutputOptData` emits `_orbitalAntiParallel_opt.dat` plus
`_orbitalParallel_opt.dat`; `interall` would use `1 0 0`
(`_orbital_general_opt.dat`). Basic/FSZ/InterAll prefixes and the non-FSZ
models are not part of the declared-window comparison and use the zero
default. The main contiguous `zqp_opt.dat` row is the same in every branch;
only the auxiliary Slater block filenames and contents depend on the triple.

The history header carries sample count, declared NPara, then widths in C
order: Gutzwiller, Jastrow, DH2 groups, DH4 groups, nine RBM sections, Slater,
OptTrans. Rows contain paired real/imaginary components of
`[Etot, Etot2, post-SR synchronized declared Para]` in chronological order.
The aggregator reconstructs metadata from canonical parsed declared slots;
this corrects the initial observer's RBM mapped-term-count header without
changing any numerical history line. The native probe validates dimensions
and finiteness, calls actual `StoreOptData` for each row, then actual
`OutputOptData`. FSZ auxiliary filenames are emitted as a generic Slater
block; only the identical contiguous main stream is currently asserted.

Window 1 has upstream's special `(real(value), 0)` pairs, including energy
and energy squared. Window >1 uses sequential complex means, then sequential
`sqrt(sum(real(delta*conj(delta)))/(window-1))`, emitted as `(real, imag,
deviation)` triples. This is sample deviation, not standard error. Energies
belong to the just-measured pre-SR state; coefficients belong to its post-SR
synchronized state (`vmcmain.c:511–512`). Julia's final-parameter-only
`zqp_opt.dat` is retained externally but is not an expected C output.

Each generated case records source/extraction/adapter/executable/history
SHA-256 hashes and compiler options in `c-window-provenance.txt`. Check in
the standalone expected output and provenance; normal Rust tests read only
those fixtures. Record actual executed Rust comparisons separately.

## Effective window and portable comparison policy

Short prefixes deliberately override **both** NSROptItrStep and NSROptItrSmp
to the prefix length (1, 2, 3 or 50), in the Julia observer call and Rust
test modpara. This is a labelled prefix workload, not the canonical long
averaging window and not `min(prefix, canonical_window)`. Model settings
record original and effective counts; the C input's first integer is the
effective window. C `readdef.c:683-684` assigns both parsed fields directly;
`vmcmain.c:511-512` stores the last NSROptItrSmp post-SR snapshots. These
positive window<=steps inputs need no reader repair or invented clamp.
The long gate keeps original steps/window and seed for all thirteen models.

`CalcAveVar` has no solver or BLAS: the fixed-history check isolates sequential
addition, complex multiplication and sqrt. Its existing #190 componentwise
budget is `abs=1e-11, rel=1e-11`, with exact dimensions and row/column layout.
For mean x, perturbing each history value by at most epsilon perturbs its
exact mean by at most epsilon; sequential-rounding error scales with window
length and mean absolute magnitude. Sample deviation is a norm divided by
sqrt(window-1): the reverse triangle inequality bounds its input propagation
by sqrt(window/(window-1))*epsilon, plus mean/rounding errors. This centered
sum of squared magnitudes does not subtract two large second moments; all
terms are nonnegative for finite inputs, so no negative cancellation is
clamped. No tolerance relaxes signs, field order, dimensions or RNG/configs.
Cross-backend history/SR error is separately constrained by the preceding
exact trajectory gates, 1e-11 parameter/energy and 1e-12 SR-buffer checks.
This budget is not a claim that arbitrary ill-conditioned histories pass;
the fixture consumer rejects nonfinite finite-contract results. No solver
residual tolerance is introduced by aggregation; SR remains its own gate.
Observed fixture scale: at most 50 snapshots, maximum input scalar magnitude
207.16141233889115. With binary64 unit roundoff, the sequential 50-term mean
rounding bound is about 1.2e-12 at this scale, below the 1e-11 absolute budget.
Nonzero C deviations range from 9.4213490715182202e-19 (near-constant columns)
to 44.878963991413649, so the absolute term is necessary near zero; relative
error alone is unsuitable. The offline fixture consumer measures actual
arithmetic differences separately; it is not model execution coverage.

Window 1 skips `CalcAveVar` entirely. C emits no deviation column and forces
each imaginary field to zero, not NaN. For larger windows, the untouched C
body retains IEEE behavior; the adapter does not clamp variance or sanitize
computed nonfinites. The current fixture domain is explicitly finite input
histories and finite C results. Nonfinite input-domain probes are separate,
not silently covered by these model fixtures.

Actual native backend audit: `ldd` on both the FSZ reference energy bridge
and this aggregation executable resolves libm and libc only—no BLAS/LAPACK.
Julia initialization, Pfaffians and SR use one-thread ILP64 OpenBLAS through
libblastrampoline, as recorded in generation provenance. C local energy
uses the serial extracted kernels with Julia's borrowed matrices; it does
not calculate them with a hidden C BLAS provider. This distinction matters:
the mixed reference is not a full native C sampling/solver executable.
