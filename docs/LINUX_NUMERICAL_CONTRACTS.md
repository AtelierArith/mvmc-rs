# Linux GNU numerical reference contracts

Issue #187 separates historical macOS/Julia reference behavior from native
Linux x86_64 C numerical authority. Existing macOS native fixtures remain
byte-for-byte intact. New `tests/fixtures/interall/*_linux_gnu.txt` files contain
independently compiled C results for the Linux GNU runtime; Cargo tests select
those files on Linux GNU and continue exact bit comparisons.

## Complex quotient and Green kernels

Apple clang 17 uses LLVM compiler-rt's exponent-scaled norm-squared complex
quotient. Ubuntu Clang 18 and GCC 13 link GNU libgcc's scaled Smith quotient,
with different rounding and subnormal/range recovery. Rust ports the GNU
algorithm on Linux GNU in `crates/mvmc-core/src/c_complex_gnu.rs`; the archived
LLVM implementation remains active on other targets. The port preserves
individual products, sums, divisions, scaling thresholds and recovery branches.
It adds no C compiler, toolbox, oracle execution or GNU runtime call to Cargo.

The standalone C quotient executable's `__divdc3` symbol was local/static
(`nm` type `t`), and `ldd` listed only libc. Clang's
`-print-libgcc-file-name` resolved to GCC 13's `libgcc.a`, whose SHA-256 is in
fixture headers. Ubuntu's separately installed shared `libgcc-s1` 14.2 package
does not identify the code used by that static quotient probe.

The native Linux fixture headers record Clang 18.1.3, glibc 2.39, archive path
and hash, exact compiler flags and upstream C-source hashes. The GCC 13.3.0
source fragment in `c_toolbox/gcc_divdc3_reference.inc` retains its full upstream
notice and extraction boundaries. The Rust port preserves the GPLv3/GCC Runtime
Library Exception notice; the repository GPL license and
`crates/mvmc-core/LICENSE-gcc-runtime.txt` contain the license text.

| Fixture | Independent native checks |
| --- | --- |
| `c_complex_division_linux_gnu.txt` | 373 normal/subnormal/range/nonfinite complex quotients |
| `c_complex_green_linux_gnu.txt` | 4,096 normal complex operators, 4 ordered InterAll sums, 4 PairHop sums |
| `c_real_green_linux_gnu.txt` | 4,096 normal real operators and 4 ordered InterAll sums |
| `c_fsz_green_linux_gnu.txt` | 768 one-body + 49,152 two-body FSZ operators and 12 ordered sums with duplicates |
| `c_fsz_real_green_linux_gnu.txt` | 384 one-body + 24,576 two-body real FSZ operators and 6 ordered sums with duplicates |

Real normal and real FSZ numerical bodies are identical to their archived macOS
counterparts; separate Linux files document their independently verified
provenance. Green probes use historical Julia wavefunction arrays only as
inputs. Expected operators and sums come from verbatim C kernels and actual C
accumulators. They cover one-process `MPI_COMM_SELF` plumbing and preserve
configuration/working buffers. They do not establish full C executable,
production FSZ dispatch, MPI sampling or SR trajectories.

The red tests with the old LLVM quotient failed all three complex gates:
quotient case 7 first, 341 normal operator/sum mismatches and 3,611 FSZ
operator/sum mismatches. The GNU port passed all five native gates without
changing tolerances. NaN arithmetic results retain the existing C-contract
classification check; every non-NaN result, including signed zero and infinity,
remains bit exact.

Separate native Clang 18 and GCC 13 runs produced identical numerical bodies
for all five quotient/Green families, including every operator and ordered sum.
Compiler metadata is still recorded separately rather than inferred from this
agreement.

## Reproduction

Inside the verified Ubuntu 24.04 Linux amd64 environment:

```sh
# Reviewable generation through the shell orchestrator; no source overwrite.
scripts/generate-numerical-references.sh --suite c-division \
  --suite c-interall-complex --suite c-interall-real \
  --suite c-fsz-complex --suite c-fsz-real --output /tmp/linux-native-review

# Verify the committed platform fixtures directly, independently of Cargo.
uv run --no-project python scripts/check_complex_division_c_parity.py
uv run --no-project python scripts/check_interall_complex_c_parity.py
uv run --no-project python scripts/check_interall_real_c_parity.py
uv run --no-project python scripts/check_fsz_green_c_parity.py
uv run --no-project python scripts/check_fsz_green_c_parity.py --real

cargo nextest run -p mvmc-core --locked --cargo-profile test-fast \
  -E 'test(scaled_complex_quotients_and_range_recovery_match_native_c) | test(native_c_green) | test(native_c_bits)'
```

The four generators detect native macOS versus Linux x86_64 glibc and select
platform-specific files. `--platform apple` or `--platform linux-gnu` asserts
the intended native platform; cross-platform requests fail. `CC` is honored;
the staging wrapper also records compiler invocations. Changing compiler or
runtime metadata can change fixture headers even if all numerical bits agree.
Use the same recorded environment for exact regeneration, and review new
platform contracts before applying them.

## Other staged C differences

The initial 19-suite Linux audit also found these differences. They are
classified separately and were not silently substituted into old expectations:

- `interall/c_reader.txt`: 15 output rows differ across three site counts.
  `NaN(123)`, `NaN(0x123)` and `-NaN(077)` have Apple scanf payload bits versus
  GNU scanf's canonical NaN. Incomplete exponent inputs `1e+` and `0x1p+`
  consume different input prefixes, changing the following imaginary field.
  Acceptance, indices and ordinary finite complete numeric conversions agree.
  The archived parser fixture documents Apple scanf behavior; it does not prove
  GNU payload/malformed-token parity.
- `orbital_general/c_general_kernels.txt`: 98 rows of derivative outputs differ
  because C `SlaterElmDiff_fsz` uses `invIP = 1.0/ip`, invoking the platform's
  complex quotient. Reader results and Slater construction rows agree. This
  derivative oracle is distinct from the Green kernel gates above.
- `orbital_general/c_rbm_prefix.txt`: two normalized complex Slater rows differ
  by rounding in `SyncModifiedParameter`'s `cabs`-based normalization. All 30
  input cases' following 624 SFMT words agree exactly. This establishes RNG
  agreement for these probes, without claiming full production normalization
  or sampling parity.

## Exact historical Julia runner parity

The first FSZ inverse divergence was the tridiagonal solve's complex division.
The Linux GNU callback now uses the tested GNU quotient without copying GPL
runtime code into the separately licensed PfaPack crate. The macOS callback
retains its archived arithmetic. Independently generated Linux Julia overlays
restore exact setup, sampling, history and SR runner comparisons, including
CG iteration/status/residual outputs and full RNG/configuration controls.
No historical tolerance was introduced. See
[`linux_gnu_julia/README.md`](../tests/fixtures/linux_gnu_julia/README.md) for
the source/BLAS manifest and preserved macOS fixture selection.
