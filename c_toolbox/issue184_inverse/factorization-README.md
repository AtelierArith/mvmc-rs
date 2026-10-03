# Fixed complex6 factorization-through-inverse contract

Optional standalone kernel acquisition for #184 M1205–1207/S261, NOT the
original unseeded Julia input or full model/sampling/MPI parity. No C/Julia
oracle or toolbox access from Cargo. Old real4/complex6 pair-pivot fixtures
remain unchanged. New stem: `complex6_factorized`.

## Authority and acquisition

C `extern/mVMC-1.3.0/src/mVMC/matrix.c:165` invokes M_ZSKTRF("U","N") then
utu2inv_z at173. Source SHA256
`849488176375102fbc3bab7001e0dc27dfa7c8e049fd90ee1d6248ebfba10764`.
The probe invokes original ZSKTRF U,N and utu2inv<std::complex<double>> on the
actual retained native factor/pivots. For small n6 the original dispatcher
uses its unblocked ZSKTF2 branch; no factorization or inverse numerical body
is rewritten/extracted. Complete unchanged Fortran zsktrf/zsktf2/zlasktrf/
zskr2/zskr2k sources are compiled, with complete original ilaenv wrapper.
All compiled units and transitive common/ltl2inv headers are hashed before
compiler invocation and checked unchanged after acquisition.

Fortran licensing/origin remains in original
`extern/mVMC-1.3.0/src/pfapack/LapackLicence` (Wimmer/LAPACK-derived modified
BSD redistribution terms); included invert.tcc retains its MPL-2.0 notice.
No vendored source edits, production FFI or distribution of compiled binaries.
The harness uses a separate C++ alias for native zsktrf_ with actual LP64
32-bit Fortran integers; original blalink_fort.h's BLIS dim_t declaration is
host-long. This ABI adapter changes no arithmetic or pivot algorithm.

Exact reproduction from repo root, requiring a NEW external destination:

```sh
OPENBLAS_NUM_THREADS=1 julia +1.13.1 --startup-file=no --project=extern/Julia-mVMC c_toolbox/issue184_inverse/generate_factorized.jl /tmp/NEW-issue184-factorized-stage
```

`factorization-provenance.txt` is the native generator-written metadata from
actual final stage `/tmp/mvmc-issue184-factorized-20261003-d`, not a fabricated
capture. It records full compiler invocations, every source/hash, project,
Manifest-v1.13 hash, linked libraries, binary and payload hashes. GCC/GFortran
13.3.0 Linux x86_64; native LP64 OpenBLAS0.3.26+ds-1ubuntu0.1 library SHA256
`bfc7492adbf84a8f567720a9e1fae2afc18f3d817da233e7f4d453683485308e`;
Julia1.13.1 ILP64 OpenBLAS0.3.30 Haswell, actual threads1, library SHA256
`4ee5ad9dcc4082b918d3e5bfc434b5a74fb43025f102447eb16a7f04ba21a12b`.
Manifest SHA256 `09ebd06dab244510094b99fe7c6efa2fe7a3d22221d1336b951123a5a6e8befc`.
Final generator SHA256
`0d7eeeaa771bdca5da10a4a6aa8938436a8a0fbcaf9d2d712b0853911fd49828`.

Input is independent literal dyadic upper(i,j)=((3i+2j)/16,(i-j)/32),
one-based i<j, skew lower=-upper, zero diagonal, with upper(1,6)=4+2i.
No RNG, Rust output, factor oracle result or inverse result constructs input.
Native C INFO0 and Julia factor INFO0 are mandatory. Actual generated pivot
is [1,1,1,1,1,6], a genuinely nonidentity sequential permutation.
Julia factor is independently checked against C factors/pivots. The independent
Julia inverse receives the SAME retained C factor/pivots as C inverse.
Rust separately executes its actual factorization then inverse from ITS OWN
factor output, compared to native expected LTL/pivots and all inverse A/M/vT.
Both C-compatible primary arithmetic and default Julia-architecture arithmetic
are separately tested; neither substitutes for a failing other path.

## Format, numerical evidence and regeneration history

Generator directly writes `.factor.txt`: native INFO, all column-major LTL
pairs, actual pivot footer. No undocumented postprocessing is needed. Rust
strictly validates INFO0, input kind/n, exact pivot footer and no extra tokens.
Input, C/J inverse outputs and original skew operator have the same strict
formats as other inverse fixtures. Zero/poison17 and twice-reused workspace
cases all compare A/M/vT and unchanged pivots; original skew residual is
independent of any inverse/factor reconstruction.

Actual final acquisition session61158 terminal0, tool chunk43d4f8 stdout
(these numbers are a tool-output transcription, not a native stdout log):

```text
factor_info_C/J=0 pivots=[1, 1, 1, 1, 1, 6] factor_max_abs=4.441027621704298e-16
inverse_all_A_M_vT_max_abs_C_J=2.220717083251841e-16
residual_inf=1.747722548140113227054635193152637576143854826412908610072001274770675406875797e-15
backward_eta=4.198524388085486667113847487015592478083076326361938575840019344564533233640466e-17
condition_inf_estimate=40.62706671657727107091760791995061951554238962872731316966190863183739658315605
```

BigFloat256 computes C inverse residual against the independently constructed
original skew matrix, not the C factor or oracle inverse-inverted operator.
Condition40.63 is an estimate for THIS matrix, not a certified universal bound.
Computed comparisons keep the existing256epsilon abs+rel bound, unchanged.
Factorization justification is separate: five <=6-wide elimination stages,
complex scalar allowance8*5*6=240 rounded to256, supported by the measured
factor difference4.45e-16 and independently small backward error4.20e-17.
Inverse uses four <=6-term solve/product stages (8*4*6=192 rounded to256),
and a separate scaled residual check prevents forward comparison alone from
hiding a wrong inverse. These budgets are retained-input rounding allowances,
not a claim that operation counts alone certify all ill-conditioned matrices.
Exact INFO/pivot/schema contracts have no numerical tolerance. No bound
increase, algorithm/order change, acceptance/RNG exemption or panel claim.

Historical stage-a failed harness ABI compilation, no numerical artifact.
Stage-b acquisition succeeded with older generator37a237d4...; its `.ltl.txt`
was deterministically prefixed with native INFO0 during initial packaging.
Stage-c added precompile source/runtime metadata, expectations unchanged.
Final stage-d directly emits `.factor.txt` and records its hash. All FIVE
checked-in payloads cmp-identical to final stage-d, combined cmp exit0:

```text
65db4e147e75c4ad53339011cc0d576e6a7f3bbfca925b07612d0bb144a813bc complex6_factorized.c.txt
d999dca36c0f8bc2152f8c90740e03d7ef31a1108ee3ac82e59580fd1287a060 complex6_factorized.factor.txt
9d1d73b6e9d2cd067b7e3ecee3ebf93b8f9b50e0a365157cc8712a480e1db0ea complex6_factorized.input.txt
8a05ff4b20d4c04d2558524ab428ddcd039cdc9a0d4d74de0233bcc7f252aef4 complex6_factorized.j.txt
19f223d72b902b6f54984fd81183aba86300c824cb7928023b6496583c3dac60 complex6_factorized.operator.txt
```

Final-source feature gates and independent parent review must be recorded
separately. Earlier 2a25ca5a four-test pass predates the final explanatory source
comment. No original12/full PfaPack/workspace/Mac/#185 completion promotion.

## Final-source agent gates

Test source SHA256
`a1e54cac03bc48bd7fcbcb27245441978e98625dfdcfca39cb609bf0d196ccfc`.
Exact ordinary crate command, independently executed with each feature flag:

```sh
cargo nextest run -p pfapack --cargo-profile test-fast --locked --no-fail-fast --retries 0
# Separately add --features simd-backend, --features blas-backend,
# or --features 'simd-backend blas-backend'.
cargo clippy -p pfapack --all-targets --features 'simd-backend blas-backend' --locked -- -D warnings
```

| Feature | Run UUID | Terminal result |
| --- | --- | --- |
| default | fabd29b4-45ce-4229-aa34-c12ade8b347d | exit0,26PASS/8skipped,.011s |
| SIMD | 5378d658-e06a-4df2-940e-7a923eb571fb | exit0,26PASS/8skipped,.010s |
| BLAS | 16dccab1-073c-4fd3-800d-04e1d3ab3531 | exit0,29PASS/8skipped,.030s |
| SIMD+BLAS | fb38ab41-621b-4c0d-8e81-10685444cc72 | exit0,29PASS/8skipped,.028s |

Strict all-targets combined clippy exit0,.29s. Separate normal-profile focused
four-test run93c6190f-218d-47a2-b688-d6e78192531c exit0,4PASS/0skip,.008s.
New fixed factorization case has TWO named tests (C-compatible and default),
not four new model scenarios. Existing ignored8 large tests were not rerun in
this final-source batch; historical preceding inverse milestone evidence is
not retroactively relabelled. Parent independent final-source review/gates
remain a separate requirement; no ledger promotion by this document.
