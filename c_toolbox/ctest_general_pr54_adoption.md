# Corrected GeneralRBM reference: scoped adoption, not completion (#180/#185)

Prepared 2026-10-03; parent approved steps1–3 and isolated validation.
General-only routing is now implemented; no historical fixture overwritten
and no numerical budget changed. Ordinary Cargo tests remain independent
of Julia and `c_toolbox/`.

## Proposed namespace and input contract

Imported **only four GeneralRBM cases** to a new
`tests/fixtures/ctest_general_pr54_3d0fd263/` namespace, with prefixes1/2/3/20.
The remaining model references stay under `ctest_reviewed20_62b/` with their
historical62b provenance. Never claim52 captures were regenerated at3d0f.

Published OPEN Julia PR54 head, independently verified through GitHub API:
`3d0fd2638fd34de2a8f9609fcfaac2504caf02d2`. Actual vmc_sampling.jl SHA256:
`e8274c5d7bce99003ed205d9ea7a854832e2e498a9b85a3d0172759e484155a1`.
The C-faithful grouped hidden-counter correction removes the prior General
OO/HO mismatch; it is an algorithm correction, not a tolerance exception.

Linux x86_64, Julia1.13.1 executable
`/home/vscode/.cache/mvmc/tools/julia-1.13.1/bin/julia`; pinned Manifest SHA256
`09ebd06dab244510094b99fe7c6efa2fe7a3d22221d1336b951123a5a6e8befc`.
Actual LBT OpenBLAS64 ILP64, one BLAS thread; Julia default1/interactive0,
serial worker1. Complete63-file manifest SHA256
`33730954e5ed3b831369723fbff577d0571fbad5067c4697c665c420a4b69605`.
The acquisition verified those original on-disk sources before and after.
Runtime read-only injected observer methods have separate hashes; do not
claim the runtime has no instrumentation merely because disk hashes match.

Canonical model `general_rbm_cmp`: seed12395, NSRCG0, NStore1, NPara102;
original steps1500/window100, explicitly overridden **both** to1/2/3/20.
Preserve the complete canonical input directory and per-case inputs.sha256,
including every namelist reference and implicitly loaded initial.def.
Initial overlay SHA256
`a71c011d75380176881e48312d4235222f5b7d29ad7df85f1c979eec447eb79e`.
No retained-coefficient, initial-value or solver kernel replacement.

## Completed captures and preserved failures

External frozen acquisition root in container73c57e563c61:
`/tmp/mvmc-general-pr54-3d0fd263.aEqff0`.
Actual100-local prefix1 capture `20509` exited0. Comparison against retained
Rust100 walkers found exact IP/Pfaffians/inverse/O/E and counter values
(Rust counter is explicitly recomputed from actual saved walkers; Julia
counter is the actual borrowed production input).

`94143` exited1 during obsolete observer-boundary preflight: **zero solves**.
Keep its actual failure output and labelled reconstructed preflight wrapper.
V2 inserted observation before original `_solve_direct_sr!` and after the
original catch. Original solve/helper arithmetic and failure policy unchanged.

`78040` exited1 **after all four original194-dimensional solves and captures
completed**, at a provenance-tail read of missing snapshot stcopt.c. Preserve
this failure and partial provenance; do not relabel the producer green.
Completed captures are at the root's sibling `-direct-v2/` directory.
Separate post-run audit succeeds against actual authoritative C root
`/home/terasaki/work/atelierarith/mvmc-rs`, stcopt.c SHA256
`43ed8790cff2715284849f0f8906e4645179dcbb100b51f819918b279d6a36f2`,
stcopt_dposv.c SHA256
`2bd48d880dcbd95ea1b1b92931178c07fd08f981b8ebf04e7db1709e909e57ca`.
These sources are metadata/numerical authority, not an executed full C solver
or sampler for these four runs.

Post-run audit and comparisons are retained separately at
`/tmp/mvmc-pr54-general-posthoc-audit.kBZ8du/`:

| Artifact | SHA256 | Actual result |
| --- | --- | --- |
| audit.cjs | ad84a8d0b5ebb9d84bf14a2c453361d8c00e6cf5e3177f2a4615213eb549616b | exit0 |
| result.json | 120e91dbe3cef6be9a552c7d59b659c8f20f2d9ad01233e2d603bb44cc407e6d | four-case input/source/settings/schema audit; producer remains exit1 |
| compare.cjs | fce46a686bc4b681135f2932187374e49de42e44350a4d96fee8c5613cdb3cf5 | exit0 |
| comparison.json | be8a8dd39c25d1c398ed1bcbdae75a46dc53c69fe94bf6f26b662f872ee8c7bd | actual Gen2 OO/HO/solve1/2 exact; historical62b discrete comparisons separately labelled |
| compare-current.cjs | 7c7dfa373787e253db79508f11f6f8506bad27c10a7ee74d1477cb26eba69c16 | exit0 |
| comparison-current.json | f13b692729c240d82bcc04516e284be41a3d0a98e8d5da3d5dcb16e8b1a2326f | fresh Rust3/20 exact controls, quantities and output rows |

All comparison results carry actual input-file hashes, shape checks and
nonzero exit on discrepancy. The external README supplies reproduction
commands. Import those code/result/provenance artifacts byte-identically
alongside fixtures; do not replace independent expectations with Rust output.

Fresh Rust3 run7d8a34f4-7c13-40be-872b-a8988fd8ab42 and Rust20
e0697ffb-f80b-4ebf-8092-c99e3f22ce4c each exited100 against the **old62b** OO
expectation. Full actual RAW624/cursor/drawcount/next624/configurations were
saved before that assertion. Separate corrected comparisons are exact for
those discrete quantities, parameters204/E2/OO85696/HO412, original final
solve A37636/b194/x194, zvo_out lengths18/120 and declared-slot var936/6240.
These are not newly passing historical gates. Test source SHA256
`580c607510ca339b0fc820b7b494ea212ebeade270814f20573346469d4dcfa6`;
focused clippy passed. No macOS execution is claimed.

## Output/reference distinctions

Keep original Julia zvo_var.dat **separate** from zvo_c_slots_var.dat, which
observes pre-SR contiguous declared C Para slots (vmcmain.c653–657), not
duplicated/reordered mapped Julia terms. The latter is the Rust full-var
expectation. Preserve both records and their explicit scope.

C windows were independently generated from corrected Julia Etot/Etot2 and
post-SR declared parameter history by unchanged extracted avevar.c bodies.
All four native probe runs exited0; main/family outputs compare exactly
against retained Rust. This is **mixed Julia history/C aggregation**, not
full native-C sampling validation. Keep compiler/source/extraction/history/
executable hashes and mean/sample-deviation semantics with every case.
See [ctest_opt_window.md](ctest_opt_window.md) for source/license boundaries.
Successful compiler: GCC Ubuntu13.3.0-6ubuntu2~24.04.1,
`-std=c11 -O0 -ffp-contract=off -lm`. An initial extra-Werror build failed
on upstream sprintf warnings; no extracted numerical source was edited.
Extraction identity independently rechecked with `diff -u` (exit0): original
avevar.c lines1–27 (copyright/license) and34–260 (complete CalcAveVar,
WriteHeader, Child_OutputOptData, StoreOptData and OutputOptData). No body
patches; only include directives and the outer guard omitted. Preserve the
upstream license text with the extraction. Native executable links system
libm/libc only: **no BLAS/LAPACK/MPI backend** in this aggregation probe.

| C-window identity | SHA256 |
| --- | --- |
| Original avevar.c | 509a573944a5eabde93864346902674faaff6c23d0c88f89867263f8372dd51a |
| ctest_opt_window_upstream.inc | 47c801ea68d9bf40af967a3a0db6189125fd54ad146820b6470d63f9317a69da |
| ctest_opt_window.c adapter | 4069961fb468957de87760db9e1a25a1587351f4c92b17c3d3e925698904e652 |
| Actual compiled executable | 7a0434c1c7a377e8a84d31fa2a02499cd5cbb1007a83ff388b7e8ef932678d75 |
| /lib/x86_64-linux-gnu/libm.so.6 | fce00b6f25f459cf4ae0b7fae4257909a1a8a86f9149e7c029a5d617baf1ccd0 |
| /lib/x86_64-linux-gnu/libc.so.6 | 3a15d66867d83762c7f2f1e37359cb8f6c5743edb369c65285cb0b1c4f7498bf |
| compare-windows.cjs | 9397f81722066c2697a72329fd2b574fdf33f7b017787685a995745a547ad12e |
| window-comparison-v2.json | 2dec0aa0b8452f5633e1463c3bcd9ef0e4c1a9e1b19f360e94fad6f7871b651c |

The comparison was executed with Node24.16.0 on native Linux x86_64. Its
report records every independent history and compared file hash. Paths in
the external script are frozen diagnostic paths; its exclusive output must
not be overwritten. For portable reproduction use a NEW exclusive output
directory and the same pinned source/history identities. Compiler command:

```sh
cc -std=c11 -O0 -ffp-contract=off c_toolbox/ctest_opt_window.c -lm \
  -o /tmp/NEW_EXCLUSIVE_DIRECTORY/ctest_opt_window
/tmp/NEW_EXCLUSIVE_DIRECTORY/ctest_opt_window \
  /path/to/corrected-general/step-N/c-window-input.txt \
  /tmp/NEW_EXCLUSIVE_DIRECTORY/zqp_c_window
```

Correct external report is window-comparison-v2.json; its predecessor
misclassified textual header tokens as nonfinite and is superseded, not
acceptance evidence. V2 checks headers literally; all actual numeric
components in these four windows are finite.

## Proposed edits and remaining acceptance gates

1. Import the four-case/input closure into the NEW namespace with a pinned
   full archive manifest, standalone C-window expectations and per-case
   provenance. Preserve the post-run-audit/failed-producer distinction.
2. Add separately pinned corrected-General verification in
   `crates/mvmc-core/tests/support/ctest_provenance.rs`: exact published head,
   Manifest/source identities, seed/raw settings/effective counts/thread
   configuration, every input reference and initial overlay, strict schemas.
   Add negative closure/settings/draw-count/archive-path mutation tests.
3. Route General only in owned `ctest_model_prefixes.rs` and
   `ctest_equivalent.rs`; all other models retain62b root/provenance. Preserve
   ignored opt-in gates and absent-env fail-closed behavior. No normal-test
   C/Julia invocation or toolbox reads. No new numerical budget.
4. Execute all four corrected public Rust gates through every numerical,
   output-family/window and primitive/discrete assertion, plus20-step public
   runner repeatability. Until executed, do not promote offline comparisons
   to full gatePASS or claim the13-model milestone complete.
5. Run focused clippy/fmt/provenance negatives and then frozen workspace
   verification. Preserve parent801/801 historical frozen-run label rather
   than treating it as validation of future fixture adoption. Native macOS
   and broader185/threaded/MPI scenario gaps remain separate acceptance work.

## Implementation and isolated validation checkpoint

At the initial implementation checkpoint, steps1–3 were implemented in the
new namespace plus owned `ctest_model_prefixes.rs`, `ctest_equivalent.rs` and
`support/ctest_provenance.rs`. Those historical13-model draft edits are
excluded from the disjoint package described below. Full archive
closure229 files, archive.sha256 identity
`5d221186641197737b69840db0876462052109fa8890342ac5963d51ff9fc411`.
Four c-window-declared-input aliases are byte-identical copies of actual
independent c-window-input histories: for this General-only model the raw
header already has declared15-family widths summing to102. No numerical
history was reconstructed or changed.

Actual corrected gate79449 terminal0, nextest
`f1c20757-d50d-4e4a-ac05-f80ef71cc6f2`: one selected test executed **all four
General prefixes1/2/3/20**, passed2.768s;13 other tests excluded. Includes
actual RAW624/cursor/drawcount/next624/configuration assertions before
parameters/energy/SR/output checks and independent C-window output checks.
Earlier33043 failed pre-numerical on absent declared-history alias; preserved
as import preflight failure, not a model numerical failure or passing gate.

Actual public20-step summary/output-repeat gate12269 terminal0, nextest
`19eace8d-2221-4565-a72c-e44dbb519973`: one selected test passed3.424s;
14 other tests excluded. This checks same-implementation/configuration
repeatability, not independently claiming its own raw-state inspection.
Raw-state evidence belongs to the corrected prefix20 gate.

Initial provenance negatives22070 terminal0, nextest
`b44b9ef7-c90f-410a-9fb1-9ace17d12f93`: three helper invocations across
three consumer binaries passed3.071s,46 excluded; these are support tests,
not model execution. Focused clippy56991 passed before final alias pin.
Finalized archive-pin negatives3695 terminal0, nextest
`d3a5a85a-dbf0-40cc-8d5c-2cb1a9315507`: three helper invocations passed3.378s,
46 excluded. Finalized three-consumer focused clippy98370 terminal0.

Remaining: parent review and frozen workspace validation after this adoption;
native macOS and broader185
scenario/threaded/MPI acceptance remain separate. Do not relabel the other
historical62b cases as3d0f or claim13-model/#185 completion from these results.

## Disjoint corrected-General packaging

For a clean four-case milestone **exclude** the existing13-model draft
ctest_equivalent.rs/ctest_model_prefixes.rs/support/ctest_provenance.rs changes
and the uncommitted historical1462-file namespace; preserve them in the shared
tree for later full-model work. The independent corrected-General closure is:

- `crates/mvmc-core/tests/ctest_general_reference.rs`;
- `crates/mvmc-core/tests/support/ctest_general_provenance.rs`;
- `crates/mvmc-core/tests/support/fixture_sha256.rs`;
- this document and the complete NEW229-file fixture manifest closure.

This target has no references to the historical52 archive. Ordinary tests
validate corrected-only provenance/schema and mutation guards. The two
numerical gates are explicitly ignored and require
`MVMC_RS_CTEST_GENERAL=1`; explicit absent/empty/skip invocation fails closed.
No C/Julia oracle or toolbox reads in Cargo. Fixture producer-source identity
reads are confined to the standalone checked-in reference bundle.

The declared-var width312 is **6 leading fields** (Etot and Etot2, each a
real/imaginary/reserved triple) plus102 declared parameter triples, i.e.
`6 + 102*3`. It is104 complex records, **not**104 parameters or six extra
parameter slots. Every reserved third field is asserted exactly zero.
C one-snapshot main output is104 pairs=208 fields, second fields exactly0;
multi-snapshot main output is104 mean/sample-deviation triples=312 fields.
Auxiliary file headers are compared as literal strings, not converted to NaN.

Bounds are unchanged copies of the existing historical consumer scope:
parameters/energy/output absolute+relative1e-11; OO/HO absolute+relative1e-12.
They are not new allowances justified by solver conditioning or downstream
Monte Carlo noise. Current corrected fixed-input values are independently
observed exact, as documented above. RNG/configuration checks stay exact.
Public repeat bytes concern same implementation/environment/configuration,
not bitwise cross-language computed-float acceptance.

Disjoint numerical gates78822 terminal0, nextest
`4a1cd5b7-070d-4d60-9a94-ffdec03136a4`: corrected allfour-prefix test passed
2.330s; public20 repeat passed3.281s; nine ordinary helper tests excluded.
An earlier ordinary run failed a mistyped known SHA input vector, not the
digest or model. That input was corrected to the standard vector; expected
digest unchanged. Independent sha256sum and OpenSSL both report
`248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1`
for the literal `abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq`.
No commit or push by the implementation owner; clean packaging awaits parent
review.

Final disjoint ordinary run87938 terminal0, nextest
`3c2bb6d9-f00f-41e3-8fd7-c9afabdebfad`: nine tests passed3.810s, two ignored
numerical gates excluded. These are three provenance/digest/mutation tests
and six shared gate-policy tests, **not nine model executions**. Final
disjoint focused clippy32548 terminal0. Actual source identities:

| Source | SHA256 |
| --- | --- |
| ctest_general_reference.rs | 182fec8fd578e40e58549b88ab3223f7d71de43648ec034b5334debaa886c97f |
| support/ctest_general_provenance.rs | bfafce041597d14ac0d368ab1b30d6cc78e3c96553c5a3caed9617cdba70bc5d |
| support/fixture_sha256.rs | 13dfd0ff698da36fc4e7da94d3c38b998e609d50b4907a0d217d3fd1a3032d21 |

Parent's801/801 frozen workspace run predates these new disjoint tests and
must retain its original snapshot label. At that owner checkpoint, clean-only-
four snapshot validation and parent review were pending; the subsequent
parent verification is recorded below without relabelling the earlier runs.

## Parent final disjoint verification

Parent current-source run `a673be61` completed **11/11 PASS** in2.875s with
the opt-in gate enabled; parent focused clippy completed exit0. This is
separate from owner87938 (ordinary9/9 PASS, two numerical gates not selected)
and owner32548 (focused clippy exit0), recorded above. Parent's first run
`23a340ec` had nine PASS and two NotRun because the parent environment lacked
the opt-in variable; preserve that result, not numerical execution evidence.
No expected digest or numerical budget changed.

The final focused package consists only of the three new General source
files, the pinned229-file archive closure and this adoption document. The
shared historical52-case draft remains excluded and preserved. Parent owns
the focused commit and subsequent authorized integration; this checkpoint
does not claim full13-model/#185 or native macOS completion.

The archived `evidence/README.md` statement “No checked-in fixture adoption”
is an **original post-run audit note retained at capture time**, before this
approved import and disjoint packaging. It is historical provenance, not a
statement about the current namespace. Preserve its bytes and archive pin;
the later adoption and executed gates are documented here separately.
