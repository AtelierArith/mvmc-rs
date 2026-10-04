# M0587 manual RBM diagnostic — scoped ordinary evidence

Related to #184 and #185. Executed base:
14a30849ae4fa9dd78900f113257389d906008ee.
Publication base: 9b6ed40754e88eb2dd8c3c3978309540df19b32d (PR294 retained).
Publication API/test bytes are identical to the successful executed sibling.
The provenance update below is later doc-only content, not executed doc bytes.
Exact publication-head workspace/CI validation remains pending.

## Actual scoped execution and retained failure

Initial receipt: /tmp/mvmc-184-manual-rbm-proof.20261004.
Initial source: /tmp/mvmc-184-manual-rbm-source, base14a308 above.
Initial test SHA256:
cc57579412d7e33a6648fbd0b679b6bb650dffc504edcee4a4be65776f4fcd58.
13 diagnostic PASS (0.007s),9 public parameter PASS (0.014s),0 skipped.
Aggregate101: two test-only field_reassign_with_default Clippy errors.
fmt/postLIST were NotRun. Source/tools/compiler-provider posts and every
reached owner cleanup0. Original frozen source/receipt are unchanged.
No retry or source fix occurred within that initial execution. The separately
approved successor below is a new source/receipt, not a relabelled initial run.

Successor receipt: /tmp/mvmc-184-manual-rbm-proof.successor20261004.
Successor source: /tmp/mvmc-184-manual-rbm-successor, same base14a308.
Both sources were uncommitted four-path overlays, not separate executed Git
commits. Successor owner handle87097 completed0.
The exclusive target /tmp/mvmc-184-manual-rbm-target.20261004 was reused only
after the initial owner groups were absent; source association was rebound to
the successor root and verified by its new LISTs. No other owner's target used.
Only two default-then-field assignments became equivalent struct initializers;
assertions, independent expected values and production bytes were unchanged.
The exact source delta is limited to ExpertModeData struct initializers in
equal_shadows_do_not_assert_dense_consistency and
signed_zero_and_nan_follow_numeric_duplicate_equality; no lint allow added.
13/13 diagnostic PASS (0.007s),9/9 public parameter PASS (0.008s),0 skipped.
Both exact LISTs, Clippy, fmt, both postLISTs, binary/runtime hash comparisons,
source/input membership+hash, tools/compiler-provider posts and aggregate0.
Each phase's owner cleanup0. Transient /proc/stat ENOENT messages from the
unchanged owner helper remain visible; no failure was hidden or retried.

Diagnostic UUID: 8eb7cc35-cff4-41ae-a626-31caa9469b12.
Public parameters UUID: bb5da64f-9453-4cc7-8c91-404b1e1b3925.
Raw full logs under the successor receipt:
diagnostic/command.stderr SHA256
a92a50d6d9b64f15f7771eb5ab54c6ec97d5141cff4dc614ebf7b39f2eeb03a1;
public/command.stderr SHA256
64396c70c2d4d3097ef9bdb884a9de197fa6128d106ae3f027e93ef4ba13e559.
API source SHA256
803a1232b323776f31c65f3985082aa54b3e4fcdf21b177fdb2288dd9fa89762;
test source SHA256
1693104a33fa6c8f738af25419286a481d55b6d7618c909ac7bac201e345fc82.

Commands: cargo nextest list/run --locked --cargo-profile test-fast
-p mvmc-core --test issue184_manual_rbm_diagnostic, then the same commands
with --test issue290_public_parameters. LIST uses --message-format json;
RUN uses --no-tests fail --no-fail-fast --retries 0.
Clippy: cargo clippy --locked --profile test-fast -p mvmc-core
--test issue184_manual_rbm_diagnostic --test issue290_public_parameters
-- -D warnings. Format: cargo fmt --all --check.
Jobs2, BLAS/OMP/MKL1, default features. Static input gitlink c0788c34a6a5753c611633a97cd1ea233203320c
was hydrated before complete source/input freeze, separately from original8bb
source authority below. No C/Julia/toolbox runtime, model/MPI/fullworkspace run.
These22 tests do not establish complete #184/#185 acceptance.

## Contract and original source

API: parameter_diagnostics::check_duplicate_consistency(&ExpertModeData)
returns Result<(), DuplicateParameterError>. The error identifies section,
raw index, zero-based first/conflicting rows and both independent values.
The first mismatch follows fixed nine-section and original row order.

Original Julia revision 8bb1b9e8ae47b1512c00b321be05664ddcac0fd1:
MVMCOptimizers.jl/src/parameter_sync.jl:202–226 and
MVMCOptimizers.jl/test_unit/test_unit_parallel.jl:295 (M0587).
Original test file SHA256:
a2b582f61dc981c318f817fc0575cfadfdb8c93e34c5679aeaf7fdae48be5050.
The helper is module-qualified, not exported. It throws on RBM values 0.1/0.2
sharing one index; it does not repair them.
parameter_sync.jl SHA256:
93886834c217f0054b956837a25ee533eb227ad27b8dbd98eb28ec0628ebe7d1.

This manual API intentionally does not check mapping ranges, declared widths,
dense/shadow agreement, finiteness, or orbital storage. Equal wrong shadows
relative to dense storage pass this diagnostic: a stronger dense invariant
would require a separately reviewed API and acceptance contract.
Rust orbital mappings have no independently mutable term value.

C uses canonical dense parameters; no C storage or numerical algorithm changes.
The diagnostic is not called by pack/unpack, setters, initialization or runners;
it accepts only an immutable borrow and has no RNG argument or RNG calls.
Complex numeric equality preserves signed-zero equality and duplicate-NaN
inequality. Singleton NaNs are outside duplicate detection, not validated input.

Proposed 13 ordinary controls cover the original Heisenberg parse context with
0.1/0.2 shadow injection, no repair, every nine-section mismatch, first row
retention through an equal intermediate duplicate, equal shadows with missing
or contradictory dense storage, signed zero/NaN, and section/index separation.
All13 controls ran successfully as bounded ordinary diagnostic evidence,
including original Heisenberg parsing and0.1/0.2 injection; not a full model run.

M0585 shared-coefficient observability is a separate public dense-storage
contract. M0586 independent orbital-value repair has no literal counterpart
in Rust orbital mappings, which do not carry independent values. Neither is
claimed covered by this M0587 RBM diagnostic. No Julia duplicate-guard proposal
has been executed or adopted; no canonical ledger status is changed here.
