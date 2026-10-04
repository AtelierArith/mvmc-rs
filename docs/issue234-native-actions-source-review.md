# Issue 234 opt-in native Actions candidate — SOURCE only

Base: `0130fdf0496742f8e49675722dea63d9cb3401d8` (PR244).
Historical initial SOURCE base was `e41ffc38265310492546b33c7df008a378a99ee9`
(tree `5e44fb5ea371bbb8c7a9b3b480c8ddac96392ed1`). The official delta modifies
only `ctest_equivalent.rs`, `support/fixture_status.rs`, and adds
`docs/ISSUE183_CTEST_SELECTION_CLASSIFICATION.md`. An actual git diff confirms
core/CLI production, the two native test sources and setup action are unchanged.
All 15 candidate bytes matched source-v3 immediately before and after the
detached-checkout reconciliation; only this provenance paragraph then changes.
No shared dirty files were copied. No dispatch, compilation,
MPI execution, commit or publication accompanies this candidate.

The opt-in workflow reuses `.github/actions/setup-rust-ci`, including
its Linux MPICH/OpenBLAS/LAPACK and nextest setup. It adds only a narrowly scoped
push trigger for `verify/issue234-native-*`, plus `workflow_dispatch` for use
after default-branch registration. It has no pull-request or ordinary main/
feature-branch push trigger. A new feature-branch workflow cannot be assumed
manually dispatchable before registration. Parent's proposed publication/
activation branch is `verify/issue234-native-0130-source`; no push or dispatch
is authorized or performed here. Explicit branch CI must prove these workflow/
script paths before merge. A future activation binds `github.sha`, recursive submodule
commits, source bytes, selected test/CLI binaries, toolchain executables and
resolved dynamic providers. It does not relabel a historical binary current.

Scope: serial healthy/direct-SR error control; ignored public SR and summary
tests on worlds 2/4; SR widths 1/2, healthy/root/last fault and two repeated
failure trials inside the test; summary widths 1/2 healthy/root readback error;
then the unchanged 260-case CLI settings matrix. The CLI launcher validates
native expected statuses, all rank markers, exact argv, output inventories
and immutable staged inputs. Positive and deliberately failing CLI cases
retain distinct expected statuses. First failure stops execution; no retries.
SR protocol validation is supplemented by the committed semantic validator.
Schema 6/48 controls are controls, not native-model evidence.

No C/Julia runtime, oracle acquisition or fixture generation is invoked.
CLI stage inputs are the checked-in `tests/fixtures/physcal_181` models
`heisenberg_chain_real`, `heisenberg_chain_fsz`, `hubbard_chain_dh_opttrans`,
including each `inputs/` tree and `zqp_opt.dat`. Their existing provenance
remains authoritative. Missing inputs fail preflight, never skip or become
Rust-generated replacements. Recursive checkout retains the exact C/Julia
gitlinks; it does not replace the reviewed 144 local hydration aliases with
unverified aliases. The CI checkout and historical local alias receipt are
different source layouts and must not be asserted identical.

External source acquisition, copied without algorithm/expectation edits:

- CLI nine-file harness: host
  `/tmp/mvmc-234-overlay-matrix-source.ZIc79d`; final historical 260-cell
  collection used CLI SHA
  `c435a4e3ea3b5d175623d46e12b06e5055e4987d53647dd974c61b76c3defb58`.
  That executable is NOT used here. Only the launch output parent changes
  from `/tmp` to the workflow artifact `TMPDIR`.
- SR protocol validator: container `happy_jackson`,
  `/tmp/mvmc-178-sr-mpi-literal.xW9KeD/validate-sr.awk`, SHA
  `1c90f4a998b53071f7a912cdc7b410b98d5082d77d1c4c0c1806aec4b109b6b8`.
- Summary validator: host
  `/tmp/mvmc-234-summary-source.9Isi5B/validate-summary.awk`.

Settings: seed 11272 in CLI stager, one optimization step, warm-up/interval
1, samples 8, two trials; all BLAS/OMP/inner worker counts 1, inner threshold
32. Original numerical budgets and public test assertions are unchanged.
Hydra metadata/help must expose the three required capture flags. OpenMPI
is unsupported by this candidate, with a hard preflight failure, not a skip.

Timeouts: job 360 minutes; build/list 1200 seconds; serial 300 seconds;
each native launch 120 seconds plus 10-second kill grace. GNU timeout and
the dedicated Actions runner are the prospective containment mechanism.
This is NOT the locally reviewed owned-group-v2 cleanup proof and must not
be described as such. Job cancellation may leave incomplete post receipts;
artifact upload is `always()`, but an interrupted job is not a PASS.

Parent-review repair: validator/inventory control roots now use the exclusive
artifact `TMPDIR`. Exactly 16 foreground notification controls
are copied from the reviewed historical source and adapted only to source the
actual `wait-cli.sh` handler used by the native wrapper. They exercise separate
cheap shell children (not a live CLI), CLI statuses 0/1/10/138 and notifications
before/during/after/concurrent; the unused asynchronous wait function is not
tested or called. The new adapter is unexecuted. Every control receipt remains
uploaded. Gate association binds current Cargo package ID/manifest, nextest
binary ID/path, exact ignored/nonignored name and source bytes for the three
native/serial gates, rather than matching a binary name alone. MPI header,
compiler/cache/linker tool bytes and resolved tool providers are checked after
execution; compiler/backend metadata is captured before compilation.
The historical foreground-control origin is
`/tmp/mvmc-234-wait-race-source.BBKTxZ/foreground-controls-SOURCE.sh`, SHA
`7fd863f0889bb24b8f26b2238392069a3523707e437ba21c7288393c9feb4014`.
No adapted control run is claimed. Shell syntax checks pass; shellcheck and
actionlint are absent from both the current host and container, so those lint
checks remain unperformed. No Rust files change; Cargo/rustfmt execution is
not part of this SOURCE-only revision.

Remaining review/execution gaps: unexecuted YAML and nextest selection;
actual Ubuntu MPICH/header path and tool availability; actual compiler
cache/backend closure; the adapted notification controls' actual execution;
semantic protocol test coverage does not replace
actual current-checkout captures. Header inventory is a declared provider
snapshot, not a traced compiler include closure. Alongside tracked-source
hashes and dirty-status equality, this candidate additionally computes
full filesystem inventories (excluding Git administrative entries, with
target/evidence outside the checkout) and compares file sets, bytes and
symlink text after execution; this new guard is SOURCE-only, unexecuted.
The future artifact must contain terminal
0, both rank matrices, exactly 260 validated CLI cells and all post checks
before acceptance; historical 260 success is insufficient. #234/#185 stay open.
