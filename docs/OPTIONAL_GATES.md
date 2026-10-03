# Optional parity gates (#183)

Phase 4/5, Lanczos, issue #181 PhysCal and MPI gates are `#[ignore]` tests. A normal nextest run
reports them as skipped/ignored, never as passed. Setting an environment variable
alone does not enable an ignored test: select it with `--run-ignored only` too.
Normal tests do not launch C, Julia or toolbox programs. These optional tests
read existing reference files and run Rust; they do not generate references.

From the repository root, list the declared gates (including ignored tests):

```sh
cargo nextest list -p mvmc-core --cargo-profile test-fast \
  -E 'binary(/phase4_zvo_gate/) | binary(phase5_zvo_gate_hubbard) | binary(lanczos_transfer_physcal) | binary(physcal_issue181) | binary(mpi_physcal)'
```

Run an individual historical 10-step fixture comparison (seed 1):

```sh
MVMC_RS_PHASE4_REAL_ZVO=1 cargo nextest run -p mvmc-core --cargo-profile test-fast --test phase4_zvo_gate --run-ignored only --no-fail-fast --retries 0 --success-output immediate --failure-output immediate
MVMC_RS_PHASE4_CMP_ZVO=1 cargo nextest run -p mvmc-core --cargo-profile test-fast --test phase4_zvo_gate_cmp --run-ignored only --no-fail-fast --retries 0 --success-output immediate --failure-output immediate
MVMC_RS_PHASE4_FSZ_ZVO=1 cargo nextest run -p mvmc-core --cargo-profile test-fast --test phase4_zvo_gate_fsz --run-ignored only --no-fail-fast --retries 0 --success-output immediate --failure-output immediate
MVMC_RS_PHASE5_HUBBARD_ZVO=1 cargo nextest run -p mvmc-core --cargo-profile test-fast --test phase5_zvo_gate_hubbard --run-ignored only --no-fail-fast --retries 0 --success-output immediate --failure-output immediate
```

Set `JULIA_MVMC_ROOT` to a reference checkout when needed. An explicitly supplied
invalid root fails rather than falling back to another checkout.

Run all three issue #181 PhysCal scenarios (six-model fixed-parameter/output
matrix, deterministic multiple samples, and non-InterAll Lanczos file contracts):

```sh
MVMC_RS_PHYSCAL_181=1 cargo nextest run -p mvmc-core --locked --cargo-profile test-fast --test physcal_issue181 --run-ignored only --no-fail-fast --retries 0 --success-output immediate --failure-output immediate
```

Run the Lanczos gate, optionally selecting one model:

```sh
MVMC_RS_LANCZOS_PHYSICAL=1 MVMC_RS_LANCZOS_MODEL=hubbard_chain_real MVMC_RS_LANCZOS_MODE=real \
  cargo nextest run -p mvmc-core --cargo-profile test-fast --test lanczos_transfer_physcal \
  --run-ignored only --no-fail-fast --retries 0 --success-output immediate --failure-output immediate
```

Models are `hubbard_chain_real`, `hubbard_chain_lanczos`, `spin_chain_lanczos`;
omitting the model requests all three. Modes are `real` (default) and `cmp`.
Unknown model/mode selections fail. Inputs and the QQQQ/energy references for
the selected model must exist. The existing comparison checks observable
references that are present; historical fixtures omit the DC reference when
there are no DC terms. A pass does not establish DC observable comparison
coverage when that reference is absent. The issue #181 scenario separately
checks the empty-output contract.

MPI needs the `mpi` feature and at least two ranks. Build and resolve exactly
one executable using Cargo's JSON output; do not launch a wildcard that could
include stale binaries or `.d` files. This example requires `jq`:

```sh
mpi_gate_binary=$(cargo test -p mvmc-core --profile test-fast --features mpi --test mpi_physcal --no-run --message-format=json | jq -r 'select(.reason == "compiler-artifact" and .target.name == "mpi_physcal" and .executable != null) | .executable')
test -n "$mpi_gate_binary" && test -x "$mpi_gate_binary" && \
  MVMC_RS_MPI_PHYSICAL=1 mpiexec -n 2 "$mpi_gate_binary" \
  --ignored --exact mpi_physcal_reduces_fixed_parameter_samples --nocapture
```

The MPI gate checks the serial PhysCal reduction and rejection of unsupported
grouped PhysCal. It does not establish numerical parity for grouped PhysCal.
Explicitly selecting the MPI gate without the feature fails as `Unsupported`:

```sh
MVMC_RS_MPI_PHYSICAL=1 cargo nextest run -p mvmc-core --cargo-profile test-fast --test mpi_physcal --run-ignored only --no-fail-fast --retries 0
```

## Interpreting reports

The harness is authoritative: ignored/skipped means coverage was not run;
`PASS` requires the comparison to finish; assertion/runner errors are failures.
Selecting an ignored gate with its selector absent or empty emits `NotRun` and
fails. A selector of `skip` emits `ExplicitSkip` and fails if executed: to skip,
omit that ignored test from selection. Missing required inputs/expected outputs
fail (fixture preflight emits `MissingFixture`; file reads/runner errors may
report a normal harness failure); unsupported selections emit `Unsupported` and fail.
Thus an explicit request cannot succeed by returning without doing the work.
Keep nextest summaries, exit codes and captured output together. Do not count
support-helper unit tests as parity coverage.

`ctest_equivalent` is owned by #180 and its optional declarations use the same
`#[ignore]`/`support::require_gate` contract through that coordinated change.
This #183 change leaves that file untouched. The legacy `select_gate` helper
was removed after all callers migrated, so new gates must use `require_gate`.

Historical fixture passes cover only the named model, mode and outputs. They
are not newly verified Julia 1.13 results or complete Julia feature coverage.
For reference-generation evidence record OS/architecture, compiler and Rust
versions, Rust features/profile, Julia version and Manifest hash, actual BLAS
provider/version, seed, model/steps, ranks/groups/threads, reference revision and
file hashes, command, exit status and artifact paths. New Julia comparisons use
`julia +1.13.1 --project=extern/Julia-mVMC` and `Manifest-v1.13.toml`.
Reference generation is a separate optional developer action. Thread matrices
must be requested and reported explicitly; these gates do not imply they ran.
