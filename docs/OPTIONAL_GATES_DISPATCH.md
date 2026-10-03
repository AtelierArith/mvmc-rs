# Explicit bounded optional-gates CI (#183)

`.github/workflows/optional-gates.yml` runs ONLY via workflow_dispatch; no PR,
push, or schedule trigger. Existing ordinary CI is unchanged. Do not dispatch
before parent review. `family` selects general/lanczos/mpi/thread, or explicitly
all four bounded jobs. A separate planning job creates only the selected matrix
entries; unselected numerical jobs are not instantiated or reported green.

| Job | Actual bounded selection | Not claimed |
| --- | --- | --- |
| general | Two exact ignored tests; General1/2/3/20 and public20 repeat | All13 models or fresh whole-matrix Julia validation |
| lanczos | One ignored test six times, explicit three models x real/cmp | Missing optional DC comparisons, InterAll, full Lanczos coverage |
| mpi | One exact test executable, actual worlds2/4, groups1/2, fixed Heisenberg real PhysCal and grouped Lanczos rejection | Full MPI model/solver matrix or independent fullC sampling |
| thread | Existing wrapper selects ONE primary runner test; workers1/2/4, steps2/samples200 | New long20/45-case outcome matrix or all threaded gates |

Lanczos models are `hubbard_chain_real`, `hubbard_chain_lanczos`,
`spin_chain_lanczos`. Their actual namelist closure is checked before execution;
InterAll is explicitly rejected, even if later introduced into a named model.
Historical numerical budgets are unchanged. Reference versions belong to
fixture provenance, not a newly executed Julia runtime: these jobs invoke no
C, Julia or toolbox programs. Offline Cargo remains independent of this script.

The script checks exact ignored identities/counts, never support-only passes.
Unknown/empty family, missing inputs, source/fixture changes, missing/empty
required artifacts and nonzero gate exits fail. Artifact roots are exclusively
new; output roots are never deleted/reused. MPI additionally receives its
required exclusive `MPI179_PHYSCAL_OUTPUT` path per actual world. Each requested
job captures commands/env, Rust/profile/features, revision/source and fixture
hashes, reference submodule revisions, actual binary hash/linkage and terminal
completion count. Actual linked OpenBLAS is loaded read-only through its
configuration API to record version/core/thread count and library SHA256;
pkg-config or environment alone is not backend-version proof. Ambiguous or
unidentifiable backend fails instead of guessing. No Julia BLAS runtime is
claimed. Artifact upload runs on failure too and missing artifacts fail.
Compilation closure includes `tests/support`, `.cargo` settings when present,
Cargo manifests/lock/toolchain and source; actual compiler/cache flags are
recorded. MPI requires exactly one PASS/zero failures/zero ignored summary per
rank and unique actual world/rank/group-width markers. The thread backend is
queried from the exact executed wrapper archive after exclusive extraction;
its binary hash must match the initial source-associated build, and the nested
wrapper terminal/fixture/source closure must succeed. No current backend is
inferred from an unrelated initial executable.

Local explicit Linux command (requires linked OpenBLAS, nextest; MPI additionally
mpiexec/mpi development libraries):

```sh
CARGO_TARGET_DIR=/absolute/checkout-specific/target \
  uv run --no-project python scripts/run_optional_gates_183.py general /tmp/exclusive-new-evidence
uv run --no-project python scripts/test_optional_gates_183.py
```

Infrastructure tests are not model coverage. Any numerical result must identify
the captured terminal artifact, scope and source snapshot. No external dispatch
or full13/Julia/MPI/thread completion is inferred from adding this workflow.

## Executed bounded validation (2026-10-03)

Final driver SHA256:
`666162c4727444864aee5dc3958cdf0b40f6cdf511df64953fde249520a5a6db`.
Workflow SHA256:
`3b21c01ddf3b4abaf2dadf8584b328832f9d26dc7e341920edee7b80195cf427`.
Infrastructure test SHA256:
`01c7e77ac756ffc92fc621d1cf4bc59361111c2130d0aeb2aaf0c243d36dda95`.
MPI diagnostic-only test SHA256:
`930e54ff1d4194af2a0b41610426bc7da3df3c2c663efc596e9042adbcc8f43e`.
No production kernel, algorithm, tolerance or reference fixture was modified.

General/thread/Lanczos executed in dedicated detached worktree
`/tmp/mvmc-optional183-frozen.e8eQuG/workspace`, base
`f3716ac004b7d82123a810334556c40c51bd69fc`, overlaid with the existing dirty
compile/test sources and these owned implementation files. This is NOT a
committed-main or full-workspace validation. It used its own target
`/tmp/mvmc-optional183-frozen.e8eQuG/target`. Frozen source manifest before/after
SHA256 is `6a06466d8a53388ed8aa81674d119d785b54537c6388896eac4062cfb6281e1b`.
Each family's fixture closure before/after matched. Linux x86_64, Rust1.99.0,
nextest0.9.146, locked/test-fast/default features, actual linked OpenBLAS0.3.26,
Haswell, one backend thread. The CLI form was:

```sh
CARGO_TARGET_DIR=/tmp/mvmc-optional183-frozen.e8eQuG/target \
  uv run --no-project python scripts/run_optional_gates_183.py FAMILY \
  /tmp/mvmc-optional183-frozen.e8eQuG/FAMILY-evidence
```

| Final handle | Actual completed scope | Retained terminal artifact |
| --- | --- | --- |
| 58512, exit0 | General two gates PASS, 3.203s; four independent prefixes plus public20 repeat; run `59db7b9a-80a2-4d7a-8104-1af311d7dde9` | `general-evidence/terminal.json`, SHA `ccb85285b58332b574ba491ec0b3c2c6f7db85bde5b85d045cf5fc363c4a7f5e` |
| 23549, exit0 | Six explicit Lanczos model/mode calls, each one ignored gate PASS | `lanczos-evidence/terminal.json`, SHA `acfc0c3bb72ea452e7c879313eac7da8eeac080855f417b57d2c8a5a08c7f2aa` |
| 36608, exit0 | ONE primary threaded gate PASS, 154.289s; workers1/2/4 actual activation, 1859 compared records; run `00564dd0-7d81-4b6f-9ed9-b5b047b009c9` | `thread-evidence/terminal.json`, SHA `7744e8c09f80de5f7d332e6b35dfaf5ad38010433b9080a8ab791c11c716415f` |

The thread wrapper executed archive SHA256
`e877ec3f06b9bf87aab1e6b5f35b71443fe2c9492a56a1ed6d1a799f2d294270`;
the exclusively extracted actual executable SHA256 is
`7215d2682cda0d2eb34f0e109b7c004695841671683a564741c4ed482b8b7ab7`.
`thread-evidence/thread-executed-archive.json` records the exact archive,
executed binary identity matching the initial build, and nested wrapper path.
Its backend API query was made against this extracted executable's linked
library; no unrelated initial-binary backend is substituted.

Final MPI handle8688 exited0 in container `73c57e563c61`, Rust1.99.0,
MPICH4.2.0, same actual OpenBLAS0.3.26/single thread, features mpi/test-fast.
Command used `CARGO_TARGET_DIR=/home/vscode/.cache/mvmc/target/issue183-owner-marker`
and `MPICC=/usr/bin/mpicc`, then `uv run --no-project python
scripts/run_optional_gates_183.py mpi /tmp/mvmc-optional183-mpi-owner-v2-20261003`.
Exact host-persisted evidence root:
`/tmp/mvmc-optional183-mpi-owner-v2-persisted-20261003`.

* `terminal.json`: `476992bcfa51c1f2f66bdf0d49391bcbbdaaaf523061fa3febb33e15b577d144`.
* `mpi-2.stdout`: `ab0ed7a3d09de411c0206acd61006c1769ad22eca0a1fb27a4add82c5791896b`.
* `mpi-4.stdout`: `298a6b199b5c60ec35ece845f90a36ad1d7f51b73042caa297ea4fad1651c42c`.
* Source before/after both `de3fa2f89a2696a58f6e90ddfda6b55cb362e30d1f104722e58d10c505874257`;
  fixture before/after both `c7a3da61d3492636edb2e7155c290121823122066a15bffc19374e8e4e1b9d29`.

Both actual worlds2/4 supplied all zero-based ranks and width1/2 markers, with
actual group size equal to world for width1 and two for width2. Exactly two/four
per-rank libtest summaries reported one PASS, zero failures, zero ignored.
The new eight-line marker observes communicator sizes only after existing
repeat/output assertions; old MPI proofs are not reassigned to this new SHA.

Infrastructure validation: `uv run --no-project python
scripts/test_optional_gates_183.py` passed9 tests (not model execution); parent
independent9 also passed. PyYAML parse/dispatch-only dynamic-matrix schema
checks passed; SHA-verified actionlint1.7.7 passed this workflow (shellcheck
integration disabled, not claimed). Focused MPI Clippy handle67679 exited0;
targeted rustfmt/diff checks passed. No external GitHub dispatch, native macOS,
full13/long45 or fresh C/Julia runtime validation was performed here.

Failures preserved separately: handle93329 failed backend preflight before
numerical work because the driver's observer originally inherited a different
thread environment; fixed by matching its library initialization environment.
The first thread attempt failed preflight on a valid commented namelist; comment
handling and its regression were added. Historical thread handle10682's numeric
gate passed166.010s but its wrapper correctly exited1 with SourceChanged:
an unrelated `constructor_contracts.rs` changed during the shared run.
All evidence remains under `/tmp/mvmc-optional183-thread-owner-v2-20261003`;
it is an infrastructure failure, NOT the final immutable-worktree PASS.
Historical Lanczos59780 and General77178 precede final driver666...;
the final table above supersedes them for this implementation only.
