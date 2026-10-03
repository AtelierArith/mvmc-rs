# #179 bounded direct/store1 evidence on main1cea33dd

Related to #179/#184/#185; #179 remains open. This stage proves bounded
same-implementation repeatability and prefix/group RNG/configuration contracts,
**not independent numerical accuracy or Julia MPI/full #179 parity**.
CG, width3, Julia capture, production changes and tolerance changes are excluded.
Historical store0/495 evidence is preserved, not relabelled as this run.

## Frozen source and build association

Container `73c57e563c61`; all artifact paths below are relative to:

```text
/home/vscode/.cache/mvmc/issue179-store1-main1cea.KQhRth
```

`snapshot/` is an independent detached clone of authoritative main
`1cea33dde10653bb946d24c52cf78f8c035e891f`, not the shared dirty checkout or an
old69 snapshot. `snapshot/git-rev-parse.txt` records the actual HEAD.
`owner-overlay-paths.txt` contains exactly these three approved script overlays:

| Script | SHA-256 |
| --- | --- |
| `mpi_issue179_prefix_evidence.py` | `02241307b41d9ffd8323cf4a088bc715de7d185d26d9f70a72ce8dd94621e91b` |
| `test_mpi_issue179_prefix_evidence.py` | `18f813929eaed0b2428223600ab7fabd9aa4b19741d5b0433c4f26cb253b9a6f` |
| `verify_mpi_issue179_prefixes.sh` | `a68c8a57764542b5a78880c0b0b63bd300ed96d444fd6b41510ae24318d6fcfa` |

They are under `snapshot/scripts/`. `owner-overlay.patch` records the difference
from main's earlier script versions. `committed-tree-before-overlay.sha256`
retains the original baseline; `snapshot/baseline.sha256` excludes only these
three permitted overlays so its actual source mutation checks remain meaningful.
The full `snapshot/sources.sha256` includes the overlays and reference contents.
Production and nonowner-script baseline diffs are empty.

Gitlinks were hydrated and verified exactly: Julia-mVMC
`8bb1b9e8ae47b1512c00b321be05664ddcac0fd1`, PfaPack.jl
`0dcf52c15caec63516d0703f36bfc8a4bc0e58d0`, mVMC-1.3.0
`d73d06bd529d3b2573f38eb5817c4a5f52971006`. No reference runtime was launched.

Build handle78750 terminal0,13.26s, own named-volume target
`/home/vscode/.cache/mvmc/target/issue179-store1-main1cea`:

```sh
timeout --kill-after=10s 600s cargo test --locked --profile test-fast \
  -p mvmc-core --features mpi --test mpi_issue179_state \
  --no-run --message-format=json
```

The unique Cargo compiler-artifact executable in `snapshot/build.json` was
copied to immutable `bin/mpi_issue179_state`. Its content hash happens to equal
the earlier69 binary, but this is a newly built main1cea artifact with its own
compiler/build/source association, not a relabelled old proof.
`build-association.json` binds the actual revision, build exit, source manifest,
binary and overlays, and hashes of `snapshot/build-command.txt`, `build.log`,
`build.exit`, `build.json`, `compiler.txt`, `git-rev-parse.txt`, `baseline.sha256`.

## Inputs, actual environment and execution

Prepare-only handle80311 terminal0 used the committed input generator with
`MPI179_PREPARE_ONLY=1`; retained `inventory-generation.log` points to original
generated root `/tmp/mvmc-issue179-evidence.7x4XzW`. This was input preparation,
**not CLI execution**. The selected `inventory/matrix.tsv` contains exactly12
real/cmp/complex-FSZ six-site Heisenberg configurations, world2/4 × width1/2,
identity QP1, seed1, warmup1, interval1, full chain samples3, InterAll absent.
`input-settings.tsv` independently checks each declaration: NSRCG0, NStore1 and
the exact requested NSplitSize, rejecting duplicate/missing declarations.
The state harness sets actual optimization steps/window to prefix1/2/3.
The whole inventory and every namelist-referenced input were hashed before
launch, not just when each cell was about to run.

Initial metadata audit returned2 because the awk variable `split` conflicted
with its built-in function. `preflight-initial.exit`,
`preflight-initial-reason.txt`, `input-settings-initial.tsv` preserve this
pre-launch metadata failure. Corrected metadata preflight handle66470 terminal0;
no MPI launch occurred before authorization. It did not change source or inputs.

Actual runtime: Linux x86_64, Rust1.99.0 b940084d7/LLVM23.1.1, MPICH/Hydra4.2.0
ch4:ucx/internal PMI1 without external PMIx. Linked OpenBLAS API reports
0.3.26, Haswell, threads1, configuration
`NO_LAPACKE DYNAMIC_ARCH NO_AFFINITY Haswell MAX_THREADS=64`.
This API observation is from a **separate metadata process** loading the
binary-linked library, not an API call inside each MPI rank. Workers1 is
explicitly configured; threshold override is unset (production default32).
OPENBLAS/OMP/MKL/BLIS thread limits are1. Actual worker/kernel snapshots remain
in each repeat directory; this stage does not claim parallel-worker activation.

One authorized stage, handle46762 harvested terminal0, no retry/expansion:

```sh
# From the retained snapshot directory:
export MPI179_SOURCE_COMMIT=1cea33dde10653bb946d24c52cf78f8c035e891f
export MPI179_CARGO_TARGET_DIR=/home/vscode/.cache/mvmc/target/issue179-store1-main1cea
export MPI179_TIMEOUT=60
bash scripts/verify_mpi_issue179_prefixes.sh ../bin/mpi_issue179_state \
  ../inventory ../direct-store1-evidence sources.sha256 \
  ../build-association.json --stage direct-store1
```

This command is retained in `launch-command.txt`; **do not replay it into the
existing evidence directory**. Actual per-launch input/output/prefix commands
are in `direct-store1-evidence/commands.txt`, selecting exactly `issue179_state`
with `--ignored --exact ... --nocapture`, timeout60s/kill grace5s.

Results:12 configurations × prefix1/2/3 × two fresh runs = **72 native exits0**,
**12 strict checker exits0**, **216 intact native PASS summary fragments**,
with no FAILED/panicked/TIMEOUT markers. Native launcher logs can interleave;
these fragments are not independently rank-tagged logs. Separate `rank-N.txt`
files identify every expected rank. Actual SR aggregate is **432 observations**,
all not-solved0, factor INFO0 and solve INFO0. The conditional no-active path
is tested synthetically but was not exercised by this MPI capture. No native101
was accepted as expected success; any such outcome would fail this driver.

## Checker contract and numerical limit

The opt-in stage reads actual SR observer settings, not manually asserted solver
flags: seed, prefix, averaging window, NSRCG0/NStore1; one system per step;
UPLO=U, NRHS1, status0, real factor/solve INFO and finite matrix/RHS/increment.
It requires actual integer flags of the parameter-derived shape, unique active
indices within those flags whose values are1, and three finite regularization
values. Full update vector lengths derive from actual initial/final parameter
records (two scalar fields per complex parameter), not reduced SR dimension.
Missing/orphan metadata and malformed/extra shapes fail. The actual direct CG
sidecar must contain only `d:events 0`; a CG `zvo_SRinfo.dat` output is rejected.

The earlier prefix/group contracts remain: initial/final raw624/cursor/count,
next624, full actual proposal/reject/Metropolis/draw trace, seven-field saved
checkpoint, group seed offset and sampler equality, common prefix trace and
unchanged initialization. Width1 is an independent per-rank chain, not a
world-wide chain-equality assertion. Measurement-local arrays and post-measurement
scratch are excluded from group equality. Intermediate checkpoint raw state is
not newly exposed; separate prefix final snapshots supply the bounded evidence.

Fresh same-config rank records and root outputs repeat exactly within this
implementation. Computed arrays are checked for finite values and record shapes,
but no independent C/Julia accumulator assembly, SR update/residual or root-output
accuracy is established. This is not a cross-language floating-point bitwise
gate and introduces no numerical bounds. Full #179 independent numerical,
CG/store combinations, uneven width3, standard QP/OptTrans, worker activation,
long20 and broader failure coverage remain outside this evidence.

## Retained filenames and parent read-only replay

| Files relative to root | Meaning |
| --- | --- |
| `launch-wrapper.exit`, `launch-terminal.exit` | Original stage and final closure exits0 |
| `direct-store1-evidence/terminal-status.txt`, `terminal-summary.txt` | Complete stage/commit,72 expected launches,exit0 |
| `direct-store1-evidence/results.tsv`, `checker-results.tsv` |36 prefix-pairs and12 checker results |
| `store1-native-audit.tsv`, `store1-native-audit-summary.txt`, `store1-artifact-audit.exit` |72 log/exit/fragment audits,216 summaries,exit0; no new MPI launches |
| `store1-sr-aggregate.tsv`, `store1-audit-artifacts.sha256` | Actual432 INFO/not-solved observations and audit hashes |
| `launch-artifacts.sha256` | Per-launch/result/checker/raw SR/RNG/config/output artifact hashes |
| `snapshot/sources.sha256`, `direct-store1-evidence/sources-before.sha256`, `source-before-check.txt`, `source-after-check.txt`; `launch-final-source-check.txt` | Full frozen source and pre/post checks |
| `snapshot/baseline.sha256`, `baseline-check.txt`, `production-baseline-diff.txt`, `nonowner-scripts-diff.txt` under evidence | Baseline excluding the permitted overlay3, empty production/nonowner differences |
| `build-association.json`, `direct-store1-evidence/build-association.json`, `build-association-check.txt`; `bin/mpi_issue179_state`, `binary.sha256`, `launch-final-binary-check.txt` | Build/source/binary association and binary postcheck |
| `inventory/matrix.tsv`, `input-settings.tsv`, `input-closure-before.sha256`; evidence `input-closure-before.sha256`, `input-closure-after-check.txt` | Declared matrix/settings and complete input pre/post closure |
| evidence `executables-checkers.sha256`, `executable-check.txt`, `selected-test-list.txt`, `stage.txt` | Exact executable/checkers/stage selection and mutation checks |
| evidence `libraries-before.sha256`, `library-after-check.txt`, `binary-ldd.txt`, `runtime-backend.json`, `runtime-backend-ldd.txt` | Actual linked backend/hash/API metadata |
| evidence `environment.txt`, `mpi-version.txt`, `rust-version.txt`, `uv-version.txt`, `commands.txt`; root `c-compiler.txt`, `mpi-compiler.txt`, `platform.txt` | Actual settings/compiler/runtime and commands |
| evidence `CELL/prefixP/w1/launchR.log`, `launchR.exit`, `repeatR/`, `inputs.sha256`, `input-check.txt`; `CELL/validation.log` | Native logs/exits,rank-SR/RNG/worker records,outputs and strict validation |

All source/input/library/binary/checker closure checks succeeded. Parent replay
below only reads saved artifacts and reruns the checker; it does not launch MPI:

```sh
cd /home/vscode/.cache/mvmc/issue179-store1-main1cea.KQhRth/snapshot
sha256sum -c sources.sha256
sha256sum -c baseline.sha256
sha256sum -c ../binary.sha256
sha256sum -c ../direct-store1-evidence/input-closure-before.sha256
sha256sum -c ../direct-store1-evidence/libraries-before.sha256
sha256sum -c ../direct-store1-evidence/executables-checkers.sha256
sha256sum -c ../launch-artifacts.sha256
(cd .. && sha256sum -c store1-audit-artifacts.sha256)
uv run --no-project python scripts/mpi_issue179_prefix_evidence.py "$PWD" \
  --build-proof ../build-association.json --binary ../bin/mpi_issue179_state \
  --manifest sources.sha256 --commit 1cea33dde10653bb946d24c52cf78f8c035e891f
checked=0
while IFS=$'\t' read -r id ranks mode width cg store rest; do
  [[ $id == cell ]] && continue
  uv run --no-project python scripts/mpi_issue179_prefix_evidence.py \
    "../direct-store1-evidence/$id" --ranks "$ranks" --width "$width" \
    --stage direct-store1 || exit 1
  checked=$((checked + 1))
done < ../inventory/matrix.tsv
(( checked == 12 ))
```

For a direct native-fragment recount, use the exact intact phrase, not anchored
complete `test result` lines. For the SR aggregate, read the original rank files:

```sh
rg --files ../direct-store1-evidence | rg '/launch[12][.]log$' |
  xargs rg -o --no-filename '1 passed; 0 failed; 0 ignored' | wc -l
# Expected216; independently also check each launch against its world2/4.
rg --files ../direct-store1-evidence | rg '/launch[12][.]log$' |
  xargs rg 'FAILED|panicked|TIMEOUT'
# Expected no matches (rg exit1), not a native process failure.
rg --files ../direct-store1-evidence | rg '/rank-[0-9]+[.]txt$' | sort |
  xargs awk '$1 ~ /^d:sr-system-[0-9]+-(not-solved|factor-info|solve-info)$/ {
    key=$1; sub(/^d:sr-system-[0-9]+-/, "", key); count[key,$2]++
  } END {for (key in count) print key,count[key]}'
# Each family has432 records with value0; original aggregate is stored above.
```

Parent independently inspected the retained terminal exits,72 native launches,
216 intact PASS summaries and432 successful INFO records. Corrected read-only
hash replays completed with exit0: `launch-artifacts.sha256` requires the
`snapshot/` working directory, whereas `store1-audit-artifacts.sha256` requires
the evidence root (the parent of `snapshot/`). An initial wrong-directory hash
replay was an observation error, not a launch failure. Parent handle84887 then
completed with exit0 and12 strict-checker PASS results (`CHECKED12`). An initial
host-to-container quoting error in the parent's tab separator was corrected;
the saved matrix and documented Bash tab separator were unchanged. These checks
read retained artifacts only; they are not additional MPI launches or independent
numerical-reference validation.

| Artifact | SHA-256 |
| --- | --- |
| `snapshot/sources.sha256` | `77ca6ae9f16361ebbe4affe29845b3a3d6dbe8189b6974ba16432c5c9c063dbc` |
| `bin/mpi_issue179_state` | `49a860661603d30fdc6a43ab30c2cdf3560a0a6eb819476300c6da24a9f760d7` |
| `build-association.json` | `fe7c91d9e066dd6007b839934668ea5646c4b1528fa86bc90fcf1b0145e25eac` |
| `inventory/matrix.tsv` | `20b6895f944ce00d948521519f13b47743e85df16344e3c65051f93d6bd90e5e` |
| `direct-store1-evidence/input-closure-before.sha256` | `6a07bd46a65e5f234f4598b81510dd8c70d910958267f9aab155ad559a014c26` |
| `direct-store1-evidence/results.tsv` | `3d82539b9a975eabba81db1a2611b3178001bfbd99373a35a1f38990b5ae3015` |
| `direct-store1-evidence/checker-results.tsv` | `ce9cb5ce17ee552e56b92c621956930f8233421266330386fc4c33c87cdfe6cc` |
| `launch-artifacts.sha256` | `81fba603131df552aba4d8a1e899f7d8bfd1df70192925d1694cb621a4adec24` |
| `store1-sr-aggregate.tsv` | `a399037f8772185f3b27cf5835e84ee961d3f3148b0509e9f97bddf5b5c82922` |
