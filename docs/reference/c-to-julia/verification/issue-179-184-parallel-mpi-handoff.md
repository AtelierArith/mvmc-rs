# Parallel74 / MPI87 individual source handoff

Related to #179 and #184. This is a reviewer handoff, not 161 execution PASS
claims. The adjacent `issue-179-184-parallel-mpi-handoff.tsv` has exactly one
row per M0518–M0591 and M1335–M1421. Chandra owns the acceptance ledger;
this artifact does not modify it. Named candidates marked related are **not**
an assertion-equivalence decision. Missing commands/evidence remain gaps.

## Source boundary

Original Julia gitlink: `8bb1b9e8ae47b1512c00b321be05664ddcac0fd1`.
This is distinct from published patched Julia PR54 `62b0f97f...`.
Shared Rust HEAD during this handoff: `78d620078f58093963945f1d8ca0493eb37b70bc`;
dirty files remain present. No current-main numerical execution is inferred.
All assertion hashes are copied from the original inventory and were checked
against the corresponding on-disk source. Full original test/launcher/worker
sources were read, including their enclosing loops and helper constructors.

Parallel test SHA: `a2b582f61dc981c318f817fc0575cfadfdb8c93e34c5679aeaf7fdae48be5050`.
Parallel helper SHA: `a7d72725f193a121f79267be6f931e2439fecdf104de01a0cb1d3a5ca8d91fc1`.
Additional helper sources read:

| Source under `MVMCOptimizers.jl/src/` | SHA-256 | Read boundary |
|---|---|---|
| parameter_sync.jl | 93886834c217f0054b956837a25ee533eb227ad27b8dbd98eb28ec0628ebe7d1 | complete file; pack/unpack, duplicate consistency, sync |
| unsupported_inputs.jl | ff43c4ae741b67d7fed97a3aa790cb0fca733841de2d148c0882ff15bf22e828 | complete file; global/optimization/PhysCal capability validators |
| weight_average.jl | 9df40ead19d28eb8e533063c7bfdae40915621878a31d4f0127de2e35d8cb972 | lines1–95; complete energy-average helpers |
| stochastic_opt.jl | d43895a547efb1d8444358b62dc762b481faee58c69afea34fcf28b3f5c9eab7 | lines781–827; complete sampled operator helper |

The MPI assertion rows originate in ten distinct files. The eleventh helper,
`test/mpi/mpi_failure_modes.jl`, has no rows in this ID interval; its full
109 lines were read. SHA:
`9b8b2c9513db534e7538c86e35e7f4970dcce682f33bda4e82927bb175d55ed3`.
This count does not exclude imports/transitive production dependencies.
Runner internals and transitive parameter-layout helpers still require deeper
mapping before claiming full model/input equivalence.

`issue-179-184-original-inputs.sha256` records 125 original input/reference
files from the eight fixture trees actually used by these source constructors.
Verify from the repo root with `sha256sum -c` on that manifest. It records
original files, **not** the worker's runtime rewritten copies or proof of C
generation. Historical reference output provenance still needs review.

## Setting keys: parallel rows

- QPRANGE: serial QP4 full; fake world4/local2 ranks0/1 QP4; fake
  world4/local4 rank3 QP1 empty. Julia one-based half-open indices adapt to
  Rust zero-based ranges, not identical tuples.
- GROUP3WIDTH2: ranks0/1/2, requested width2, group IDs0/0/1.
- SERIALCTX/SERIALWRAP: Julia serial context with null communicators and all
  sizes1/ranks0; complex array `[1+2i,3]`, scalar42, counter1..11.
  Rust uses typed/in-place reducers and does not return Julia object identity.
- ENV0530–0551 (TSV keys use full numeric suffix): cleared Julia flag,
  OMPI/PMIx/PMI/Slurm variables; then each exact `withenv` in original lines
  77–136. Clean auto, OMPI size2, PMIx rank1/size2 + Slurm2, PMI size2,
  PMI rank0/size1 + Slurm2, PMI singleton without/with Slurm1, Slurm-only2,
  NPROCS-only2, explicit flag1, flag0 clean/PMIx/PMI/Slurm, invalid `yes`.
  A launcher rank detector does not implement Julia's entire mode policy.
- CHUNK0558–0561: lengths0/3/8/9, max chunk4; corresponding literal ranges.
- SEED0562–0567: default11272, zero0, positive123, negative-1 clock,
  explicit777 override, fake group3 with base100→103.
- VALIDATION0568–0577: fake MPI world2/local1/cross2; standard CG width1
  accepted; serial-context standard CG accepted; grouped direct width2
  accepted globally and by parallel validator; grouped CG globally valid but
  capability rejected; solver2 globally capability rejected. Missing grouped
  reducer is separate from invalid public input. Grouped CG's C unwritten
  store dependency is not a universal C-invalid-input designation.
- DEFAULTSEED: parser default11272.
- PACK0579–0591: parsed Heisenberg real fixture; parameter count/pack length,
  first and last slot delta `.25-.5i`, all-slot `.01+.02i` perturb/restore,
  short-vector rejection, duplicate orbital repair, deliberately inconsistent
  manual charge-RBM duplicate, signed negative-zero real/imaginary, and serial
  context sync versus legacy sync. Rust private flat-vector helpers and typed
  data layout need explicit assertion mappings, not API-name matching.

## Setting keys: MPI rows and loop expansion

- REAL4 / REAL4FILES / REAL4LOG: Heisenberg real original para-opt fixture,
  runtime nsteps4/nsmp4. Serial sanity then world2; separate world4 launch.
  Original fixture modpara is retained, unlike the repeat matrix's sample3
  rewritten inputs. Root readback has4 rows; non-root result is minimal/NaN.
- HUBBARD4 / HUBBARD4FILES: Hubbard real fixture, world2, nsteps4/nsmp4.
- IDENTITY1: Hubbard real and Heisenberg FSZ; each invokes `(width,store,world)`
  `(1,0,2),(1,1,2),(2,0,4),(2,1,4)`. Worker replaces NSPGaussLeg1/NMPTrans1,
  identity qptrans, step/window1, warmup1, sample4, directSR0.
- STANDARD1: Heisenberg real, tetragonal momentum-projected Hubbard real,
  Heisenberg complex; same four width/store/world combinations, step/window1,
  warmup1/sample4/direct0, but **retains** projector inputs. Optional NSP override
  is absent from the current three caller tuples.
- PHYSCALGROUP: Heisenberg real/complex physcal_ref inputs plus preset
  `zqp_opt.dat`; width1/world2 and width2/world4; quantity windows1,
  warmup1/sample4, Lanczos0. Compare five arrays: out/var/one-body/two-body/
  factored two-body. PHYSREAL/PHYSREALLOG retain original real PhysCal settings.
- FILESETS: expand helper calls into six para-opt filenames and five PhysCal
  filenames in the original launcher; existence is not exclusive writer proof.
- GROUPCOMPARE: identity family two fixtures × three comparisons × two fields;
  standard family three fixtures × three comparisons × three fields; PhysCal
  family two fixtures × five fields. Length and maxdiff assertions execute for
  each pair. Historical `1e-8`/`5e-8` bounds are not adopted Rust budgets.
- CREFCG/CREFCGFILES: real `heisenberg_chain_real_nsrcg`, world2 one step/window1;
  original C MPI2 reference files, six energy columns,14 packed parameters,
  expanded orbital output. Original column bounds1e-10 (columns3/4 1e-9),
  parameter1e-2 and exact SRinfo text are historical, not current acceptance.
- LITERALCG: world2/width1; n2/local sample2, invW.25, stabilizer.1,
  mean `[.25,-.5]`, diagonal `[2,3]`; rank-specific real and imaginary sample
  matrices and non-root search poison999. Expected products real
  `[-15.30625,-22.7375]`, complex `[-14.4859375,-24.2375]`.
- WEIGHT2/WEIGHT2LOG: world2/width1, local weight rank+1, energy multiples
  10/100/2/3, expected global weight3; WEIGHTZERO adds wc0/energy0 and checks
  one root warning. Generic reductions are not those normalization assertions.
- REJECT1401–1408: each world2 process independently validates before MPI.Init:
  solver2; grouped CG width2; grouped OptTrans two sectors; grouped FSZ
  NSP1/NMP2. Errors caught by worker yield launcher success and empty output;
  this is not generic nonzero-exit/callback-failure semantics.
- GUARDEDSERIAL: Julia flag0 under world2 must terminate nonzero; Rust no-MPI
  feature refusal is a different availability boundary, not flag emulation.
- FORCEDSINGLETON: Julia flag1 without launcher; Rust explicit MPI-context
  singleton/CLI choices need intentional architecture mapping.

## Rust candidates and execution labels

Candidate sources inspected (hashes bind this source review, not new execution):

| File | SHA-256 |
|---|---|
| src/parallel.rs | 8ca276c5ab24cf6bc59a9ee6b7acab03b7f5ca5babb22aa5696eb3ba400aca2f |
| src/reducer.rs | 3d0e4af324a1a660cfa8975369eae7dd39b48754d7367d530e8ae6ed4916beb7 |
| src/sync.rs | 7e0b3ce22c69f2d4e7093b5ad062c2bc7810e9649bb6888a10da6b3e69f15b0c |
| tests/runtime_contract.rs | 50c7d3fad4b69e6b04977aacba8e2fa6ac4afc3cabbd0c4da2722ec9361aae29 |
| tests/mpi_issue179_mapping.rs | bd91020f911c76878d650a488fd6a68c35bd9917ae4695283692ea68548ba363 |
| tests/mpi_issue179_literal_operator.rs | 43e519c563473c5b1f859607a139c7d60d95d205c843fc0c313b7ff0f04eea4e |

Paths above are relative to `crates/mvmc-core`. Literal source has only import
format changes relative to its executed historical snapshot; these source hashes
must not overwrite that snapshot's binary/test hashes.

Relevant named gate reproduction (requires correctly frozen feature-enabled
executable; do not let cfg-disabled selection silently pass zero tests):

```sh
timeout --kill-after=5s 120s mpiexec -n 2 "$LITERAL_BINARY" --ignored --exact issue179_literal_two_rank_cg_products --nocapture
timeout --kill-after=5s 120s mpiexec -n 2 "$MAPPING_BINARY" --ignored --exact issue179_group_width_endpoints --nocapture
timeout --kill-after=5s 120s mpiexec -n 4 "$MAPPING_BINARY" --ignored --exact issue179_group_width_endpoints --nocapture
```

Historical actual build3051 and short28851 terminal0 bind
`/home/vscode/.cache/mvmc/issue179-current-groups.02Msjl`: base
`b0fda9e8697ccf1c193cbab4e0c343f610dc8e29` plus recorded overlay,
not new main. Actual source manifest
`db35058d4209b6facac48821b32e2f362ee28fcf3c3088566bb0a7bb22a49d94`,
overlay `bdaec1ebbfd569e403c4de8ba26bbc527c42c3d52bf9ce887e17857f91d869e7`.
Read `short-results.tsv`, `short-binaries.sha256`, `short-binary-check.txt`,
`short-source-check.txt` there. Literal world2 and mapping world2/4 are bounded
actual passes; eleven short rows do not prove the 161 original assertions.
Separate callback31914 six configurations prove a post-init injected callback
boundary only; no pre-init rejection or model accuracy inference.

The four literal operator rows are retained historical focused evidence.
All other entries remain source candidates, explicit API differences, or gaps
pending exact conditions and execution review. No arbitrary historical Julia
budget, exact solver iteration-count requirement, or same-language495 repeat
result is promoted to cross-configuration accuracy or 161-row completion.

## Corrected assertion-text extraction and mutation guards

The initial TSV SHA `f6c228c5d55ae2efac59399cc8c0a7d1192b273e3cac6622b8742661954416e3`
was defective: its assertion column copied ledger column19 (byte end), not
column20 (`assertion_full_source_escaped`). Do not adopt that draft as an
assertion-text review. The correction replaces **only column5**, preserving the
nine-field schema and the other eight columns of all161 rows. It does not
upgrade any proof classification. Corrected TSV SHA:
`490f3e71e7113f35a8b5ce89fced439c174e4eb61f411fc6772d59321c780419`.

Correction-time HEAD: `bdf35145979e57ec17a27909f8f5064261b28cd9` with shared
dirty files retained. Mutation guards compared the original in-memory TSV
against the corrected file: header and all eight non-assertion columns were
unchanged. Ledger SHA before/after was
`3f9d82e291005fc1fdbd39351862a5fd368787fc57b1615fa9ac767906f3a839`;
the input manifest and all11 assertion-source files also remained unchanged.
The ledger remains reviewer-owned and was not edited by this correction.

The original byte intervals are **1-based inclusive**: Python slice
`raw[start-1:end]`, not an exclusive-end source interval. An initial exploratory
exclusive-end comparison exposed the final-character mismatch; it was not
acceptance evidence. The corrected verification below exited0 under uv0.12.21.
It verifies all161 original escaped texts, byte spans, start lines and source
hashes, not just row/field counts. Independent input-manifest verification and
`git diff --check` also exited0. Reproduce from the repo root:

```sh
uv run --no-project python - <<'PY'
import csv, hashlib, re
from pathlib import Path
base = Path('docs/reference/c-to-julia/verification')
with (base / 'issue-184-assertion-audit.tsv').open() as f:
    ledger = {r['audit_id']: r for r in csv.DictReader(f, delimiter='\t')}
with (base / 'issue-179-184-parallel-mpi-handoff.tsv').open() as f:
    reader = csv.DictReader(f, delimiter='\t')
    assert len(reader.fieldnames) == 9
    rows = list(reader)
expected = ({f'M{i:04d}' for i in range(518, 592)} |
            {f'M{i:04d}' for i in range(1335, 1422)})
assert len(rows) == 161 and {r['id'] for r in rows} == expected
paths = set()
for r in rows:
    assert None not in r
    original = ledger[r['id']]
    assert (r['julia_path'], r['julia_line'], r['julia_source_sha256']) == (
        original['julia_source'], original['julia_line'], original['julia_source_sha256'])
    assert r['assertion'] == original['assertion_full_source_escaped']
    assert not re.fullmatch(r'\d+', r['assertion'])
    raw = (Path('extern/Julia-mVMC') / r['julia_path']).read_bytes()
    paths.add(r['julia_path'])
    assert hashlib.sha256(raw).hexdigest() == r['julia_source_sha256']
    start, end = int(original['assertion_byte_start']), int(original['assertion_byte_end'])
    assert 1 <= start <= end <= len(raw)
    text = raw[start - 1:end].decode()
    escaped = (text.replace('\\', '\\\\').replace('\t', '\\t')
               .replace('\r', '\\r').replace('\n', '\\n'))
    assert escaped == r['assertion'], r['id']
    assert raw[:start - 1].count(b'\n') + 1 == int(r['julia_line'])
    assert text.startswith(('@test', '@assert'))
assert len(paths) == 11
assert sum(r['review_status'] == 'LIVE_HISTORICAL_LITERAL' for r in rows) == 4
print('PASS assertion-text provenance161 sourceSHA11; NOT execution PASS161')
PY
sha256sum -c docs/reference/c-to-julia/verification/issue-179-184-original-inputs.sha256
git diff --check
```
