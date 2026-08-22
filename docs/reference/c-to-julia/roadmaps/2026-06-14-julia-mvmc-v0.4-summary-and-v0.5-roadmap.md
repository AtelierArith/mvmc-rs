---
date: 2026-06-14
datetime: 2026-06-14 20:44 JST
model: GPT-5 Codex
status: plan
topic: Julia-mVMC v0.4.0 到達点整理と v0.5 以降の作業候補
updated:
  - datetime: 2026-06-15 13:10 JST
    model: GPT-5 Codex
    note: Selected v0.4.1 MPI hardening as the next release target.
  - datetime: 2026-06-23 14:26 JST
    model: GPT-5 Codex
    note: Refreshed roadmap after v0.4.1 release and PR #39/#40 merges.
  - datetime: 2026-06-23 14:32 JST
    model: GPT-5 Codex
    note: Audited post-v1.4.0 C-mVMC develop features and added intake order.
  - datetime: 2026-06-23 14:47 JST
    model: GPT-5 Codex
    note: Split C-mVMC v1.4.0 baseline parity items from post-v1.4.0 develop intake.
  - datetime: 2026-06-23 18:00 JST
    model: GPT-5 Codex
    note: Retargeted the current release prep from v0.5.0 to v0.4.2 and reserved v0.5.0 for C-mVMC v1.2 parity minus BackFlow.
  - datetime: 2026-06-23 18:28 JST
    model: GPT-5 Codex
    note: Expanded the C-mVMC v1.2 parity backlog and added the Full Lanczos implementation plan link.
  - datetime: 2026-06-29 16:48 JST
    model: GPT-5 Codex
    note: Marked v0.4.2 release complete and fixed the v0.4.3 P0 scope to Full Lanczos R0/R1.
  - datetime: 2026-06-29 21:44 JST
    model: GPT-5 Codex
    note: Numbered the v0.5 remaining work buckets as V05-1..V05-4.
  - datetime: 2026-06-30 22:06 JST
    model: GPT-5 Codex
    note: Marked PR #43/#44 progress, PR #45 status, and kept NLanczosMode=2 plus PhysCal/Green residuals as v0.5.0 requirements.
  - datetime: 2026-07-01 16:52 JST
    model: GPT-5 Codex
    note: Marked PR #45 merged to develop and moved the immediate v0.5 focus to NLanczosMode=2 plus PhysCal/Green residuals.
  - datetime: 2026-07-08 11:21 JST
    model: GPT-5 Codex
    note: Added the C-mVMC v1.3 parity progress dashboard link after PR #48 merged.
---

# Julia-mVMC release summary and v0.4.3/v0.5 roadmap

Progress dashboard:
`docs/roadmaps/2026-07-08-c-mvmc-v13-parity-roadmap.html`

## Current state

2026-07-01 16:52 JST 時点の状態:

- Latest release: `v0.4.2`
  - Release: `https://github.com/tmisawa/Julia-mVMC/releases/tag/v0.4.2`
  - Release prep PR: `tmisawa/Julia-mVMC#41`
  - Release PR: `tmisawa/Julia-mVMC#42`
  - Release merge commit on `main`: `44754192edb15834592bb7f8cf745ec3256e1751`
  - Published: `2026-06-29`
- Current public branches:
  - `origin/main = 44754192edb15834592bb7f8cf745ec3256e1751`
  - `origin/develop = ad4db59` (PR #45 merge 後)
  - `origin/feature/v0.5-nsplit-standard-projection = d76f7af` (PR #45 source branch)
- `v0.4.2` includes the post-`v0.4.1` MPI work:
  - PR #39 `feature/v0.5-srcg-cg`: MPI-compatible SR-CG support
    (`NSRCG = 1`, `NSplitSize = 1` scope).
  - PR #40 `feature/v0.5-nsplit-nstore`: direct-SR `NSplitSize > 1`
    with `NStore = 0/1` for the `NQPFull = 1` R1 scope.
- Post-`v0.4.2` develop state:
  - PR #43 merged Full Lanczos R1 (`NLanczosMode = 1` PhysCal energy / QQQQ).
  - PR #44 merged PairHop C-reference parity gate.
  - PR #45 merged sz-conserved standard-projection `NQPFull > 1`
    grouped QP split under direct SR.
- C-mVMC reference state checked on 2026-06-23:
  - `mVMC/origin/master = e83aa6c` (`v1.4.0` release line).
  - `v1.4.0` baseline includes `reweight`, Twist/Lattice PhysCal output,
    t-J update paths (`NExUpdatePath = 4/5`, `NCond` convention), and
    doublon-only pair hopping (`NExUpdatePath = 6`).
  - `mVMC/origin/develop = 5d59c06`, with PR #104/#106/#108/#109/#110/#111/#112
    merged after `v1.4.0`; PR #107 BackFlow remains open.

## Decision update: v0.4.2 complete and v0.4.3 P0

v0.4.2 は release 済み。v0.4.2 には、当初 P0 候補だった MPI SR-CG
(`NSRCG = 1`, `NSplitSize = 1`) と direct-SR NSplitSize/NStore R1
(`NSplitSize > 1`, `NStore = 0/1`, `NQPFull = 1`) が入った。

ただし、v0.5.0 を「C-mVMC v1.2 parity 到達点」として扱うなら、現在の
`develop` はまだそこまで到達していない。v0.5.0 条件から BackFlow は外す。
理由は、C 側でも v1.2 の BackFlow は不完全で、v1.4 以降の separate track として
実装が進んでいるため。

次の patch release は **v0.4.3** として進める。v0.4.3 P0 は
Full Lanczos の R0/R1 に固定する:

1. R0: C v1.2 の `SpinChainLanczos` / `HubbardChainLanczos` 入出力、
   unsupported combination、output contract の監査。
2. R1: serial `NSplitSize = 1` の `NLanczosMode = 1` energy/output path
   (`zvo_ls.dat`, `zvo_ls_qqqq.dat`) と C-reference gate。

`NLanczosMode = 2` の Lanczos Green list / Green accumulator、MPI/FSZ broader
mode は v0.4.3 P0 から外し、R2 以降の別 PR / 別 gate として扱う。
v0.4.3 以降で以下を C-mVMC v1.2 parity へ向けて順に積む。以後、
v0.5.0 までの残タスクはこの番号で参照する。

1. **V05-1 Full Lanczos**: PR #43 で R1 (`NLanczosMode = 1` PhysCal
   energy / QQQQ) は完了。残りは v0.5.0 必須の R2/R3
   (`NLanczosMode = 2` Lanczos Green)。
2. **V05-2 PairHop parity**: PR #44 で C-reference gate は完了。v0.5.0
   release 前に manual / non-FSZ / FSZ scope の closeout を確認する。
3. **V05-3 MPI / NSplit residuals**: PR #45 で `NQPFull > 1`
   sz-conserved standard-projection grouped QP split は merge 済み。残りは
   `NSplitSize > 1` + SR-CG。FSZ / OptTrans-derived QP split は当面 reject。
4. **V05-4 PhysCal / Green residuals**: `VMCPhysCal NSplitSize > 1`、
   FSZ `TwoBodyGEx`、Lanczos Green output を v0.5.0 必須として扱う。

これらが揃った段階で v0.5.0 を切る。`NLanczosMode = 2` Lanczos Green と
PhysCal/Green residuals は v0.5.0 条件から外さない。RBM、`reweight`、Twist、
NBodyG、NBodyInterAll、t-J / doublon-only、BackFlow は v1.4+ / develop feature
intake として v0.5.0 条件から切り分ける。

### C-mVMC v1.2 parity backlog

v0.5.0 は、BackFlow を除いた C-mVMC v1.2 parity の節目として扱う。
現時点で足りない、または検証が必要な項目は以下。

| Item | C-mVMC v1.2 state | Julia-mVMC current state | Roadmap action |
|------|-------------------|--------------------------|----------------|
| V05-1 Full Lanczos | `NLanczosMode > 0` path、Lanczos energy、Lanczos Green、`SpinChainLanczos` / `HubbardChainLanczos` tests がある | PR #43 で R1 (`NLanczosMode = 1` PhysCal energy / QQQQ) は merge 済み。`NLanczosMode = 2` Lanczos Green は未対応 | `docs/plans/2026-06-23-julia-mvmc-full-lanczos-plan.md` を正として、R2/R3 (`NLanczosMode = 2` Lanczos Green) を v0.5.0 必須の別 PR / 別 gate で実装する |
| V05-2 PairHop parity | Hamiltonian の PairHop 項が実装済み。`pairhop.def` 読み込み時に `(i,j)` と `(j,i)` を内部展開する | PR #44 で C-reference gate は merge 済み | manual / non-FSZ / FSZ scope の closeout を行い、必要なら補足 fixture を追加する |
| V05-3 MPI / NSplit residuals | v1.2 の MPI split path と `NSRCG = 1` がある。後続 v1.4 では SR-CG split の扱いがさらに整理された | `NSRCG = 1` は `NSplitSize = 1` のみ。PR #45 で sz-conserved standard-projection `NQPFull > 1` grouped QP split は merge 済み | 残る `NSplitSize > 1` + SR-CG を別 track で設計する。FSZ / OptTrans-derived QP split は当面 reject |
| V05-4 PhysCal / Green residuals | `NVMCCalMode = 1` で normal Green、Lanczos Green、MPI split output を扱う | `TwoBodyGEx` は non-FSZ のみ対応。FSZ `TwoBodyGEx`、Lanczos Green、`VMCPhysCal NSplitSize > 1` は未対応 | v0.5.0 必須として、PhysCal grouped split、FSZ `TwoBodyGEx`、Lanczos Green output を C-reference fixtures で固定する |

## What v0.4.0 includes

### Public release and verification base

- Public Julia-mVMC repository, submodule-based layout, license/citation/manual
  structure, GitHub Release 運用を確立した。
- CI は Ubuntu/macOS × Julia 1.11/1.12 の unit / integration /
  ctest-equivalent / PhysCal gate を持つ。
- `Manifest-v1.11.toml` を reproducibility snapshot として維持している。
- `CHANGELOG.md`, `CITATION.cff`, README/manual の release-facing metadata を
  v0.4.0 へ更新済み。

### C-reference coverage

- `VMCParaOpt` は supported standard fixtures で C reference gate を持つ。
- C ctest-equivalent integration runner と committed reference data がある。
- `VMCPhysCal` は experimental 扱いだが、one-body / direct two-body /
  factored-product two-body Green output の C-referenced fixtures を持つ。
- DH2/DH4 projection layout, `greentwoex.def` / `TwoBodyGEx`, `InOrbitalParallel`,
  warn-only `OptTrans` 系など、v0.3 で主要 input path を拡張した。

### Threading policy

- C-mVMC 忠実性を優先し、sample-level `VMCMainCal` threading は無効化した。
- 保守的な inner-loop threading helper と CI guard を追加した。
- `JULIA_MVMC_INNER_THREADS` / `JULIA_MVMC_PFAPACK_THREADS` は opt-in / triage
  扱いで、default serial C-parity path を正とする。

### MPI support

- MPI.jl を通常依存に追加し、`ParallelContext` / `SerialContext` と通信 wrapper を
  `MVMCOptimizers.jl/src/parallel.jl` に集約した。
- common launcher env と `JULIA_MVMC_MPI` に基づく MPI detection / fail-fast policy を
  実装した。
- `VMCParaOpt` と `VMCPhysCal` は MPI.jl-compatible launcher 下で
  `NSplitSize = 1` の sample-parallel execution を実行できる。
- rank0-only output/readback, rank0-only logging/warning, comm0 allreduce,
  Green reduce-to-root, seed offset handling を実装した。
- `test/mpi/run_mpi_smoke.jl` は `mpiexec -n 2`, `mpiexec -n 4`, WeightAverage,
  PhysCal, stdout/warning gate, `JULIA_MVMC_MPI` policy を確認する。
- Genkai では Intel MPI 2021.10 / PMI (`PMI_SIZE=4`, `PMI_RANK=0..3`) で
  v0.4.0 release-candidate gate を pass した。
- Post-release Genkai benchmark では、`L=16`, ranks `1/2/4` の first-10-step
  strict trajectory gate が pass した。`L=24/32` は strict pass/fail gate から外し、
  C ctest と同様に tail-summary (`Etot`, `Etot2`) と timing の診断対象にした。
- C-mVMC v1.4.0 `icc` / `gcc` build との 300-step timing matrix では、Julia は
  C `gcc` build より明確に速い。一方、C `icc` build は Genkai primary baseline として
  速く、Julia は概ね同等からやや遅い。fixed-total の Julia rank 4 speedup は
  `3.74-3.93` だった。

### NSRCG and parameter layout

- Serial `NSRCG = 1` SR-CG path を C-reference fixture へ載せた。
- `xdot` の C-source-order accumulation を unit guard した。
- post-CG parameter update は bit parity ではなく `NSRCG_PARAM_TOL = 1e-2` の
  tolerance gate として明文化した。
- variational parameter flat-layout walker を共有化し、MPI sync / pack / unpack /
  parser count の重複を減らした。
- `InOrbital` overlay は array position ではなく variational-parameter index で適用する
  よう修正した。

## What landed after v0.4.0

### v0.4.1 release / MPI hardening

- MPI smoke and failure-mode tests were hardened and kept green on CI.
- v0.4.1 release metadata, README/manual, CITATION, and Manifest snapshot were
  updated and released through `develop -> main`.
- The release flow now requires a `develop -> main` release PR and tags the
  resulting `main` merge commit.

### v0.4.2 release head

- MPI SR-CG support landed for `VMCParaOpt`, `NSRCG = 1`, `NSplitSize = 1`.
  The implementation follows the C-compatible broadcast/allreduce ordering in
  `operate_by_s!`, keeps rank0 update + sync semantics, and retains guards for
  unsupported CG submodes.
- Direct-SR `NSplitSize > 1` support landed for `VMCParaOpt` with `NStore = 0/1`,
  non-FSZ/FSZ, and `NQPFull = 1`. The implementation splits main-cal samples
  through `comm1`, uses stored-O global sample offsets, and adds four-way MPI
  self-consistency gates.

## C-mVMC v1.4.0 baseline parity audit

2026-06-23 に `mVMC/v1.4.0` tag を確認した。Julia 側では、C v1.4.0 release line の
機能と post-v1.4.0 `develop` の新機能を分けて扱う。

| C feature | C status | Julia state | Intake order |
|-----------|----------|-------------|--------------|
| RBM correlation factor | C では既存機能。v1.4.0 では reweight/RBM 周辺の修正も入っている | Parser / `In*` overlay / parameter init / RBM counters / log ratio / SR derivative path は実装済み。ただし C-style `initial.def` の RBM block は拒否され、`GeneralRBM_cmp` ctest は deferred | B0: finish `initial.def` RBM block and enable `GeneralRBM_cmp` gate |
| `reweight` | v1.4.0 PR #69 系。`modpara.def` keyword と SR / SR-CG accumulation の weighting | Julia は `reweight` keyword と C-compatible weighted accumulation をまだ持たない | B1: small SR-related parity feature after release |
| Twist/Lattice PhysCal output | v1.4.0 PR #77/#92 系。`lattice.def`, `twist.def`, `zvo_twist_yyy.dat` | Julia は parser / data model / PhysCal output が未実装 | B2: measurement/output feature after reweight or in parallel design |
| t-J update paths | v1.4.0。`NExUpdatePath = 4/5`, `NCond` convention, BackFlow/LocSpin/Lanczos restrictions | `NCond` / `NExUpdatePath` fields exist, but update-type logic supports only 0-3 | B4: separate sampling-sector design |
| doublon-only pair hopping | v1.4.0。`NExUpdatePath = 6`, doublon-only constraints and Green tests | Pair-hopping / doublon-only update path is not implemented | B5: after or together with t-J design |

RBM は「Julia-mVMC に入っているか」という問いには **partially yes** と答えるのが正しい。
実行 path の主要部は入っているが、C reference の `GeneralRBM_cmp` を通すための
`initial.def` RBM block 読み込みが未完了なので、release-facing には partial support と書く。

## C-mVMC post-v1.4.0 feature intake audit

2026-06-23 に `mVMC/origin/develop` を fetch し、`v1.4.0` 後の merged PR と open PR を
確認した。Julia 側へ取り込む順序は、依存関係・既存実装との距離・release scope の
膨張リスクで決める。

| C PR | C status | Feature | Julia state | Intake order |
|------|----------|---------|-------------|--------------|
| #106 | merged | `lattice.def` product mismatchを abort から warning へ緩和、Twist docs 修正 | Julia は standard Twist/lattice parser を主経路にしていないため直接影響は小さい | C0: compatibility note。将来 standard/Twist input を広げる時に追従 |
| #110/#111/#112 | merged | `NSplitSize` validation/doc、direct SR で `NStore` x `NSplitSize > 1` を許可、SR-CG split は reject | Julia PR #40 で R1 (`NQPFull = 1`) は追従済み。PR #45 で sz-conserved standard-projection `NQPFull > 1` も追従済み | C0 done / R2 merged; SR-CG split pending |
| #108 | merged | `NBodyG` variable-order correlation measurement、`zvo_NBodyG_yyy.dat` output、non-FSZ/FSZ/MPI tests | Parser / storage / `GreenFuncN` output が未実装 | C1: first post-v1.2-parity develop intake candidate |
| #109 | merged | `NBodyInterAll` local-energy interaction、complex normal/FSZ support、real/BackFlow/Lanczos reject | `InterAll` parserはあるが `NBodyInterAll` と general local-energy `GreenFuncN` が未実装 | C2: after NBodyG core |
| #104 | merged | t-J / doublon-only sector projection for Power Lanczos | PR #43 で Full Lanczos R1 は merge 済み。t-J / doublon-only sector projection は未対応 | C5: after Full Lanczos R2/R3 and sampling-sector design |
| #107 | open | minimum viable BackFlow implementation with narrow supported scope | BackFlow is globally unsupported and entry points fail fast | C4: keep outside v0.5.0; wait for newer C implementation track to settle, then separate design |

### Recommended intake sequence

1. **v0.4.2 release readiness** is complete. PR #39/#40 were released as
   `v0.4.2`, and `v0.4.2` is tagged on `main`.
2. **V05-1 Full Lanczos** started with PR #43, which merged R0/R1:
   C-reference audit plus `NLanczosMode = 1` PhysCal energy/QQQQ output for
   serial `NSplitSize = 1`. The remaining V05-1 work is mandatory R2/R3:
   `NLanczosMode = 2` Lanczos Green list reconstruction and output.
3. **V05-2 PairHop parity validation** landed through PR #44. The remaining
   action is closeout audit: confirm manual wording and non-FSZ / FSZ scope
   match the C-reference gate.
4. **V05-3 `NSplitSize > 1` + SR-CG / grouped QP split** is the current MPI
   semantics track. PR #45 landed sz-conserved standard-projection
   `NQPFull > 1`; `NSplitSize > 1` + SR-CG remains.
5. **V05-4 `VMCPhysCal NSplitSize > 1` and PhysCal/Green residuals** remains a
   mandatory v0.5.0 track. It covers PhysCal grouped split, FSZ `TwoBodyGEx`,
   and Lanczos Green output.
6. **v0.5.0 release readiness** should start only after items 2-5 are
   implemented. BackFlow is explicitly scoped out of this v0.5.0 criterion;
   `NLanczosMode = 2` and PhysCal/Green residuals are not scoped out.
7. **RBM `initial.def` block support** remains the first small v1.4+ C-parity
   cleanup after the v1.2 parity track. Most RBM machinery is already
   implemented, and this should unblock `GeneralRBM_cmp`.
8. **reweight** should follow as the SR-related v1.4 baseline feature. It is
   closer to the current MPI SR-CG / direct-SR work than t-J or doublon-only.
9. **Twist/Lattice PhysCal output** is the next low-semantic-risk v1.4 baseline
   feature. It is parser/output work and should stay separate from sampling
   sector changes.
10. **NBodyG measurement** should be the first post-v1.4.0 develop feature
   intake. It introduces the reusable `GreenFuncN` machinery needed by later
   work.
11. **NBodyInterAll local energy** should follow NBodyG. It reuses the same
   general operator-product evaluation idea but changes Hamiltonian/local-energy
   code, so it should not be implemented before the measurement primitive is
   tested.
12. **t-J (`NExUpdatePath = 4/5`) and doublon-only (`NExUpdatePath = 6`)** should
   get a separate sampling-sector design. They change allowed moves and Hilbert
   space constraints, so they should not be bundled with measurement features.
13. **BackFlow MVP** should wait for the newer C implementation track to settle
   and then get its own design. It touches sampling, optimization, parameter
   layout, restart, output, and performance-sensitive update kernels.

## Current known limitations

- Current `develop` `NSplitSize > 1` support is still direct-SR only:
  - supported in v0.4.2: `VMCParaOpt`, direct SR (`NSRCG = 0`), `NStore = 0/1`,
    non-FSZ/FSZ, `NQPFull = 1`;
  - PR #45 adds sz-conserved standard-projection `NQPFull > 1` grouped QP split
    on `develop`;
  - still not supported: `NSplitSize > 1` + SR-CG, FSZ standard-projection
    `NQPFull > 1`, OptTrans-derived QP sectors, and PhysCal split.
- MPI SR-CG is `NSplitSize = 1` only. `NSplitSize > 1 && NSRCG != 0` remains
  unsupported.
- SR-CG support is still limited to the implemented submode. `NSRCG >= 2`,
  `useDiagScale != 0`, and `RescaleSmat != 0` remain unsupported.
- `VMCPhysCal NSplitSize > 1` remains unsupported, but is a v0.5.0 requirement.
- C-mVMC `develop` の `NBodyG` measurement と `NBodyInterAll` local-energy
  interaction are not ported yet.
- C-mVMC `v1.4.0` baseline の `reweight`, Twist/Lattice output,
  t-J update paths, and doublon-only pair hopping are not ported yet.
- RBM is partially implemented, but C-style `initial.def` with RBM triples is
  still rejected; `GeneralRBM_cmp` remains deferred.
- MPI strict trajectory gate は小さい系 (`L=16`) に限定する。大きい系では MC sample
  と SR update の微小差が数 step 後に分岐し得るため、`L=24/32` の row-by-row
  strict equality は release blocker にしない。
- `L=32`, ranks `4` は known sensitive diagnostic case。C の multi-rank trajectory
  自体も repeat 間で揺れることがあり、Julia/C の微小差が早期に増幅し得る。
- Serial `NSRCG = 1` post-CG parameter gate は coarse tolerance で、CPU / BLAS /
  FMA / reduction-order 差に敏感。
- `VMCPhysCal` は C-reference gate を持つが、release status は experimental のまま。
- Full Lanczos R1 (`NLanczosMode = 1` PhysCal energy/QQQQ) は PR #43 で
  merge 済み。`NLanczosMode = 2` Lanczos Green は v0.5.0 requirement。
  FSZ/general-orbital Lanczos と ParaOpt Lanczos は未対応。
- BackFlow は未対応。
- sample-level independent-chain threading は C の逐次 Markov chain semantics を変えるため
  v0.4 scope では採用していない。
- `Pkg.add(url=..., subdir=...)` は workspace `[sources]` と submodule coupling のため
  supported install path ではない。

## v0.4.3+ / v0.5 candidate themes

v0.4.2 は PR #39/#40 の MPI work を含む patch release として完了。v0.5.0 は
C-mVMC v1.2 parity の節目として残す。v0.5.0 までの残タスクは
V05-1..V05-4 の番号で管理する。BackFlow は C 側でも後続実装 track のため、
v0.5.0 条件から外す。

### P0 candidates

1. **V05-1 Full Lanczos**

   - PR #43 で R0/R1 (`NLanczosMode = 1` PhysCal energy / QQQQ) は完了。
   - 残る R2/R3 で `NLanczosMode = 2` の Lanczos Green list reconstruction、
     Green accumulator、`zvo_ls_cisajs*` output を C-reference gate に載せる。
     これは v0.5.0 必須で、先送りしない。

2. **V05-2 PairHop parity validation**

   - PR #44 で C-reference gate は完了。
   - manual wording と non-FSZ / FSZ scope が C-reference gate と一致しているか
     closeout audit を行う。
   - 必要なら補足 fixture を追加する。

3. **V05-3 `NSplitSize > 1` + SR-CG / grouped QP split**

   - PR #45 で sz-conserved standard-projection `NQPFull > 1` を `comm1` 内で
     split し、`CalculateIP` / accept-reject probability を comm1-reduced QP sum に
     揃えた。
   - 次に `NSplitSize > 1` with `NSRCG = 1` を C-compatible にするか、
     C v1.2/v1.4 の behavior 差を明示して scope を決める。
   - 同一 `comm1` group の rank が Markov-chain lock-step を保つことを synthetic /
     integration test で固定する。

4. **V05-4 `VMCPhysCal NSplitSize > 1` and PhysCal/Green residuals**

   - PhysCal の grouped MPI split semantics を C 実装で確認する。
   - Green output reduce-to-root、rank0 output、sample/QP split の責務を分けて design を書く。
   - FSZ `TwoBodyGEx` と Lanczos Green output を C v1.2 parity 残件として扱い、
     v0.5.0 必須から外さない。
   - ParaOpt R1 の validator 緩和を PhysCal へ流用しない。専用 plan と gate を用意する。

5. **v0.5.0 release readiness**

   - v0.4.3+ の C v1.2 parity items を統合し、BackFlow を out-of-scope として明記する。
     `NLanczosMode = 2` と PhysCal/Green residuals は out-of-scope にしない。
   - release metadata / docs / CHANGELOG / Manifest を v0.5.0 へ更新する。
   - C v1.2 parity matrix と残り caveat を manual / release note に反映する。

6. RBM `initial.def` block support

   - C `initial.def` / `zqp_opt.dat` の `NProj -> NRBM -> NSlater -> NOptTrans`
     triples layout を Julia loader に実装する。
   - 既存 RBM parser / init / sampling / SR derivative path と接続し、
     `GeneralRBM_cmp` を ctest-equivalent runner へ昇格する。
   - RBM x reweight / BackFlow / doublon-only の unsupported combination は
     C v1.4.0 の behavior に合わせて明示する。

7. `reweight`

   - `modpara.def` の `reweight` keyword を parser / runtime data model に追加する。
   - C v1.4.0 の SR / SR-CG / store-O weighted accumulation を Julia 側へ反映する。
   - direct SR vs SR-CG の reweight comparison gate を C test に合わせて追加する。

8. Twist/Lattice PhysCal output

   - `lattice.def` / `twist.def` parser と validation を追加する。
   - `zvo_twist_yyy.dat` output を `VMCPhysCal` に追加し、non-FSZ / FSZ / MPI を分けて gate する。
   - BackFlow との組み合わせは、現段階では C と同様に unsupported/warn-equivalent policy を決める。

9. NBodyG measurement

   - `nbodyg.def` parser / data model / namelist wiring を追加する。
   - non-FSZ / FSZ の general `GreenFuncN` evaluator を実装し、`N=1/2` が既存
     Green output と一致する unit / integration gate を作る。
   - `zvo_NBodyG_yyy.dat` output と MPI reduce-to-rank0 を実装する。
   - BackFlow との組み合わせは C と同様に fail-fast する。

10. NBodyInterAll local energy

   - `nbodyinterall.def` parser / data model / MPI broadcast 相当を追加する。
   - complex normal / FSZ path から `GreenFuncN` を使って local-energy contribution を足す。
   - C と同様に real / BackFlow / Lanczos path は初期 scope では reject する。
   - NBodyG の `GreenFuncN` primitive と fixtures が揃ってから着手する。

11. t-J / doublon-only sampling sectors

   - C v1.4.0 の `NExUpdatePath = 4/5` t-J update と `NExUpdatePath = 6`
     doublon-only pair hoppingを別 design として読む。
   - `NCond` convention、BackFlow / LocSpin / Lanczos / RBM / FSZ restrictions、
     sector-preserving Green tests を整理する。
   - Sampling move semantics を変えるため、RBM / reweight / Twist / NBodyG とは別 PR にする。

12. `NSRCG = 1` tolerance follow-up

   - 実 run の `operate_by_S` 入力 (`stcOs`, `x`, scalar correction) を保存した
     C/Julia 比較を作る。
   - Linux / OpenBLAS / Julia 1.12 で post-CG residual distribution を測る。
   - coarse e2e tolerance に頼りすぎない semantic gate を設計する。
   - 例: `xdot` source-order unit, SR-CG residual diagnostics, parameter layout guard,
     first-row energy tight gate の組み合わせ。

13. C-mVMC upstream / parity feedback

   - `NSplitSize > 1`、`NStore`、`NSRCG` の C 側方針が public C release へ入った後、
     Julia 側 docs の parity wording を更新できるか確認する。
   - SR-CG x split の C-side behavior は、Julia 側では引き続き unsupported として分離する。
   - 必要なら再現手順付き upstream issue / note を作る。

### P1 candidates

1. `VMCPhysCal` production hardening

   - 現在の experimental status を解除できる条件を定義する。
   - FSZ caveat, product-side two-body Green, unsupported inputs, output contract を再確認する。
   - C reference fixture を増やし、manual の「experimental」文言を解除できるか判断する。

2. Threading and performance follow-up

   - default / opt-in / OMP guard を CI か release gate に追加する。
   - `JULIA_NUM_THREADS=1/2/4` default numerical gate と L16/L24/L32 benchmark sweep を更新する。
   - native PfaPack の static state と `JULIA_MVMC_PFAPACK_THREADS` の扱いを切り分ける。
   - sample-level independent-chain threading は、C fidelity とは別 design として扱う。

3. Packaging and install path

   - `Pkg.add(url=..., subdir=...)` を将来 support するか判断する。
   - submodule coupling, workspace `[sources]`, package registration, native artifacts の方針を整理する。
   - GitHub-generated source archives が submodule を含まない問題の user-facing mitigation を強化する。

4. Release engineering

   - release-gate command set を script 化する。
   - clean clone / submodule / Manifest / examples 50-step / Genkai MPI gate の再現手順をまとめる。
   - `main` / `develop` / tag / release branch の運用方針を明文化する。
   - stale feature branches / remote branches の cleanup 方針を決める。

## Beyond v0.5

- BackFlow support.
- `multiDef` / multi-model workflow support.
- independent-chain sampling / stochastic semantics redesign.
- Standard-mode input generation or parser/writer expansion.
- Package registration / artifact-based native dependency distribution.
- Larger benchmark matrix on ohtaka / Genkai / local macOS.

## Suggested next step

PR #45 は CI green 後に develop へ merge 済み。
次は、v0.5.0 必須残件として **V05-1 R2/R3
(`NLanczosMode = 2` Lanczos Green)** と **V05-4 PhysCal / Green residuals**
を進める。これらは v0.5.0 から先送りしない。

Recommended immediate target:

1. **V05-1 Full Lanczos R2/R3 plan**: `NLanczosMode = 2` の Lanczos Green list reconstruction と
   `zvo_ls_cisajs*` output を扱う。既存の Full Lanczos plan を正として更新する。

Mandatory v0.5 targets:

2. **V05-4 PhysCal NSplitSize / Green residual design**: `VMCPhysCal NSplitSize > 1`、
   FSZ `TwoBodyGEx`、Lanczos Green output を扱う。
3. **V05-3 NSplitSize + SR-CG split design**: `NSplitSize > 1` with SR-CG を扱う。
4. **V05-2 PairHop closeout**: PR #44 後の manual / non-FSZ / FSZ scope を確認し、
   必要なら補足 gate を追加する。
5. **v0.5.0 release prep**: C v1.2 parity items が揃った段階で BackFlow out-of-scope、
   `NLanczosMode = 2` / PhysCal-Green support scope、残り caveat を明記して進める。

Post-v0.5 / v1.4+ targets:

6. **RBM `initial.def` block plan**: C `GeneralRBM_cmp` を基準に、RBM triples loader と ctest gate を設計する。
   RBM の実行 path はすでに大部分があるため、v1.4+ parity cleanup として扱う。
7. **reweight design / plan**: C v1.4.0 の `reweight` keyword と weighted SR accumulation を設計する。
8. **Twist/Lattice PhysCal design**: C v1.4.0 の `lattice.def` / `twist.def` / `zvo_twist_yyy.dat` を基準に設計する。
9. **NBodyG design / plan**: C PR #108 を基準に parser・`GreenFuncN`・output・MPI reduce を設計する。
   NBodyInterAll と BackFlow の土台になるため、post-v1.4.0 develop feature intake の最初に置く。
10. **NBodyInterAll design / plan**: C PR #109 を基準に local-energy `GreenFuncN` contribution を設計する。
   NBodyG の primitive を先に入れてから着手する。
11. **t-J / doublon-only sampling-sector design**: C v1.4.0 の `NExUpdatePath = 4/5/6`、
   `NCond` convention、unsupported combinations、sector-preserving gates を整理する。
12. **BackFlow design**: newer C implementation track が固まった後に、初期 supported scope を Julia 側でも独立設計する。
13. **SR-CG tolerance follow-up**: coarse tolerance の背景をさらに診断し、semantic gate を増やす。

次に作るべき文書は、選んだ主題に対応する design / plan:

- V05-1 Full Lanczos:
  正の計画書は `docs/plans/2026-06-23-julia-mvmc-full-lanczos-plan.md`
- V05-2..V05-4 / v1.4+ feature topics:
  `docs/specs/YYYY-MM-DD-julia-mvmc-v0.5-<topic>-design.md` and
  `docs/plans/YYYY-MM-DD-julia-mvmc-v0.5-<topic>-plan.md`
