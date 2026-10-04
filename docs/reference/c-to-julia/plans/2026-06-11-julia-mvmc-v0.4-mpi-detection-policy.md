---
date: 2026-06-11
datetime: 2026-06-11 01:38 JST
model: Claude Fable 5
status: design
topic: Julia-mVMC v0.4 MPI 起動検出 policy の改訂（review F3/F5 対応、ohtaka spike 前提）
review: docs/reviews/2026-06-11-julia-mvmc-v0.4-mpi-r0-implementation-review.md (F3/F5)
spec: docs/specs/2026-06-10-julia-mvmc-v0.4-mpi-design.md (§4.1 の改訂提案)
---

# MPI 起動検出 policy の改訂設計（F3/F5）

## 問題（review F3/F5）

現行 `parallel.jl` の検出は
`MPI_ENV_KEYS = ("OMPI_COMM_WORLD_SIZE", "PMI_SIZE", "PMI_RANK")` の **key 存在**のみ。

- **F3（under-detection、High）**: srun 直接起動（PMIx。OpenMPI on SLURM の標準
  構成）は `PMIX_RANK` / `SLURM_PROCID` / `SLURM_NTASKS` を設定するが上記 3 キーは
  設定しない → 全 rank が `:serial` 判定 → 全員 `is_output_rank == true` → 同一
  `zvo_*.dat` へ並行書き込み（silent corruption）。F12 guard は `:serial` のままなので
  到達不能。ohtaka は srun 運用のため、site の SLURM MpiDefault が pmix の場合に
  直撃する。
- **F5（over-detection、Medium）**: SLURM MpiDefault=pmi2 では slurmstepd が
  **非 MPI の task にも** `PMI_RANK`/`PMI_SIZE`/`PMI_FD` を注入する。`srun -n 1 julia`
  （v0.3 では純 serial）が `:mpi` 判定になり、`MPI.Init()` + is_mpi sync 経路を通る
  （挙動変化 + site PMI への MPICH_jll Init 失敗リスク）。

## 設計方針

検出は「ヒントとして使い、矛盾したら fail-fast」に改める。**silent corruption を
作らないことを最優先**にし、確信が持てないケースは error で止めてユーザに
`JULIA_MVMC_MPI` の明示を求める。

### 提案する判定表（`resolve_mpi_mode` の改訂）

判定材料:

- `mpi_launch_detected()`: 「MPI launcher 配下」の強いシグナル =
  `OMPI_COMM_WORLD_SIZE`（mpirun/prte）、`PMI_SIZE`（hydra/pmi2、ただし下記の
  多重 task 条件と組み合わせ）、`PMIX_RANK`（PMIx）**を追加**。
- `slurm_multi_task()`: `SLURM_NTASKS`（または `SLURM_NPROCS`）> 1。
  「複数 task が同時起動されている」ことの直接シグナル（launcher 非依存）。

| 条件 | 判定 |
|------|------|
| `JULIA_MVMC_MPI=1` | `:mpi`（従来どおり、無条件） |
| `JULIA_MVMC_MPI=0` + 何らかの検出 | `:mpi_guarded_serial`（従来どおり: size>1 abort） |
| `JULIA_MVMC_MPI=0` + 未検出 | `:serial` |
| auto + `OMPI_COMM_WORLD_SIZE` or `PMIX_RANK` あり | `:mpi` |
| auto + `PMI_SIZE` あり **かつ** (`PMI_SIZE>1` or `slurm_multi_task()`) | `:mpi` |
| auto + `PMI_SIZE==1` かつ単 task | `:serial`（**F5 対策**: pmi2 注入の単独実行を serial 扱い） |
| auto + 上記いずれも不検出 **かつ** `slurm_multi_task()` | **error（fail-fast）**（**F3 対策**） |
| auto + 完全未検出 | `:serial`（従来どおり） |

F3 対策行の error message には「srun の MPI plugin（pmi2/pmix）が認識できなかった。
MPI で走らせる場合は `JULIA_MVMC_MPI=1`、単独 process × job array 等の意図的な
非 MPI 実行なら `JULIA_MVMC_MPI=0` を設定せよ」を明記する。

### F5 の補足判断

`PMI_SIZE==1` 単 task を `:serial` に落とすのは「v0.3 と同じ挙動を保つ」ため。
`JULIA_MVMC_MPI=1` 明示時は従来どおり size-1 MPI として動く（A7 の test 維持）。
`PMIX_RANK` は PMIx が rank 概念を持つ process にのみ注入するため、`:mpi` 直行で
よい（pmi2 のような無差別注入はしない）。ohtaka spike で実測し、必要なら
`PMIX_SIZE`/`PMIX_LOCAL_SIZE` の併用条件を追加する。

### 変更対象

- `MVMCOptimizers.jl/src/parallel.jl`: `MPI_ENV_KEYS` 廃止 →
  `mpi_launch_detected()` / `slurm_multi_task()` の 2 関数 + 判定表の実装。
  `mpi_env_detected()` は後方互換のため「強シグナルのみ」へ再定義して残す
  （F7 guard と smoke が使用）。
- `test_unit_parallel.jl`: 判定表の withenv 全行 test（PMIX_RANK、PMI_SIZE=1 単 task、
  SLURM_NTASKS=2 + 無シグナル → error、等）。
- spec §4.1 の表を本設計で置換（spec 改訂は実装 PR と同時）。

### 実装しないこと

- `MPI.Init()` を呼んでからの size ベース判定（「Init してみて size を見る」）は
  採用しない。非 MPI 環境での MPICH_jll Init は環境次第で hang し得るため、
  env ベース + fail-fast を維持する。
- SLURM 以外の scheduler（PBS_NODENUM 等）への対応は ohtaka に不要なので R1 以降。

## 実施順序

1. 本設計のユーザ / 独立レビュー確認
2. `parallel.jl` 実装 + unit test（withenv で全行）
3. macOS 上の回帰（unit / integration / mpiexec smoke — hydra は `PMI_SIZE>1` 経路）
4. その後に Task 10（ohtaka spike）: 「srun --mpi=pmi2 / pmix それぞれで rank env を
   実測 → 判定表の妥当性確認」を spike 項目に追加

## 未決事項（spike で確定）

- ohtaka の SLURM MpiDefault（pmi2 / pmix / none）と、srun 配下で実際に注入される
  env key の一覧
- ohtaka の system MPI（MPIPreferences 設定後）で `MPI.Init()` が pmi2/pmix 双方の
  起動形態に対して成功するか
