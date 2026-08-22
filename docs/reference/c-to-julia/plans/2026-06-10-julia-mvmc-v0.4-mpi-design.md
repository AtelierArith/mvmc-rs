---
date: 2026-06-10
datetime: 2026-06-10 19:19 JST
model: Claude Fable 5
status: design
topic: Julia-mVMC v0.4 MPI 並列（sample 並列 + NSplitSize）の設計
revision: |
  v3.1 (2026-06-10 19:19 JST): v3 再レビュー（B1-B2、Low）を反映。FSZ 側 Counter reset
  の C 参照行を `vmcmake_fsz.c:142` / `vmcmake_fsz_real.c:133` へ訂正（`vmcmake.c:679`
  は BackFlow 側、B1）、counter writeback test に global rank0 の Counter 出力 parity
  assertion を追加（B2）。
  v3 (2026-06-10 19:11 JST): v2 追加レビュー（同レビュー文書の「追記: v2 再レビュー」、
  A1-A8）を反映。unsupported 条件を `NSplitSize>1 && (NStore!=0 || NSRCG!=0)` へ拡大
  （A1）、ReduceCounter writeback root を `ctx.rank2 == 0` へ訂正し誤った「二重計上」
  理由付けを削除（A2）、C の MakeSample 内 InvM/PfM local-compressed layout と Julia
  global layout + adapter 方針（A3）、既存 unsupported mode の gate 残し（A4）、
  `RndSeed` 解決表（A5）、依存更新範囲（A6）、`JULIA_MVMC_MPI` 3 値 semantics（A7）、
  `split_loop` edge case test（A8）。
  v2 (2026-06-10 18:09 JST): レビュー
  `docs/reviews/2026-06-10-julia-mvmc-v0.4-mpi-design-review-and-mitigation.md`
  （F1-F13）の対応策を反映。
  v1 (2026-06-10 17:39 JST): 初版（brainstorming 合意内容）。
---

# Julia-mVMC v0.4 MPI 並列 設計文書

brainstorming（superpowers/brainstorming skill 準拠）でユーザと合意した v0.4 の設計を記録する。
v2 でレビュー（F1-F13）、v3 で v2 追加レビュー（A1-A8）の対応策を全面反映した。
レビューは `docs/reviews/2026-06-10-julia-mvmc-v0.4-mpi-design-review-and-mitigation.md`
（GPT-5 Codex）。各節はユーザ承認済み。

## 1. 目的とスコープ

C-mVMC の MPI 並列のうち、以下を Julia-mVMC（実体は `Julia-mVMC/MVMCOptimizers.jl`）へ導入する。

- **sample 並列**: group（= 独立 chain）ごとに独立 Markov chain を走らせ、物理量・SR moments を
  全 rank で重み付き平均する（C-mVMC の NSplitSize=1 デフォルト動作）。
- **NSplitSize（QP 分割）**: 同一グループ内の rank が同じ chain を共有し、`VMCMakeSample` の
  QP 重み計算と `VMCMainCal` の sample 処理を分担する。現在 error にしている `NSplitSize > 1`
  を制限付きで解除する（下記 unsupported combinations を参照）。

**スコープ外（defer / unsupported）**:

- multiDef mode（複数 parameter dir の一括実行、`initMultiDefMode` 相当）→ v0.5 以降。
- ScaLAPACK 相当（`stcopt_pdposv.c`）→ 対象外。非 ScaLAPACK path（冗長 solve）に合わせる。
- `VMCMakeSample` の independent-chain threading（v0.3 で defer 済み）→ 本設計の MPI
  sample 並列がその役割を C 忠実な形で担う。
- **`NSplitSize > 1 && (NStore != 0 || NSRCG != 0)`** → v0.4 では **unsupported
  （実行前に明示 error）**。C の store 経路は `NStoreO != 0` だけでなく `NSRCG != 0` でも
  使われ（`mVMC/src/mVMC/vmccal.c:227`, `310`）、store slot semantics に疑義があるため
  （§3.1 / F2 / A1）。**`NSRCG=1` の MPI 対応は `NSplitSize=1` のみが対象**。
- **`useDiagScale != 0` / `RescaleSmat != 0`（SR-CG）** → 実装しない場合は unsupported
  （実行前に明示 error）。
- **既存 runtime が serial で既に unsupported とする mode は v0.4 でも unsupported のまま**
  （A4）: `NLanczosMode > 0`（`Julia-mVMC/MVMCOptimizers.jl/src/unsupported_inputs.jl:54-62`）、
  BackFlow 入力（MakeSample / MainCal stub error）、FSZ / general-orbital mode の
  `TwoBodyGEx`（`Julia-mVMC/MVMCOptimizers.jl/src/green_func_calc.jl:105-114`）、その他現行
  runtime が reject する機能。MPI 化で support 範囲は広がらない。

## 2. C 忠実性の目標

**同 rank 数で数値一致**を目標とする（v0.1〜v0.3 の ctest_equivalent 路線の拡張）。

- rank ごとの SFMT seed 生成・Markov chain・reduce の意味論を C と一致させる。
- C-mVMC を同じ `mpiexec -n N` で実行した結果（`zvo_out.dat` 等）と許容誤差内で一致することを
  gate とする。
- ただし MPI 実装間（MPICH / OpenMPI）で allreduce の縮約順序が異なりうるため、bit 一致では
  なく従来水準の tolerance gate とする。
- C 側の挙動自体に疑義がある組み合わせ（§3.1）は「C を再現する」のではなく unsupported と
  して切り分ける（correctness 優先、レビュー F2 の推奨 2 を採用）。
- 内部データ layout は C と一致させることを目的にしない（外部挙動の一致が目標）。C と layout
  が異なる箇所は明示し、adapter test で同じ計算対象であることを保証する（§5-4 / A3）。

## 3. C-mVMC の MPI 構造（確認済みの事実）

設計の根拠として、C 側の構造を記録する。

- **communicator 分割**（`mVMC/src/mVMC/vmcmain.c:239-246`）:
  - `group1 = rank0 / NSplitSize`、`MPI_Comm_split(comm0, group1, ...)` → `comm1`
    （同一グループ内、**QP / sample 分担用**）。
  - `group2 = rank1`、`MPI_Comm_split(comm0, group2, ...)` → `comm2`
    （グループ横断、**counter 集約用**）。comm2 は rank1 値ごとに複数存在する。
  - comm1 の size は通常 NSplitSize。ただし `size0 % NSplitSize != 0` のとき C は
    **warning のみで継続**し、最後の group の comm1 は NSplitSize より小さくなる
    （`mVMC/src/mVMC/vmcmain.c:248-250`、F8）。Julia も error にしない。
  - NSplitSize=1 では comm1 = self、comm2 = comm0 に退化し、純 sample 並列になる。
- **RNG seed**: `init_gen_rand(RndSeed + group1)`（`mVMC/src/mVMC/vmcmain.c:257`）。
  同一グループ内の rank は同じ seed = 同じ chain を冗長に歩く。**独立 chain 数は group 数**
  （= `ceil(size0 / NSplitSize)`）であり rank 数ではない（F1）。
  `RndSeed` の決定は: default `11272`（`mVMC/src/mVMC/readdef.c:1967`）、入力で明示されれば
  その値（`RndSeed == 0` も seed 0 のまま使う）、**`RndSeed < 0` のときだけ** `time(NULL)`
  へ置換（`mVMC/src/mVMC/readdef.c:2161-2163`、A5）。
- **SplitLoop**（`mVMC/src/mVMC/splitloop.c:37-60`）: `mpiSize >= loopLength` のとき
  `rank >= loopLength` の rank は**空 range** `[loopLength, loopLength)` を受け取る（A8）。
- **VMCMakeSample(comm1)**: 冒頭で `Counter[0:Counter_max)` を毎回 reset する
  （`mVMC/src/mVMC/vmcmake.c:144`、FSZ 側は `mVMC/src/mVMC/vmcmake_fsz.c:142`、
  `mVMC/src/mVMC/vmcmake_fsz_real.c:133`。なお `mVMC/src/mVMC/vmcmake.c:679` の reset は
  BackFlow 側 `VMC_BF_MakeSample` のもので、v0.4 では BackFlow unsupported のため参考情報、
  B1）。`SplitLoop(&qpStart, &qpEnd, NQPFull,
  rank1, size1)` で **QP を分割**（`mVMC/src/mVMC/vmcmake.c:66`）。エラー flag は
  `MPI_Allreduce(MAX)` → `MPI_Abort`（`mVMC/src/mVMC/vmcmake.c:413-420`）。
  QP split は IP 計算と不可分: `CalculateLogIP_*` / `CalculateIP_*` は担当 QP の partial sum
  を comm1 で `MPI_Allreduce(SUM)` し、accept/reject 判定は全 rank が同じ global IP を使う
  （`mVMC/src/mVMC/qp.c:90-127`、`mVMC/src/mVMC/qp_real.c:34-70`、F5）。
- **MakeSample 中の配列 layout（A3）**: C は `SlaterElm` を global QP index
  `qpidx + qpStart` で読み（`mVMC/src/mVMC/matrix.c:345`）、`InvM` / `PfM` は
  **local-compressed**（担当 chunk を配列先頭に詰める: `InvM + qpidx*Nsize*Nsize`、
  `PfM[qpidx]`）で書く（`mVMC/src/mVMC/matrix.c:348`, `377`）。`CalculateLogIP` /
  `CalculateIP` は local-compressed `pfM[qpidx]` と global `QPFullWeight[qpidx+qpStart]` を
  組み合わせる（`mVMC/src/mVMC/qp.c:90-127`）。
- **VMCMainCal(comm1)**: `SplitLoop(&sampleStart, &sampleEnd, NVMCSample, rank1, size1)` で
  **sample を分割**（`mVMC/src/mVMC/vmccal.c:108`）。QP は全範囲で計算し、IP は
  `CalculateIP(..., MPI_COMM_SELF)`（`mVMC/src/mVMC/vmccal.c:143-145`）。
  **MainCal 内で comm1 allreduce してはいけない**（F5）。
- **SR moments の経路選択（A1）**: direct OO 計算は `NSRCG==0 && NStoreO==0` のときのみ
  （`mVMC/src/mVMC/vmccal.c:227`）。`NSRCG!=0 || NStoreO!=0` では `SROptO_Store` ベースの
  `calculateOO_Store*` 経路に入る（`mVMC/src/mVMC/vmccal.c:310`）。つまり **`NSRCG=1` は
  `NStore=0` でも store 経路を使う**。
- **WeightAverageWE / WeightAverageSROpt(comm_parent=comm0)**: 重み Wc 込みの和を
  allreduce → 除算。全 rank が結果を持つ。
- **WeightAverageGreenFunc(comm0)**: こちらは **`MPI_Reduce`（reduce-to-rank0）** であり、
  normalized result を持つのは rank0 のみ（`mVMC/src/mVMC/average.c:209-245`、F9）。
- **ReduceCounter(comm_child2)**: `Counter[6]`（`Counter_max=6`、hopping try/accept,
  exchange try/accept, local spin flip try/accept）のみを comm2 で allreduce する
  （`mVMC/src/mVMC/include/global.h:369-373`、`mVMC/src/mVMC/vmcmake.c:496-508`）。
  **書き戻しは渡された communicator（comm2）内の rank 0 のみ**（`MPI_Comm_rank(comm,&rank)`
  後の `if(rank==0)`）。global rank0 ではない点に注意: 例えば size0=4, NSplitSize=2 では
  rank2==0 となるのは global rank 0 と 1（group 0 の全 rank）である（A2）。
  Counter は VMCMakeSample 冒頭で毎 step reset されるため、writeback root の違いは累積には
  影響しないが、C 忠実性（どの rank がどの counter 値を持つか）に影響する。
- **StochasticOptCG(comm_parent)**: `fn_operate_by_S`（`mVMC/src/mVMC/stcopt_cg_impl.c:481-545`）
  が rank-local sample store に対する GEMV 2 発 + `SafeMpiAllReduce` で S·x を組み立てる。
  S 行列は陽に作らない。x は反復ごとに `MPI_Bcast`（root=0）。normalization は
  `invW = 1.0 / Wc`（Wc は `WeightAverageWE(comm0)` 後の global sum）。
  C には `useDiagScale`（`mVMC/src/mVMC/stcopt_cg_impl.c:189-192`）と `RescaleSmat`
  （`fn_Rescale4SRCG`）の分岐がある（F7）。
- **SyncModifiedParameter(comm0)**（`mVMC/src/mVMC/parameter.c:134-175`）: 冒頭で
  `MPI_Bcast(Para, NPara, MPI_DOUBLE_COMPLEX, 0, comm)` を行い、その後全 rank が同じ
  parameter に対して shift（DH/GJ）→ rescale（Slater）→ normalize（OptTrans）を実行（F4）。
- **dposv path**（NSRCG=0）: `SROptOO` 等は WeightAverage 済みで全 rank 同値 → 冗長 solve。
- **出力**: `if(rank==0) outputData()`。
- **SafeMpiAllReduce**（`mVMC/src/mVMC/safempi.c`）: 大バッファを `D_MpiSendMax` で chunk
  分割して allreduce。
- **NVMCSample は「chain（= group）あたり」の保存サンプル数**（F1）:
  - NSplitSize=1 では group 数 = rank 数なので総独立サンプル数は `NVMCSample × size0`。
  - NSplitSize>1 では同一 group 内 rank は同じ chain を分担し、総独立サンプル数は
    `NVMCSample × group 数`。

### 3.1 C 側の疑義: store 経路（`NStoreO != 0 || NSRCG != 0`）と `NSplitSize > 1`（F2 + A1）

C の `VMCMainCal` は sample split した各 rank が **global sample index** で store へ書く
（`SROptO_Store_real[int_i + sample*SROptSize]`、`mVMC/src/mVMC/vmccal.c:241`）一方、
`calculateOO_Store_real(..., sampleSize)` は store の**先頭から** `sampleSize = sampleEnd -
sampleStart` 個を読む（`mVMC/src/mVMC/vmccal.c:310-318`、`660-688`）。`sampleStart > 0` の
rank は自分が書いた slot を読まず、group 内 rank0 だけ偶然整合する。

この store 経路は `NStoreO != 0` だけでなく **`NSRCG != 0` でも使われる**
（`mVMC/src/mVMC/vmccal.c:310`、A1）。SR-CG の `fn_operate_by_S` も `SROptO_Store` を前提に
する（`mVMC/src/mVMC/stcopt_cg_impl.c:481-545`）。

レビューの実走確認（heisenberg_chain_real、`OMP_NUM_THREADS=1`）でも、`NStore=1` では
`mpiexec -n 1, split=1` と `-n 2, split=2` の `zvo_SRinfo.dat` が大きく異なり、`NStore=0`
では一致した。

**v0.4 の方針（レビュー推奨 2 + A1 を採用）**: `NSplitSize > 1 && (NStore != 0 ||
NSRCG != 0)` は unsupported として実行前に明示 error にする。`NSRCG=1` の MPI 対応は
`NSplitSize=1` のみを対象とする。C 完全忠実（疑義込み再現）は採らない。C 側の挙動は R0 で
正式に実走確認をタスク化し、結論が出るまで release gate に入れない。upstream（C-mVMC）への
issue 報告は v0.4 とは切り離して別途検討する。

## 4. アーキテクチャ（§1、承認済み + F3/F12/F13/A6/A7/A8 反映）

**方式: C 構造直写し + 薄い通信層**（backend 抽象化・MPI 直書きは不採用）。

### 4.1 `MVMCOptimizers.jl/src/parallel.jl`（新規、1 ファイル完結）

```julia
struct ParallelContext
    is_mpi::Bool
    comm0; comm1; comm2          # MPI.Comm（serial 時は nothing）
    rank0::Int; size0::Int       # 全体
    rank1::Int; size1::Int       # comm1: QP / sample 分担（グループ内）
    rank2::Int; size2::Int       # comm2: counter 集約（グループ横断）
    group1::Int                  # = rank0 ÷ NSplitSize（seed offset）
end
```

- **起動判定と `JULIA_MVMC_MPI` の 3 値 semantics（F12 + A7、F3/F5 改訂）**:
  検出は `docs/specs/2026-06-11-julia-mvmc-v0.4-mpi-detection-policy.md` の判定表に従う
  （review 2026-06-11 F3/F5 対応。本表は同設計で置換済み）。判定材料は
  `mpi_launch_detected()`（強シグナル `OMPI_COMM_WORLD_SIZE` / `PMIX_RANK`、および
  `PMI_SIZE` + 多重 task 条件）と `slurm_multi_task()`（`SLURM_NTASKS` /
  `SLURM_NPROCS` > 1）。

  | 条件 | 判定 |
  |------|------|
  | `JULIA_MVMC_MPI=1` | `:mpi`（無条件。size 1 でも MPI mode、Init 失敗は error） |
  | `JULIA_MVMC_MPI=0` + launch 検出 | `:mpi_guarded_serial`（MPI.Init 後 size>1 なら rank0 が error + `MPI.Abort`、size==1 なら serial fallback） |
  | `JULIA_MVMC_MPI=0` + 未検出 | `:serial`（job array 等の意図的な非 MPI 実行） |
  | auto + `OMPI_COMM_WORLD_SIZE` or `PMIX_RANK` あり | `:mpi` |
  | auto + `PMI_SIZE` あり かつ (`PMI_SIZE>1` or `slurm_multi_task()`) | `:mpi` |
  | auto + `PMI_SIZE==1` かつ単 task | `:serial`（F5: pmi2 の無差別注入対策） |
  | auto + 無シグナル かつ `slurm_multi_task()` | error（F3 fail-fast: `JULIA_MVMC_MPI` の明示を要求） |
  | auto + 完全未検出 | `:serial` |

  SerialContext（is_mpi=false、全 size=1、rank=0）では **MPI API を一切呼ばない**。
- **MPI lifecycle（F13）**: `MPI.Initialized()` guard を使い、二重 Init を避ける。library
  code では明示 `MPI.Finalize()` を呼ばない（MPI.jl が Julia exit 時に自動 finalize する。
  同一 process 内の複数 test / 他 package との衝突を避ける）。fatal collective mismatch では
  `MPI.Abort(ctx.comm0, code)`。test runner は MPI.jl の `mpiexec()` ラッパを使い、project
  の MPI backend と一致させる。
- **通信 wrapper**: `allreduce_sum!` / `reduce_sum_to_root!` / `bcast!` / `barrier` /
  `reduce_counter!` / `abort` を parallel.jl に集約。SerialContext では no-op。
  C の `SafeMpiAllReduce` 相当の chunk 分割 allreduce も同所に実装し、chunk サイズは C の
  `D_MpiSendMax` に合わせる。
- **`split_loop(n, rank, size)`**: C の `SplitLoop` と同一の分割式を再現する（parity の要）。
  unit test で以下の edge case を C と一致させる（F8 + A8）:
  - `size0 % NSplitSize != 0`（非整除、例 `size0=3, NSplitSize=2`）
  - `loopLength = 1, mpiSize = 4`（空 range の rank が出る）
  - `NQPFull < size1` / `NVMCSample < size1`（空 QP / sample range）
  - `loopLength = 0` は Julia 側で許すか error にするかを R0 で明示する
- **依存（A6）**: 以下を R0 で更新する。
  - `Julia-mVMC/MVMCOptimizers.jl/Project.toml` の通常 deps に MPI.jl を追加
    （package extension / 別パッケージ化は不採用）。
  - root workspace `Julia-mVMC/Project.toml` にも、test runner が `MPI.mpiexec()` を使う
    起動形態に合わせて MPI.jl / MPIPreferences の追加要否を確認し反映する。
  - `Julia-mVMC/Manifest-v1.11.toml`（Julia 1.11 reproducibility snapshot）の更新方針を
    決め、CI の Julia 1.11 / 1.12 で resolve 差がないことを確認する。
  - MPI.jl は既定で MPICH_jll を同梱しビルド不要。ohtaka では MPIPreferences で system MPI
    に切替（ドキュメント化、R0 で疎通確認）。

### 4.2 MPI context の導入位置（F3）

既存 entry point（`Julia-mVMC/MVMCOptimizers.jl/src/run_para_opt_from_namelist.jl:117-184`）は
`vmc_para_opt!` より前に seed → `init_parameter!` → initial.def / In*.def overlay →
`sync_modified_parameter!` → `init_qp_weight!` を実行する。C は communicator split 後に
`init_gen_rand(RndSeed + group1)` → `InitParameter()` → overlay → `SyncModifiedParameter(comm0)`
の順である。したがって:

- `ParallelContext` は **high-level runner の parse 直後、seed / `init_parameter!` の前**に作る。
- seed は runner が `actual_seed = resolved_RndSeed + ctx.group1` で作る（解決表は §5-1）。
- `vmc_para_opt!` に ctx を渡すだけでは不十分で、`run_para_opt_from_namelist` /
  `run_phys_cal_from_namelist` 自体を MPI-aware にする。
- 低レベル `vmc_para_opt!(data; rng=...)` は後方互換の serial API として残し、MPI parity は
  high-level runner を正とする。

## 5. データフロー（§2、承認済み + F4-F11/A2/A3/A5 反映）

`vmc_para_opt!` / `vmc_phys_cal!` の各段に ctx を通す。

1. **RNG（A5 で seed 解決表を明確化）**: MPI-aware high-level runner の seed 解決は C に
   揃える:

   ```text
   RndSeed missing/default -> 11272（注 2026-06-11 review F2: Julia parser の
                              kwdef default は 12345 だったため R0 実装で 11272 へ
                              修正。C は readdef.c:1967）
   RndSeed < 0             -> rank0 で C 相当の時刻 seed を決めて comm0 bcast
   RndSeed == 0            -> seed 0（現行 Julia の `> 0 ? : 11272` fallback は 0 を
                              11272 に化けさせるため、MPI runner では C に合わせて修正）
   RndSeed > 0             -> 指定 seed
   actual_seed             -> resolved_RndSeed + ctx.group1
   ```

   SFMT.jl は C と bit 一致のため、同 rank 数で C と同一 chain が走る。
2. **parameter sync（F4）**: `sync_modified_parameter!(ctx, data)` を追加し、C と同じ順序
   （rank0 parameter vector の comm0 bcast → DH/GJ shift → Slater rescale → OptTrans
   normalize）にする。初期化後と各 SR step 後の両方で呼ぶ。Julia の structured parameter
   arrays と C の contiguous `Para` 相当の対応は pack/unpack helper として明示する。
3. **MakeSample**: 既存実装に `qp_start:qp_end`（comm1 の `split_loop`）を導入。
   `calculate_log_ip_fcmp/_real`・`calculate_ip_fcmp/_real` に ctx.comm1 を渡し、partial sum
   の comm1 allreduce を実装する（F5。現行 Julia 実装は single-process 前提で MPI
   communication skipped と明記されている: `Julia-mVMC/MVMCOptimizers.jl/src/vmc_sampling.jl:1620-1751`）。
   エラー flag の allreduce(MAX) → abort も C と同位置に再現。NSplitSize=1 では全範囲となり
   現行コードと一致。
4. **QP index convention（F6 + A3）**: Julia の低レベル `calculate_m_all_*` は「配列は QP
   範囲に slice 済み・関数内 index 1 から」という契約
   （`Julia-mVMC/MVMCOptimizers.jl/src/calculate_m_all.jl:256-264`）、一方 update 系
   （`calculate_new_pf_m2!` 等）は global QP index 前提であり、契約が混在している。
   R3 の前に以下の convention に統一する:

   ```text
   state arrays（inv_m / pf_m / slater_elm）は全 QP 分を global index で保持する。
   QP split rank は [qp_start, qp_end) のみを書き換える。
   low-level PfaPack wrapper へ渡すときだけ local view に変換する。
   IP partial sum は global weight[qp] と global pf_m[qp] から計算する。
   ```

   **この global layout は C の MakeSample 内部 layout（InvM / PfM を local-compressed で
   先頭詰め、SlaterElm / QPFullWeight は global、§3 / A3）とは異なる**。Julia は安全性を
   優先して global layout を採り、C と同じ計算対象になることを adapter（local view 変換）と
   unit test で保証する。R3 の unit test には以下を含める:
   - `qp_start > 0` の rank で write/read 位置が正しい
   - `NQPFull < size1` で空 QP range の rank が壊れない
   - `CalculateMAll` / `CalculateNewPfM*` / `UpdateMAll*` / `CalculateLogIP` /
     `CalculateIP` 相当が同じ QP convention を共有する
5. **MainCal**: sample range を comm1 で `split_loop`。QP は全範囲・IP は COMM_SELF 相当
   （comm1 allreduce しない、F5）。v0.3 の per-worker accumulator 構造
   （`VMCThreadAccumulator`）を rank-local accumulator として流用し、merge 先を thread-merge
   から MPI allreduce へ置き換える。SR store の layout（rank-local か global slot か）は
   R3 で明文化する（C の疑義 §3.1 を踏まえ、Julia は自己整合な layout を選ぶ）。
6. **WeightAverage**: `WeightAverageWE` / `WeightAverageSROpt(_real)` 相当を comm0 の
   `allreduce_sum!` で実装（Wc 込み和の allreduce → 除算、C と同順序）。
   **Green function は C と同じ reduce-to-rank0**（`reduce_sum_to_root!`）とし、非 rank0 の
   Green arrays は local のままでよい（F9。rank0 のみ output するため十分）。
7. **counter（F10 + A2）**: `reduce_counter!(ctx, counter)` は **comm2 で allreduce し、
   `ctx.rank2 == 0` の rank だけへ先頭 6 counters を書き戻す**（C の
   `MPI_Comm_rank(comm,&rank); if(rank==0)` は comm2 内 rank であり global rank0 ではない、
   §3 参照）。テストでは `size0=4, NSplitSize=2` のように rank2==0 が global rank 0 以外
   （global 0 と 1）にも存在する case を確認する。
   Julia が burn flag として流用している `counter[11]`
   （`Julia-mVMC/MVMCOptimizers.jl/src/vmc_para_opt.jl:187-192`）は reduction 対象外。
   可能なら burn flag を counter 配列から分離する（最善策）。
8. **SR solve**:
   - **NSRCG=0（dposv 相当）**: `SROptOO` は WeightAverage で全 rank 同値のため、現行
     `stochastic_opt!`（`Julia-mVMC/MVMCOptimizers.jl/src/stochastic_opt.jl:227`）の serial
     solve を各 rank で冗長実行。**solve 本体は変更ゼロ**。
   - **NSRCG=1（SR-CG、F7 + A1）**: **対象は `NSplitSize=1` のみ**（`NSplitSize>1` との
     組み合わせは §3.1 により unsupported）。**serial SR-CG の C parity gate を先に確立し、
     MPI 化はその後**とする。現行 Julia CG は `inv_w = 1.0 / n_vmc_sample` 固定
     （`Julia-mVMC/MVMCOptimizers.jl/src/stochastic_opt.jl:853-868`）であり、C の
     `invW = 1.0 / Wc`（global Wc）へ揃える。MPI 化では `operate_by_s!(ctx, ...)` に
     `bcast!(x, root=0, comm0)` と GEMV 後の `allreduce_sum!(z_local, comm0)` を挿入
     （serial では no-op）。`stcO`・`sdiag`・`g` は WeightAverage 済み global moments から
     構築。`stochastic_opt_cg_init!` の store コピー範囲は rank-local sample chunk とし、
     layout を明文化する。v0.4 の NSRCG=1 対象入力は `useDiagScale = 0`・`RescaleSmat = 0`・
     store layout が C parity で説明可能なものに限定する。全 rank の反復回数・収束判定が
     一致することを unit test で確認する。
9. **出力 / readback（F11）**: `zvo_*.dat` / CalcTimer / 進捗 print は rank0 のみ
   （C と同じ）。rank0 output 後に `barrier(ctx.comm0)` を置き、runner の `zvo_out.dat`
   readback（`Julia-mVMC/MVMCOptimizers.jl/src/run_para_opt_from_namelist.jl:209-239`）は
   rank0 のみ実行する。非 rank0 は minimal result または `nothing` を返す。テスト用 runner
   は MPI 実行時に rank0 の result だけを parent process へ伝える。
10. **PhysCal**: `vmc_phys_cal!` も同型（NDataQtySmp ループ内で同じ split + average 構造、
    Green は reduce-to-rank0）。

serial 時は wrapper が全て no-op のため、**v0.3 までの数値と bit 一致が保たれる**こと
（既存 annotation・GEMM 経路に触らない）を不変条件とする。

## 6. エラー処理・I/O（§3、承認済み + F12/F13 反映）

1. **collective 整合**: C が flag allreduce → `MPI_Abort` で守っている箇所は同位置に再現。
   それ以外の予期しない例外は run driver 最上位の try/catch で捕捉して `MPI.Abort(comm0)`
   （1 rank だけ死んで残りが hang するのを防止）。serial 時は従来どおり再 throw。
   rank-divergent な例外 path は R0 の設計レビューで網羅確認する。
2. **入力**: seed 解決は §5-1 の表に従う（`RndSeed < 0` は rank0 生成 + bcast）。def
   ファイル読み込みは初版では全 rank read（結果不変・実装最小）。rank0 read + bcast 化は
   ohtaka でメタデータ負荷が見えたら R4 で対応。
3. **出力**: rank0 のみ書き込み + readback の rank0-only 化（§5-9）。
4. **unsupported combinations は実行前に明示 error**（§10 Release Gate 7 参照）。

## 7. テスト・CI・ベンチマーク（§4、承認済み + F1/F2/A1/A8 反映）

1. **serial 単体テスト**（mpiexec 不要）: `split_loop` の C 式一致表（§4.1 の edge case
   一式: 非整除・空 range・`loopLength=0` 方針、F8 + A8）、SerialContext wrapper の no-op
   性、chunk 分割境界。serial 数値の v0.3 bit 不変は既存 ctest_equivalent が gate。
2. **MPI テスト** `test/mpi/`（MPI.jl の `mpiexec()` ラッパで起動）:
   - **C parity gate**: C-mVMC を同条件（`OMP_NUM_THREADS=1`）で実行した reference fixture
     をローカル生成・コミットし、Julia 側を同 rank 数で実行して tolerance 内一致を確認
     （ctest_equivalent の MPI 版）。MPI 実装間の縮約順序差のため bit 一致は要求しない。
   - **seed parity test（A5）**: `RndSeed` missing / `-1` / `0` / positive の 4 case で
     C と同じ seed 解決になることを確認（R0 gate）。
   - **NSplit self-consistency（F1 で修正）**: 比較条件は
     `mpiexec -n G, NSplitSize=1` vs `mpiexec -n G*S, NSplitSize=S`
     （独立 chain 数 = G を揃える）。同じ `-n N` での split 1 vs 2 比較は独立 chain 数が
     変わるため**不適切であり使わない**。NSplit fixture は `NStore=0, NSRCG=0` から開始する
     （§3.1）。
   - **counter writeback test（A2 + B2）**: `size0=4, NSplitSize=2` で rank2==0 が global
     rank 0 と 1 に存在する case の root 配置を確認する。さらに同じ fixture で、
     **global rank 0 が保持する reduced `Counter` が C 出力と一致する**ことを assertion に
     追加する（C の最終出力は global rank 0 の `Counter` を参照するため、root 配置確認
     だけでは rank1 lane 間の counter 同一性の崩れによる出力差を見逃しうる）。可能なら
     global rank 0 と 1 の writeback root の counter 値が、同一 chain 条件で一致することも
     確認する。
   - **hybrid smoke**: `mpiexec -n 2` × `JULIA_MVMC_INNER_THREADS=2` で no-crash +
     threads=1 と tolerance 内一致。
3. **CI 常設化**: GitHub Actions に `mpiexec -n 2`（NSplitSize=1）の小型 fixture ジョブを
   毎 PR で追加（MPICH_jll 使用、system MPI 不要）。小型 `NSplitSize=2` fixture は
   `NStore=0, NSRCG=0` から開始。Julia 1.11 / 1.12 で dependency resolve 差がないことを
   確認（A6）。
4. **ohtaka scaling**: v0.3 benchmark 手順を踏襲し、L24/L32 で ranks = 1/2/4/…/64
   （multi-node 含む）の strong scaling を C-mVMC 同条件と比較。metadata に MPI backend /
   MPIPreferences 設定・rank 配置・`OMP_NUM_THREADS=1` を記録。

## 8. v0.3 threading との関係（承認済み方針）

- **MPI × 1 thread を正**とする。v0.4 の parity gate / benchmark は threads=1 で実施。
- 既存 opt-in threading（`JULIA_MVMC_INNER_THREADS` 等）との併用は「壊れないこと」の確認
  （hybrid smoke）に留め、hybrid 性能最適化は v0.5 以降。
- experimental の sample-level `JULIA_MVMC_MAINCAL_THREADS` は引き続き default off /
  experimental 扱いで、v0.4 の対象外（未解決の thread race は v0.4 に影響しない）。

## 9. フェーズ分割（レビュー修正版 + v2 追補で差し替え）

v0.3 の R 方式（branch + review 文書 + ユーザ承認）を踏襲する。

### R0 — MPI infra + initialization semantics

目的: MPI context を high-level runner の初期化順序へ導入し、serial bit parity を保つ。

必須作業:

- `parallel.jl` 追加: `ParallelContext` / `SerialContext`、`split_loop`、chunked allreduce /
  reduce-to-root / bcast / barrier / abort wrappers。
- MPI detection: MPI env 検出、`JULIA_MVMC_MPI` 3 値 policy（F12 + A7）、`MPI.Initialized()`
  guard（F13）。
- 依存更新（A6）: `MVMCOptimizers.jl/Project.toml` + root `Julia-mVMC/Project.toml`（要否
  確認のうえ）+ `Manifest-v1.11.toml` 更新方針 + CI Julia 1.11/1.12 resolve 確認。
- seed: §5-1 の解決表を実装。`actual_seed = resolved_RndSeed + ctx.group1`。
- parameter sync: pack/unpack `Para` 相当 helper、`sync_modified_parameter!(ctx, data)` で
  rank0 bcast + local sync（F4）。
- output/readback: rank0-only write policy、runner readback の rank0-only 化（F11）。
- **C 側 store 経路（`NStore!=0 || NSRCG!=0`）× `NSplitSize>1` の実走確認を正式タスク化**
  （§3.1 の結論を出す、F2 + A1）。
- ohtaka で MPI.jl + system MPI（MPIPreferences）の疎通 spike。

R0 gate:

- serial 既存 integration pass（bit 不変）。
- `split_loop` table が C と一致（非整除・空 range・`loopLength=0` 方針を含む、A8）。
- seed parity test: `RndSeed` missing / `-1` / `0` / positive（A5）。
- `mpiexec -n 2` smoke: rank0-only output、duplicate write なし。

### R1 — sample 並列（NSplitSize=1）

目的: rank ごとに独立 chain を走らせ、global weighted average を C と一致させる。

必須作業:

- `VMCMakeSample` は各 rank full QP、seed は `RndSeed + rank0`（NSplitSize=1 では
  group1 = rank0）。
- `VMCMainCal` は full local `NVMCSample`。
- `WeightAverageWE` / `WeightAverageSROpt` は comm0 allreduce。
- `WeightAverageGreenFunc` は reduce-to-rank0（F9）。
- `ReduceCounter` は C の 6 counters のみ、`ctx.rank2 == 0` writeback（F10 + A2。
  NSplitSize=1 では comm2 = comm0 のため rank2==0 は global rank0 と一致する）。

R1 gate:

- `mpiexec -n 2/4`, `NSplitSize=1`, `NSRCG=0`, `NStore=0/1` の C parity。
- PhysCal energy + Green output が rank0 で C parity。

### R2 — SR-CG（serial parity 先行、その後 MPI 化。対象は NSplitSize=1 のみ）

目的: `NSRCG=1` の C parity を確立する（`NSplitSize=1` 限定、A1）。

前提:

- **serial SR-CG が C と一致していること**（normalization を global Wc へ修正、F7）。
- `useDiagScale=0`, `RescaleSmat=0` に限定するか、未実装分岐を実装する。

必須作業:

- `operate_by_s!(ctx, ...)`: `bcast!(x, root=0, comm0)` → local GEMV →
  `allreduce_sum!(z_local, comm0)`。normalization は global `Wc`。
- `stcO`, `sdiag`, `g` は WeightAverage 済み global moments から構築。
- store layout（rank-local / global slot）を明文化。

R2 gate:

- serial NSRCG=1 C parity。
- `mpiexec -n 2`, `NSplitSize=1`, NSRCG=1 C parity。
- all ranks が同一 iteration count / parameter update になること。

### R3 — NSplitSize / QP split

目的: 同一 group 内 rank で同一 chain を共有し、QP と sample を分担する。

必須作業:

- QP index convention の統一（§5-4、F6）を先に実施。**C の local-compressed layout との
  違いと adapter 方針を明文化し、adapter test（`qp_start > 0` / `NQPFull < size1` / 全関数の
  convention 共有）を置く**（A3）。
- `comm1` QP split: `CalculateMAll` / `CalculateNewPfM*` / `UpdateMAll*`、
  **`CalculateLogIP` / `CalculateIP` の comm1 allreduce**（F5）。
- `comm1` sample split: `VMCMainCal` sample range、SR store layout の明確化。
- `comm2` counter reduction: C の 6 counters のみ、`ctx.rank2 == 0` writeback の multi-group
  test（A2）。
- `NSplitSize > 1 && (NStore != 0 || NSRCG != 0)` の事前 error（§3.1 + A1）。

R3 gate:

- `NStore=0`, `NSRCG=0`: `mpiexec -n G, split=1` vs `mpiexec -n G*S, split=S` の
  self-consistency（F1 修正版の条件）。
- C parity fixture も同条件で確認。
- store 経路（`NStore!=0 || NSRCG!=0`）× `NSplitSize>1` は F2/A1 の結論が出るまで
  release gate に入れない。

### R4 — release gate / documentation / benchmark

目的: CI 常設化、ohtaka scaling、manual 整備。

必須作業:

- CI: `mpiexec -n 2`, `NSplitSize=1` を毎 PR。小型 `NSplitSize=2` fixture は
  `NStore=0, NSRCG=0` から開始。
- ohtaka: MPI backend / MPIPreferences / job script を明記、`OMP_NUM_THREADS=1` を
  metadata に記録。
- manual: `NVMCSample` は group あたりであること、`NSplitSize` と rank 数の関係、
  unsupported combinations（既存 unsupported mode 含む）、`JULIA_MVMC_MPI` semantics、
  MPI backend setup（mpiexec / MPIPreferences / ohtaka job script 例）。
- CHANGELOG / README 更新。

## 10. 完了条件（Release Gate、レビュー修正版 + v2 追補で差し替え）

1. serial 既存 C parity が bit / tolerance で維持される。
2. `mpiexec -n 2/4`, `NSplitSize=1`, `NSRCG=0` が C parity。
3. `mpiexec -n G, split=1` vs `mpiexec -n G*S, split=S` が C parity / self-consistency。
   ただし初期 gate は `NStore=0, NSRCG=0`。
4. `NSRCG=1` は serial parity が成立した入力に限定して MPI parity。**対象は
   `NSplitSize=1` のみ**（A1）。
5. PhysCal は energy + Green output が rank0 で C parity。
6. rank0-only output / readback / barrier がテストで確認される。
7. `ReduceCounter` は comm2 allreduce + `ctx.rank2 == 0` writeback がテストで確認される
   （A2）。
8. QP split は C layout（local-compressed）と Julia layout（global）の違いが明文化され、
   adapter test が置かれている（A3）。
9. unsupported combinations は実行前に明示 error:
   - `NSplitSize < 1`
   - multi-rank mpiexec 下の `JULIA_MVMC_MPI=0`
   - 当面の `NSplitSize > 1 && (NStore != 0 || NSRCG != 0)`（F2/A1 を解決するまで）
   - 未実装の `useDiagScale != 0` / `RescaleSmat != 0`（実装しない場合）
   - **既存 unsupported mode（A4）**: `NLanczosMode > 0`、BackFlow 入力、FSZ /
     general-orbital mode の `TwoBodyGEx`、その他現行 runtime が reject する機能
10. seed parity test（`RndSeed` missing / `-1` / `0` / positive）が R0 で pass（A5）。
11. CI での MPI テスト常設化（毎 PR、mpiexec -n 2）。
12. ohtaka での strong scaling 測定と C-mVMC との比較報告。
13. マニュアル / ドキュメント整備（§9 R4 参照）。

## 11. リスクと留意点

- **MPI 実装差による FP 差**: allreduce 縮約順序の違いで bit 一致は保証されない →
  tolerance gate で運用（§2 のとおり）。
- **C 側の store 経路 × `NSplitSize>1` 疑義（F2 + A1）**: v0.4 では `NStore!=0 || NSRCG!=0`
  との組み合わせを unsupported。R0 で実走確認をタスク化し、結論が出るまで gate へ入れない。
  upstream issue 報告は別途検討。
- **ohtaka の system MPI 切替**: MPIPreferences の疎通を R0 で早期確認（後工程での手戻り防止）。
- **collective deadlock**: rank-divergent な例外 path は R0 の設計レビューで網羅確認する。
- **QP index 契約の混在（F6）と C との layout 差（A3）**: R3 着手前に convention 統一・
  adapter 明文化と unit test を済ませる。
- **NVMCSample 意味論**: 「chain（group）あたり」の数である点をマニュアルに明記
  （総独立サンプル数は group 数で変わる）。C-mVMC と同じ仕様であり変更しない。

## 12. 関連文書

- レビュー: `docs/reviews/2026-06-10-julia-mvmc-v0.4-mpi-design-review-and-mitigation.md`
  - 初回レビュー F1-F13（v2 で反映）
  - 追記「v2 再レビュー」A1-A8（v3 で反映）
  - 追記「v3 再レビュー」B1-B2（本 v3.1 で反映）
