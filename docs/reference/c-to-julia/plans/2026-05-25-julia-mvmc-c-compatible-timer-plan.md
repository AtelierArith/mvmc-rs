---
date: 2026-05-25
datetime: 2026-05-25 21:21 JST
status: design plan
topic: Julia-mVMC C-compatible lightweight timer instrumentation
---

# Julia-mVMC C-compatible Timer 設計プラン

## レビュー結果（2026-05-25 追記）

- reviewer model: Claude Opus 4.7 (1M context) / `claude-opus-4-7[1m]`
- reviewed datetime: 2026-05-25 21:36 JST
- scope: C 側 timer 実装（`mVMC/src/mVMC/vmcclock.c`, `vmcmain.c`, `vmcmake_*.c`, `vmccal.c`, `stcopt*.c`）と Julia 側関数群・並列モデル・既存 timer インフラとの突き合わせ

section 対応表（id↔関数）はおおむね正確で、C 側の id/label・呼び出し構造、Julia 側の関数の実在を確認した。ただし以下の問題を解決しないと実装が破綻する。特に C1 は最優先。

| # | 重大度 | 内容 |
|---|---|---|
| C1 | 🔴 | 既存 `MVMC_TIMER` / `TIMER_ENABLED[]` / `VMC_TIMER` / `@timeit` インフラをプランが無視。置換 or 共存方針が必須 |
| C2 | 🔴 | `time_ns()` は `UInt64` を返す。プランの `Int64` 保持は型的に不健全 |
| C3 | 🔴 | timer struct を型パラメータ化しないと `Val` no-op が動的ディスパッチ化し設計目的が崩壊 |
| M1 | 🟠 | timer は呼び出し箇所に巻く（関数本体に巻かない）ことをルールとして明記 |
| M2 | 🟠 | 今回の保存済み benchmark は C/Julia とも 1 process / 1 thread なので絶対秒比較可。将来 C を MPI 多ランクで測る場合は正規化が必要 |
| M3 | 🟠 | 共有 timer への非アトミック `+=` は現状安全だが、将来のサンプルレベル並列で破綻 |
| M4 | 🟠 | `elapsed_ns[id+1]` の配列長は使用最大 id（fsz=603）を含めること |
| L1 | 🟡 | C 側 `vmcmake_fsz.c:338` に timer id typo（`StopTimer(610)`）。fsz 内訳比較は信用不可 |
| L2 | 🟡 | `[25] SR` は `[21]` の内側、`[24] cal` は `[4]` の内側。inclusive ネストを表に明示 |
| L3 | 🟡 | 単一プロセスなので C の `rank0==0` ガードは不要（意図的に落とす旨を明記）。書式は `%12.5lf` 互換 |
| L4 | 🟡 | repeats を同一プロセス内でループする場合、timer を run ごとに reset/再生成しファイルも上書き |
| L5 | 🟡 | `Val` を call-site まで通すと `vmc_make_sample_*` / `vmc_main_cal_*` ほか広範なシグネチャ改変が波及 |

### C1. 既存 TimerOutputs インフラとの関係（最優先）

Julia-mVMC のホットループには **すでに timer が実装済み**:

- `Julia-mVMC/MVMCOptimizers.jl/src/vmc_sampling.jl:12-13`: `const TIMER_ENABLED = Ref(false)` / `const VMC_TIMER = TimerOutput()`
- `vmc_sampling.jl:61-65` の `__init__` が環境変数 `MVMC_TIMER=1` を読んで有効化
- ホットループ（`vmc_sampling.jl:5754-6005` ほか）は `if TIMER_ENABLED[]` で **timed/untimed の2経路に完全分岐**し、`@timeit VMC_TIMER` を `make_candidate_*` / `update_ele_config!` / `calculate_new_pf_m2_*!` / `calculate_log_ip_*` / `update_m_all_*!` に巻いている

本プランの「非目的」「軽量化設計」が避けよと言う `@timeit` / `Ref(false)` のホットループ読みは、**現状コードがまさにやっていること**。新 timer を `Val(false)` で no-op 化しても、既存の `if TIMER_ENABLED[]`（Ref 読み）がホットループに残る限り「無効時ゼロオーバーヘッド」は成立せず、1% 回帰判定のベースライン自体が既存 Ref 分岐込みになる。

**方針（本プランの採用）**: 既存の `@timeit` / `TIMER_ENABLED` ベースの計装を **新 C-timer で置き換える**。これにより (a) ホットループから Ref 読みを除去でき本プランの非目的と整合、(b) 同一箇所の二重計装を回避、(c) コードを timed/untimed 2経路から単一経路（`Val` 特殊化）に統合できる。`MVMC_TIMER`（旧）と `MVMC_C_TIMER`（新）の env var を併存させず、移行期間中のみ `MVMC_TIMER` を deprecated alias とする。

**実装状況（2026-05-25, feature/c-compatible-timer）**: Phase 1+2+C1 と Phase 3/4（real 経路）を実装・ローカル commit 済み（`2b53bf5` infra+section、`3d04516` 旧経路撤去、`a7d7450` inner id）。

- Phase 1+2+C1: `vmc_sampling.jl` から `TIMER_ENABLED`/`VMC_TIMER`/`@timeit`/`enable_timer!` 系/`__init__` を削除し hot loop を untimed 経路へ collapse（net -350行・数値不変）、`MVMC_TIMER` は真の deprecated alias 化。
- Phase 3/4（real 経路）: `CTimer` を typed 引数でホットループへ渡し（function barrier）、無効時は `ctimer_start!` が `return nothing` に inline・10M 回 start/stop で **0 alloc**（C3 実証）。`vmc_make_sample_real!` に `[30][31][32][60-63][33][65-68][34][35]`、`vmc_main_cal!` に `[24][40][41][42][43][45]`、`calculate_hamiltonian` に `[70][71][72]` を call site で計装。timer は call site に巻き、`[31]` は reject `continue` 前に閉じ `[32]/[33]/[35]` は `continue` の後で開始（タイマー開きっぱなし回避）。
- 検証（julia 1.12）: 旧シンボル未定義、`zvo_out.dat` は 無効/`MVMC_C_TIMER=1`/`MVMC_TIMER=1` で全バイト一致、inner id は inclusive ネスト整合（親≥子の和）。Heisenberg は exchange モデルなので `[33][66][68][72]` が支配的・hopping `[32][60-63]` は 0（正しい）。統合 120/120・ctest 61/61・単体 248 pass。完了条件 id `[2][3][4][40][41][42][66][68][72]` を C と同じ id/label で出力。
- 追加実装（`9997095` SR+`[69]`、`54409ff` complex-sz/fsz）: SR 直接ソルバ `stochastic_opt!` の内訳 `[50]`preprocess / `[51]`stcOptMain ⊃ `[56]`calculate S and g / `[57]`DPOSV / `[52]`postprocess。`[69]` MAll（real↔complex 変換）。complex-sz sampling `vmc_make_sample!` に `[30-35][60-68]`、fsz main-cal `vmc_main_cal_fsz!` に `[24][40][41][42][43]`、`calculate_hamiltonian_fsz` に `[70][71][72]`。全 sampling/main-cal 変種へ `c_timer` を thread 済み。検証: 全モデルで `zvo_out` byte 一致、complex-sz で `[66]/[68]`・fsz で `[40]/[41]/[42]/[72]` が populate、統合 120/120・ctest 61/61・単体 248 pass。
- `TimerOutputs` 依存除去（`89623a8`）: `MVMCOptimizers.jl/Project.toml` の `[deps]`/`[compat]` から削除し、manifest（subpkg / workspace 1.12 / `Manifest-v1.11.toml`）を resolve で再生成（`TimerOutputs` + 推移依存 `ExprTools` を prune）。これで「CTimer, not TimerOutputs」方針と依存グラフが一致。検証: 単体 248・統合 120/120 pass。
- 未実装（低優先・非 benchmark）: fsz **sampling** 変種（`vmc_make_sample_fsz!` / `vmc_make_sample_fsz_real!`）の inner id。HOPPING/EXCHANGE/LOCALSPINFLIP の 3 分岐 + lspinflip `[36]/[600]-[603]` で、C 側 fsz id へのマッピングが曖昧、かつ C は `vmcmake_fsz.c:338` に `StopTimer(610)` typo（L1）があり比較が信頼できない。CG ソルバ `stochastic_opt_cg!` の SR 内訳（`[5]` 全体は計測済）。
- 次: ohtaka L=16/24/32 で `MVMC_C_TIMER=1` の Julia 内訳を取り、C の `zvo_CalcTimer.dat` と per-sample 正規化（M2）で比較。無効時 benchmark regression（<1%）も ohtaka で確認。

### C2 / C3 の修正は下記「軽量化設計」「出力形式」に反映済み（`UInt64` 化・型パラメータ化）。

### M2. MPI 多ランク実行時の比較方法論

- 今回の保存済み ohtaka benchmark は C-mVMC も `mpiexec -n 1`、Slurm も `#SBATCH -n 1` / `#SBATCH -c 1` で、Julia-mVMC も 1 process / 1 thread。したがって `[2] VMCParaOpt` 以下の絶対秒はそのまま比較してよい。
- Julia-mVMC は **MPI を一切使わない単一プロセス**（`using MPI` なし。`reduce_counter!` は no-op: `counter.jl:24`）。スレッド並列は `calculate_m_all` 内の QP ループ（`calculate_m_all.jl:440` の `@threads for qpidx`）に閉じる。
- 将来 C-mVMC を MPI 多ランクで測る場合、全ランクが timer を回し、rank0 のみ `OutputTimerParaOpt()` を出力する。この場合、C の `[3]`/`[4]` は rank0 の担当分を含むランクローカル timer、Julia の `[3]`/`[4]` は単一プロセス全体 timer なので、絶対秒の直接比較は誤解を招く。

MPI 多ランク比較を行う場合は、比較表（後述）に `nproc` / `nthreads` / `n_sample_per_proc` 列を追加し、必要に応じて per-sample 正規化値で比較する。`[55] initBLACS` / `[58] gatherParaChange` / `[21]` の MPI reduction 相当は Julia では ~0 になるので N/A 扱い。

### M3. スレッド安全性の前提条件

サンプリングループ（`vmc_make_sample_*`）と `vmc_main_cal_*` には `Threads.@threads` が無いことを確認済み。スレッド並列は `calculate_m_all` 内の QP ループに閉じ、timer stop 前に join される。よって **現状は** 単一 timer struct への `+=` で安全。ただし:

- 将来サンプルレベル並列が入ると共有 `elapsed_ns` への並行 `+=` がデータレースになる
- timer id を `@threads` 領域の内側に置くと同様にレース

→ 「サンプルループが単一スレッドであること」を前提として明記し、call-site 計装（M1）を徹底する。サンプル並列化時は thread-local timer へ移行する。

## 目的

ohtaka i8cpu benchmark で Julia-mVMC が C-mVMC より速い理由を、C 側の `zvo_CalcTimer.dat` と同じ粒度で比較できるようにする。

現状は C-mVMC だけが `VMCMakeSample` / `VMCMainCal` / `UpdateMAllTwo` / `CalculateMAll` などの内部 timer を出している。Julia 側は外側の elapsed しか保存していないため、差分が Pfaffian 更新、M 再計算、局所エネルギー評価、SR solver のどこで出ているかを切り分けられない。

## 非目的

- 通常実行を遅くしない。
- 常時 profiling を有効化しない。
- `TimerOutputs.@timeit` を hot loop に増やさない。
- まずは benchmark / performance diagnosis 用の計測であり、API の主要機能として前面に出さない。
- C-mVMC の timer 定義を変更しない。

## 基本方針

Julia 側に C 互換の軽量 timer を追加する。出力ファイルは C と同じ `zvo_CalcTimer.dat` 形式を基本とし、C の既存集計スクリプトを流用できるようにする。

有効化は環境変数で明示する。

```bash
MVMC_C_TIMER=1 julia --project=... scripts/run_julia_repeats.jl ...
```

デフォルトでは無効。無効時の hot path に timer 分岐や `time_ns()` 呼び出しを残さない設計にする。

## 軽量化設計

### Timer 実装

`TimerOutputs`（既存 `VMC_TIMER`）ではなく、`time_ns()` ベースの累積 timer を使う。**既存の `@timeit` / `TIMER_ENABLED[]` 計装は本 timer で置き換える**（C1 参照）。

想定する内部構造（**型パラメータ `E` で特殊化する。C3**）:

```julia
# NTIMER は C の NTimer=1000 に合わせる。少なくとも使用最大 id + 1（fsz lspinflip = 604）を満たすこと（M4）
const CTIMER_N = 1000

struct CTimer{E}                      # E は Val{true} / Val{false}
    enabled::E
    elapsed_ns::Vector{UInt64}        # timer id ごとの累積時間（UInt64。C2）
    start_ns::Vector{UInt64}          # timer id ごとの開始時刻（UInt64。C2）
end

CTimer(enabled::Bool) =
    CTimer(Val(enabled), zeros(UInt64, CTIMER_N), zeros(UInt64, CTIMER_N))
```

`time_ns()` は **`UInt64`** を返すので、配列も `UInt64` で統一する（`Int64` だと暗黙 convert と signed/unsigned 混在が起きる。C2）。秒変換は出力時に一度だけ `value / 1e9` で行う。

`StartTimer(id)` / `StopTimer(id)` 相当は以下の薄い関数にする。`enabled` をフィールドではなく **第一引数の `Val` で dispatch** し、struct 自体も `CTimer{E}` で具象化されているため、無効時は完全に no-op として inline される。

```julia
@inline ctimer_start!(::Val{false}, timer, id) = nothing
@inline ctimer_stop!(::Val{false}, timer, id) = nothing

@inline function ctimer_start!(::Val{true}, timer, id)
    @inbounds timer.start_ns[id + 1] = time_ns()
    return nothing
end

@inline function ctimer_stop!(::Val{true}, timer, id)
    @inbounds timer.elapsed_ns[id + 1] += time_ns() - timer.start_ns[id + 1]
    return nothing
end

# 呼び出し側は timer.enabled を渡す（CTimer{E} で具象化 → コンパイル時に解決）
@inline ctimer_start!(t::CTimer, id) = ctimer_start!(t.enabled, t, id)
@inline ctimer_stop!(t::CTimer, id)  = ctimer_stop!(t.enabled, t, id)
```

重要なのは、無効時に `Val(false)` が（型パラメータ経由で）伝播して no-op として inline されること。`Ref(false)` を hot loop で毎回読む形にはしない。

**M1: timer は呼び出し箇所（call site）に巻く。関数本体には巻かない。** `calculate_m_all_*` は `vmc_make_sample_*`（id 30/34）と `vmc_main_cal_*`（id 40）の両方から、計 12 箇所以上呼ばれる（C 側も `vmccal.c:122` 等の call site で計時し、関数本体は無計時）。関数本体に入れると id を区別できず二重計上になる。同一 id を再帰的・ネストして再入させない（`start_ns[id]` は 1 スロットのみ。C と同じ制約）。

### 有効/無効の分岐位置

`MVMC_C_TIMER` の判定は run の入口で一度だけ行う。

- `run_para_opt_from_namelist` で `timer_enabled = get(ENV, "MVMC_C_TIMER", "0") != "0"` を判定
- `CTimer(timer_enabled)`（= `CTimer{Val{true/false}}`）を生成し、`vmc_para_opt!` 以下へ渡す
- `Val(timer_enabled)` の生成は **run 入口の一度だけ**（実行時 Bool → 型への変換は意図的な動的ディスパッチ点）。直後に `vmc_para_opt!` が timer の型パラメータで特殊化される **function barrier** を置く（C3）
- hot loop の中では `if ENV[...]` や `if TIMER_ENABLED[]` を見ない

既存 public API の互換性は保つ。

```julia
vmc_para_opt!(data; ..., c_timer = nothing)
```

`c_timer === nothing` の場合はデフォルト無効（`CTimer(false)`）。`run_para_opt_from_namelist` だけが環境変数から明示的に有効化する。

**C1: 既存 `MVMC_TIMER` / `TIMER_ENABLED[]` / `VMC_TIMER` インフラとの統合**

- 新 timer 導入と同時に、ホットループの `if TIMER_ENABLED[]` 二経路分岐（`vmc_sampling.jl:5754-6005` ほか）と `@timeit VMC_TIMER` を撤去し、`ctimer_start!/stop!` 呼び出しへ一本化する
- `vmc_sampling.jl:61-65` の `__init__` による `MVMC_TIMER` のプロセスグローバル有効化は廃止（per-run の `CTimer` に置換）。移行期間は `MVMC_TIMER` を `MVMC_C_TIMER` の deprecated alias とし、両方未設定なら無効
- これにより無効時のホットループから Ref 読み・分岐が消え、「無効時ゼロオーバーヘッド」が実際に成立する

**C3 補足: timer を既存 struct（`ExpertModeData` / `VMCOptimizationState`）のフィールドに型パラメータ無しで埋め込むと特殊化を失う。** timer は引数で渡すか、埋め込むなら struct 側も型パラメータ化する。

**L5: シグネチャ波及。** `Val` を call site まで通すため、`vmc_make_sample!` / `vmc_make_sample_real!` / `vmc_make_sample_fsz!` / `vmc_make_sample_fsz_real!` / `vmc_main_cal!` / `vmc_main_cal_fsz!` など（いずれも現状 timer 引数なし）に `c_timer` を追加する必要がある。Phase 1 完了条件にシグネチャ変更の影響範囲確認を含める。

### Timer 有効時の overhead

timer 有効時は細粒度計測のため多少遅くなるのは許容する。ただし以下を守る。

- `time_ns()` 呼び出しは C timer と同等の start/stop 境界だけに置く。
- array allocation を timer 境界で発生させない。
- string label lookup を計測中に行わない。
- 出力 formatting は run 終了後に一度だけ行う。

## 出力形式

C の `OutputTimerParaOpt()` と同じ label / id / indentation / 秒単位を基本にする。

例:

```text
All                         [0]    130.12345
Initialization              [1]      0.12345
  read options             [10]      0.00000
  ReadDefFile              [11]      0.00000
  SetMemory                [12]      0.00000
  InitParameter            [13]      0.00000
VMCParaOpt                  [2]    129.99999
  VMCMakeSample             [3]     70.00000
...
```

Julia 側の output directory に `zvo_CalcTimer.dat` を書く。C と同名にすることで、比較スクリプトは engine / path だけで切り替えられる。

実装上の注意:

- C は `{CDataFileHead}_CalcTimer.dat`（例 `zvo_CalcTimer.dat`）。Julia の output prefix が `zvo` であることを確認する
- 書式は C の `%12.5lf`。Julia では `value_ns / 1e9` を `@sprintf("%12.5f", …)` で出す（C2）
- 単一プロセスなので C の `if(rank0==0)` ガードは不要。意図的に rank ガードを落とす（L3）
- repeats を同一プロセス内でループする場合、`CTimer` を run ごとに再生成（または `fill!(elapsed_ns, 0)` で reset）し、ファイルは run ごとに上書き（"w"）する。モジュールグローバルに持つと repeat 間で累積する（L4）

## C Timer 対応表

### Top-level

| id | C label | Julia 対応候補 | 優先度 |
|---:|---|---|---|
| 0 | `All` | `run_para_opt_from_namelist` 全体 | 中 |
| 1 | `Initialization` | parse / seed / init / read input / sync / init QP weight | 中 |
| 10 | `read options` | namelist path resolve / mode validation など。厳密対応は難しい | 低 |
| 11 | `ReadDefFile` | `parse_expert_mode_files` | 中 |
| 12 | `SetMemory` | `VMCOptimizationState(...)` と workspace 確保 | 中 |
| 13 | `InitParameter` | `init_parameter!` + `read_initial_def!` + `read_input_parameters!` + initial sync | 中 |
| 2 | `VMCParaOpt` | `vmc_para_opt!` optimization loop 全体 | 高 |

Initialization は C が Standard mode から expert files を生成する場合と Julia が `namelist.def` を直接読む場合で処理が違う。主比較は `[2] VMCParaOpt` 以下とする。

### VMCParaOpt loop

| id | C label | Julia 対応候補 | 優先度 |
|---:|---|---|---|
| 20 | `UpdateSlaterElm` | `update_slater_elm_*` + `update_qp_weight!` | 中 |
| 3 | `VMCMakeSample` | `vmc_make_sample_*` | 高 |
| 4 | `VMCMainCal` | `vmc_main_cal!*` | 高 |
| 21 | `WeightAverage` | `weight_average_we!` + `weight_average_sr_opt*!` + `reduce_counter!` | 低 |
| 25 | `SR` | `weight_average_sr_opt*!` 部分 | 低 |
| 22 | `outputData` | `output_data!` | 低 |
| 5 | `StochasticOpt` | `stochastic_opt!` / `stochastic_opt_cg!` | 中 |
| 23 | `SyncModifiedParameter` | `sync_modified_parameter!` | 低 |

ohtaka の差分支配項は `[3]` と `[4]` なので、まずここを完全対応させる。

### VMCMakeSample

| id | C label | Julia 対応候補 | 優先度 |
|---:|---|---|---|
| 30 | `makeInitialSample` | initial sample / burn sample copy + initial `calculate_m_all_*` + initial logIP | 中 |
| 31 | `make candidate` | `make_candidate_hopping` / `make_candidate_exchange` / fsz variants | 高 |
| 32 | `hopping update` | hopping branch 全体 | 中 |
| 60 | `UpdateProjCnt` | hopping branch の config/proj 更新 | 中 |
| 61 | `CalculateNewPfM2` | `calculate_new_pf_m2*` | 中 |
| 62 | `CalculateLogIP` | hopping branch の `calculate_log_ip*` | 中 |
| 63 | `UpdateMAll` | `update_m_all*` | 中 |
| 33 | `exchange update` | exchange branch 全体 | 高 |
| 65 | `UpdateProjCnt` | exchange branch の config/proj 更新 | 高 |
| 66 | `CalculateNewPfMTwo2` | `calculate_new_pf_m_two2*` | 高 |
| 67 | `CalculateLogIP` | exchange branch の `calculate_log_ip*` | 高 |
| 68 | `UpdateMAllTwo` | `update_m_all_two*` | 最高 |
| 34 | `recal PfM and InvM` | periodic `calculate_m_all*` + logIP 更新 | 高 |
| 35 | `save electron config` | sample 保存 loop | 低 |
| 69 | `MAll` | real/complex array copy around sample phase | 低 |

ohtaka L=32 の C では `[68] UpdateMAllTwo` が約 73.4 s、`[66] CalculateNewPfMTwo2` が約 38.5 s、`[34] recal PfM and InvM` が約 8.8 s。Julia が速い理由を調べる最優先は `[68]` と `[66]`。

### VMCMainCal

| id | C label | Julia 対応候補 | 優先度 |
|---:|---|---|---|
| 24 | `cal` | `clear_phys_quantity!` | 低 |
| 40 | `CalculateMAll` | sample ごとの `calculate_m_all*` | 最高 |
| 41 | `LocEnergyCal` | `calculate_hamiltonian*` | 高 |
| 70 | `CalHamiltonian0` | diagonal terms: CoulombIntra / CoulombInter / Hund | 中 |
| 71 | `CalHamiltonian1` | Transfer / one-body terms | 中 |
| 72 | `CalHamiltonian2` | PairHopping / Exchange / InterAll two-body terms | 高 |
| 42 | `ReturnSlaterElmDiff` | `slater_elm_diff_*` | 高 |
| 43 | `calculate OO and HO` | `calculate_oo*` / store HO accumulation | 中 |
| 45 | `multiply store OO` | `finalize_oo_store*` | 低 |

ohtaka L=32 の C では `[40] CalculateMAll` が約 65.7 s で、mac 比約 3.0x と大きい。Julia 側 timer の最重要候補。

### StochasticOpt

| id | C label | Julia 対応候補 | 優先度 |
|---:|---|---|---|
| 50 | `preprocess` | real→complex conversion、diag/cut、mapping 作成 | 中 |
| 51 | `stcOptMain` | solver main 全体 | 中 |
| 55 | `initBLACS` | Julia local LAPACK では非対応、0 または N/A | 低 |
| 56 | `calculate S and g` | `build_s_matrix_and_g_vector!` | 中 |
| 57 | `DPOSV` | `potrf!` + `potrs!` | 中 |
| 58 | `gatherParaChange` | local vector gather 相当、ほぼ 0 | 低 |
| 52 | `postprocess` | SR info / finite check / parameter update | 中 |

今回の Heisenberg chain では `[5] StochasticOpt` は 0.1% 程度なので、初期実装では top-level `[5]` だけでもよい。必要になったら細分化する。

## 実装フェーズ案

### Phase 1: Timer infrastructure と top-level

- `MVMCOptimizers.jl/src/c_timer.jl` を追加（`CTimer{E}` を型パラメータ化。C3）。
- `MVMCOptimizers.jl/src/MVMCOptimizers.jl` で include。
- `run_para_opt_from_namelist` に `MVMC_C_TIMER` 判定を追加し、`CTimer(enabled)` を生成して渡す。
- **既存 `TIMER_ENABLED` / `VMC_TIMER` / `@timeit` / `__init__` の `MVMC_TIMER` 読みを撤去または deprecated alias 化（C1）。**
- `vmc_para_opt!` に `c_timer` kwarg を追加し、function barrier で型特殊化されることを `@code_warntype` 等で確認（C3）。
- `zvo_CalcTimer.dat` 出力関数を追加（`%12.5f`・rank ガード無し・run ごと上書き）。
- `[0]`, `[1]`, `[2]` だけ出せる状態にする。
- **`vmc_make_sample_*` / `vmc_main_cal_*` への `c_timer` 引数追加の波及範囲を確認（L5）。**
- 無効時 benchmark で回帰なし（既存インフラ撤去により同等以上）を確認する。

### Phase 2: VMCMakeSample / VMCMainCal の大枠

- `[3] VMCMakeSample`
- `[4] VMCMainCal`
- `[20] UpdateSlaterElm`
- `[21] WeightAverage`
- `[22] outputData`
- `[5] StochasticOpt`
- `[23] SyncModifiedParameter`

ここで C の major section と Julia の major section が並ぶ。

### Phase 3: 差分支配項の細分化

優先順:

1. `[68] UpdateMAllTwo`
2. `[66] CalculateNewPfMTwo2`
3. `[40] CalculateMAll`
4. `[41] LocEnergyCal`
5. `[72] CalHamiltonian2`
6. `[42] ReturnSlaterElmDiff`
7. `[34] recal PfM and InvM`

この段階で ohtaka の Julia/C 差分の主因を判断できるはず。

### Phase 4: 残りの C label を埋める

- hopping 系 `[32]`, `[60]`-`[63]`
- candidate `[31]`
- save config `[35]`
- OO/HO `[43]`, `[45]`
- SR 内訳 `[50]`-`[58]`

## 検証手順

### Correctness

timer 無効:

```bash
cd Julia-mVMC
julia --project=. -e 'using Pkg; Pkg.test()'
```

timer 有効:

```bash
cd Julia-mVMC
MVMC_C_TIMER=1 julia --project=. -e 'using Pkg; Pkg.test()'
```

timer 有効時にも `zvo_out.dat` の内容が timer 無効時と一致することを確認する。

### Performance regression

timer 無効の benchmark を実施し、現行値からの回帰を確認する。

判定基準:

- L=16/24/32 の Julia median が既存値から 1% 以内
- 1% を超える場合は hot loop に timer branch が残っている可能性を調査する

**ベースラインの注意（C1）**: 比較対象の「既存値」は、既存 `@timeit`/`TIMER_ENABLED[]` インフラを撤去する前の現行 main で測ったもの。本プランは既存インフラを新 timer に置換するので、無効時はホットループから `if TIMER_ENABLED[]` の Ref 読みが消える。理想的には**置換後（timer 無効）が現行 main 以上に速い**（Ref 読み分岐が無くなるため）。遅くなっていれば `Val(false)` no-op が inline されていない（C3 の型不安定）疑い。

timer 有効時は overhead を測るが、通常 benchmark の代表値には使わない。

### Compare script

既存の C `zvo_CalcTimer.dat` parser を Julia output にも使う。比較表は以下の形を想定する。今回の 1 process / 1 thread benchmark では `median_s` をそのまま比較できる。将来 C を MPI 多ランクで測る場合に備えて、`nproc` / `nthreads` / `n_sample_per_proc` 列と正規化値（per-sample）も持たせる。

| env | engine | L | nproc | nthreads | n_sample_per_proc | id | label | median_s | s_per_sample | pct_of_vmc |
|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|
| ohtaka | C-mVMC | 32 | ... | 1 | ... | 40 | CalculateMAll | ... | ... | ... |
| ohtaka | Julia-mVMC | 32 | 1 | ... | ... | 40 | CalculateMAll | ... | ... | ... |

1 process / 1 thread 条件では `median_s` 同士を主比較にする。MPI 多ランク条件では `s_per_sample`（または同一コア数で総サンプル数を揃えた値）も併記する。`pct_of_vmc` は `[2] VMCParaOpt` を 100% とした内訳割合（inclusive ネストを考慮）。

まずは `benchmark/heisenberg_chain_issue1_ohtaka/scripts/summarize_results.py` を拡張するのが自然。

## 既知の注意点

- C timer は parent section と child section がどちらも elapsed を積む inclusive timer。Julia も同じ考え方にする。
- **inclusive ネストを正しく対応させる（L2）**: `[25] SR` は `[21] WeightAverage` の内側、`[24] cal` は `[4] VMCMainCal` の内側、`[30]`-`[35]`/`[60]`-`[68]` は `[3]` の内側。フラットに足すと `pct_of_vmc` の合計が合わない。
- C の `[1] Initialization` は Standard mode から expert files を作る経路を含む場合がある。Julia benchmark は `namelist.def` 直指定なので、初期化 timer は完全一致を期待しない。
- C の `[55] initBLACS` / `[58] gatherParaChange` / `[21]` の MPI reduction 相当は Julia 単一プロセスでは ~0。0 または N/A 扱いにする（M2）。
- **MPI 多ランク実行時の注意（M2）**: 今回の保存済み benchmark は C/Julia とも 1 process / 1 thread なので絶対秒比較可。将来 C を MPI 多ランクで測る場合は、C の `[3]`/`[4]` が rank-local timer になるため、`nproc` / `nthreads` / `n_sample_per_proc` を揃えるか per-sample 正規化して比較する。
- **C 側 fsz timer のバグ（L1）**: `mVMC/src/mVMC/vmcmake_fsz.c:338` が `StopTimer(601)` であるべきところ `StopTimer(610)` になっており、C の `[601]` は止まらず `[610]` は無効値。fsz lspinflip 経路（id 36, 600-603）の内訳を C と比較する場合、C 側の値は信用できない。Heisenberg chain（sz 保存・exchange のみ）では当該経路を踏まないので影響なし。可能なら upstream に報告する。
- timer 有効 run は計測 overhead を含むため、最終的な速度 benchmark ではなく内訳調査用と位置づける。

## 完了条件

- `MVMC_C_TIMER=1` の Julia run が `zvo_CalcTimer.dat` を出す。
- 少なくとも `[2]`, `[3]`, `[4]`, `[40]`, `[41]`, `[42]`, `[66]`, `[68]`, `[72]` が C と同じ id/label で出る。
- timer 無効時の Julia benchmark に 1% 超の回帰がない。
- ohtaka L=16/24/32 で C / Julia の timer 内訳比較表を作れる。
