---
date: 2026-06-09
datetime: 2026-06-09 22:24 JST
model: Claude Opus 4.8 (1M context)
status: review
topic: VMCMainCal sample-level threading の物理量破壊 race — commit d3e0ac9「Fix VMCMainCal threaded ele_num snapshots」の root-cause 検証
---

# 2026-06-09: VMCMainCal threaded race — root-cause 検証

## 対象

- Repository: `Julia-mVMC` (`MVMCOptimizers.jl`)
- Branch: `investigate/v0.3-maincal-thread-race`
- 検証対象 commit: `d3e0ac9`「Fix VMCMainCal threaded ele_num snapshots」(親 `203f7c1` = old)
  - `MVMCOptimizers.jl/src/vmc_main_cal.jl`: 各 worker の per-sample `ele_num` を、共有 buffer からの `copyto!` ではなく per-sample の `ele_cfg` から再構成する変更(+ half-filling validation block 削除)。
  - `MVMCOptimizers.jl/test_unit/test_unit_threading.jl`: `vmc_main_cal!` requested-thread self-consistency を `@test_skip`(broken)→ enabled へ。
- commit メッセージの主張:
  1. threaded VMCMainCal が物理量を壊す(native PfaPack 競合 triage の結論として)。
  2. root cause は per-sample `ele_num` snapshot が copied configuration と不整合になること。
  3. `ele_cfg` から rebuild すれば threaded == serial になる。

## 検証環境

- Julia 1.12.6(`~/.juliaup`)、`OMP_NUM_THREADS=1` 固定(CLAUDE.md の macOS ルール準拠)。
- fixture: `test_unit_threading.jl` の `run_threading_maincal_fixture`(4-site / 2 electrons per spin / Gutzwiller+Jastrow projection / 非同一 sample)を流用し、`requested_threads=1`(serial 分岐)と `>1`(threaded 分岐)の snapshot を `threading_snapshots_match`(atol=rtol=1e-10)で比較。
- stress harness は fixture の関数定義だけを取り出し、`n_samples` と `requested_threads` を拡大して多数回反復。

## 結論(要約)

| 主張 | 判定 |
|---|---|
| ① threaded VMCMainCal が物理量を壊す | **正しい**(深刻・非決定的・大きな誤差) |
| ② 修正で threaded == serial になる | **現状テスト上は真**(全テストで 0 fail、16 workers でも安定) |
| ③ root cause は `ele_num` snapshot | **支持できない**(full state isolation でも再現。修正は値不変で、rebuild ループの存在という timing 摂動でのみ症状が消える) |

→ commit はバグの**存在**を正しく捉え症状を消すが、**stated root cause(`ele_num` snapshot)は支持されない**。修正は値不変で、効くのは rebuild ループの実行という timing 摂動のみ。なお正の race site は本検証では特定に至っていない（後述「追加検証」で `data` 共有・PfaPack/BLAS contention は否定。残る候補は unlocked な green/Hamiltonian 評価部の Julia scheduling/closure/timing 依存）。「native PfaPack 競合の方が真相に近い」とは**言えない**（active な numeric kernel は lock 内・pure Julia・BLAS も lock 内で直列化されている）。

## ① バグの実在 — 確認(深刻)

old(`203f7c1`)を本物の複数 OS thread で stress:

| 条件 | old (copyto!) | fixed (d3e0ac9) |
|---|---|---|
| 2 threads, 4 samples, ×12 | 2/12 fail | 0(累計 36/36 pass) |
| 8 threads, 8 workers, 32 samples, ×48 | **14/48 fail** | 0/48 pass |
| 16 threads, 16 workers, 96 samples, ×120 | — | 0/120 pass |

誤差の規模(old, 8 threads):

- `energy` absdiff ≈ 1.69〜2.0(scale ≈ 38〜41、**約 5%**)
- `sr_opt_ho` ≈ 0.75〜2.0(scale ≈ 18)
- `phys_cis_ajs` / `phys_cis_ajs_ckt_alt` ≈ 1.0(scale ≈ 15)

FP reduction noise(~1e-12)ではなく **sample 単位 O(1) の粗い破壊**。`mode=0/1`・`complex/real` の全 case で非決定的に発生。→ **「物理量を壊す」は事実**。

## ② 修正の効果 — 現状テスト上は真

fixed は本検証の全 run(2/8/16 threads、最大 16 workers × 96 samples、累計 ~240 比較)で **0 fail**。`test_unit_threading.jl` の self-consistency を enabled に戻したのは、現状のテスト範囲では妥当。

## ③ root-cause 切り分け — 「ele_num snapshot」は支持されない

`ele_num` 修正は **値を変えない**:rebuild は `ele_num[i] = ele_cfg[i] < 0 ? 0 : 1` で、共有 buffer の値と恒等。runtime 計測でも worker entry の整合性 BADINIT=0、run 後の永続破壊 POST_PERM=0(共有 `ele_num` は前後で `ele_cfg` と完全整合)。

old の `copyto!` 読みを保持したまま、疑わしい共有資源を個別に worker-local 化(全て 8 threads × 48 比較):

| Variant | 変更点 | 結果 |
|---|---|---|
| OLD | 共有 buffer から `copyto!` | 14/48 fail |
| B | OLD + validation block 削除 | 8/48 fail |
| C | OLD + `ele_num` を worker-local copy | 7/48 fail |
| G | OLD + 全 config 配列(idx/cfg/num/proj/spn)worker-local | 8/48 fail |
| J | OLD + `slater_elm`/`slater_elm_real` worker-local | 6/48 fail |
| **N** | OLD + **全 state field deepcopy**(config + slater_elm + phys_quantities) | **7/48 fail** |
| **P** | OLD + **per-worker `deepcopy(data)`**(共有 `data::ExpertModeData` を分離) | **4/48 fail** |
| A | OLD で `ele_num` だけ rebuild に置換(validation 保持) | **0/48 pass** |
| FIXED | rebuild + validation 削除(= d3e0ac9) | **0/48 pass** |
| L | rebuild **後に** buffer 値で `copyto!` 上書き(最終値 = buffer) | **0/48 pass** |
| M | rebuild + `GC.safepoint()` | **0/48 pass** |

判定:

- **`VMCOptimizationState` のどの共有 field も race 元ではない。** N で全 field を deepcopy しても fail。`workspace` は constructor が per-worker 確保で独立(`src/types.jl:335`)、`inv_m`/`pf_m` も per-worker copy、native PfaPack 呼び出しは `with_pfapack_call_lock` で無条件直列化(`src/vmc_sampling.jl:1404`)。
- **効いているのは「rebuild の for ループが実行されるか」だけ。** 最終値は無関係(L は buffer 値で pass)、`copyto!`/safepoint の有無も無関係(L・M とも pass)。
- つまり修正は **共有データ競合を是正していない**。`copyto!`→ループ置換による **タイミング摂動で race window をずらし症状を消している**(timing masking)。

残る race の在処は state 配列ではない。当初は「共有 native/global 状態(PfaPack/BLAS 内部)」を疑ったが、後述「追加検証」でこれは**否定**された(numeric kernel は lock 内・pure Julia・BLAS も lock 内）。`data::ExpertModeData` の per-worker 複製(Variant P)でも race は残る。**正の race site は本検証では特定に至らず**(per-worker state / data からの排除のみ確定)。残る候補は unlocked な green/Hamiltonian 評価部に関わる Julia scheduling / closure capture / timing 依存だが未確定。

## 追加コメント(2026-06-09 22:29 JST, GPT-5 Codex)

本文の stress 結果は、`d3e0ac9` を「root-cause fix」と断定しないための重要な反証材料として有用。ただし、現行コード照合上は以下の留保を置く。

- 「native PfaPack 競合が真相に近い」という表現はまだ強い。非FSZの `real` / `fcmp` path では `calculate_m_all_real!` / `calculate_m_all_fcmp!` 呼び出し全体が `with_pfapack_call_lock` 内にあり、現行の通常 path は `cimpl_utu2inv!` / Fortran `dsktf2` ではなく pure Julia `utu2inv!` / `julia_dsktf2!` を使っている。したがって、現時点で言えるのは「state 共有 alias だけでは説明できず、`data` 共有・未ロック native path・BLAS/LAPACK・Julia scheduling/closure/timing 依存を含む別要因が残る」まで。
- `ele_num` rebuild が「値を変えない」ことは、永続 buffer が `ele_cfg` と整合していることの説明としては妥当。一方で、old path の `copyto!` 後に worker-local `ele_num` が常に同じ内容だったことまでは意味しない。前段調査では、親の保存済み配列は正しいまま、local snapshot だけが別 sample 相当に見える症状を観測しているため、「永続 buffer 破壊ではない」「final value だけでは説明しきれない」と表現するのが安全。
- Variant N の結果から言えるのは「`VMCOptimizationState` の単純な alias だけでは説明できない」までで、「state 由来ではない」「native/global 状態が原因」とまでは確定しない。`data::ExpertModeData` の複製、FSZ path の未ロック `cimpl_utu2inv!`、`BLAS.trmm!` を含む pure Julia `utu2inv!`、PfaPack を完全 bypass した fixture などを分けて追加検証する必要がある。
- 再現手順は方向性として十分だが、強い結論の根拠にするなら stress harness / variant patch / 実行コマンド / seed / case list を保存するのが望ましい。現状の文書だけだと、第三者が `L` / `N` の exact variant を再現しにくい。

作業上の判断としては、`d3e0ac9` は現状の integration / stress を安定化するため revert 不要。ただし PR / release note では「`ele_num` snapshot を直した root-cause fix」と言い切らず、「`ele_cfg` から occupation を再構成して threaded path の observed instability を抑止し、self-consistency gate を有効化した。sample-level MAINCAL threading は引き続き experimental/off」と書くのが妥当。

## 追加検証(2026-06-09 22:36 JST, 上記コメントを受けて / Claude Opus 4.8)

コメントの留保は妥当。現行コードを照合し、以下を確定・修正した。本文の③結論(root cause は ele_num snapshot でない)は維持し、race 在処の表現を弱めた。

- **「native PfaPack 競合」は否定。** active path は pure Julia(`MVMCOptimizers.jl/src/calculate_m_all.jl:202` の `utu2inv!` が有効、`:203` の `cimpl_utu2inv!` は comment-out。decomposition は `julia_zsktf2!`/`julia_dsktf2!`)。numeric kernel(`calculate_m_all_*_pfapack!` → `calculate_m_all_child_*!` → `utu2inv!`/`julia_*sktf2!`)は `with_pfapack_call_lock` 内で直列化され(`MVMCOptimizers.jl/src/vmc_sampling.jl:1404`)、`utu2inv!` 内の `BLAS.trmm!`(`PfaPack.jl/src/utu2.jl:163`)も lock 内。よって numeric/native/BLAS の並行 contention は失敗 path では起きない。→ コメントの指摘どおり、本文の「native PfaPack 競合の方が真相に近い」は撤回。
- **`data` 共有も否定(Variant P)。** `process_sample_range!` が closure capture する `data::ExpertModeData` を per-worker `deepcopy(data)` にしても **4/48 fail**。よって `data` の共有/遅延 mutation も race 元ではない。
- **module-level mutable global は無し。** `MVMCOptimizers.jl/src/*.jl` と `PfaPack.jl/src/*.jl` に `const/global` の Ref/Dict/Vector/Array 等の共有可変 global は見当たらず(`@turbo` は lock 内 kernel のみ)。
- **失敗率は isolation 強化で逓減するが 0 にはならない**(OLD 14 → N 7 → P 4 /48)。一方 rebuild ループは 0(累計 ~240 比較で 0)。これは「特定の共有資源を除去した」のではなく「timing 摂動の強さの違い」と整合的。
- コメントの「old path の copyto! 後に local snapshot が別 sample 相当に見える」という前段観測は、本検証の Variant 群(C/N/P すべて fail)とも整合する。永続 buffer は前後で整合(BADINIT=0/POST_PERM=0)である一方、threaded path の local snapshot 経路だけが observed instability を示す、という表現が安全。

**現時点の正味の結論**:race は実在し深刻だが、(a) `VMCOptimizationState` の全 state field、(b) `data`、(c) lock 済 numeric/BLAS kernel、(d) module global、のいずれでも説明できない。残る候補は unlocked な green/Hamiltonian 評価部に絡む Julia scheduling / closure capture / timing 依存。**正の site は未特定**であり、強い断定はしない。

再現一式(harness / variant generator / コマンド / case list)は [`2026-06-09-maincal-thread-race-repro/`](2026-06-09-maincal-thread-race-repro/) に保存(Variant A/B/C/G/J/L/M/N/P を `203f7c1`/`d3e0ac9` から決定論的に生成)。

## 追加検証 その2 — closure boxing 仮説(2026-06-09 22:45 JST, Claude Opus 4.8)

前節で残した「unlocked 並行部の closure capture / boxed 変数」仮説を検証。`process_sample_range!` は `vmc_main_cal!` 内のネスト closure で、`@threads :static` により worker タスクから並行に呼ばれる。Julia では、closure 内で代入され かつ 親スコープでも参照される変数は `Core.Box` 化されて **全 worker 間で共有**されるため、これが race 元になり得る(有力仮説だった)。

検証方法:`vmc_main_cal!` の kwarg body(closure を含む実体)を `Base.bodyfunction(which(vmc_main_cal!, Tuple{ExpertModeData, VMCOptimizationState, CTimer}))` で取得し、その lowered CodeInfo(`Base.uncompressed_ir`)を `Core.Box` で走査。

結果:body method は `#vmc_main_cal!#101`、**`Core.Box` 出現 = 0**。entry wrapper(2/3-arg)も 0。→ **`process_sample_range!` に boxed capture は無く、worker 間で共有される可変 box は存在しない。closure-boxing 仮説は否定**。

追認(2026-06-09 22:46 JST, GPT-5 Codex):現行 `d3e0ac9` でも `Base.bodyfunction(which(vmc_main_cal!, Tuple{ExpertModeData,VMCOptimizationState,CTimer})) == #vmc_main_cal!#101` を確認し、その body method に対する `Base.uncompressed_ir` 走査で `Core.Box count = 0` を再確認した。closure boxing 仮説を否定する本文の結論は妥当。

これで除外済みの候補は次のとおり:(a) `VMCOptimizationState` の全 state field(Variant N)、(b) `data::ExpertModeData`(Variant P)、(c) lock 済 numeric / native / `BLAS.trmm!` kernel、(d) module-level mutable global、(e) closure boxed capture。**いずれでも説明できない。** 残るのは Julia 1.12 の threading/codegen 起因の可能性、または現状の isolation 群では捕捉できない極めて微妙な共有(要 ThreadSanitizer 級の動的検出)。本検証単独では **正の race site の特定に至らず**、強い断定はしない。

注:worker の計算は理屈上 deterministic(chunk 固定・per-worker local matrices・merge 順固定・numeric kernel は lock 直列)であるはずなのに threaded 結果が ~15–29% で serial と乖離する、という観測自体が「並行領域内のどこかで shared mutable を読む/書く真の race」を強く示唆する。にもかかわらず上記 (a)–(e) で捉えられない点が本件の核心的な未解明部分。

## リスク

- 修正は **無害**(値不変・現状テスト緑)だが **root-cause fix ではない**。race はコードに残存し、CPU / thread 数 / Julia・compiler / PfaPack version / GC 挙動 / 近傍コード変更次第で再浮上し得る。
- self-consistency 回帰テストは **timing 依存**(old でも `nthreads=1` では pass していた)。緑でも真の race を確実には守れない。

## 推奨

1. d3e0ac9 は revert 不要(無害)。ただし「ele_num snapshot を直した root-cause fix」とは扱わない。commit/PR 文面の因果説明は訂正が望ましい。
2. **production で sample-level MAINCAL threading の opt-in を有効化しない。** CI の default C-parity guard(opt-in off)を安全網として維持。
3. 真の原因究明(「追加検証」「その2」で `data`・native/BLAS・state field・global・closure boxing は除外済み。残る方向):
   - Julia 1.12 の threading/codegen 起因の可能性(別 Julia version / `-O1` / `--check-bounds=yes` での再現差、最小再現の切り出し)。
   - 動的 race detection:全共有 buffer の per-iteration checksum、`@threads` を逐次 `for` に置換した場合との bit 一致確認、または TSan 相当ツール。
   - `merge_thread_accumulators!` / `finalize_oo_store!` / `clear_sropt_store!` など barrier 後処理と per-worker accumulator の境界を、worker 数 1〜N で bit 単位比較。

## 再現手順(参考)

1. old src: `git show 203f7c1:MVMCOptimizers.jl/src/vmc_main_cal.jl`。
2. stress harness: `test_unit_threading.jl` の `using`(1–14 行)+ fixture 定義(368–587 行)を抽出し、`run_threading_maincal_fixture(; requested_threads, n_samples, case...)` を `requested_threads=1` と `8` で比較、`threading_snapshots_match` で判定するループを付加。
3. 実行: `OMP_NUM_THREADS=1 JULIA_NUM_THREADS=8 julia --project=@. <harness>`。
4. Variant は `make_vmc_main_cal_worker_state`(`src/vmc_main_cal.jl:2592`)のエイリアス代入を `copy(...)` / `deepcopy(...)` に、ele_num block(`copyto!` 行)を rebuild ループに置換して生成。

(検証終了時、working tree は committed fix `d3e0ac9` のまま変更なしに復元済み。)
