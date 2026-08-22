---
date: 2026-06-06
datetime: 2026-06-06 10:31 JST
model: Claude Opus 4.8 (1M context)
status: review
topic: Julia-mVMC v0.3 DH2/DH4 input compatibility plan のバグ・不具合レビュー
target:
  repository: Julia-mVMC
  reviewed_document: docs/superpowers/plans/2026-06-06-julia-mvmc-dh2-dh4-compat-plan.md
  c_reference: mVMC (projection.c / parameter.c / readdef.c / vmcmain.c)
  method: multi-agent adversarial review (58 agents / 52 findings) + 独立ソース確認
---

# Julia-mVMC DH2/DH4 互換 plan — バグ・不具合レビュー

対象文書: [`docs/superpowers/plans/2026-06-06-julia-mvmc-dh2-dh4-compat-plan.md`](../superpowers/plans/2026-06-06-julia-mvmc-dh2-dh4-compat-plan.md)
（warn-only stub からの DH2/DH4・InDH2/InDH4 input compatibility 実装 plan、GPT-5 (Codex) 作成）

## 0. 調査方法

- C-mVMC のソース（`mVMC/src/mVMC/{projection.c, parameter.c, readdef.c, vmcmain.c, setmemory.c}`）を ground truth として、index 式・layout offset・count・shift・file format・overlay 順序を確定。
- 現行 Julia 実装（`MVMCExpertModeParsers.jl/src/**`, `MVMCOptimizers.jl/src/**`）と plan の主張を突き合わせ。
- 6 次元（C-contract fidelity / Julia 現状 fidelity / layout-data 設計 / counts-sampling 設計 / loaders-sync 設計 / edge-case hunt）× adversarial verification の multi-agent review（58 agents・52 findings・1 refuted）を実行し、各 finding を cited source で再検証。
- 上記に加え、orchestrator が独立に C・Julia ソースを直接読んで主要主張を確認。

**重要な前提:** multi-agent が出した blocker/high の多くは「現状 Julia が DH 未実装」を再確認しただけで、これは plan の動機そのもの（＝plan の欠陥ではない）。本レビューは **plan 自身の誤り・不足・曖昧さ**だけを抽出する。

## 1. 総評

**plan の中核は正しい。** 以下はすべて C と一致することを確認した:

- projection layout の offset 連鎖 `Gutz | Jastrow | SpinJastrow | DH2 | DH4 | RBM | Slater | OptTrans`
- DH2/DH4 の index 式 `idx = offset + xn + (xi + 2*xm) * N...Idx`、bin 数 6 (DH2) / 10 (DH4)
- count ロジック（singly-occupied center skip、holon center は近傍 doublon を数え、doublon center は近傍 holon を数える）
- `UpdateProjCnt` の「DH tail を全消去して eleNum から再計算」戦略
- `shiftDH2`/`shiftDH4` の bin grouping（DH2 は 3 bins、DH4 は 5 bins を平均で引く）と、`gShift` を Gutzwiller に足してから GJ shift する順序
- `LogProjVal`/`LogProjRatio` の `creal(Proj)` 重み付け
- In-overlay が initial.def を上書きする順序（C は `InitParameter → ReadInitParameter → ReadInputParameters`、`mVMC/src/mVMC/vmcmain.c:264-268`）
- loader の float 数 `6 + 3*(NProj + NRBM + NSlater + NOptTrans)`、`AllComplexFlag` の構成

multi-agent が出した「shiftDH4 は 5 bins ではない」という指摘は**誤検出**で、plan の記述（5 bins）が正しい。

**一方、実装するとバグになる/数値が C と静かに分岐する欠陥が複数存在する。** 以下に severity 順で示す。最優先は D1・D2。

### Severity 凡例

| 記号 | 意味 |
|------|------|
| High | 実装後、DH ありで数値が C と分岐、またはメモリ破壊。plan の文面修正が必須 |
| Medium | 取り違えやすい契約・順序。明記しないと実装者がバグを入れる確率が高い |
| Low | 曖昧さ・補足。実装前に潰すと安全 |

---

## 2. A. 実装すると bug になる plan の欠陥（要修正）

### D1【High】DH-shift 有効化条件に「Gutzwiller が全て最適化」前提が抜けている

- **plan 該当:** §2.5（L134-135）, §4.6（L275）
- **plan の記述:** 「shift flags は Gutzwiller が存在し、対応 DH slice の real OptFlag が全て最適化なら有効」
- **C の実際**（`mVMC/src/mVMC/parameter.c:255-312`, とくに L262-267）:

  ```c
  void SetFlagShift() {
    FlagShiftGJ = FlagShiftDH2 = FlagShiftDH4 = 0;
    if (NGutzwillerIdx == 0) return;
    start = 0; end = start + NGutzwillerIdx;
    for (i = start; i < end; i++) {
      if (OptFlag[2*i] != 1) return;   // ← Gutzwiller が 1 つでも固定なら全 flag=0 で early return
    }
    ... (Jastrow → SpinJastrow skip → DH2 → DH4)
  }
  ```

  すなわち DH-shift は「**Gutzwiller が全パラメータ最適化(`OptFlag[2*i]==1`)**」が必須前提。Gutzwiller を 1 つでも固定すると `FlagShiftDH2`/`FlagShiftDH4` は DH slice が全最適化でも 0 のまま。なお `FlagShiftGJ` が（Jastrow 固定で）0 でも DH flag は独立に立ち得る。早期 return を起こすのは **Gutzwiller の固定のみ**。
- **影響:** Gutzwiller を一部固定した DH 入力で、Julia は DH-shift を発火させ C と分岐する。gauge 取り方が変わり optimization 軌道がずれる。
- **修正:** §2.5/§4.6 の条件を次に置換 —
  「DH-shift flag が立つのは (a) `NGutzwillerIdx > 0` **かつ (b) 全 Gutzwiller real OptFlag==1** かつ (c) 当該 DH slice の real OptFlag が全て==1 のとき。Gutzwiller が 1 つでも固定なら `FlagShiftDH2`/`FlagShiftDH4` は 0。」
- **検出:** 4 agent が独立指摘 + orchestrator 独立確認。

### D2【High】projection-count バッファを `layout.n_proj` で確保する要求が明示されていない

- **plan 該当:** §4.1, §4.5
- **plan の状態:** helper `n_projection_parameters(data)` を定義し、log 関数で「count vector 長 == layout.n_proj を assert/guard」とは書くが、**count バッファ自体の確保サイズ**には言及がない。
- **Julia の実際:** `ele_proj_cnt` / `tmp_ele_proj_cnt` / `burn_ele_proj_cnt` / `proj_cnt_new` は `MVMCOptimizers.jl/src/types.jl:128-162, 250` で `zeros(Int, n_proj)`（と `n_sample*n_proj`）として確保され、`n_proj` は constructor 引数（`types.jl:117, 321`）。呼び出し側は `n_proj = n_gutzwiller + n_jastrow`（`MVMCOptimizers.jl/src/initial_params.jl:58` ほか）。
- **影響:** ここを DH 込みに直さないと、`_count_dh2!`/`_count_dh4!` が `proj_cnt[dh2_offset + …]` で**配列外書き込み**（`BoundsError` か `@inbounds` 下でメモリ破壊）。さらに下流の `MVMCOptimizers.jl/src/vmc_main_cal.jl` は `n_proj = length(ele_proj_cnt)` を信頼するため、バッファ長の誤りが全 offset 計算に伝播する。
- **修正:** §4.5 に明示 —
  「`VMCOptimizationState` / `ElectronConfiguration` / `SamplingWorkspace` を構築する `n_proj` は必ず `projection_layout(data).n_proj`（DH 込み）であること。constructor 内に `@assert n_proj == projection_layout(data).n_proj` を入れる。DH 入力で `ElectronConfiguration` の配列サイズが count 操作と一致することを確認する unit test を追加。」
- **検出:** 2 agent（partially-confirmed）+ orchestrator が `types.jl` の確保箇所を直接確認。

### D3【Medium-High】opt-table 第1列の扱いが def ファイルと In-overlay で正反対なのに plan が区別していない

- **plan 該当:** §2.2（opt-table 行形式 `local_param_index opt_flag`）, §4.3, §4.4
- **C の実際（2 つの読み取りで第1列の意味が逆）:**
  - **def ファイルの opt-table** — `GetInfoOpt`（`mVMC/src/mVMC/readdef.c:2309-2324`）: 第1列を変数 `i` に読むが**捨てる**。`fidx` をインクリメントしながら**ファイル行順**で `ArrayOpt[2*fidx]` に詰める。→ **index 列は無視**。
  - **In-overlay** — `InDH2`/`InDH4`（`mVMC/src/mVMC/readdef.c:1449-1451, 1462-1464`）: 第1列 `idx` を**そのまま使い** `Proj[idx+count]` に書く。→ **index 列を target offset として使用**。
- **問題:** plan §4.3「parse the DH optimization table into local DH opt flags」と §4.4「local indices … overwrite `params[local+1]`」は個別には C と整合するが、**両者の対比を書いていない**。同じ `(index, value)` 形式の 2 表が逆の規約を持つため、実装者が片方の規約を両方に適用しやすい。def opt-table を index 列で詰めると、行が順不同のとき C と分岐する。
- **修正:**
  - §2.2/§4.3 に「def の opt-table はファイル行順で逐次格納し、第1列(local index)は C では discard される（cosmetic）」を明記。
  - §4.4 に「In-overlay は第1列を target local index として使用する」を明記し、両者を対比。

### D4【Medium】InDH2/InDH4 overlay の検証要件が未指定（C は無検証 → fail-loud と C fixture 再現の緊張）

- **plan 該当:** §4.4（L242-247）
- **C の実際:** In-overlay は range / duplicate / missing を**一切検証しない**（`mVMC/src/mVMC/readdef.c:1449-1451, 1462-1464`）。out-of-range の `idx` で `Proj[idx+count]` は buffer overflow（C 側の潜在バグ）。これは `InSpinJastrow` が `seen[]` 配列で範囲・重複・欠落を検証している（`readdef.c:1408-1437`）のと対照的。
- **plan の状態:** overlay については「file count == n_dh2 を検証」しか書いておらず、local index の検証要件が無い。§2.2(L94) の「Julia は out-of-range index 等で fail loud」は **def ファイル**側の記述で、overlay には適用が明示されていない。
- **注意（双方向のリスク）:** plan は strict runner で malformed を hard fail 推奨（open Q5）。だが fail-loud を入れる場合、**committed C fixture が range 内・全件揃っていることを保証しないと、C が受理する fixture を Julia が誤って弾く**。
- **修正:** §4.4 に明示 —
  「InDH2/InDH4 overlay は local index が `[0, 6*n_dh2)` / `[0, 10*n_dh4)`・重複なし・欠落なし・行数一致を検証し、違反は hard error。`fscanf` 相当が途中で尽きた(file too short)場合も error。」
  test plan §5 に out-of-range / duplicate / short-file の overlay failure test を追加。委員 C fixture は range 内・全件であることを fixture 生成時に保証。

### D5【Medium】FSZ の `ri==rj` early-return を plan が扱っていない

- **plan 該当:** §2.3（FSZ path）, §4.5（FSZ make/update equivalents）
- **plan の記述:** 「FSZ path has the same DH recompute requirement」
- **C の実際:** `UpdateProjCnt_fsz`（`mVMC/src/mVMC/projection.c:308-454`）は on-site spin flip（`ri==rj`）で SpinJastrow 更新後 **L361 で return** し、DH を再計算しない（占有数 `n0+n1` 不変のため DH count 不変、正しい）。DH 再計算は `ri!=rj` のみ。
- **影響:** 数値的には無条件再計算でも `ri==rj` で同一結果になる（占有不変）ため致命ではないが、(1) C 構造を模倣する port は early-return を正しく入れる必要があり、(2) 「update 後 == fresh `make_proj_cnt!`」の不変条件 test を `ri==rj` ケースでも回すべき。現行 FSZ 関数は `MVMCOptimizers.jl/src/vmc_sampling.jl` の `update_proj_cnt_fsz!`（概ね L4486-4592）に存在し DH 未対応。
- **修正:** §2.3/§4.5 に `ri==rj` の早期 return 注記と、FSZ 不変条件 test（on-site spin flip を含む）を明記。
- **検出:** 3 agent + orchestrator 独立確認。

### D6【Medium】PR DH-1 の中間状態が未定義 — offset ずれ / BackFlow 誤ルートのリスク

- **plan 該当:** §6 PR DH-1（「DH opt flag を global `optimization_flags` に適用」「sampling は guarded のまま」）
- **問題 1（offset half-state）:** DH opt flag を入れて layout(`n_proj`)を DH 込みに変えると、DH-2 が来る前の DH-1 時点で **Slater/RBM の OptFlag offset と `init_parameter!` の RNG 消費が DH 込みで動く**一方、count/param は未配線という half-state になる。DH 入力に対し Slater が誤った最適化/固定判定を受ける、RNG 系列が C と分岐する等が起こり得る。
- **問題 2（BackFlow 誤ルート）:** 現行 `n_proj_bf` は DH terms を数えて BackFlow path に分岐する（`MVMCOptimizers.jl/src/vmc_phys_cal.jl:92`, `vmc_para_opt.jl:108`, ガードは同 `:214/223/235`, `:233/249/264`）。DH-1 で DH を parse すると `n_proj_bf > 0` になり、**DH 入力が BackFlow path（error stub: `vmc_main_cal.jl:3058`, `vmc_sampling.jl:6647/6665`）に流れる**。
- **修正:** §6 DH-1 の exit criteria に明記 —
  「DH-1 の段階では DH 入力は完全に reject/guard（半処理にしない）。`n_proj_bf` は DH-1 で DH を数えないよう（BackFlow 専用へ）変更し、no-DH の offset / RNG / 出力が不変であることを test で確認。」
  no-DH の不変性は安全だが、**DH 入力の PR 境界をまたいだ挙動**を明文化すること。

---

## 3. B. 明確化すべき曖昧さ（Low〜Medium、実装前に潰す）

### D7【Low-Med】SpinJastrow zero-slot は SpinJastrow ファイル present 時に fail-loud にする

- 現行 Julia は SpinJastrow を parse しない（`MVMCOptimizers.jl/src/initial_params.jl:20`）。layout に 0 slot を持つ設計は正しいが、**SpinJastrow.def が与えられても `n_spinjastrow=0` のままになり**、C（`NSpinJastrowIdx>0`）と DH offset が静かに分岐する。
- **修正:** open Q3 を「SpinJastrow ファイルが present なら hard fail」で確定し、§4.1 の layout helper に「SpinJastrow 入力検出で error」を明記。

### D8【Low】複素 DH の imag OptFlag 規約が未記載

- C は `ArrayOpt[2*fidx+1] = (iComplxFlag>0 ? ArrayOpt[2*fidx] : 0)`（`mVMC/src/mVMC/readdef.c:2314-2318`）。すなわち complex のとき imag flag は real flag をミラー、real のとき 0。
- **修正:** open Q2 を「imag flag は complex 時 real flag をミラー、それ以外 0」で確定し、§4.3 に明記。`AllComplexFlag` への DH 寄与の畳み込みと併せて記述。

### D9【Low】keyword 配線（DH2/DH4）が具体性不足

- `MVMC_KEYWORDS`（`MVMCExpertModeParsers.jl/src/utils/constants.jl`）に `DH2`/`DH4` の file エントリが無く、`parse_file_by_type!` は long name `DoublonHolon2Site` / `DoublonHolon4Site` で分岐（`MVMCExpertModeParsers.jl/src/MVMCExpertModeParsers.jl:656-664`）。namelist は keyword をそのまま file_type として渡す。
- **修正:** §4.3 に「keyword dict **と** `parse_file_by_type!` の branch の**両方**を `DH2`/`DH4` で配線（legacy long-name branch は撤去 or legacy 明示）」と具体化。

### D10【Low】neighbors 行列の格納向きを明記

- §4.2 に「DH-index `n` ごとに 1 つの `DoublonHolon2SiteIndex`、`neighbors[i, :] = [ArrayIdx[n][2*i], ArrayIdx[n][2*i+1]]`（0-based site id、転置に注意）」と明記。全 DH-index を 1 行列にまとめる/転置する誤実装を防ぐ。

### D11【Low】LogProjVal/Ratio の長さチェックは MUST(assert) に確定

- §4.5 の「assert or guard」を、strict runner では MUST(`@assert`)と確定。C は assert しない（`projCnt` 長を仮定）が、Julia は防御的に固定すべき。

---

## 4. C. plan が正しく捉えている点（バグではない確認）

- index 式・layout offset・count ロジック・shift bin grouping・`creal` 重み・overlay 勝ち順序・float-count `6 + 3*(…)`・`AllComplexFlag` 構成は **すべて C と一致**。
- §3 の現状 gap 列挙（value-bearing term、`n_proj` 重複、DH→BackFlow ルート、loader 拒否、In* warn-skip、GJ-shift のみ）は実コードと一致。
  - `n_proj = length(gutzwiller_terms) + length(jastrow_terms)` の重複は **約 15 箇所**: `opt_flag_utils.jl:25/101`, `stochastic_opt.jl:48/208/720`, `vmc_phys_cal.jl:124`, `vmc_para_opt.jl:143`, `vmc_main_cal.jl:2490/2887`, `MVMCExpertModeParsers.jl:228`, `read_input_parameters.jl:325` ほか。§4.1 の「helper へ全置換」要求は妥当だが取りこぼしリスクが高い → **置換漏れ検知の test を必須化**推奨（DH ありで n_proj が DH 込みになることを各 call site で検証）。
- 「shiftDH4 は 5 bins ではない」指摘は **誤検出**（plan が正しい）。

---

## 5. 推奨アクションと優先順位

| 優先 | ID | 内容 | 性質 |
|------|----|------|------|
| 1 | D1 | DH-shift 条件に「Gutzwiller 全最適化」前提を追加 | High / 数値分岐 |
| 1 | D2 | count バッファを `layout.n_proj` で確保する要求を明記 + assert | High / メモリ破壊 |
| 2 | D3 | def opt-table（index 列無視）と In-overlay（index 列使用）の対比を明記 | Med-High / 契約取り違え |
| 2 | D4 | In-overlay の range/dup/missing 検証要件と fixture 整合を明記 | Med |
| 3 | D5 | FSZ `ri==rj` early-return の注記 + 不変条件 test | Med |
| 3 | D6 | PR DH-1 の中間状態（DH 入力 reject、`n_proj_bf` 非 DH 化）を明記 | Med |
| 4 | D7-D11 | SpinJastrow fail-loud / imag flag 規約 / keyword 配線 / 行列向き / assert | Low |

最優先は **D1（shift 前提）と D2（バッファ sizing）**。どちらも「DH なしは合うが DH ありで静かに分岐 or 破損」する種類で、no-DH fixtures が green でも検出できない。

---

## 付録: 確認済み C contract（実装リファレンス）

| 項目 | 値 / 規則 | C ソース |
|------|-----------|----------|
| layout offset | `Gutz(0) | Jastrow(+NGutz) | SpinJastrow(+NJastrow) | DH2(+NSpinJastrow) | DH4(+6*NDH2)` | projection.c:74-163 |
| DH2 index | `offset + xn + (xi+2*xm)*NDH2`、xi=occ/2、xm∈0..2 | projection.c:128 |
| DH4 index | `offset + xn + (xi+2*xm)*NDH4`、xm∈0..4 | projection.c:159 |
| count center skip | `xi = n0+n1; if(xi==1) continue;` | projection.c:115-117 |
| count holon/doublon | holon→近傍 doublon、doublon→近傍 holon | projection.c:121-126 |
| Update DH | tail を 0 化して eleNum から全再計算 | projection.c:243-300 |
| FSZ ri==rj | SpinJastrow 後 return、DH 再計算なし | projection.c:361 |
| LogProjVal/Ratio | `creal(Proj[idx])` のみ | projection.c:32-48 |
| dh2.def 本体 | `i x0 x1 n` を `Nsite*NDH2` 行、`ArrayIdx[n][2*i]=x0`, `[2*i+1]=x1` | readdef.c:2474-2476 |
| dh4.def 本体 | `i x0 x1 x2 x3 n` を `Nsite*NDH4` 行、`ArrayIdx[n][4*i+k]=xk` | readdef.c:2503-2506 |
| opt-table 行数 | DH2:`6*NDH2`, DH4:`10*NDH4`、第1列 discard・**file 順** | readdef.c:2309-2324, 2486, 2516 |
| opt flag imag | complex 時 real をミラー、else 0 | readdef.c:2314-2318 |
| In-overlay header | line2 の整数 == `NDH2`(resp `NDH4`) を検証 | readdef.c:1444, 1457 |
| In-overlay 書込 | `Proj[idx+count]`（idx は **file の第1列**、無検証） | readdef.c:1449-1451, 1462-1464 |
| shiftDH2 | `(d/h, xn)` ごとに 3 bins(step `2*NDH2`)を平均で引き gShift 加算 | parameter.c:202-224 |
| shiftDH4 | 5 bins(step `2*NDH4`) | parameter.c:228-253 |
| SetFlagShift | Gutz 全最適化が必須、固定 1 つで全 flag=0 early return | parameter.c:262-267 |
| Sync 順序 | shiftDH2→shiftDH4→`Proj[Gutz]+=gShift`→shiftGJ→Slater rescale | parameter.c:147-161 |
| 初期化 RNG | Proj は 0 化のみ(RNG 消費なし)、RNG は RBM/Slater のみ | parameter.c:35-92 |
| param 読込 | header 6 + `NProj/NRBM/NSlater/NOptTrans` の triple | parameter.c:104-125 |
| 呼出順序 | `InitParameter → ReadInitParameter → ReadInputParameters`（In が勝つ） | vmcmain.c:264-268 |
