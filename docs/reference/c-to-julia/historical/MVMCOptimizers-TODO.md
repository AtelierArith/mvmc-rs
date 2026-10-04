# TODO

最終更新: 2026-01-25

## 🎉 VMCParaOpt 移植完了 (2024-12-08)

### 検証結果サマリー

**C実装とJulia実装の完全一致を確認しました！**

| 検証項目 | 状態 | 詳細 |
|----------|------|------|
| 乱数シーケンス同期 | ✅ 完全一致 | `makeInitialSample`の`mi`, `si`値が一致 |
| Step 0 エネルギー期待値 | ✅ 完全一致 | `0.800848976685722...` (15桁精度) |
| Step 0 SR最適化 (S行列) | ✅ 完全一致 | 5x5行列が完全一致 |
| Step 0 SR最適化 (g vector) | ✅ 完全一致 | 最初の10要素が完全一致 |
| Step 0 SR最適化 (解r) | ✅ 完全一致 | 最初の10要素が完全一致 |
| Step 1 SR最適化 | ✅ 完全一致 | S, g, rすべて一致 |
| パラメータ更新 | ✅ 完全一致 | 更新前後の値が一致 |
| 出力ファイル (zvo_out.dat) | ✅ 完全一致 | エネルギー値が完全一致 |

### 修正した箇所

#### 1. SFMT19937.jl (`SFMT19937.jl/src/SFMT19937.jl`)
- **問題**: `rand(rng, range)`が rejection sampling を使用し、C実装の`gen_rand32() % N`と異なる乱数消費パターン
- **修正**: simple modulo方式（`gen_rand32() % len`）に変更
- **影響**: 乱数シーケンスの完全同期を実現

#### 2. パラメータ初期化 (`MVMCExpertModeParsers.jl/src/utils/parameter_init.jl`)
- **問題**: `init_parameter!`が`orbital_terms`の数（256個）だけ乱数を消費していた
- **修正**: `n_orbital_idx`（64個）だけ乱数を消費するように変更
- **影響**: C実装の`InitParameter()`と同じ乱数消費パターンを実現

#### 3. 交換候補生成 (`MVMCOptimizers.jl/src/vmc_sampling.jl`)
- **問題**: `make_candidate_exchange`のロジックがC実装と異なり、HeisenbergChainで無限ループ
- **修正**: C実装の`makeCandidate_exchange`と同じロジックに書き換え
- **影響**: 交換更新が正しく動作

#### 4. LocSpinパーサー (`MVMCExpertModeParsers.jl/src/parsers/locspin_parser.jl`)
- **問題**: `locspn.def`の5行ヘッダーをスキップしていなかった
- **修正**: `IGNORE_LINES_IN_DEF = 5`を追加
- **影響**: `loc_spn`配列が正しく読み込まれる

#### 5. QPTransパーサー (`MVMCExpertModeParsers.jl/src/utils/orbital_qptrans_utils.jl`)
- **問題**: `qptransidx.def`の3列フォーマット（符号なし）に対応していなかった
- **修正**: 3列の場合は`itmpsgn = 1`をデフォルトに
- **影響**: QPTransマッピングが正しく読み込まれる

### テスト環境
- **主要テストケース**: HeisenbergChain (16サイト)
  - 乱数シード: 123456789
  - 最適化ステップ: 3ステップ
  - サンプル数: 1000
- **利用可能なexampleファイル**:
  - `example_heisenbergchain.jl` - Heisenberg鎖模型
  - `example_heisenbergsquare.jl` - Heisenberg正方格子模型
  - `example_hubbard_square.jl` - Hubbard正方格子模型
  - `example_hubbard_triangular.jl` - Hubbard三角格子模型
  - `example_kondo_chain.jl` - Kondo鎖模型
  - `example_spin_kagome.jl` - Kagome格子スピン模型
  - `example_spin_kitaev.jl` - Kitaevスピン模型
- **ベースラインテストファイル**:
  - `test_example_*_baseline.jl` - 各モデルでのベースライン検証用

---

## Slater Matrix Update (`slater_update.jl`)

### OrbitalSgn の完全な実装 ✅
- **実装完了**: `MVMCExpertModeParsers.jl` のパーサーで OrbitalSgn を読み取り、`ExpertModeData.orbital_sgn` に格納する機能を実装
- **実装内容**:
  - `OrbitalTerm` に `sign` フィールドを追加（デフォルト値: 1）
  - `orbital_parser.jl` で4列目の符号を読み取り
  - `build_orbital_sgn_matrix!()` 関数で OrbitalSgn マトリックスを構築
  - APFlag を考慮した符号の設定（APFlag == 0 の場合は全て +1）
- **関連ファイル**:
  - `MVMCExpertModeParsers.jl/src/types/expert_types.jl`: `OrbitalTerm` 型定義
  - `MVMCExpertModeParsers.jl/src/parsers/orbital_parser.jl`: パーサー
  - `MVMCExpertModeParsers.jl/src/utils/orbital_qptrans_utils.jl`: マトリックス構築関数
  - `MVMCOptimizers.jl/src/slater_update.jl`: `update_slater_elm_fcmp!` 関数

### QPTrans マッピングの完全な実装 ✅
- **実装完了**: `MVMCExpertModeParsers.jl` のパーサーで QPTrans マッピングを読み取り、`ExpertModeData` に格納する機能を実装
- **実装内容**:
  - `ExpertModeData` に `qp_trans`, `qp_trans_inv`, `qp_trans_sgn`, `qp_opt_trans`, `qp_opt_trans_sgn` フィールドを追加
  - `build_qp_trans_mappings!()` 関数で `qptransidx.def` から QPTrans マッピングを読み取り
  - APFlag を考慮した符号の設定（APFlag == 0 の場合は全て +1）
- **関連ファイル**:
  - `MVMCExpertModeParsers.jl/src/types/expert_types.jl`: `ExpertModeData` 型定義
  - `MVMCExpertModeParsers.jl/src/utils/orbital_qptrans_utils.jl`: マッピング構築関数
  - `MVMCExpertModeParsers.jl/src/MVMCExpertModeParsers.jl`: パース後の呼び出し
  - `MVMCOptimizers.jl/src/slater_update.jl`: `update_slater_elm_fcmp!` 関数

## 実装済み機能
- ✅ OrbitalSgn の読み取りとマトリックス構築
- ✅ QPTrans マッピングの読み取りと構築
- ✅ APFlag を考慮した符号の設定
- ✅ `update_slater_elm_fcmp!` での ExpertModeData からのデータ取得

## VMCParaOpt 移植状況

### 概要
Cコードの `VMCParaOpt` 関数（`mVMC/src/mVMC/vmcmain.c:331-528`）をJuliaに移植する作業。

**移植対象**: `vmc_para_opt!` 関数（`MVMCOptimizers.jl/src/vmc_para_opt.jl`）

### 移植進捗率（概算）
- **構造・フロー**: 100% ✅
- **データ構造**: 100% ✅
- **関数実装**: 100% ✅ (コア機能完了)

### 完全実装済み ✅
1. **`update_slater_elm_fcmp!` / `update_slater_elm_fsz!`** → `slater_update.jl`
   - Slater行列要素の更新（sz保存版/一般化版）

2. **`update_qp_weight!`** → `qp_weight_update.jl`
   - 量子射影重みの更新

3. **`weight_average_we!`** → `weight_average.jl`
   - エネルギーの重み付き平均

4. **`weight_average_sr_opt!` / `weight_average_sr_opt_real!`** → `weight_average.jl`
   - SR最適化データの重み付き平均（複素数版/実数版）

5. **`stochastic_opt!` / `stochastic_opt_cg!`** → `stochastic_opt.jl`
   - 確率的最適化（LAPACK版/CG版）

6. **`sync_modified_parameter!`** → `parameter_sync.jl`
   - パラメータの同期

7. **`store_opt_data!`** → `data_io.jl`
   - 最適化データの保存

8. **VMCサンプリング関数** → `vmc_sampling.jl`
   - `vmc_make_sample!` - 複素数版、sz保存 ✅
   - `vmc_make_sample_fsz!` - 複素数版、一般化 ✅（スタブ実装）
   - `vmc_make_sample_real!` - 実数版、sz保存 ✅
   - `vmc_make_sample_fsz_real!` - 実数版、一般化 ✅（スタブ実装）
   - `vmc_bf_make_sample!` - Back Flow版、複素数 ✅（スタブ実装）
   - `vmc_bf_make_sample_real!` - Back Flow版、実数 ✅（スタブ実装）
   - **実装内容**:
     - ✅ Metropolis-Hastingsアルゴリズム
     - ✅ 電子ホッピング更新
     - ✅ 初期サンプル生成
     - ✅ 電子配置更新関数
     - ✅ プロジェクションカウント更新
     - ✅ Pfaffian計算（完全実装完了）
     - ✅ 逆行列更新（完全実装完了）
     - ✅ 交換更新（完全実装完了）
     - ❌ 局所スピンフリップ更新（未実装）

9. **メイン計算関数** → `vmc_main_cal.jl`
   - `vmc_main_cal!` - sz保存版 ✅
   - `vmc_main_cal_fsz!` - 一般化版 ✅（スタブ実装）
   - `vmc_bf_main_cal!` - Back Flow版 ✅（スタブ実装）
   - **実装内容**:
     - ✅ 各サンプルでのエネルギー計算
       - Transfer項、Coulomb項、Hund項、Exchange項、InterAll項の計算
       - Green関数（1体/2体）の計算
     - ✅ SR最適化量（O, OO, HO）の計算
       - プロジェクション項（Gutzwiller, Jastrow）の対数微分
       - Slater項の対数微分
       - Back Flow項の対数微分（スタブ実装）
     - ✅ 重み付き平均の累積
     - ⚠️ InterAllTermのスピン情報はデフォルト値で処理（TODO: ファイルからパース）
     - ⚠️ FSZ版の`slater_elm_diff_fsz`が未実装（TODO）

10. **データ出力関数** → `data_io.jl`
    - `output_data!` - データ出力（zvo_out.dat, zvo_var.dat） ✅
    - `output_opt_data!` - 最適化パラメータ出力（zqp_opt.dat） ✅
    - **実装内容**:
      - ✅ ファイルの作成/オープン
      - ✅ 18桁精度でのデータ出力（C実装と一致）

### 未実装（低優先度）❌
1. **タイマー機能**
   - `StartTimer` / `StopTimer` - タイマー管理
     - Cコード: 複数箇所で使用
     - Julia: 未実装（`TimerOutputs.jl`を使用可能）
   - `OutputTime(step)` - タイマー出力
     - Cコード: 342行目、518行目で使用
     - Julia: 未実装

2. **ファイルI/O機能**
   - `FlushFile(step, rank)` - ファイルフラッシュ
     - Cコード: 515行目で使用
     - Julia: 未実装（必要に応じて`flush()`を使用可能）

3. **MPI機能**
   - `ReduceCounter(comm_child2)` - MPIカウンター集約
     - Cコード: 435行目で使用
     - Julia: 未実装（MPI対応が必要、`reduce_counter!`はシングルプロセス版）

### 主な相違点
1. **MPI処理**: CコードはMPI通信を使用、Julia実装は現状シングルプロセス想定
2. **タイマー機能**: Cコードは詳細なタイマー管理、Julia実装は未実装
3. **デバッグ出力**: Cコードは`#ifdef`で制御、Julia実装はファイル出力で対応

### 関連ファイル
- **Cコード**: `mVMC/src/mVMC/vmcmain.c:331-528`
- **Julia実装**: `MVMCOptimizers.jl/src/vmc_para_opt.jl`
- **データ構造**: `MVMCOptimizers.jl/src/types.jl`
- **グローバル変数依存関係**: `VMCParaOpt_global_variables.md`

## VMCサンプリング実装詳細

### 実装完了 ✅

#### 1. データ構造の拡張
- `ElectronConfiguration`に以下を追加:
  - 一時配列（`tmp_ele_idx`, `tmp_ele_cfg`, `tmp_ele_num`, `tmp_ele_proj_cnt`, `tmp_ele_spn`）
  - バーンイン用配列（`burn_ele_idx`, `burn_ele_cfg`, `burn_ele_num`, `burn_ele_proj_cnt`, `burn_ele_spn`）
  - カウンター配列（`counter`）

#### 2. ヘルパー関数
- `get_loc_spn_array()` - LocSpn配列の取得
- `make_proj_cnt!()` - プロジェクションカウントの計算
- `update_proj_cnt!()` - プロジェクションカウントの更新
- `log_proj_ratio()` - プロジェクション比の計算
- `update_ele_config!()` / `revert_ele_config!()` - 電子配置の更新/復元
- `make_candidate_hopping!()` - ホッピング候補の生成
- `make_candidate_exchange()` - 交換候補の生成 ✅ (C実装と同じロジック)
- `get_update_type()` - 更新タイプの選択

#### 3. 初期サンプル生成
- `make_initial_sample!()` - 初期電子配置の生成 ✅
  - 局所スピンサイトの処理
  - 遍歴電子のランダム配置
  - プロジェクションカウントの計算
  - **C実装と完全同期**: 同じ乱数シードで同じ`eleIdx`を生成

#### 4. メインサンプリングループ
- `vmc_make_sample!()` / `vmc_make_sample_real!()` - Metropolis-Hastingsループ ✅
  - ホッピング更新の実装
  - 交換更新の実装
  - 受容/棄却判定
  - サンプル保存機能

#### 5. Pfaffian計算関数 ✅
- `calculate_m_all_fcmp!()` - 初期Pfaffian計算
  - **実装完了**: MVMCPfaPack.jlの`calculate_m_all_fcmp!`を統合
- `calculate_new_pf_m2!()` - ホッピング後のPfaffian更新
  - **実装完了**: Sherman-Morrison公式を使用した効率的な更新
- `calculate_new_pf_m_two2!()` - 2電子交換後のPfaffian計算
  - **実装完了**: Cコードの`CalculateNewPfMTwo2_fcmp`をJuliaに移植
- `calculate_log_ip_fcmp!()` - 内積計算
  - **実装完了**: QPFullWeightを使用した正確な内積計算

#### 6. 逆行列更新関数 ✅
- `update_m_all!()` - ホッピング後の逆行列更新
  - **実装完了**: Sherman-Morrison公式を使用した効率的な更新
- `update_m_all_two!()` - 2電子交換後の逆行列更新
  - **実装完了**: Cコードの`UpdateMAllTwo_fcmp`をJuliaに移植

### 未実装 ❌

#### その他の機能
1. **局所スピンフリップ更新**
   - `vmc_sampling.jl`の`get_update_type()`で`LOCALSPINFLIP`が未実装
   - Cコード: `vmcmake.c`の`makeCandidate_localspinflip`を参照
   - 優先度: 中（局所スピンを持つモデルで必要）

2. **バーンインサンプルの保存/復元**
   - `ElectronConfiguration`に`burn_*`配列は定義済みだが未使用
   - Step 0では新規初期サンプル、Step 1+ではバーンインサンプルを使用する仕様
   - 優先度: 低（現状は動作している）

3. **Doublon-Holon相関因子の完全なサポート**
   - `make_proj_cnt!`と`update_proj_cnt!`で部分的に実装済み
   - `vmc_sampling.jl`の`make_proj_cnt!`にTODOコメントあり（192行目、314行目、335行目）
   - 2-siteと4-siteのDoublon-Holon項の完全な統合が必要
   - 優先度: 中（特定のモデルで必要）

4. **FSZ版（一般化版）の完全実装**
   - `vmc_make_sample_fsz!`と`vmc_main_cal_fsz!`はスタブ実装
   - `vmc_main_cal.jl`の`slater_elm_diff_fsz`が未実装（1997行目にTODO）
   - 優先度: 中（一般化スピン模型で必要）

5. **InterAllTermのスピン情報のパース**
   - `vmc_main_cal.jl`の`calculate_hamiltonian`でスピン情報をデフォルト値で処理（852行目にTODO）
   - InterAllファイルからスピン情報を読み取る機能が必要
   - 優先度: 低（現状は動作している）

### 実装ファイル
- **メイン実装**: `MVMCOptimizers.jl/src/vmc_sampling.jl`
- **データ構造**: `MVMCOptimizers.jl/src/types.jl`
- **Cコード参考**:
  - `mVMC/src/mVMC/vmcmake.c` - メインサンプリング関数
  - `mVMC/src/mVMC/matrix.c` - Pfaffian計算
  - `mVMC/src/mVMC/pfupdate.c` - Pfaffian更新
  - `mVMC/src/mVMC/projection.c` - プロジェクションカウント

---

## 今後の課題（優先度順）

### 高優先度 🔴
1. **他のモデルでの検証**
   - Hubbard模型（`example_hubbard_square.jl`, `example_hubbard_triangular.jl`）
   - t-J模型
   - より大きなシステムサイズ
   - 複数のexampleファイルが存在するが、全モデルでの検証が必要

2. **パフォーマンス最適化**
   - プロファイリングによるボトルネック特定
   - メモリ割り当ての最適化
   - BLAS/LAPACK呼び出しの効率化
   - `@turbo`マクロの活用範囲の拡大

### 中優先度 🟡
3. **FSZ版（一般化版）の完全実装**
   - `vmc_make_sample_fsz!`の完全実装
   - `vmc_main_cal_fsz!`の完全実装
   - `slater_elm_diff_fsz`の実装（`vmc_main_cal.jl`）
   - 一般化スピン模型での動作確認

4. **局所スピンフリップ更新の実装**
   - `make_candidate_localspinflip!`の実装
   - `get_update_type()`での`LOCALSPINFLIP`の有効化
   - 局所スピンを持つモデルでの動作確認

5. **Doublon-Holon相関因子の完全な統合**
   - `make_proj_cnt!`での2-site/4-site項の完全な処理
   - `update_proj_cnt!`での更新ロジックの実装
   - プロジェクションカウント計算への統合

6. **MPI並列化**
   - `MPI.jl`を使用した並列化
   - C実装と同じ通信パターンの実装
   - `reduce_counter!`のMPI対応

7. **複素数版の検証**
   - AllComplexFlag = 1 の場合のテスト
   - 複素数パラメータの最適化
   - 複素数版と実数版の結果の一致確認

### 低優先度 🟢
8. **Back Flow版の実装**
   - `vmc_bf_make_sample!`の完全実装
   - `vmc_bf_make_sample_real!`の完全実装
   - `vmc_bf_main_cal!`の完全実装
   - Back Flow項の完全なサポート

9. **バーンインサンプルの保存/復元**
   - Step間でのサンプル状態の保存
   - バーンイン期間の管理
   - サンプル品質の向上

10. **InterAllTermのスピン情報のパース**
    - InterAllファイルからのスピン情報読み取り
    - スピン依存のハミルトニアン項の正確な処理

11. **タイマー機能の実装**
    - `StartTimer` / `StopTimer`の実装
    - `OutputTime(step)`の実装
    - パフォーマンス測定の詳細化

12. **デバッグ機能の強化**
    - より詳細なデバッグ出力
    - エラーハンドリングの改善
    - 検証用のユーティリティ関数
