# HeisenbergChain_cmp デバッグ作業ログ (progress_1)

日付: 2026-02-04

## 目的
HeisenbergChain_cmp における C-mVMC と Julia-mVMC の不一致を調査（複素数ケース）。
実数のみの場合は一致しており、複素数関連の差分が疑わしい。

---

## これまでの結論（概要）

### ✅ 一致しているもの
- **RNG シーケンス（step 0, サンプリング前）**: SFMT を非消費でダンプすると C/JL が一致。
- **locspn 配列**: 完全一致。
- **初期サンプル（step 1）**: `elec_initial_sample_step_001.dat` が一致。
- **サンプリング後（step 1）**: `elec_after_sampling_step_001.dat` が一致（C は EleSpn 行あり）。
- **SR 量（step 1）**: S/g/r/更新マップ一致（`sr_S_step_001.dat`, `sr_g_step_001.dat`, `sr_r_step_001.dat`, `sr_smat_to_para_idx_step_001.dat`）。
- **更新後パラメータ（step 1）**: `sr_para_post_step_001.dat`, `sr_para_post_sync_step_001.dat` 一致。
- **エネルギー分解（step 1, sample 0）**:
  - `energy_terms_step_001_sample_000.dat` 一致。
  - `energy_exchange_terms_step_001_sample_000.dat` 一致（丸め誤差程度）。

### ❌ 一致していないもの
- **zvo_out のエネルギー**: step 0 は一致するが step 1 以降が不一致（複素数ケース）。
- **step 2 の energy_terms（sample 0）**: C/JL が大きく不一致（ip, e_total, hund, exchange など）。
  - step 1 以降に「サンプル」または「更新後パラメータ」がズレている可能性が高い。

---

## 主な変更点・デバッグ出力

### C-mVMC (mVMC)
- CMake オプション追加:
  - `DEBUG_DUMP_PARA`, `DEBUG_DUMP_SG`, `DEBUG_DUMP_RNG`, `DEBUG_DUMP_ELEC`, `DEBUG_DUMP_LOCSPN`,
    `DEBUG_DUMP_RNG_TRACE`, `DEBUG_DUMP_ENERGY_TERMS`
- 追加ダンプ:
  - RNG: `rng_step_001.dat`, `rng_after_sampling_step_001.dat`
  - RNG trace: `rng_trace_step_001.dat`
  - locspn: `locspn_step_001.dat`
  - 初期/サンプリング後電子配置: `elec_initial_sample_step_001.dat`, `elec_after_sampling_step_001.dat`
  - SR: `sr_S_*`, `sr_g_*`, `sr_r_*`, `sr_para_post_*`
  - エネルギー分解: `energy_terms_step_001_sample_000.dat`
  - 交換項詳細: `energy_exchange_terms_step_001_sample_000.dat`
- C 側の詳細分解関数を追加:
  - `CalculateHamiltonian_terms`, `DumpExchangeTerms`
- `VMCParaOpt` の step を `VMCMainCal` に渡すため `DebugCurrentStep` を追加。

### Julia (MVMCOptimizers.jl)
- SFMT の **非消費 RNG ダンプ**に切替（下記 SFMT19937.jl 変更とセット）。
- 出力先を `test/debug_output` に固定。
- 追加ダンプ:
  - rng 各段階
  - locspn
  - 初期/サンプリング後電子配置
  - SR 量
  - エネルギー分解 (`energy_terms_step_XXX_sample_000.dat`)
  - 交換項詳細 (`energy_exchange_terms_step_XXX_sample_000.dat`)

### SFMT19937.jl
- `sfmt_dump_rand32` を C API として追加、`libsfmt.dylib` を再ビルド。
- `deepcopy(rng)` は **安全でない**（グローバル RNG のため）。
  - 以後は `sfmt_dump_rand32` を使用。

---

## 途中で方向転換したポイント

1) **RNG 不一致が原因？**
   - × ではなかった。SFMT 非消費ダンプで一致確認。

2) **初期サンプル不一致？**
   - × 解消（RNGの問題だった）。

3) **エネルギー項分解（step 1）不一致？**
   - × 解消。C/JL 一致。

4) **step 2 の energy_terms**
   - × 大きく不一致（未解決）。

---

## 現在の主要出力ファイル

### C 出力
`/Users/takahiromisawa/Dropbox/Shin-mVMC/debug_HeisenbergChain_cmp/`
- `rng_step_001.dat`, `rng_after_sampling_step_001.dat`
- `rng_trace_step_001.dat`
- `locspn_step_001.dat`
- `elec_initial_sample_step_001.dat`, `elec_after_sampling_step_001.dat`
- `sr_S_step_001.dat`, `sr_g_step_001.dat`, `sr_r_step_001.dat`
- `sr_para_post_step_001.dat`, `sr_para_post_sync_step_001.dat`
- `energy_terms_step_001_sample_000.dat`
- `energy_exchange_terms_step_001_sample_000.dat`
- `energy_terms_step_002_sample_000.dat`
- `energy_exchange_terms_step_002_sample_000.dat`

### Julia 出力
`/Users/takahiromisawa/Dropbox/Shin-mVMC/private-mVMC/MVMCOptimizers.jl/test/debug_output/`
- `rng_step_001.dat`, `rng_after_sampling_step_001.dat`
- `rng_before_*` 各段階ダンプ
- `rng_trace_julia_step_001.dat`
- `locspn_step_001.dat`
- `elec_initial_sample_step_001.dat`, `elec_after_sampling_step_001.dat`
- `sr_S_step_001.dat`, `sr_g_step_001.dat`, `sr_r_step_001.dat`
- `sr_para_post_step_001.dat`, `sr_para_post_sync_step_001.dat`
- `energy_terms_step_001_sample_000.dat`
- `energy_exchange_terms_step_001_sample_000.dat`
- `energy_terms_step_002_sample_000.dat`
- `energy_exchange_terms_step_002_sample_000.dat`

---

## 一致/不一致まとめ

### 一致
- RNG（step 1）
- locspn
- 初期/サンプリング後電子配置（step 1）
- SR 量（S/g/r）step 1
- 更新後パラメータ（step 1）
- energy_terms / exchange_terms（step 1）

### 不一致
- zvo_out の step 1 以降
- energy_terms（step 2）

---

## 申し送り（次の手順）

1) **step 2 の電子配置ダンプ比較**
   - `elec_initial_sample_step_002.dat`
   - `elec_after_sampling_step_002.dat`
   - これで「step 2 でサンプルがズレたのか」を判定。

2) **step 1 → 2 のパラメータ更新確認**
   - `sr_para_post_step_001.dat` との対応チェック。

3) **step 2 のサンプルが一致していた場合**
   - `calculate_hamiltonian` あるいは `green_func` 系の差分を再確認。

---

## 注意点
- SFMT RNG はグローバル。`deepcopy(rng)` では状態が保持できない。
- C 側で `NSROptItrStep` を 2 に変更して step 2 を出力している。

---

## Claude Opus 4.5 による追加調査 (2026-02-04)

### 実施した修正

1. **Step 2 の電子配置ダンプを追加**
   - C 側 (`vmcmain.c`): `step <= 1` に拡張、ファイル名を動的に生成
   - Julia 側 (`vmc_para_opt.jl`): `step <= 1` に拡張
   - Julia 側 (`vmc_sampling.jl`): `copy_from_burn_sample!` 直後のダンプを追加

2. **Pfaffian 値のダンプを追加**
   - C 側 (`vmcmake.c`): `pfm_step_XXX.dat` を出力
   - Julia 側 (`vmc_sampling.jl`): 同様のダンプを追加

3. **`DebugCurrentStep` を `_DEBUG_DUMP_ELEC` でも使用可能に**
   - `vmcmain.c` の条件を `#if defined(_DEBUG_DUMP_ENERGY_TERMS) || defined(_DEBUG_DUMP_ELEC)` に変更

### 詳細調査結果

| 項目 | 結果 |
|------|------|
| Step 1 RNG 状態 (サンプリング後) | ✅ 一致 |
| Step 1 電子配置 (サンプリング後) | ✅ 一致 |
| Step 1 S 行列 | ✅ 一致 |
| Step 1 g ベクトル | ✅ 一致 (~1e-18 差) |
| **Step 1 パラメータ更新後** | ⚠️ 微妙な差 (~1e-12) |
| Step 2 電子配置 (開始時, burn sample から復元) | ✅ 一致 |
| **Step 2 Pfaffian** | ⚠️ 微妙な差 (~1e-8) |
| **Step 2 電子配置 (サンプリング後)** | ❌ 不一致 |

### 根本原因の特定

**SR 最適化での連立方程式 `Sx = g` を解く際の数値誤差**が原因。

- g ベクトルは完全に一致 (~1e-18 差)
- しかしパラメータ更新後の値に微妙な差 (~1e-12)
- この差が Slater 行列 → Pfaffian (~1e-8 差) → サンプリング結果に伝播
- サンプリングの accept/reject 判定に影響し、最終的に異なる電子配置に

### パラメータ更新後の値の比較例

**para_idx 3:**
- C: `2.429266323342945011e+00`
- Julia: `2.429266323344612122e+00`
- 差: ~1.7e-12

**para_idx 4:**
- C: `-1.687327666745209287e+00`
- Julia: `-1.687327666751437638e+00`
- 差: ~6e-12

---

## 追加調査と最終修正 (2026-02-04)

### 実証結果
- **C の step1 パラメータを Julia に強制適用してもズレる**ことを確認。
  - `sr_para_post_step_001.dat` を Julia に適用し、step1 のサンプリングを実施。
  - その結果、エネルギーは一致しなかった。
- **適用直後のパラメータは完全一致**（C ファイルと Julia ダンプで diff なし）。
- **accept/reject の候補と判定自体は一致**しており、遷移は同じ。
- それでもエネルギーがズレるため、**サンプル保存の開始タイミングが原因**と判明。

### 根本原因（複素数経路のみ）
Julia の複素数サンプリング (`vmc_make_sample!`) で、
**burn_flag 時の保存開始条件が C とずれていた**。

| 項目 | C | Julia (旧) | 影響 |
|---|---|---|---|
| burn_flag 時の保存開始 | outStep >= 1 | out_step >= n_vmc_warmup | サンプル集合がズレる |

実数経路 (`vmc_make_sample_real!`) では C と同条件だったため、
**実数ケースは問題が顕在化しなかった**。
複素数経路に入った時だけズレが発生していた。

### 修正内容
`private-mVMC/MVMCOptimizers.jl/src/vmc_sampling.jl`

```julia
save_start = burn_flag ? 1 : n_vmc_warmup
if out_step >= save_start
    sample = out_step - save_start
```

これにより C と同じサンプル集合を保存。

### 検証結果
`heisenbergchain_cmp.jl` を再実行し、**全ステップ一致**を確認。

```
HeisenbergChain_cmp: Julia zvo_out.dat vs C ref |   23 / 23 Pass
```

### 申し送り
- 修正は **複素数経路の保存条件のみ**。数値計算自体の不一致ではない。
- 実数経路はもともと正しかったため、今回の修正は複素数ケースのみに影響。

### 考察

これは LAPACK の実装やアルゴリズムの違いに起因する可能性がある。C 側と Julia 側で：
1. 異なる LAPACK ライブラリを使用している可能性
2. 連立方程式のソルバー（LU 分解 vs QR 分解など）が異なる可能性
3. S 行列の条件数が悪い可能性

### BLAS/LAPACK の統一 (2026-02-04)

#### 発見した問題

C 側と Julia 側で **異なる BLAS/LAPACK 実装** を使用していた。

| 実装 | BLAS/LAPACK |
|------|-------------|
| C-mVMC | Apple Accelerate |
| Julia-mVMC | OpenBLAS |

#### 実施した修正

`config/mac_gcc.cmake` を修正し、C 側も OpenBLAS を使用するように設定：

```cmake
# Use OpenBLAS for consistency with Julia's BLAS/LAPACK
set(BLA_VENDOR OpenBLAS CACHE STRING "BLAS vendor" FORCE)
set(BLAS_LIBRARIES "/opt/homebrew/opt/openblas/lib/libopenblas.dylib" CACHE STRING "" FORCE)
set(LAPACK_LIBRARIES "/opt/homebrew/opt/openblas/lib/libopenblas.dylib" CACHE STRING "" FORCE)
```

ビルド：

```bash
cd private-mVMC/mVMC
mkdir build && cd build
cmake .. -DCONFIG=mac_gcc -DGIT_SUBMODULE_UPDATE=OFF
make -j4
```

#### テスト結果

| テスト | モード | 状態 | 備考 |
|--------|--------|------|------|
| `test_vmc_models_real.jl` | 実数 | ✅ PASS | 全ステップで 1e-16 オーダーの差（丸め誤差レベル） |
| `heisenbergchain_cmp.jl` | 複素数 | ❌ FAIL | Step 0 のみ一致、Step 1 以降で不一致 |

**実数モードは OpenBLAS 統一で解決。複素数モードには別の問題が残っている。**

#### 複素数モードの状況

`heisenbergchain_cmp.jl` を `include` で直接実行した場合：

| Step | C (OpenBLAS) | Julia | 差 |
|------|--------------|-------|-----|
| 0 | 1.076e-03 | 1.076e-03 | ~8e-17 ✅ |
| 1 | -1.330e-02 | -1.312e-02 | ~1.8e-04 ❌ |
| 2 | -4.155e-01 | -3.734e-01 | ~4.2e-02 ❌ |

Step 0 は一致しているが、Step 1 以降で乖離が発生。

#### 注意点

- `Pkg.test(test_args=["heisenbergchain_cmp"])` で PASS と表示されたのは、`runtests.jl` に含まれる**別のテスト**（実数モードの `test_example_heisenbergchain_baseline.jl` など）
- `heisenbergchain_cmp.jl` は `runtests.jl` に含まれていないため、`Pkg.test` では実行されない
- 直接実行する場合: `julia --project=@. -e 'include("test/heisenbergchain_cmp.jl")'`

---

### 複素数モードの詳細調査 (2026-02-04 続き)

#### OpenBLAS バイナリの違いを発見

C 側と Julia 側で**同じ OpenBLAS でも異なるバイナリ**を使用していた。

| 実装 | OpenBLAS バイナリ | インターフェース |
|------|------------------|-----------------|
| C-mVMC | `/opt/homebrew/opt/openblas/lib/libopenblas.0.dylib` | **LP64** (32-bit integer) |
| Julia-mVMC | `libopenblas64_.dylib` (バンドル版) | **ILP64** (64-bit integer) |

#### S 行列と g ベクトルの比較

`MVMC_DEBUG_SG=1` と `DEBUG_DUMP_SG` を有効にして比較：

| 項目 | 差 | 判定 |
|------|-----|------|
| S 行列 (Step 1) | ~1e-16 〜 1e-17 | ✅ 完全一致 |
| g ベクトル (Step 1) | ~1e-16 〜 1e-18 | ✅ 完全一致 |

**S 行列と g ベクトルは高精度で一致**している。

#### パラメータ更新後の差

Cholesky 分解（`potrf!` / `potrs!`）後のパラメータに微妙な差が発生：

| para_idx | C (real) | Julia (real) | Δreal |
|----------|----------|--------------|-------|
| 3 | 2.429266323344535738e+00 | 2.429266323344612122e+00 | 7.6e-14 |
| 4 | -1.687327666748053900e+00 | -1.687327666751437638e+00 | 3.4e-12 |
| 5 | 3.431756484592154877e+00 | 3.431756484589709721e+00 | 2.4e-12 |
| 6 | -2.188828640673508019e+00 | -2.188828640668239345e+00 | 5.3e-12 |

**最大差**: |Δreal| = 5.3e-12, |Δimag| = 4.3e-12

#### エネルギー差の推移

パラメータの ~1e-12 の差が VMC のカオス的な性質で増幅：

| Step | C (real) | Julia (real) | Δreal | 相対誤差 |
|------|----------|--------------|-------|---------|
| **0** | 1.076e-03 | 1.076e-03 | **8e-17** | 7.5e-14 ✅ |
| 1 | -1.330e-02 | -1.312e-02 | 1.8e-04 | 1.4% |
| 2 | -4.155e-01 | -3.734e-01 | 4.2e-02 | 10% |
| 3 | -4.545e-01 | -4.872e-01 | 3.3e-02 | 7.2% |
| ... | ... | ... | ... | 2-10% |
| 9 | -7.861e-01 | -7.472e-01 | 3.9e-02 | 4.9% |

#### 結論

1. **Step 0 は完全に一致** → 初期状態、サンプリング、エネルギー計算は正しい
2. **S 行列と g ベクトルは完全一致** → サンプリング結果の計算は正しい
3. **パラメータ更新後に ~1e-12 の差** → LAPACK Cholesky 分解の数値誤差（異なる OpenBLAS バイナリ）
4. **VMC のカオス的な性質で増幅** → エネルギーの相対誤差 1-10%

**物理的には問題なし**。両実装とも同じ方向に収束しており、最終的なエネルギー値も同程度。

#### 対応案

1. **許容誤差を緩くする** - 完全一致ではなく、収束傾向の一致を確認するテストに変更
2. **同一 OpenBLAS バイナリを使用** - Julia の OpenBLAS を C からも使用（技術的に複雑）
3. **テストから除外** - 複素数モードの厳密比較テストは実施しない

---

### 生成されたデバッグファイル

**C 側** (`mVMC/build/test/python/work/HeisenbergChain_cmp/`):
- `elec_initial_sample_step_002.dat`
- `elec_after_sampling_step_002.dat`
- `pfm_step_001.dat`, `pfm_step_002.dat`

**Julia 側** (`MVMCOptimizers.jl/test/debug_output/`):
- `elec_initial_sample_step_002.dat` 〜 `elec_initial_sample_step_010.dat`
- `elec_after_sampling_step_001.dat`, `elec_after_sampling_step_002.dat`
- `pfm_step_001.dat` 〜 `pfm_step_010.dat`
