# VMCPhysCal 実装計画

## 概要

C 実装の `VMCPhysCal` 関数（`mVMC/src/mVMC/vmcmain.c:531-637`）を Julia に移植する。
この機能は `NVMCCalMode=1`（物理量測定モード）に対応し、最適化後のパラメータを使って
グリーン関数などの物理量を高精度に計算する。

## ディレクトリ構造

### MVMCOptimizers.jl（メインパッケージ）

```
MVMCOptimizers.jl/
├── src/
│   ├── MVMCOptimizers.jl      # [修正] vmc_phys_cal.jl の include と export 追加
│   ├── types.jl               # [修正] PhysicalQuantities 型を追加
│   │
│   │  # === 既存ファイル（VMCParaOpt 用、流用可能） ===
│   ├── vmc_para_opt.jl        # [既存] パラメータ最適化メイン
│   ├── vmc_sampling.jl        # [既存] モンテカルロサンプリング（共用）
│   ├── vmc_main_cal.jl        # [修正] calculate_green_func! を追加
│   ├── slater_update.jl       # [既存] Slater行列更新（共用）
│   ├── weight_average.jl      # [修正] weight_average_green_func! を追加
│   ├── stochastic_opt.jl      # [既存] SR最適化（VMCParaOpt専用）
│   ├── parameter_sync.jl      # [既存] パラメータ同期（VMCParaOpt専用）
│   ├── qp_weight_update.jl    # [既存] 量子射影重み（共用）
│   ├── counter.jl             # [既存] カウンター（共用）
│   │
│   │  # === 新規ファイル（VMCPhysCal 用） ===
│   ├── vmc_phys_cal.jl        # [新規] 物理量計算メイン ★
│   ├── green_func_calc.jl     # [新規] グリーン関数累積計算 ★
│   └── data_io.jl             # [修正] output_green_func! を追加
│
├── test/
│   ├── runtests.jl            # [修正] VMCPhysCal テストを追加
│   ├── test_vmc_phys_cal.jl   # [新規] VMCPhysCal 単体テスト ★
│   └── test_green_func.jl     # [新規] グリーン関数計算テスト ★
│
├── examples/
│   └── example_phys_cal.jl    # [新規] 物理量計算の使用例 ★
│
├── PLAN.md                    # 本ファイル
└── TODO.md                    # 実装履歴
```

### MVMCExpertModeParsers.jl（パーサーパッケージ）

```
MVMCExpertModeParsers.jl/
├── src/
│   ├── parsers/
│   │   ├── green_parser.jl    # [既存] greenone.def, greentwo.def パーサー ✓
│   │   └── ...
│   │
│   └── types/
│       └── expert_types.jl    # [確認] GreenOneTerm, GreenTwoTerm 型 ✓
│
└── test/
    └── samples/
        └── HeisenbergChain/
            ├── greenone.def   # [新規] テスト用グリーン関数定義 ★
            └── greentwo.def   # [新規] テスト用2体グリーン関数定義 ★
```

### C 実装（リファレンス）

```
mVMC/src/mVMC/
├── vmcmain.c          # VMCPhysCal() 定義 (L531-637)
├── calgrn.c           # CalculateGreenFunc() - 1体・2体グリーン関数
├── calgrn_fsz.c       # CalculateGreenFunc_fsz() - FSZ版
├── average.c          # WeightAverageGreenFunc() - 重み付き平均
├── initfile.c         # InitFilePhysCal() - 出力ファイル初期化
└── include/
    └── global.h       # PhysCisAjs[], NCisAjs 等のグローバル変数
```

### ファイル修正・追加一覧

| ファイル | 操作 | 内容 |
|----------|------|------|
| `src/MVMCOptimizers.jl` | 修正 | `include`, `export` 追加 |
| `src/types.jl` | 修正 | `PhysicalQuantities` 型追加 |
| `src/vmc_phys_cal.jl` | **新規** | メイン関数 |
| `src/green_func_calc.jl` | **新規** | グリーン関数累積 |
| `src/vmc_main_cal.jl` | 修正 | `calculate_green_func!` 追加 |
| `src/weight_average.jl` | 修正 | `weight_average_green_func!` 追加 |
| `src/data_io.jl` | 修正 | `output_green_func!` 追加 |
| `test/test_vmc_phys_cal.jl` | **新規** | 単体テスト |
| `examples/example_phys_cal.jl` | **新規** | 使用例 |

### 依存関係図

```
vmc_phys_cal.jl (メイン)
├── types.jl
│   └── PhysicalQuantities
├── slater_update.jl
│   └── update_slater_elm_fcmp!() [既存]
├── vmc_sampling.jl
│   └── vmc_make_sample!() [既存]
├── vmc_main_cal.jl
│   ├── vmc_main_cal!() [既存]
│   └── calculate_green_func!() [新規]
├── green_func_calc.jl [新規]
│   ├── green_func1() [既存・vmc_main_cal.jl内]
│   └── green_func2() [既存・vmc_main_cal.jl内]
├── weight_average.jl
│   ├── weight_average_we!() [既存]
│   └── weight_average_green_func!() [新規]
└── data_io.jl
    ├── output_data!() [既存]
    └── output_green_func!() [新規]
```

## C 実装の処理フロー

```
VMCPhysCal(comm_parent, comm_child1, comm_child2)
│
├── UpdateSlaterElm_fcmp() / UpdateSlaterElm_fsz()
│   └── Slater行列要素の更新
│
└── for ismp = 0 to NDataQtySmp-1:
    │
    ├── InitFilePhysCal(ismp, rank)
    │   └── 測定用出力ファイルの初期化
    │
    ├── VMCMakeSample() / VMCMakeSample_fsz() / VMC_BF_MakeSample()
    │   └── モンテカルロサンプリング（パラメータ更新なし）
    │
    ├── VMCMainCal() / VMCMainCal_fsz() / VMC_BF_MainCal()
    │   ├── エネルギー計算
    │   └── CalculateGreenFunc() ← 物理量測定モードで追加
    │       ├── 1体グリーン関数 <c†_i c_j> の計算
    │       └── 2体グリーン関数 <c†_i c_j c†_k c_l> の計算
    │
    ├── WeightAverageWE(comm_parent)
    │   └── エネルギーの重み付き平均
    │
    ├── WeightAverageGreenFunc(comm_parent) ← 新規実装
    │   └── グリーン関数の重み付き平均
    │
    ├── ReduceCounter(comm_child2)
    │
    └── outputData() ← グリーン関数出力を追加
        ├── zvo_cisajs.dat (1体グリーン関数)
        ├── zvo_cisajscktalt.dat (2体相関)
        └── zvo_cisajscktaltex.dat (2体相関・拡張)
```

## 実装フェーズ

### Phase 1: データ構造の拡張

#### 1.1 物理量格納用の型定義 (`types.jl`)

```julia
"""
物理量測定用のデータ構造
"""
mutable struct PhysicalQuantities
    # 1体グリーン関数: <c†_{ri,s} c_{rj,s}>
    # サイズ: NCisAjs
    local_cis_ajs::Vector{ComplexF64}      # 各サンプルでの局所値
    phys_cis_ajs::Vector{ComplexF64}       # 重み付き平均の累積

    # 2体グリーン関数 (直積): <c†_i c_j> × <c†_k c_l>
    # サイズ: NCisAjsCktAlt
    phys_cis_ajs_ckt_alt::Vector{ComplexF64}

    # 2体グリーン関数 (ダイレクト): <c†_i c_j c†_k c_l>
    # サイズ: NCisAjsCktAltDC
    local_cis_ajs_ckt_alt_dc::Vector{ComplexF64}
    phys_cis_ajs_ckt_alt_dc::Vector{ComplexF64}

    # Lanczos法用 (Phase 3で実装)
    # qqqq, q_cis_ajs_q, q_cis_ajs_ckt_alt_q, etc.
end
```

#### 1.2 VMCOptimizationState の拡張

```julia
# VMCOptimizationState に追加
phys_quantities::Union{PhysicalQuantities, Nothing}
```

#### 1.3 インデックス配列の追加 (`ExpertModeData` に既存、確認のみ)

- `green_one_terms::Vector{GreenOneTerm}` - 1体グリーン関数の測定対象
- `green_two_terms::Vector{GreenTwoTerm}` - 2体グリーン関数の測定対象

### Phase 2: グリーン関数計算の実装

#### 2.1 グリーン関数計算関数 (`vmc_main_cal.jl` に追加)

```julia
"""
    calculate_green_func!(data, state, w, ip, ele_idx, ele_cfg, ele_num, ele_proj_cnt)

グリーン関数を計算し、重み付きで累積する。
C実装: calgrn.c:CalculateGreenFunc()
"""
function calculate_green_func!(
    data::ExpertModeData,
    state::VMCOptimizationState,
    w::Float64,
    ip::ComplexF64,
    ele_idx::Vector{Int},
    ele_cfg::Vector{Int},
    ele_num::Vector{Int},
    ele_proj_cnt::Vector{Int}
)
    phys = state.phys_quantities
    
    # 1体グリーン関数の計算
    for (idx, term) in enumerate(data.green_one_terms)
        ri, rj = term.site1, term.site2
        s = term.spin1 == :up ? 0 : 1
        
        # GreenFunc1 を呼び出し
        local_val = green_func1(ri, rj, s, ip, ele_idx, ele_cfg, ele_num, 
                                ele_proj_cnt, data, state)
        phys.local_cis_ajs[idx] = local_val
        phys.phys_cis_ajs[idx] += w * local_val
    end
    
    # 2体グリーン関数（ダイレクト）の計算
    for (idx, term) in enumerate(data.green_two_terms)
        ri, rj, rk, rl = term.site1, term.site2, term.site3, term.site4
        s = term.spin1 == :up ? 0 : 1
        t = term.spin3 == :up ? 0 : 1
        
        local_val = green_func2(ri, rj, rk, rl, s, t, ip, ele_idx, ele_cfg, 
                                ele_num, ele_proj_cnt, data, state)
        phys.local_cis_ajs_ckt_alt_dc[idx] = local_val
        phys.phys_cis_ajs_ckt_alt_dc[idx] += w * local_val
    end
    
    # 2体グリーン関数（直積）の計算
    # CisAjsCktAltIdx[idx] = (idx0, idx1) の形式で
    # PhysCisAjsCktAlt[idx] += w * LocalCisAjs[idx0] * conj(LocalCisAjs[idx1])
    for (idx, (idx0, idx1)) in enumerate(data.cis_ajs_ckt_alt_idx)
        phys.phys_cis_ajs_ckt_alt[idx] += w * phys.local_cis_ajs[idx0] * 
                                          conj(phys.local_cis_ajs[idx1])
    end
end
```

#### 2.2 重み付き平均関数 (`weight_average.jl` に追加)

```julia
"""
    weight_average_green_func!(state)

グリーン関数の重み付き平均を計算。
C実装: average.c:WeightAverageGreenFunc()
"""
function weight_average_green_func!(state::VMCOptimizationState)
    phys = state.phys_quantities
    
    # 1体 + 2体直積 + 2体ダイレクト
    n = length(phys.phys_cis_ajs) + 
        length(phys.phys_cis_ajs_ckt_alt) + 
        length(phys.phys_cis_ajs_ckt_alt_dc)
    
    if n == 0
        return
    end
    
    # 単一プロセスの場合は単純な平均
    # MPI実装時は AllReduce を追加
    weight_average_reduce!(phys.phys_cis_ajs)
    weight_average_reduce!(phys.phys_cis_ajs_ckt_alt)
    weight_average_reduce!(phys.phys_cis_ajs_ckt_alt_dc)
end
```

### Phase 3: メイン関数の実装

#### 3.1 `vmc_phys_cal!` 関数 (`vmc_phys_cal.jl` 新規作成)

```julia
"""
    vmc_phys_cal!(data::ExpertModeData;
                  callback::Union{Nothing, Function}=nothing,
                  rng::AbstractRNG) -> Int

VMC 物理量計算モード。
パラメータを固定したまま、グリーン関数などの物理量を高精度に計算する。

C実装: vmcmain.c:VMCPhysCal()

# Arguments
- `data::ExpertModeData`: 初期化済みの Expert Mode データ
- `callback`: オプションのコールバック関数
- `rng`: 乱数生成器

# Returns
- `info::Int`: 戻り値 (0 = 成功)

# Output Files
- `zvo_out_XXX.dat`: エネルギー
- `zvo_var_XXX.dat`: パラメータ
- `zvo_cisajs_XXX.dat`: 1体グリーン関数
- `zvo_cisajscktalt_XXX.dat`: 2体相関関数
"""
function vmc_phys_cal!(data::ExpertModeData;
                       callback::Union{Nothing, Function}=nothing,
                       rng::AbstractRNG)::Int
    # パラメータ取得
    n_data_qty_smp = data.modpara.n_data_qty_smp  # サンプリング回数
    all_complex = get_all_complex_flag(data)
    i_flg_orbital_general = data.i_flg_orbital_general
    n_proj_bf = get_n_proj_bf(data)
    
    # 状態初期化
    state = initialize_vmc_state(data)
    
    # 物理量格納用配列の初期化
    initialize_phys_quantities!(state, data)
    
    info = 0
    
    # Slater行列要素の更新
    if i_flg_orbital_general == 0
        update_slater_elm_fcmp!(data, state)
    else
        update_slater_elm_fsz!(data, state)
    end
    
    # サンプリングループ
    for ismp in 0:(n_data_qty_smp - 1)
        println("Sampling: $ismp / $n_data_qty_smp")
        
        # 物理量配列のリセット
        reset_phys_quantities!(state)
        
        # ファイル初期化
        init_file_phys_cal!(data, ismp)
        
        # サンプリング
        if !all_complex
            if i_flg_orbital_general == 0
                if n_proj_bf == 0
                    vmc_make_sample_real!(data, state, rng)
                else
                    vmc_bf_make_sample_real!(data, state, rng)
                end
            else
                vmc_make_sample_fsz_real!(data, state, rng)
            end
        else
            if n_proj_bf == 0
                if i_flg_orbital_general == 0
                    vmc_make_sample!(data, state, rng)
                else
                    vmc_make_sample_fsz!(data, state, rng)
                end
            else
                vmc_bf_make_sample!(data, state, rng)
            end
        end
        
        # メイン計算（エネルギー + グリーン関数）
        if n_proj_bf == 0
            if i_flg_orbital_general == 0
                vmc_main_cal_phys!(data, state)  # 物理量計算版
            else
                vmc_main_cal_fsz_phys!(data, state)
            end
        else
            vmc_bf_main_cal_phys!(data, state)
        end
        
        # 重み付き平均
        weight_average_we!(state)
        weight_average_green_func!(state)
        
        # カウンター集約
        reduce_counter!(state)
        
        # データ出力
        output_data_phys!(data, state, ismp)
        
        # ファイルクローズ
        close_file_phys_cal!(ismp)
        
        # コールバック
        if callback !== nothing
            callback(ismp, data, state.energy.etot, info)
        end
    end
    
    return info
end
```

### Phase 4: 出力関数の実装

#### 4.1 グリーン関数出力 (`data_io.jl` に追加)

```julia
"""
    output_green_func!(data, state, ismp)

グリーン関数を出力ファイルに書き込む。

出力ファイル:
- zvo_cisajs_XXX.dat: 1体グリーン関数 <c†_i c_j>
- zvo_cisajscktalt_XXX.dat: 2体相関 (直積)
- zvo_cisajscktaltdc_XXX.dat: 2体相関 (ダイレクト)
"""
function output_green_func!(data::ExpertModeData, state::VMCOptimizationState, ismp::Int)
    phys = state.phys_quantities
    output_dir = data.modpara.output_dir
    
    # zvo_cisajs_XXX.dat
    if !isempty(data.green_one_terms)
        filename = joinpath(output_dir, @sprintf("zvo_cisajs_%03d.dat", ismp))
        open(filename, "w") do io
            for (idx, term) in enumerate(data.green_one_terms)
                val = phys.phys_cis_ajs[idx]
                @printf(io, "%d %d %d %d % .18e  % .18e \n",
                        term.site1, term.spin1 == :up ? 0 : 1,
                        term.site2, term.spin2 == :up ? 0 : 1,
                        real(val), imag(val))
            end
        end
    end
    
    # zvo_cisajscktalt_XXX.dat (2体相関・直積)
    if !isempty(phys.phys_cis_ajs_ckt_alt)
        filename = joinpath(output_dir, @sprintf("zvo_cisajscktalt_%03d.dat", ismp))
        open(filename, "w") do io
            for val in phys.phys_cis_ajs_ckt_alt
                @printf(io, "% .18e  % .18e ", real(val), imag(val))
            end
            println(io)
        end
    end
    
    # zvo_cisajscktaltdc_XXX.dat (2体相関・ダイレクト)
    if !isempty(data.green_two_terms)
        filename = joinpath(output_dir, @sprintf("zvo_cisajscktaltdc_%03d.dat", ismp))
        open(filename, "w") do io
            for (idx, term) in enumerate(data.green_two_terms)
                val = phys.phys_cis_ajs_ckt_alt_dc[idx]
                @printf(io, "%d %d %d %d %d %d %d %d % .18e % .18e\n",
                        term.site1, term.spin1 == :up ? 0 : 1,
                        term.site2, term.spin2 == :up ? 0 : 1,
                        term.site3, term.spin3 == :up ? 0 : 1,
                        term.site4, term.spin4 == :up ? 0 : 1,
                        real(val), imag(val))
            end
        end
    end
end
```

### Phase 5: エクスポートと統合

#### 5.1 モジュールへの追加 (`MVMCOptimizers.jl`)

```julia
# Include files
include("vmc_phys_cal.jl")

# Export
export vmc_phys_cal!
```

#### 5.2 ユニファイドインターフェース（オプション）

```julia
"""
    vmc!(data; mode=:optimize, ...)

統合インターフェース。mode で動作を切り替え。

- `:optimize` (NVMCCalMode=0): パラメータ最適化 → vmc_para_opt!
- `:measure` (NVMCCalMode=1): 物理量測定 → vmc_phys_cal!
"""
function vmc!(data::ExpertModeData; mode::Symbol=:optimize, kwargs...)
    if mode == :optimize
        return vmc_para_opt!(data; kwargs...)
    elseif mode == :measure
        return vmc_phys_cal!(data; kwargs...)
    else
        error("Unknown mode: $mode. Use :optimize or :measure")
    end
end
```

## 実装優先順位

| 優先度 | タスク | 依存関係 | 推定工数 |
|--------|--------|----------|----------|
| 🔴 P1 | `PhysicalQuantities` 型定義 | なし | 小 |
| 🔴 P1 | `vmc_phys_cal!` 基本構造 | P1 | 中 |
| 🔴 P1 | `calculate_green_func!` (1体) | P1 | 中 |
| 🟡 P2 | `calculate_green_func!` (2体) | P1 | 中 |
| 🟡 P2 | `weight_average_green_func!` | P1 | 小 |
| 🟡 P2 | `output_green_func!` | P1, P2 | 小 |
| 🟢 P3 | FSZ版対応 | P1, P2 | 中 |
| 🟢 P3 | BackFlow版対応 | P1, P2 | 大 |
| 🟢 P3 | Lanczos法対応 | P1, P2 | 大 |

## 検証計画

### テストケース

1. **HeisenbergChain** (既存テストケース)
   - greenone.def, greentwo.def を追加
   - C実装の出力と比較

2. **Hubbard模型**
   - 電荷・スピン相関関数の検証

### 検証項目

| 項目 | 検証方法 |
|------|----------|
| 1体グリーン関数 | C出力 `zvo_cisajs.dat` と数値比較 |
| 2体グリーン関数 | C出力 `zvo_cisajscktalt.dat` と数値比較 |
| エネルギー | 最適化モードの結果と一致確認 |

## 関連ファイル

### C 実装

- `mVMC/src/mVMC/vmcmain.c:531-637` - VMCPhysCal メイン
- `mVMC/src/mVMC/calgrn.c` - CalculateGreenFunc
- `mVMC/src/mVMC/average.c:209-` - WeightAverageGreenFunc
- `mVMC/src/mVMC/initfile.c` - InitFilePhysCal

### Julia 実装（既存）

- `MVMCExpertModeParsers.jl/src/parsers/green_parser.jl` - greenone/greentwo パーサー
- `MVMCOptimizers.jl/src/vmc_main_cal.jl` - green_func1, green_func2
- `MVMCOptimizers.jl/src/weight_average.jl` - weight_average_we!

### Julia 実装（新規作成）

- `MVMCOptimizers.jl/src/vmc_phys_cal.jl` - メイン関数
- `MVMCOptimizers.jl/src/types.jl` - PhysicalQuantities 追加

## 備考

- `NDataQtySmp` は `modpara.def` の `NDataQtySmp` パラメータで指定
- 現在のパーサーで `n_data_qty_smp` が読み込まれているか要確認
- Lanczos法（`NLanczosMode > 0`）は Phase 3 で対応
