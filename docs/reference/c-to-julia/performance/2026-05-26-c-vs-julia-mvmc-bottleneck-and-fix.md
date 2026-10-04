---
date: 2026-05-26
datetime: 2026-05-26 10:53 JST
topic: C-mVMC vs Julia-mVMC ベンチマーク・ボトルネック分析・解消法
audience: セミナー発表用レポート
model: Claude Opus 4.7 (1M context)
---

# C-mVMC vs Julia-mVMC: なぜ Julia が速かったか、そして C でどう追いついたか

Heisenberg 鎖の VMC 最適化ベンチマークで Julia-mVMC が C-mVMC より 1.4–1.8 倍速かった。
本レポートは、(1) ベンチマーク結果、(2) ボトルネックの特定とアルゴリズムの説明（C/Julia の簡易コード）、
(3) ボトルネックの解消法（C/Julia の簡易コード）を、プロファイル実測に基づいてまとめる。

---

## 0. 要旨（TL;DR）

- **結果**: ohtaka (AMD EPYC 7702) で Julia-mVMC は C-mVMC の **1.38 / 1.62 / 1.77 倍**速い（L=16/24/32、単一プロセス）。
- **ボトルネック**: 速度差のほぼ全ては **交換モンテカルロ更新の Pfaffian カーネル** ―
  `UpdateMAllTwo`（逆行列 M⁻¹ の O(N²) 更新）と `CalculateNewPfMTwo2`（Pfaffian 比）― に集中。
  C ではこの 2 つで全実行時間の **過半**を占め、特に `UpdateMAllTwo` は L=32 で全体の **32 %**。
- **原因はアルゴリズムではない**。C と Julia は**同一アルゴリズム**（同じ O(N²) 反対称ランク更新の式）。
  差は内側ループの **SIMD コード生成**にあった:
  1. C-mVMC のベクトル化指示は **Fujitsu コンパイラ専用 pragma**（`#pragma loop noalias` 等。京 / FX / Fugaku 系のチューニング資産）で、
     ohtaka の gcc/Intel コンパイラは**無視**する。
  2. さらに **Intel classic コンパイラ × AMD CPU** の組み合わせが C を一段不利にしていた。
  Julia 側は `@turbo`（LoopVectorization.jl）がこれらの制約を外して AVX2+FMA を積極生成していた。
- **解消法**: C の該当 2 ループに **`#pragma omp simd`**（移植可能なベクトル化指示）を入れ、
  `-march=znver2 -funsafe-math-optimizations -ffp-contract=fast` でビルドするだけで、
  **C が Julia とほぼ同等**になった（L=32 で全体 133.6 s vs Julia 134.3 s、`UpdateMAllTwo` は同点）。
- **結論**: Julia の速さは言語の本質的優位ではなく、**「C 側が現行コンパイラ向けにベクトル化指示を持っていなかった」**ことの帰結だった。数行の pragma で C-mVMC は約 1.5 倍速くなる（移植可・低リスク）。

---

## 1. セットアップ

| 項目 | 内容 |
|---|---|
| モデル | Heisenberg 鎖、L = 16 / 24 / 32、`Lsub=4`、`J=1`、2Sz=0 |
| VMC 条件 | `NVMCSample=1000`、`NSROptItrStep=300`、`NSROptItrSmp=30` |
| マシン | ISSP ohtaka i8cpu（AMD EPYC 7702、Zen2、AVX2+FMA / AVX-512 なし） |
| 並列 | **単一プロセス・単一スレッド**（C: `mpiexec -n 1`+`OMP_NUM_THREADS=1`、Julia: 1 thread）。比較は完全に同条件 |
| C-mVMC | Intel oneAPI classic 2023（icc, MKL）。後段で gcc 10.1 でも再ビルド |
| Julia-mVMC | Julia 1.12.1、`feature/c-compatible-timer` ブランチ |
| 計測 | C は組込みタイマ `zvo_CalcTimer.dat`、Julia は同形式の C 互換タイマを本作業で実装 |

C・Julia の両方が**同一区間 id・同一ラベル**の `zvo_CalcTimer.dat` を吐くので、セクション単位で 1:1 比較できる。

---

## 2. ベンチマーク結果

### 2.1 全体（タイマ無し本番計測、10 repeat 中央値）

| L | Julia [s] | C-mVMC (icc) [s] | C / Julia |
|--:|--:|--:|--:|
| 16 | 36.34 | 50.18 | **1.38** |
| 24 | 74.17 | 120.41 | **1.62** |
| 32 | 130.75 | 231.43 | **1.77** |

Julia が一貫して速く、**差は系のサイズとともに拡大**する。

### 2.2 セクション内訳（C 互換タイマ、ohtaka L=32、中央値 [s]）

タイマ有効化のオーバーヘッドは ~4–7 %（比率は妥当）。`[id]` は C-mVMC の `zvo_CalcTimer.dat` の区間番号。

| id | 区間 | C-mVMC (icc) | Julia | C / Julia | 全体に占める C の割合 |
|--:|---|--:|--:|--:|--:|
| 2 | VMCParaOpt（全体） | 229.6 | 134.3 | 1.71 | 100 % |
| 3 | VMCMakeSample（サンプリング） | 128.1 | 46.1 | **2.78** | 55.8 % |
| 33 | └ exchange update | 118.2 | 37.9 | 3.12 | 51.5 % |
| **68** | **　└ UpdateMAllTwo（M⁻¹ 更新）** | **73.4** | **14.4** | **5.09** | **32.0 %** |
| 66 | 　└ CalculateNewPfMTwo2（Pf 比） | 38.5 | 15.5 | 2.48 | 16.8 % |
| 4 | VMCMainCal（物理量計算） | 100.8 | 86.2 | 1.17 | 43.9 % |
| 40 | └ CalculateMAll（全再計算） | 65.8 | 51.3 | 1.28 | 28.6 % |
| 42 | └ ReturnSlaterElmDiff | 4.4 | 12.1 | **0.36**（Julia が遅い） | 1.9 % |

**読み取り**:
- C の実行時間は **サンプリング `[3]`（55.8 %）が支配的**で、その中の `UpdateMAllTwo [68]` 単独で **全体の 32 %**。
- C/Julia 比が最も大きいのも `[68]`（**5.09 倍**）。**ここが主ボトルネック**。
- `[42] ReturnSlaterElmDiff` だけは Julia が遅い（C の 2.4 倍）が、全体の 2 % 程度で大勢に影響しない（Julia 側の最適化候補）。

---

## 3. ボトルネックの説明：交換更新の Pfaffian カーネル

### 3.1 何を計算しているか

mVMC は Pfaffian-Slater 波動関数を使い、Metropolis 法で電子配置をサンプリングする。
Heisenberg 模型では「2 電子の交換（スピン交換に相当）」が主な遷移である。
1 回の遷移ごとに次が必要になる:

- **受理確率**: 行列 M の Pfaffian 比 `Pf(M_new) / Pf(M_old)` → `CalculateNewPfMTwo2`（区間 `[66]`）
- **受理後の更新**: 逆行列 `M⁻¹` の更新 → `UpdateMAllTwo`（区間 `[68]`）

M⁻¹ をゼロから作ると **O(N³)**。しかし 2 電子交換は M の 2 行・2 列しか変えないので、
**反対称版の Sherman–Morrison–Woodbury 公式**で **O(N²)** で更新できる。これがこのカーネルの正体である。
このカーネルは Metropolis 内側ループで **遷移提案のたび（=数百万回）**呼ばれるため、サンプリング時間を支配する。

### 3.2 カーネルの構造（O(N²) の 2 つの内側ループ）

`UpdateMAllTwo` は本質的に次の 2 つの 2 重ループからなる（N = 電子数 ×2 程度）:

1. **行列ベクトル積（縮約）**: `vecP[i] = Σ_j M⁻¹[i,j]·a[j]`、`vecQ[i] = Σ_j M⁻¹[i,j]·b[j]`
2. **反対称ランク更新（6 項）**: 各成分 `M⁻¹[i,j]` に 6 個の反対称項を加える

### 3.3 C と Julia は同一アルゴリズム（簡易コード）

**C-mVMC**（`mVMC/src/mVMC/pfupdate_two_real.c`、本質を抜粋）:

```c
/* (1) 行列ベクトル積: vecP[i]=Σ_j Minv[i][j]*a[j], vecQ[i]=Σ_j ...*b[j] */
for (i = 0; i < N; i++) {
  for (j = 0; j < N; j++) {
    vecP[i] += Minv[i*N+j] * a[j];
    vecQ[i] += Minv[i*N+j] * b[j];
  }
}

/* (2) 反対称ランク更新（6項）。p,q,s,t はベクトル、ca..cf はスカラ係数 */
for (i = 0; i < N; i++) {
  for (j = 0; j < N; j++) {
    Minv[i*N+j] += ca*(q[i]*t[j] - q[j]*t[i])
                 + cb*(q[i]*s[j] - q[j]*s[i])
                 + cc*(p[i]*t[j] - p[j]*t[i])
                 + cd*(p[i]*s[j] - p[j]*s[i])
                 + ce*(s[i]*t[j] - s[j]*t[i])
                 + cf*(p[i]*q[j] - q[i]*p[j]);
  }
}
```

**Julia-mVMC**（`MVMCOptimizers.jl/src/vmc_sampling.jl`、本質を抜粋）:

```julia
# (1) 行列ベクトル積（@turbo がスカラ縮約を SIMD 化）
@turbo for i in 1:N
    p = 0.0; q = 0.0
    for j in 1:N
        p += Minv[i,j]*a[j]
        q += Minv[i,j]*b[j]
    end
    vecP[i] = p; vecQ[i] = q
end

# (2) 反対称ランク更新（同じ 6 項）
@turbo for i in 1:N, j in 1:N
    Minv[i,j] += ca*(q[i]*t[j] - q[j]*t[i]) +
                 cb*(q[i]*s[j] - q[j]*s[i]) +
                 cc*(p[i]*t[j] - p[j]*t[i]) +
                 cd*(p[i]*s[j] - p[j]*s[i]) +
                 ce*(s[i]*t[j] - s[j]*t[i]) +
                 cf*(p[i]*q[j] - q[i]*p[j])
end
```

**式・計算量とも完全に同じ**（Julia は C を行単位で移植したもの）。違いは内側ループに付く一語、
C の `#pragma loop noalias` と Julia の `@turbo` だけである。

---

## 4. なぜ Julia が速かったか：SIMD コード生成の差

### 4.1 `@turbo` / LoopVectorization.jl とは

`@turbo` は単なる「ベクトル化してね」ヒントではなく、**専用のループ最適化エンジン**。
コンパイル時に SIMD 幅・アンロール量・FMA・メモリアクセスを自前でモデル化し、
**非エイリアス**と**浮動小数の再結合可**を前提に、AVX2+FMA を積極的に生成する。
今回のような素直な数値ループでは手書き intrinsics 並みのコードを出す。

### 4.2 C 側でベクトル化が効いていなかった理由

C-mVMC のベクトル化指示 `#pragma loop noalias` / `#pragma procedure serial` / `#pragma loop swp` 等は、
**Fujitsu コンパイラ専用の最適化指示**（京 / FX10 / FX100 / Fugaku 系でのチューニング由来。`loop swp`＝software pipelining や `procedure serial` は Fujitsu 固有）で、**gcc / Intel コンパイラは無視**する。
その結果:

1. **エイリアス情報の欠落**: 更新先 `Minv[i*N+j]` が読み元 `p/q/s/t` と別物だとコンパイラが証明できず
   （`restrict` も無い）、更新ループを**保守的（スカラ寄り）**に生成しがち。`@turbo` は非エイリアス前提で SIMD 化。
2. **Intel コンパイラ × AMD CPU**: icc の命令ディスパッチは非 Intel（EPYC）で狭い経路に落ちやすい。

### 4.3 切り分け実験：gcc で再ビルド（icc / gcc / Julia の 3-way）

同一ソースを **gcc 10.1 `-O3 -march=znver2`** で再ビルドして比較した（L=32、[s]）:

| id | 区間 | icc-C | gcc-C | Julia | icc/gcc |
|--:|---|--:|--:|--:|--:|
| 68 | UpdateMAllTwo | 73.4 | 48.4 | 14.4 | **1.52** |
| 66 | CalcNewPfMTwo2 | 38.5 | 48.4 | 15.5 | **0.80** |

- `[68]` は **gcc が icc より 1.5 倍速** → icc×AMD のコード生成劣化が実在。
- ただし gcc-C でも **Julia の 3.4 倍遅い** → 残りは「ベクトル化指示の欠落」（`@turbo` 優位）。
- `[66]` は **gcc が icc より遅い** → これはコンパイラ vendor の問題ではなく純粋にベクトル化の差。

→ 速度差は **(a) icc×AMD ビルド劣化 ＋ (b) C ソースが現行コンパイラ向けにベクトル化されていない**の合算、と確定。

---

## 5. ボトルネックの解消法：C に `#pragma omp simd` を入れる

### 5.1 修正方針

`@turbo` が前提にしている「非エイリアス + 再結合可」を、**移植可能な標準指示**で C に与える:

- 内側ループに **`#pragma omp simd`**（縮約ループは `reduction` 付き）。
  これは Fujitsu `#pragma loop noalias` がやりたかったことを、gcc/clang/icc 共通の形で実現する。
- ビルドフラグに **`-march=znver2 -funsafe-math-optimizations -ffp-contract=fast -fopenmp-simd`**
  （FMA + 再結合を許可。`@turbo` と同じ自由）。

> 注: `-ffast-math` は不可。`-ffinite-math-only` を含み、StdFace の「キーワード未設定 = NaN」判定を壊して
> `Keyword j is duplicated` で落ちる。NaN 処理を保つ `-funsafe-math-optimizations` を使う。

### 5.2 簡易コード（C: 修正前 → 修正後）

**修正前**（Fujitsu pragma は無視され、コンパイラは保守的なまま）:

```c
#pragma loop noalias        /* Fujitsu 専用 — gcc/icc は無視する */
for (i = 0; i < N; i++) {
  for (j = 0; j < N; j++) {
    Minv[i*N+j] += ca*(q[i]*t[j]-q[j]*t[i]) + ... ;   /* スカラ実行に留まりがち */
  }
}
```

**修正後**（標準の `#pragma omp simd` を内側ループに付ける）:

```c
/* (2) 反対称ランク更新: 内側 j ループは各成分が独立 → そのまま SIMD 化 */
for (i = 0; i < N; i++) {
  #pragma omp simd
  for (j = 0; j < N; j++) {
    Minv[i*N+j] += ca*(q[i]*t[j]-q[j]*t[i]) + ... ;   /* → AVX2 + FMA */
  }
}

/* (1) 行列ベクトル積: スカラ縮約なので reduction を明示 */
for (i = 0; i < N; i++) {
  double pacc = 0.0, qacc = 0.0;
  #pragma omp simd reduction(+:pacc,qacc)
  for (j = 0; j < N; j++) {
    pacc += Minv[i*N+j]*a[j];
    qacc += Minv[i*N+j]*b[j];
  }
  vecP[i] = pacc; vecQ[i] = qacc;
}
```

ビルド:

```bash
gcc -O3 -march=znver2 -funroll-loops \
    -funsafe-math-optimizations -ffp-contract=fast -fopenmp-simd ...
```

### 5.3 Julia 側（参照＝既にこの状態）

Julia は同じことを `@turbo` 一語で達成している（§3.3 のコード）。
つまり今回の C 修正は「Julia が `@turbo` で得ている SIMD を、C にも標準 pragma で持たせる」作業に等しい。

### 5.4 結果：C が Julia とほぼ同等に（4-way、L=32、[s]）

| id | 区間 | icc-C | gcc-C | **gcc+simd** | Julia | gcc+simd / Julia |
|--:|---|--:|--:|--:|--:|--:|
| 2 | 全体 | 229.6 | 205.2 | **133.6** | 134.3 | **1.00** |
| **68** | **UpdateMAllTwo** | 73.4 | 48.4 | **15.7** | 14.4 | **1.09** |
| 66 | CalcNewPfMTwo2 | 38.5 | 48.4 | 27.6 | 15.5 | 1.78 |
| 40 | CalculateMAll | 65.8 | 50.5 | 49.1 | 51.3 | 0.96 |
| 42 | ReturnSlaterElmDiff | 4.4 | 5.3 | 5.3 | 12.1 | 0.44 |

- **`UpdateMAllTwo` は `#pragma omp simd` だけで Julia に並んだ**（plain gcc 比 **3.1 倍速**、対 Julia 1.09）。
  元の icc 比 5 倍差が解消。
- **全体でも C が Julia と同等**（133.6 s vs 134.3 s。L=16/24 では gcc+simd がわずかに速い）。
- 残る Julia 優位は **`[66]` のみ（~1.8 倍）**。これは「行ごとに行列ベクトル積 → 縮約」という構造で、
  単純な `#pragma omp simd reduction` より `@turbo` のループブロッキングが効くため。
  pragma だけでは届かず、ループ再構成（融合・ブロック化）が必要。

---

## 6. 正当性と注意点

- **数値一致**: gcc-C と gcc+simd の最終エネルギーは **約 14 桁一致**（Δ ≈ 1.4×10⁻¹⁴）。
  再結合による順序差のみで、物理は不変。
- **回帰テストへの影響**: C↔Julia の比較も mVMC 自身の ctest も **3σ 統計許容**（bit 一致ではない）ので、
  `-funsafe-math-optimizations` の導入で壊れる回帰テストは無い。
- **計測条件**: タイマ ON 計測（オーバーヘッド ~4–7 %、比率は妥当）。`[40]`/SR は BLAS 依存
  （icc→MKL、gcc/Julia→OpenBLAS）なので、コンパイラの純粋効果は BLAS 非依存の `[66]`/`[68]` で判断している。
- **AVX-512 なし**: ohtaka は Zen2 で AVX2(256bit/4 double) まで。これは C・Julia 双方に共通の上限。

---

## 7. 結論と提言

1. **Julia-mVMC が速かった主因は SIMD コード生成**であり、言語やアルゴリズムの本質的優位ではない。
   内訳は「icc×AMD のビルド劣化（~1.5×）」＋「C ソースが現行コンパイラ向けにベクトル化されていない（~3×）」。
2. **C 側で容易に追いつける**: 該当 2 ループ（`pfupdate_two_real.c`、および hopping 版 `pfupdate_real.c`）に
   `#pragma omp simd` を入れ、`-march=native -funsafe-math-optimizations -ffp-contract=fast` でビルドするだけで、
   C-mVMC は EPYC で **現行 icc ビルド比 ~1.5 倍**、**Julia とほぼ同等**になる。小・移植可・低リスク。
3. **upstream への提言**: Fujitsu 専用 pragma に加えて標準の `#pragma omp simd` を併記すれば、
   Fujitsu コンパイラ以外（gcc/clang/Intel on x86・ARM 等）でも自動的に SIMD が効く。
4. **残課題**: `[66] CalculateNewPfMTwo2` の ~1.8 倍差（ループ再構成で詰められる余地）。
   Julia 側は `[42] ReturnSlaterElmDiff`（C の 2.4 倍遅）が最適化候補。

**一言**: 「Julia が速い」ではなく「**C がベクトル化指示を持っていなかった**」。
正しく指示すれば C も Julia も同じハードの AVX2+FMA を等しく使い切る。

---

## 付録 A: 全 4-way 内訳（L=16/24/32）

`benchmark/heisenberg_chain_issue1_ohtaka/calc_timer_compare/four_way_icc_gcc_gccsimd_julia.txt` を参照。
L=16/24/32 で同傾向（`UpdateMAllTwo` の gcc+simd/Julia 比 = 1.00 / 1.03 / 1.09、全体比 = 0.93 / 0.94 / 1.00）。

## 付録 B: 再現手順（要点）

- C 互換タイマ（Julia 側）: `MVMC_C_TIMER=1` で `zvo_CalcTimer.dat` を出力（ブランチ `feature/c-compatible-timer`）。
- gcc ビルド: `cmake -DCMAKE_C_COMPILER=mpicc -DUSE_GEMMT=OFF -DUSE_SCALAPACK=OFF -DTesting=OFF
  -DBLAS_LIBRARIES=/usr/lib64/libopenblas.so.0 -DCMAKE_C_FLAGS="-O3 -march=znver2 -funroll-loops
  -funsafe-math-optimizations -ffp-contract=fast -fopenmp-simd"`。
- 比較: `benchmark/heisenberg_chain_issue1/scripts/compare_calc_timer.py`、
  `benchmark/heisenberg_chain_issue1_ohtaka/scripts/compare_three_way.py`。
- 詳細レポート: `benchmark/heisenberg_chain_issue1_ohtaka/calc_timer_compare/`
  （`README.md`, `README_gcc_rebuild.md`, `README_gcc_simd_experiment.md`）。

## 付録 C: 関連ソース

| 区間 | C-mVMC | Julia-mVMC |
|---|---|---|
| `[68]` UpdateMAllTwo | `mVMC/src/mVMC/pfupdate_two_real.c` `updateMAllTwo_child_real` | `MVMCOptimizers.jl/src/vmc_sampling.jl` `update_m_all_two_real!` |
| `[66]` CalcNewPfMTwo2 | 同上 `calculateNewPfMTwo_child_real` | 同上 `calculate_new_pf_m_two2_real!` |
| タイマ定義/出力 | `mVMC/src/mVMC/vmcclock.c` | `MVMCOptimizers.jl/src/c_timer.jl` |

---

## 付録 D: icc+simd 検証（6-way、2026-05-27 追試）

**問い**: §5 の `#pragma omp simd` 修正は **gcc ビルド**で検証した。では (1) 同じ移植可能 pragma は **icc でも効くのか**、(2) §4 で示した **icc×AMD のディスパッチ劣化はベクトル化後も残るのか** を直接確かめる。

**方法**: §5 と同じ修正済みソース（`mVMC-gcc-simd`、`#pragma omp simd` 入り）を **icc（oneAPI classic 2023.0.0 = icc 2021.8.0）**で再ビルド。AVX2 ターゲットの指定を 2 通り用意して、ディスパッチの影響を分離した:

- **icc+simd (march)**: `-march=core-avx2` — ベンダ判定なしで AVX2 を直接生成（gcc 流）。
- **icc+simd (axdisp)**: `-axCORE-AVX2` — Intel マルチパス・ディスパッチ。GenuineIntel のみ AVX2 経路、AMD では baseline(SSE2) 経路に落ちる＝「Intel 流チューニングビルドを AMD で走らせた」状況。

> **注記**: 最初に試した厳格指定 **`-xCORE-AVX2`（大文字 X）は AMD EPYC で起動を拒否**した（実行時に *"Please verify that … the processor support Intel … AVX2 instructions"* で abort）。Zen2 は全命令をサポートするが、Intel のベンダチェックが GenuineIntel 以外を弾くため。よって「走るが格下げ」を見るには `-axCORE-AVX2` を用いる。

いずれも OpenBLAS・単一プロセス・`OMP_NUM_THREADS=1`・timer-ON・5 reps 中央値。StdFace 条件は §1 と同一。

### D.1 コンパイルオプション（全 6 ビルド）

| ビルド | コンパイラ | ソース（pragma） | 主フラグ | BLAS |
|---|---|---|---|---|
| icc-C | icc 2021.8 (oneAPI classic 2023) | mVMC（Fujitsu pragma、omp simd なし） | MateriApps モジュール既定（当方制御外） | MKL |
| gcc-C | gcc 10.1.0 | mVMC-gcc（omp simd なし） | `-O3 -march=znver2 -funroll-loops` | OpenBLAS |
| gcc+simd | gcc 10.1.0 | mVMC-gcc-simd（omp simd 入り） | `-O3 -march=znver2 -funroll-loops -funsafe-math-optimizations -ffp-contract=fast -fopenmp-simd` | OpenBLAS |
| **icc+simd (march)** | icc 2021.8 | mVMC-gcc-simd（omp simd 入り） | `-O3 -march=core-avx2 -funroll-loops -qopenmp-simd -Wno-unknown-pragmas` | OpenBLAS |
| **icc+simd (axdisp)** | icc 2021.8 | mVMC-gcc-simd（omp simd 入り） | `-O3 -axCORE-AVX2 -funroll-loops -qopenmp-simd -Wno-unknown-pragmas` | OpenBLAS |
| Julia | Julia 1.12.1 | Julia-mVMC（`@turbo`） | LoopVectorization `@turbo`（JIT、静的フラグなし） | OpenBLAS |

共通 CMake: `-DCMAKE_BUILD_TYPE=Release -DTesting=OFF -DUSE_SCALAPACK=OFF -DBLAS_LIBRARIES=/usr/lib64/libopenblas.so.0`。

### D.2 結果（L=32、timer-ON、中央値 [s]）

| id | 区間 | icc-C | gcc-C | **icc+simd (march)** | **icc+simd (axdisp)** | gcc+simd | Julia |
|--:|---|--:|--:|--:|--:|--:|--:|
| 2 | VMCParaOpt（全体） | 229.6 | 205.2 | **160.8** | **167.6** | 133.6 | 134.3 |
| 68 | UpdateMAllTwo | 73.4 | 48.4 | **17.2** | **27.5** | 15.7 | 14.4 |
| 66 | CalcNewPfMTwo2 | 38.5 | 48.4 | **39.8** | **32.3** | 27.6 | 15.5 |
| 40 | CalculateMAll | 65.8† | 50.5 | 52.4 | 52.2 | 49.1 | 51.3 |
| 42 | SlaterElmDiff | 4.4 | 5.3 | 4.4 | 5.6 | 5.3 | 12.1 |

†icc-C のみ MKL。注目カーネル `[66]`/`[68]` は BLAS 非依存なので比較は妥当。

L=16 / L=24 も同傾向（全体 [2]: icc+simd march = 36.6 / 82.4、axdisp = 40.9 / 86.6、対 gcc+simd 35.7 / 74.7、Julia 38.4 / 79.2）。`UpdateMAllTwo` [68]: march = 4.0 / 7.8、axdisp = 6.5 / 13.1。

### D.3 発見

1. **`#pragma omp simd` は icc でも効く（主ボトルネックで劇的）。** `UpdateMAllTwo` は icc-C 73.4 → icc+simd(march) **17.2 s（4.3×）**で、gcc+simd(15.7)/Julia(14.4) にほぼ並ぶ。移植可能 pragma は icc でも fire する。

2. **icc×AMD ディスパッチ劣化はベクトル化後も残る。** 同じ simd 入りでも `[68]` は march **17.2** vs axdisp **27.5 s ＝ 1.6×**。AMD で baseline 経路に落ちる代償がカーネルに残る。§4 の「icc×AMD ~1.5×」を simd 後の条件で再確認。

3. **icc は `[66]` を simd でも速くできない（予想外）。** `CalcNewPfMTwo2` は icc-C 38.5 → icc+simd(march) **39.8（ほぼ不変）**。gcc は 48.4 → 27.6 と改善するのに、icc は AVX2 でこの入れ子縮約を効率化できない（axdisp の baseline 32.3 の方がむしろ速い＝icc の AVX2 codegen が逆効果）。これが icc+simd(march) 全体を gcc+simd/Julia まで届かせない主因（`[66]` で +12s、`[72]` で +9s）。

### D.4 結論

- simd は icc でも有効（icc-C 全体 229.6 → 160.8 ≈ **1.4×**）。修正の**移植可能性は実証**された。
- ただし EPYC では icc+simd は gcc+simd/Julia に **~20% 及ばない**（②ディスパッチ劣化 ＋ ③icc の `[66]` 非ベクトル化）。
- **EPYC での最良ビルドは gcc+simd**。icc を使う限りは AMD 上で penalty が残るため、`#pragma omp simd` 化に加えて **コンパイラ選択（gcc/clang）またはターゲット指定（`-march=` を使い `-x`/`-ax` を避ける）**が要る。

### D.5 注意点

- `[40] CalculateMAll`: icc+simd ビルドは `-DUSE_GEMMT` を明示せず既定のまま（gcc ビルドは `USE_GEMMT=OFF`）。`[40]` は GEMMT/BLAS 依存なので厳密比較には不向きだが、本検証の主眼 `[66]`/`[68]` は BLAS 非依存で影響なし。
- `-xCORE-AVX2`（厳格）は AMD で起動不可（D 冒頭の注記）。本表の "axdisp" は `-axCORE-AVX2`（マルチパス）。

### D.6 再現情報

- ソース: `$HOME/shin-mvmc-benchmark/source/mVMC-gcc-simd`（ビルド: `build_icc_march` / `build_icc_axdisp`）。
- ベンチ: `$HOME/iccsimd_bench/`（`results/{march,axdisp}/L{16,24,32}/run_*/output/zvo_CalcTimer.dat`）。
- Slurm jobids: march L16/24/32 = 2923080/2923081/2923082、axdisp L16/24/32 = 2923083/2923084/2923090。
- 環境: ohtaka i8cpu（AMD EPYC 7702）、`mvmcvars.sh`（oneAPI classic 2023.0.0 + openmpi-4.1.5）。
