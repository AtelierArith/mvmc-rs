---
date: 2026-06-20
datetime: 2026-06-20 23:11 JST
model: GPT-5 Codex
status: investigation
topic: C-mVMC NStore/NSplitSize/SR-CG implementation status
---

# C-mVMC `NStore` / `NSplitSize` / SR-CG 調査

## 0. 結論

現行 `mVMC/`（branch `feature/nbodyinterall`, HEAD `8c7f95a`）で、Julia-mVMC
開発時に問題になった `NStore`、`NSplitSize`、SR-CG (`NSRCG`) の C 側実装を調査した。

結論は以下。

| 条件 | C-mVMC の現状 | 判定 |
|---|---|---|
| `NSplitSize=1`, `NSRCG=0`, `NStore=0/1` | direct/store 経路は一致 | 実装済み |
| `NSplitSize=1`, `NSRCG=1` | SR-CG は MPI でも direct SR と一致可能。ただし入力によって `NSROptCGMaxIter` 調整が必要 | 実装済み |
| `NSplitSize=1`, `NSRCG=2` | `NSRCG=1 + useDiagScale=1` に正規化され、DiagScale-CG として動作 | 実装済み |
| `NSplitSize>1`, `NStore=0`, `NSRCG=0` | store を使わない direct SR は整合 | 実装済み |
| `NSplitSize>1`, `NStore!=0` | store slot の index 不整合で SR 行列が silent に壊れる | バグ |
| `NSplitSize>1`, `NSRCG!=0` | SR-CG も stored `O` を使うため silent wrong | バグ |

したがって、Julia-mVMC 側で

```text
NSplitSize > 1 && (NStore != 0 || NSRCG != 0)
```

を unsupported にしている判断は妥当。C 側は、修正するまではこの組み合わせを hard
reject すべき。

用語上の注意:

- `NStoreOO` という C-mVMC 入力名はない。入力ファイルでは `NStore`、内部 global は
  `NStoreO`。
- `Nsplisize` ではなく、実装名は `NSplitSize`。

## 1. 実装経路

### 1.1 入力名と default

`NStore` / `NSplitSize` / `NSRCG` は StdFace から `modpara.def` へ出力される。

- StdFace parser: `mVMC/src/StdFace/src/StdFace_main.c:2794`,
  `mVMC/src/StdFace/src/StdFace_main.c:2798`,
  `mVMC/src/StdFace/src/StdFace_main.c:2799`
- StdFace output: `mVMC/src/StdFace/src/StdFace_main.c:1505-1507`
- StdFace default: `mVMC/src/StdFace/src/StdFace_main.c:1950-1952`

expert/readdef 側では `NStore` が内部 `NStoreO` に入る。

- global: `mVMC/src/mVMC/include/global.h:45-48`
- parse: `mVMC/src/mVMC/readdef.c:2443-2458`
- default: `mVMC/src/mVMC/readdef.c:2323-2328`
- broadcast: `mVMC/src/mVMC/readdef.c:872-878`

`NSRCG=2` は readdef 後に `NSRCG=1` と `useDiagScale=1` へ変換される。

- `mVMC/src/mVMC/readdef.c:1086-1103`

また `RescaleSmat=1` は `NSRCG==1` を要求する。

- `mVMC/src/mVMC/readdef.c:1105-1116`

### 1.2 `NSplitSize` の communicator 構造

MPI 実行では `NSplitSize` によって `comm1` / `comm2` が作られる。

- `mVMC/src/mVMC/vmcmain.c:239-246`

```c
group1 = rank0/NSplitSize;
MPI_Comm_split(comm0, group1, rank0, &comm1);
...
group2 = rank1;
MPI_Comm_split(comm0, group2, rank0, &comm2);
```

`VMCMakeSample` と `VMCMainCal` は `comm_child1`、つまり `comm1` 内で実行される。

- `mVMC/src/mVMC/vmcmain.c:381-418`

`VMCMainCal` 側は `comm1` の rank/size で sample を分割する。

- `mVMC/src/mVMC/vmccal.c:102-109`
- `mVMC/src/mVMC/vmccal_fsz.c:56-61`
- `mVMC/src/mVMC/splitloop.c:31-68`

### 1.3 `NStore` / `NSRCG` の store 経路

SR optimization のメモリ確保では、`NSRCG!=0` の場合 `SROptOO` は full matrix ではなく
`<O_i>` と対角だけを保持する。`NStoreO!=0` または `NSRCG==1` の場合、
sample ごとの `O` を `SROptO_Store[_real]` に保存する。

- allocation: `mVMC/src/mVMC/setmemory.c:465-500`
- free: `mVMC/src/mVMC/setmemory.c:565-577`

direct non-store では sample loop 内で `calculateOO[_real]` が full `OO` を更新する。

- `mVMC/src/mVMC/vmccal.c:237-242`
- `mVMC/src/mVMC/vmccal.c:845-858`

store 経路では `SROptO_Store[_real]` に `sqrt(w) * O` を保存し、後で
`calculateOO_Store[_real]` で `OO` を作る。

- store write: `mVMC/src/mVMC/vmccal.c:243-260`
- post-loop store multiply: `mVMC/src/mVMC/vmccal.c:317-329`
- real store multiply: `mVMC/src/mVMC/vmccal.c:673-721`
- complex store multiply: `mVMC/src/mVMC/vmccal.c:725-773`

`vmccal_fsz.c` も同型。

- `mVMC/src/mVMC/vmccal_fsz.c:187-218`
- `mVMC/src/mVMC/vmccal_fsz.c:232-248`

### 1.4 SR-CG の実装

SR-CG は real / complex それぞれ `stcopt_cg_impl.c` の macro instantiation で実装されている。

- wrapper: `mVMC/src/mVMC/stcopt_cg.c:51-66`
- main selection / fallback: `mVMC/src/mVMC/stcopt_cg_impl.c:81-351`
- normal CG: `mVMC/src/mVMC/stcopt_cg_impl.c:488-595`
- DiagScale-CG: `mVMC/src/mVMC/stcopt_cg_impl.c:356-484`
- matrix-vector product `z = S*x`: `mVMC/src/mVMC/stcopt_cg_impl.c:599-671`
- initialization from stored `O`: `mVMC/src/mVMC/stcopt_cg_impl.c:674-752`
- `RescaleSmat`: `mVMC/src/mVMC/stcopt_cg_impl.c:776-879`

MPI 対応は `operate_by_S` にある。rank 0 の探索ベクトル `x` を broadcast し、各 rank が
local stored `O` から `z_local` を作り、`SafeMpiAllReduce` で `z` を合算する。

- `MPI_Bcast(x, ...)`: `mVMC/src/mVMC/stcopt_cg_impl.c:627`
- local GEMV: `mVMC/src/mVMC/stcopt_cg_impl.c:637-649`
- `SafeMpiAllReduce(z_local, z, ...)`: `mVMC/src/mVMC/stcopt_cg_impl.c:651-653`

このため、`NSplitSize=1` の MPI sample-parallel SR-CG は設計として C 側に存在する。

## 2. `NSplitSize>1` store/SR-CG の root cause

問題は、stored `O` の index が global sample slot と local sample count の間で不整合になる点。

`VMCMainCal` は `SplitLoop` で各 rank の担当 sample 範囲を決める。

```c
SplitLoop(&sampleStart, &sampleEnd, NVMCSample, rank, size);
```

store write は global `sample` を使う。

```c
SROptO_Store_real[int_i + sample*SROptSize] = sqrtw*SROptO_real[int_i];
```

一方で post-loop の direct store multiply は、rank-local の `sampleSize` だけを渡す。

```c
sampleSize = sampleEnd - sampleStart;
calculateOO_Store_real(..., sampleSize);
```

`calculateOO_Store_real` は `j=0..sampleSize-1` の leading columns を読む。

```c
for (j=0; j<sampleSize; ++j) {
  o = srOptO_Store_real[i + j*srOptSize];
  ...
}
```

したがって `NSplitSize>1` で `sampleStart>0` の rank は、後方 global slot に正しく
書いたにもかかわらず、先頭 slot を読んで `OO` を作る。これは crash ではなく silent
wrong になる。

SR-CG ではさらに `StochasticOptCG_Init` が `i=0..NVMCSample-1` で stored `O` を読む。

```c
offset = i*OFFSET*SROptSize;
stcOs_real[idx] = CREAL(srOptO_Store[offset + pi + OFFSET]);
```

`NSplitSize>1` では各 rank が担当していない sample slot も含めて読むため、stored `O`
の未初期化・不一致領域を CG 入力に混ぜる。

`NSplitSize=1` では `comm1` 内の size が 1 なので `sampleStart=0` となり、この不整合は
表面化しない。

## 3. 実走検証

### 3.1 環境

- source: `mVMC/`
- branch: `feature/nbodyinterall`
- HEAD: `8c7f95a Add NBodyInterAll local energy support`
- build: `mVMC/build-nbodyinterall`
- fixture: `mVMC/test/python/data/HubbardChain_DiagCG/StdFace.def`
- command: `env OMP_NUM_THREADS=1 mpiexec -n {1,2} ...`
- scratch:
  - `tmp/c-mvmc-nsplit-store-probe/`
  - `tmp/c-mvmc-nsplit1-store-probe/`
  - `tmp/c-mvmc-mpi-srcg-nsplit1-probe/`
  - `tmp/c-mvmc-mpi-srcg-nsplit2-probe/`

### 3.2 `NSplitSize=1`: direct vs store

`NSplitSize=1`, `NSRCG=0` で `NStore=0` と `NStore=1` を比較。

| 比較 | 結果 |
|---|---:|
| `zqp_opt.dat` max abs diff | `3.552713678800501e-15` |
| `zqp_opt.dat` mean abs diff | `4.166640577536674e-16` |
| `zvo_SRinfo.dat` max abs diff | `0.0` |

判定: `NSplitSize=1` の store 経路は direct 経路と一致する。

### 3.3 `NSplitSize=2`: direct vs store

`mpiexec -n 2`, `NSplitSize=2`, `NSRCG=0` で `NStore=0` と `NStore=1` を比較。

| 比較 | 結果 |
|---|---:|
| `zqp_opt.dat` max abs diff | `4.6096478353043455` |
| `zqp_opt.dat` mean abs diff | `0.8062498699127654` |
| `zvo_SRinfo.dat` max abs diff | `18.0` |

`zvo_SRinfo.dat`:

```text
direct: [19, 19, 19, 0, 1.0924, 0, 0.187345, 36]
store : [19, 19, 19, 0, 0.7404, 0, 16.381, 18]
```

判定: `NSplitSize=2 && NStore=1` は step 0 の SR 行列診断から別物になっている。
energy/output だけでは初期 step で気づきにくいが、最適化 trajectory は壊れる。

### 3.4 `NSplitSize=1`: MPI SR-CG

`mpiexec -n 2`, `NSplitSize=1`, `NStore=0`, `NSROptCGMaxIter=128` で direct SR と
SR-CG を比較。

| 条件 | `zqp_opt.dat` max abs diff | mean abs diff |
|---|---:|---:|
| `NSRCG=1` | `4.188314139952354e-10` | `3.3777623472701424e-11` |
| `NSRCG=2` | `1.21914922601718e-10` | `1.1964823642433907e-11` |

`SRinfo` の CG 反復数:

```text
NSRCG=1: info=28
NSRCG=2: info=16
```

補足: `NSROptCGMaxIter` を明示しない default (`max_iter=nSmat=19`) では、この MPI 入力の
`NSRCG=1` は未収束で abort した。これは solver 未実装ではなく、反復上限の問題。

判定: `NSplitSize=1` の MPI SR-CG は C 側に実装されている。

### 3.5 `NSplitSize=2`: MPI SR-CG

`mpiexec -n 2`, `NSplitSize=2`, `NStore=0`, `NSROptCGMaxIter=128` で direct SR と
SR-CG を比較。

| 条件 | `zqp_opt.dat` max abs diff | mean abs diff |
|---|---:|---:|
| `NSRCG=1` | `0.13063990431522488` | `0.01591057072369149` |
| `NSRCG=2` | `0.13063990428586192` | `0.015910570729117397` |

`zvo_SRinfo.dat`:

```text
direct: [19, 19, 19, 0, 1.0924, 0, 0.187345, 36]
NSRCG=1: [19, 19, 0, 0, 0.7404, 0.00374, 0.206403, 9, 27]
NSRCG=2: [19, 19, 0, 0, 0.7404, 0.00374, 0.206403, 9, 18]
```

判定: `NSplitSize=2 && NSRCG!=0` は SR-CG 入力が壊れている。`NStore=0` でも SR-CG
内部で stored `O` を使うため、`NStore` の指定では回避できない。

## 4. テスト網の状況

C-mVMC の現行 tests には、SR-CG direct-vs-CG 比較と `RescaleSmat` smoke はある。

- `mVMC/test/python/runtest_sr.py`
- `mVMC/test/python/runtest_rescale.py`
- `mVMC/test/python/CMakeLists.txt:260-274`

ただし、これらは `NSplitSize>1` と `NStore` / `NSRCG` を掛け合わせていない。

MPI smoke は存在するが、基本的に `NSplitSize=1` の通常経路や physics smoke であり、
`NSplitSize>1 && (NStore!=0 || NSRCG!=0)` の silent wrong を検出しない。

## 5. Julia-mVMC への含意

Shin-mVMC 側の既存判断は維持すべき。

1. `NSplitSize>1 && (NStore!=0 || NSRCG!=0)` は Julia-mVMC で実装しない。
   C 側が silent wrong なので、C 忠実性の対象にしてはいけない。
2. `NSplitSize>1` を将来解禁する場合、最初の安全 subset は
   `NStore=0`, `NSRCG=0` に限定するのが妥当。
3. MPI SR-CG は `NSplitSize=1` 限定なら C 側に実装がある。Julia 側で対応するなら、
   `operate_by_S` 相当の broadcast/allreduce、`NSRCG=2`/`useDiagScale`/`RescaleSmat`
   の parser/gate、反復上限・許容誤差の検証を先に設計する必要がある。

## 6. C 側の修正方針

安全な方針は 2 つ。

### 方針 A: hard reject

最小修正として、readdef 後に以下を abort する。

```text
NSplitSize > 1 && (NStoreO != 0 || NSRCG != 0)
```

これは silent wrong を止めるには十分。default `NStore=1` なので、`NSplitSize>1` を使う
ユーザを明示的に保護できる。

### 方針 B: store indexing 修正

本修正を入れるなら、少なくとも以下が必要。

- `SROptO_Store[_real]` を rank-local dense column に書く、または local sample offset を
  store multiply / SR-CG init に渡す。
- local sample 数を `calculateOO_Store[_real]` だけでなく SR-CG init / `operate_by_S`
  / `Rescale4SRCG` まで一貫して扱う。
- 未担当 sample slot を読む設計を残すなら、毎 step で stored `O` の未使用 slot を zero
  clear する。

単純に write index だけを `sample - sampleStart` へ変えると、direct store multiply は直るが、
SR-CG が `NVMCSample` 列を読む設計や `Rescale4SRCG` の loop と整合しない可能性がある。
SR-CG まで含めた local sample count 設計が必要。

## 7. 推奨 regression

C 側修正または reject を入れる場合、最低限以下を追加する。

| test | 期待 |
|---|---|
| `mpiexec -n 2`, `NSplitSize=2`, `NStore=0`, `NSRCG=0` | 現行 direct 経路と同程度の FP tolerance で整合 |
| `mpiexec -n 2`, `NSplitSize=2`, `NStore=1`, `NSRCG=0` | 修正するなら direct と整合。reject 方針なら明示 error |
| `mpiexec -n 2`, `NSplitSize=2`, `NStore=0`, `NSRCG=1` | 修正するなら direct/SR-CG tolerance で整合。reject 方針なら明示 error |
| `mpiexec -n 2`, `NSplitSize=2`, `NStore=0`, `NSRCG=2` | DiagScale-CG として同様 |
| `mpiexec -n 2`, `NSplitSize=2`, `RescaleSmat=1` | 修正範囲に含めるなら local sample count と整合 |

## 8. 既存 Shin-mVMC 文書との照合

今回の現行 C-mVMC 再調査は、Shin-mVMC 側の以下の既存結論と一致する。

- `docs/reports/2026-06-11-c-mvmc-nsplit-store-verification.md`
- `docs/reports/2026-06-14-dev-mvmc-nsplit-store-nsrcg-issue.md`

今回の差分は、現行 `mVMC/feature/nbodyinterall` HEAD でも同じ問題が残っていること、
および `NSplitSize=1` の MPI SR-CG は `NSROptCGMaxIter=128` で direct SR と整合することを
再確認した点。

## 9. 対応方針メモ

対応は 2 段階に分ける。

### 9.1 先行安全策: hard reject

まず silent wrong を止めるため、C-mVMC 側に hard reject を入れるのが安全。

```text
NSplitSize > 1 && (NStoreO != 0 || NSRCG != 0)
```

この条件を入力エラーにする。場所は `readdef.c` の `NSRCG==2` 正規化後、
`RescaleSmat` / `useDiagScale` 周辺の validation block が自然。

理由:

- 現行 default は `NStore=1` なので、ユーザが `NSplitSize>1` だけを指定すると
  問題経路に入る。
- エラーを出さずに最適化 trajectory が壊れるため、warning では不十分。
- direct store / SR-CG の本修正前でも、公開 C-mVMC と Julia-mVMC の安全 contract を
  揃えられる。

### 9.2 本修正の大枠

本修正では、stored `O` を「global sample slot に書き、rank-local leading columns を読む」
現在の不整合から外す必要がある。

最も保守的な方向は、store 配列の確保量は当面 `NVMCSample` 分のまま維持し、
読み側へ `sampleStart` / `sampleSize` を明示的に渡す方法。

direct store 経路では、`calculateOO_Store[_real]` に渡す pointer を
`SROptO_Store[_real] + sampleStart * stride` へずらし、`sampleSize` 列だけを読む。

SR-CG 経路では、`StochasticOptCG_Init` / `operate_by_S` / `Rescale4SRCG` の
`NVMCSample` 固定 loop / GEMV 次元を、rank-local の `sampleSize` に置き換える必要がある。
`sampleSize==0` rank では local contribution を 0 として allreduce する。

ただし direct store 修正と SR-CG 修正は影響範囲が異なるため、詳細設計は分けて書く。

### 9.3 文書化予定

後続で個別文書として整理する。

1. direct store 修正案:
   `NStore=1`, `NSRCG=0`, `NSplitSize>1` を対象に、`calculateOO_Store[_real]`
   の local sample range 修正と regression を設計する。
2. SR-CG 修正案:
   `NSRCG!=0`, `NSplitSize>1` を対象に、local sample count を CG workspace /
   `operate_by_S` / `RescaleSmat` へ通す設計と regression を整理する。

この順序にする理由は、direct store は `OO` 構築だけで閉じやすい一方、SR-CG は workspace
layout、matrix-vector product、rescale、fallback まで含むため、同時に直すと検証範囲が
大きくなりすぎるため。
