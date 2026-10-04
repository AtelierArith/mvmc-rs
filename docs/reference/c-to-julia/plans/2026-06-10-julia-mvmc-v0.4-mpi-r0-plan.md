---
date: 2026-06-10
datetime: 2026-06-10 19:29 JST
model: Claude Fable 5
status: plan
topic: Julia-mVMC v0.4 MPI R0（MPI infra + initialization semantics）実装計画
spec: docs/specs/2026-06-10-julia-mvmc-v0.4-mpi-design.md (v3.1)
review: docs/reviews/2026-06-10-julia-mvmc-v0.4-mpi-design-review-and-mitigation.md
revisions:
  - datetime: 2026-06-10 21:58 JST
    model: Claude Fable 5
    note: |
      plan review (docs/reviews/2026-06-10-julia-mvmc-v0.4-mpi-r0-plan-review.md)
      の F1-F7 を反映。絶対パス修正、Bcast!/mpiexec snippet を MPI.jl documented
      API へ、validate_supported_modpara の呼び出し順序、stdout の rank0 gate、
      MPI worker の abort guard、duplicate orbital idx invariant check を追加。
  - datetime: 2026-06-10 22:12 JST
    model: Claude Fable 5
    note: |
      plan review addendum C1 を反映。check_duplicate_consistency の検査対象を
      orbital_terms + RBM 9 sections へ拡張（_duplicate_checked_sections）。
      unit test に synthetic RBM duplicate の fail-fast ケースを追加。
---

# Julia-mVMC v0.4 MPI R0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** spec v3.1 §9 R0 — MPI infrastructure（`parallel.jl`）と initialization semantics
（seed / parameter sync / rank0-only output）を導入し、serial bit parity を保ったまま
`mpiexec -n 2` smoke を通す。

**Architecture:** C 構造直写し + 薄い通信層。新規 `parallel.jl` に
`ParallelContext`（comm0/comm1/comm2 + rank/size/group1）、`split_loop`（C `SplitLoop`
と同式）、chunked allreduce 等の wrapper を集約。mpiexec 非検出時は `SerialContext`
（MPI API を一切呼ばない no-op）で従来動作と bit 一致。high-level runner
（`run_para_opt_from_namelist`）の parse 直後に ctx を作り、seed 解決
（`resolved_RndSeed + group1`）→ `sync_modified_parameter!(ctx, ...)`（rank0 bcast）→
rank0-only output/readback まで通す。

**Tech Stack:** Julia 1.11+, MPI.jl（MPICH_jll 同梱、ohtaka では MPIPreferences で
system MPI）, SFMT.jl（C bit-compatible RNG）。

**R0 で対象外（spec の R1 以降）:** WeightAverage の allreduce 化、`vmc_phys_cal!` /
`run_phys_cal_from_namelist` の MPI-aware 化（PhysCal は R1 の Green reduce-to-rank0 と
一緒に行う）、QP/sample split（R3）、SR-CG MPI（R2）。R0 の MPI 実行では各 rank が
独立に全計算を行い、rank0 だけが出力する（rank0 の seed は `RndSeed + 0` なので
**rank0 の出力は serial 実行と bit 一致**する — これが smoke gate の判定基準）。

---

## 前提条件（着手前にユーザ確認）

- [ ] **Julia-mVMC repo の branch 状態確認。** 現在 working tree は
  `feature/v0.3-remove-maincal-sample-threading`（2026-06-10 19:29 時点）。v0.3 の
  release 作業が `develop` へ merge 済みになってから、最新 `develop` を base に
  `feature/v0.4-mpi-r0` を切る。**v0.3 が未 merge のまま着手しないこと**（ユーザに base
  branch を確認）。
- [ ] 動作確認用 Julia: `julia --version` が 1.11 以降であること。

## 規約（全タスク共通）

- 実行 dir: `/Users/misawatakahiro/Dropbox/Projects/Shin-mVMC/Julia-mVMC`（以下 repo 相対）。
- テストは必ず `JULIA_NUM_THREADS=1 OMP_NUM_THREADS=1` で実行（macOS の既知問題）。
- unit test: `cd MVMCOptimizers.jl && JULIA_NUM_THREADS=1 julia --project -e 'using Pkg; Pkg.test()'`
- integration test: `JULIA_NUM_THREADS=1 OMP_NUM_THREADS=1 julia --project test/integration/runtests.jl`
- commit は local のみ。`git push` / PR 操作は実行前にユーザ確認（CLAUDE.md）。
- C 参照は `../mVMC/src/mVMC/`（プロジェクトルート相対 `mVMC/src/mVMC/`）。

## File Structure

| 操作 | パス | 責務 |
|------|------|------|
| Create | `MVMCOptimizers.jl/src/parallel.jl` | MPI infra 全部（ctx、detection、split_loop、wrapper、seed 解決）。1 ファイル完結 |
| Create | `MVMCOptimizers.jl/test_unit/test_unit_parallel.jl` | parallel.jl の serial unit tests（mpiexec 不要） |
| Create | `test/mpi/mpi_smoke.jl` | mpiexec 配下で走る smoke worker（workspace level、integration fixture を使用） |
| Create | `test/mpi/run_mpi_smoke.jl` | serial 基準生成 → `mpiexec -n 2` 起動 → bit 比較の driver |
| Modify | `MVMCOptimizers.jl/Project.toml` | MPI.jl を deps + compat に追加 |
| Modify | `Project.toml`（root workspace） | MPI / MPIPreferences を追加（driver 用） |
| Modify | `MVMCOptimizers.jl/src/MVMCOptimizers.jl` | `include("parallel.jl")` + export |
| Modify | `MVMCOptimizers.jl/src/stochastic_opt.jl` | `get_parameter_value` 追加（`update_parameter_value` の read 版、:40 の直前） |
| Modify | `MVMCOptimizers.jl/src/parameter_sync.jl` | pack/unpack + `sync_modified_parameter!(ctx, data)` |
| Modify | `MVMCOptimizers.jl/src/run_para_opt_from_namelist.jl` | ctx 構築・seed 解決・rank0-only readback |
| Modify | `MVMCOptimizers.jl/src/vmc_para_opt.jl` | `ctx` kwarg、output の rank0 gate、ctx 版 sync 呼び出し |
| Modify | `MVMCOptimizers.jl/test/runtests.jl` | `test_unit_parallel.jl` の include 追加 |

---

### Task 1: branch + MPI.jl 依存追加

**Files:**
- Modify: `MVMCOptimizers.jl/Project.toml`
- Modify: `Project.toml`（root）

- [ ] **Step 1: branch 作成**

```bash
cd /Users/misawatakahiro/Dropbox/Projects/Shin-mVMC/Julia-mVMC
git checkout develop && git status -sb   # clean であること（dirty なら停止しユーザ確認）
git checkout -b feature/v0.4-mpi-r0
```

- [ ] **Step 2: MVMCOptimizers.jl/Project.toml に MPI を追加**

`[deps]` に追記:

```toml
MPI = "da04e1cc-30fd-572f-bb4f-1f8673147195"
```

`[compat]` に追記:

```toml
MPI = "0.20"
```

- [ ] **Step 3: root Project.toml（JuliaMVMCWorkspace）に driver 用依存を追加**

`[deps]` に追記:

```toml
MPI = "da04e1cc-30fd-572f-bb4f-1f8673147195"
MPIPreferences = "3da0fdf6-3ccc-4f1b-acd9-58baa6c99267"
```

- [ ] **Step 4: resolve + precompile（Manifest 更新）**

```bash
cd MVMCOptimizers.jl && julia --project -e 'using Pkg; Pkg.resolve(); Pkg.precompile()' && cd ..
julia --project -e 'using Pkg; Pkg.resolve(); Pkg.precompile()'
```

Expected: エラーなし。`MVMCOptimizers.jl/Manifest.toml` と root `Manifest.toml` に
MPI 系エントリが追加される。
**Manifest-v1.11.toml の方針（spec A6）:** 実行 Julia が 1.12 系の場合、
`juliaup` 等で 1.11 を起動し root で `julia +1.11 --project -e 'using Pkg; Pkg.resolve()'`
を実行して `Manifest-v1.11.toml` も更新する。1.11 が手元にない場合は更新せず、
LOG.md handoff に「Manifest-v1.11.toml 未更新」と明記して CI（R4）で確認する。

- [ ] **Step 5: serial 全テストで既存動作の不変を確認**

```bash
cd MVMCOptimizers.jl && JULIA_NUM_THREADS=1 julia --project -e 'using Pkg; Pkg.test()' && cd ..
```

Expected: 全 pass（v0.3 時点と同数）。

- [ ] **Step 6: Commit**

```bash
git add MVMCOptimizers.jl/Project.toml MVMCOptimizers.jl/Manifest.toml Project.toml Manifest.toml
# Manifest-v1.11.toml を更新した場合はそれも add
git commit -m "build: add MPI.jl dependency for v0.4 MPI R0"
```

---

### Task 2: `split_loop`（C `SplitLoop` の再現、TDD）

**Files:**
- Create: `MVMCOptimizers.jl/src/parallel.jl`
- Create: `MVMCOptimizers.jl/test_unit/test_unit_parallel.jl`
- Modify: `MVMCOptimizers.jl/src/MVMCOptimizers.jl`
- Modify: `MVMCOptimizers.jl/test/runtests.jl`

- [ ] **Step 1: failing test を書く**

`MVMCOptimizers.jl/test_unit/test_unit_parallel.jl` を新規作成:

```julia
# Unit tests for src/parallel.jl (serial paths only; no mpiexec needed).
using Test
using MVMCOptimizers
using MVMCOptimizers: split_loop, split_range

@testset "split_loop matches C SplitLoop (mVMC/src/mVMC/splitloop.c:33-63)" begin
    # (loop_length, size) => [(ist, ien) for rank 0..size-1]   (0-based, half-open)
    cases = Dict(
        (10, 2) => [(0, 5), (5, 10)],                       # divisible
        (10, 3) => [(0, 3), (3, 6), (6, 10)],               # non-divisible (imod=1)
        (3, 2)  => [(0, 1), (1, 3)],                        # F8: size0=3, NSplitSize=2 相当
        (5, 4)  => [(0, 1), (1, 2), (2, 3), (3, 5)],        # non-divisible (imod=1)
        (4, 4)  => [(0, 1), (1, 2), (2, 3), (3, 4)],        # size == loop
        (2, 4)  => [(0, 1), (1, 2), (2, 2), (2, 2)],        # A8: 空 range (size > loop)
        (1, 4)  => [(0, 1), (1, 1), (1, 1), (1, 1)],        # A8: loop=1, size=4
        (0, 2)  => [(0, 0), (0, 0)],                        # A8: loop=0 → 全 rank 空 (C と同じ)
    )
    for ((n, sz), expected) in cases
        for r in 0:(sz - 1)
            @test split_loop(n, r, sz) == expected[r + 1]
        end
    end
end

@testset "split_range is the 1-based Julia view of split_loop" begin
    @test split_range(10, 0, 3) == 1:3
    @test split_range(10, 2, 3) == 7:10
    @test isempty(split_range(2, 3, 4))     # 空 range → empty Julia range
end

@testset "group1 assignment (C vmcmain.c:239)" begin
    # size0=3, NSplitSize=2: 最後の group が小さい (F8, warning のみで継続)
    @test [div(r, 2) for r in 0:2] == [0, 0, 1]
end
```

- [ ] **Step 2: include 配線をして test が FAIL することを確認**

`MVMCOptimizers.jl/src/MVMCOptimizers.jl` の `include("c_timer.jl")`（line 35）の直後に
追記:

```julia
include("parallel.jl")
```

`MVMCOptimizers.jl/test/runtests.jl` の `include("../test_unit/test_unit_threading.jl")`
（line 71）の直後に追記:

```julia
    include("../test_unit/test_unit_parallel.jl")
```

空の `MVMCOptimizers.jl/src/parallel.jl` を作って実行:

```bash
cd MVMCOptimizers.jl && JULIA_NUM_THREADS=1 julia --project -e \
  'using Pkg; Pkg.test()' 2>&1 | tail -20
```

Expected: FAIL（`split_loop` not defined / UndefVarError）。

- [ ] **Step 3: `split_loop` を実装**

`MVMCOptimizers.jl/src/parallel.jl` の先頭に:

```julia
"""
Parallel (MPI) infrastructure for Julia-mVMC v0.4.

Mirrors C-mVMC's communicator structure (mVMC/src/mVMC/vmcmain.c:239-257):
comm0 = whole run, comm1 = QP/sample split within a group (size NSplitSize),
comm2 = cross-group counter reduction. Serial runs (no mpiexec) use a no-op
SerialContext and never touch MPI APIs, preserving v0.3 bit parity.

Design doc: docs/specs/2026-06-10-julia-mvmc-v0.4-mpi-design.md (v3.1)
"""

using MPI: MPI

# C safempi.c:29  #define D_MpiSendMax 1048576 (elements per allreduce chunk)
const D_MPI_SEND_MAX = 1_048_576

"""
    split_loop(loop_length, mpi_rank, mpi_size) -> (ist, ien)

C `SplitLoop` (mVMC/src/mVMC/splitloop.c:33-63) と同一の分割。0-based half-open
`[ist, ien)` を返す。`mpi_size >= loop_length` のとき `mpi_rank >= loop_length` の
rank は空 range を受け取る。`loop_length == 0` も C と同じく全 rank 空 range。
"""
function split_loop(loop_length::Int, mpi_rank::Int, mpi_size::Int)
    if mpi_size < loop_length
        imod = loop_length % mpi_size
        if imod == 0
            idiv = loop_length ÷ mpi_size
            ist = idiv * mpi_rank
            ien = ist + idiv
        else
            idiv = (loop_length - imod) ÷ mpi_size
            if mpi_rank < mpi_size - imod
                ist = idiv * mpi_rank
                ien = ist + idiv
            else
                ist = idiv * mpi_rank + mpi_rank - (mpi_size - imod)
                ien = ist + idiv + 1
            end
        end
    else
        if mpi_rank < loop_length
            ist = mpi_rank
            ien = mpi_rank + 1
        else
            ist = loop_length
            ien = loop_length
        end
    end
    return ist, ien
end

"""
    split_range(loop_length, mpi_rank, mpi_size) -> UnitRange{Int}

`split_loop` の 1-based Julia range 版（`(ist+1):ien`）。
"""
function split_range(loop_length::Int, mpi_rank::Int, mpi_size::Int)
    ist, ien = split_loop(loop_length, mpi_rank, mpi_size)
    return (ist + 1):ien
end
```

- [ ] **Step 4: test pass を確認**

```bash
cd MVMCOptimizers.jl && JULIA_NUM_THREADS=1 julia --project -e 'using Pkg; Pkg.test()'
```

Expected: 全 pass（新規 split_loop testset 含む）。

- [ ] **Step 5: Commit**

```bash
git add MVMCOptimizers.jl/src/parallel.jl MVMCOptimizers.jl/src/MVMCOptimizers.jl \
        MVMCOptimizers.jl/test_unit/test_unit_parallel.jl MVMCOptimizers.jl/test/runtests.jl
git commit -m "feat(mpi-r0): add split_loop matching C SplitLoop with edge-case table"
```

---

### Task 3: `ParallelContext` / detection / `JULIA_MVMC_MPI` policy（TDD）

**Files:**
- Modify: `MVMCOptimizers.jl/src/parallel.jl`
- Modify: `MVMCOptimizers.jl/test_unit/test_unit_parallel.jl`

- [ ] **Step 1: failing test を追記**

`test_unit_parallel.jl` に追記:

```julia
using MVMCOptimizers: ParallelContext, serial_context, is_output_rank,
                      mpi_env_detected, resolve_mpi_mode

@testset "serial_context" begin
    ctx = serial_context()
    @test ctx.is_mpi == false
    @test ctx.comm0 === nothing && ctx.comm1 === nothing && ctx.comm2 === nothing
    @test (ctx.rank0, ctx.size0) == (0, 1)
    @test (ctx.rank1, ctx.size1) == (0, 1)
    @test (ctx.rank2, ctx.size2) == (0, 1)
    @test ctx.group1 == 0
    @test is_output_rank(ctx)
end

@testset "JULIA_MVMC_MPI policy (spec §4.1, F12+A7)" begin
    clean = ("JULIA_MVMC_MPI" => nothing, "OMPI_COMM_WORLD_SIZE" => nothing,
             "PMI_SIZE" => nothing, "PMI_RANK" => nothing)
    withenv(clean...) do
        @test !mpi_env_detected()
        @test resolve_mpi_mode() === :serial                       # auto + 未検出
    end
    withenv(clean..., "OMPI_COMM_WORLD_SIZE" => "2") do
        @test mpi_env_detected()
        @test resolve_mpi_mode() === :mpi                          # auto + 検出
    end
    withenv(clean..., "PMI_SIZE" => "2") do
        @test resolve_mpi_mode() === :mpi                          # MPICH hydra
    end
    withenv(clean..., "JULIA_MVMC_MPI" => "1") do
        @test resolve_mpi_mode() === :mpi                          # =1 は常に MPI 必須
    end
    withenv(clean..., "JULIA_MVMC_MPI" => "0") do
        @test resolve_mpi_mode() === :serial                       # =0 + 未検出 → serial
    end
    withenv(clean..., "JULIA_MVMC_MPI" => "0", "PMI_RANK" => "0") do
        @test resolve_mpi_mode() === :mpi_guarded_serial           # =0 + 検出 → guarded
    end
    withenv(clean..., "JULIA_MVMC_MPI" => "yes") do
        @test_throws ErrorException resolve_mpi_mode()             # 不正値は明示 error
    end
end
```

- [ ] **Step 2: FAIL を確認**

```bash
cd MVMCOptimizers.jl && JULIA_NUM_THREADS=1 julia --project -e 'using Pkg; Pkg.test()' 2>&1 | tail -5
```

Expected: FAIL（`ParallelContext` not defined）。

- [ ] **Step 3: 実装を追記**

`parallel.jl` に追記:

```julia
"""
    ParallelContext

C-mVMC の comm0/comm1/comm2 構造（mVMC/src/mVMC/vmcmain.c:239-246）の直写し。
serial 実行では `serial_context()`（is_mpi=false、全 size=1）を使い、通信 wrapper は
すべて no-op になる。
"""
struct ParallelContext
    is_mpi::Bool
    comm0::Union{Nothing,MPI.Comm}
    comm1::Union{Nothing,MPI.Comm}   # QP / sample 分担（グループ内、size=NSplitSize）
    comm2::Union{Nothing,MPI.Comm}   # counter 集約（グループ横断）
    rank0::Int
    size0::Int
    rank1::Int
    size1::Int
    rank2::Int
    size2::Int
    group1::Int                      # = rank0 ÷ NSplitSize（seed offset, vmcmain.c:257）
end

serial_context() =
    ParallelContext(false, nothing, nothing, nothing, 0, 1, 0, 1, 0, 1, 0)

"出力（zvo_*.dat / readback / 進捗 print）を担う rank か（C: `if(rank==0)`)。"
is_output_rank(ctx::ParallelContext) = ctx.rank0 == 0

# mpiexec 起動の検出に使う環境変数（spec §4.1）。
const MPI_ENV_KEYS = ("OMPI_COMM_WORLD_SIZE", "PMI_SIZE", "PMI_RANK")

mpi_env_detected() = any(k -> haskey(ENV, k), MPI_ENV_KEYS)

"""
    resolve_mpi_mode() -> :serial | :mpi | :mpi_guarded_serial

`JULIA_MVMC_MPI` の 3 値 semantics（spec §4.1、F12+A7）:
auto/unset → 検出時のみ :mpi。`0` → 未検出なら :serial、検出時は
:mpi_guarded_serial（MPI.Init 後に size>1 なら error+Abort、size==1 なら serial）。
`1` → 常に :mpi（Init 失敗はそのまま error）。
"""
function resolve_mpi_mode()
    raw = strip(get(ENV, "JULIA_MVMC_MPI", ""))
    detected = mpi_env_detected()
    if raw == "1"
        return :mpi
    elseif raw == "0"
        return detected ? :mpi_guarded_serial : :serial
    elseif raw == "" || raw == "auto"
        return detected ? :mpi : :serial
    else
        error("JULIA_MVMC_MPI must be \"0\", \"1\", \"auto\", or unset; got \"$raw\"")
    end
end

"""
    build_parallel_context(nsplit_size) -> ParallelContext

C vmcmain.c:239-257 の comm split を再現する。MPI mode では `MPI.Initialized()`
guard 付きで `MPI.Init()` し（二重 Init 回避、F13）、library 側からは
`MPI.Finalize()` を呼ばない（MPI.jl が Julia exit 時に自動 finalize）。
`size0 % nsplit_size != 0` は C と同じく warning のみで継続する（F8）。
"""
function build_parallel_context(nsplit_size::Int)
    mode = resolve_mpi_mode()
    mode === :serial && return serial_context()

    MPI.Initialized() || MPI.Init()
    comm0 = MPI.Comm_dup(MPI.COMM_WORLD)
    rank0 = MPI.Comm_rank(comm0)
    size0 = MPI.Comm_size(comm0)

    if mode === :mpi_guarded_serial
        # F12: mpiexec 配下の JULIA_MVMC_MPI=0。size>1 は出力破壊防止のため abort。
        if size0 > 1
            rank0 == 0 && @error "JULIA_MVMC_MPI=0 under a multi-rank mpiexec run is " *
                                 "not allowed (each rank would write the same output " *
                                 "files). Unset JULIA_MVMC_MPI or run without mpiexec."
            MPI.Abort(comm0, 1)
        end
        return serial_context()
    end

    nsplit = max(nsplit_size, 1)
    if size0 % nsplit != 0 && rank0 == 0
        @warn "load imbalance. MPI size0=$size0 NSplitSize=$nsplit"   # C vmcmain.c:248-250
    end
    group1 = rank0 ÷ nsplit
    comm1 = MPI.Comm_split(comm0, group1, rank0)
    rank1 = MPI.Comm_rank(comm1)
    size1 = MPI.Comm_size(comm1)
    group2 = rank1
    comm2 = MPI.Comm_split(comm0, group2, rank0)
    rank2 = MPI.Comm_rank(comm2)
    size2 = MPI.Comm_size(comm2)
    return ParallelContext(true, comm0, comm1, comm2,
                           rank0, size0, rank1, size1, rank2, size2, group1)
end
```

`MVMCOptimizers.jl` 本体（module file）の export 節に追記:

```julia
export ParallelContext, serial_context, build_parallel_context, is_output_rank
```

- [ ] **Step 4: test pass を確認 → Commit**

```bash
cd MVMCOptimizers.jl && JULIA_NUM_THREADS=1 julia --project -e 'using Pkg; Pkg.test()'
cd .. && git add -A MVMCOptimizers.jl/src MVMCOptimizers.jl/test_unit
git commit -m "feat(mpi-r0): ParallelContext, MPI detection, JULIA_MVMC_MPI 3-value policy"
```

---

### Task 4: 通信 wrapper（serial no-op + chunked allreduce、TDD）

**Files:**
- Modify: `MVMCOptimizers.jl/src/parallel.jl`
- Modify: `MVMCOptimizers.jl/test_unit/test_unit_parallel.jl`

- [ ] **Step 1: failing test を追記**

```julia
using MVMCOptimizers: bcast!, bcast_scalar, allreduce_sum!, reduce_sum_to_root!,
                      barrier, reduce_counter!, _chunk_ranges

@testset "serial wrappers are no-ops" begin
    ctx = serial_context()
    v = ComplexF64[1.0 + 2.0im, 3.0]
    @test bcast!(ctx, v) === v && v == ComplexF64[1.0 + 2.0im, 3.0]
    @test bcast_scalar(ctx, 42) == 42
    @test allreduce_sum!(ctx, v) === v && v == ComplexF64[1.0 + 2.0im, 3.0]
    @test reduce_sum_to_root!(ctx, v) === v
    @test barrier(ctx) === nothing
    c = collect(1:11)
    @test reduce_counter!(ctx, c) === c && c == collect(1:11)   # F10: 不変
end

@testset "_chunk_ranges (C SafeMpiAllReduce, safempi.c:29 D_MpiSendMax)" begin
    @test _chunk_ranges(0, 4) == UnitRange{Int}[]
    @test _chunk_ranges(3, 4) == [1:3]
    @test _chunk_ranges(8, 4) == [1:4, 5:8]
    @test _chunk_ranges(9, 4) == [1:4, 5:8, 9:9]
end
```

- [ ] **Step 2: FAIL を確認**（前タスクと同じコマンド。Expected: UndefVarError）

- [ ] **Step 3: 実装を追記**

`parallel.jl` に追記:

```julia
@inline function _comm(ctx::ParallelContext, which::Symbol)
    which === :comm0 && return ctx.comm0
    which === :comm1 && return ctx.comm1
    which === :comm2 && return ctx.comm2
    throw(ArgumentError("which must be :comm0, :comm1, or :comm2; got $which"))
end

"C SafeMpiAllReduce の chunk 分割（1 chunk あたり最大 `maxlen` 要素）。"
function _chunk_ranges(n::Int, maxlen::Int)
    n <= 0 && return UnitRange{Int}[]
    return [i:min(i + maxlen - 1, n) for i in 1:maxlen:n]
end

"C `MPI_Bcast` 相当。serial では no-op で buf を返す。"
function bcast!(ctx::ParallelContext, buf::AbstractArray; root::Int = 0,
                which::Symbol = :comm0)
    ctx.is_mpi || return buf
    # MPI.jl stable docs 掲載形式（keyword root）。positional 形式も実装上は存在するが
    # documented API に揃える（2026-06-10 plan review F2）。
    MPI.Bcast!(buf, _comm(ctx, which); root = root)
    return buf
end

"scalar の rank0 → 全 rank broadcast（seed 配布用、spec §5-1）。"
function bcast_scalar(ctx::ParallelContext, x::T; which::Symbol = :comm0) where {T<:Number}
    ctx.is_mpi || return x
    buf = T[x]
    MPI.Bcast!(buf, _comm(ctx, which); root = 0)
    return buf[1]
end

"C SafeMpiAllReduce 相当（sum、D_MpiSendMax 要素ごとに chunk 分割、in-place）。"
function allreduce_sum!(ctx::ParallelContext, buf::AbstractArray;
                        which::Symbol = :comm0)
    ctx.is_mpi || return buf
    comm = _comm(ctx, which)
    for r in _chunk_ranges(length(buf), D_MPI_SEND_MAX)
        MPI.Allreduce!(view(buf, r), +, comm)
    end
    return buf
end

"C weightAverageReduce 相当の reduce-to-root（Green 用、F9）。root のみ合計値を持つ。"
function reduce_sum_to_root!(ctx::ParallelContext, buf::AbstractArray;
                             root::Int = 0, which::Symbol = :comm0)
    ctx.is_mpi || return buf
    comm = _comm(ctx, which)
    for r in _chunk_ranges(length(buf), D_MPI_SEND_MAX)
        MPI.Reduce!(view(buf, r), +, comm; root = root)
    end
    return buf
end

function barrier(ctx::ParallelContext; which::Symbol = :comm0)
    ctx.is_mpi && MPI.Barrier(_comm(ctx, which))
    return nothing
end

"""
    reduce_counter!(ctx, counter)

C `ReduceCounter` 相当（mVMC/src/mVMC/vmcmake.c:496-508）: 先頭 6 要素
（C `Counter_max=6`、global.h:369-373）だけを comm2 で allreduce し、
**`ctx.rank2 == 0`（comm2-local root）だけ**へ書き戻す（A2。global rank0 ではない）。
`counter[7:end]`（Julia の burn flag `counter[11]` を含む）は対象外（F10）。
"""
function reduce_counter!(ctx::ParallelContext, counter::AbstractVector{<:Integer})
    ctx.is_mpi || return counter
    n = min(length(counter), 6)
    n == 0 && return counter
    recv = MPI.Allreduce(counter[1:n], +, ctx.comm2)
    if ctx.rank2 == 0
        counter[1:n] .= recv
    end
    return counter
end

"fatal error 時の全 rank 停止（F13）。serial は単に error。"
function abort_parallel(ctx::ParallelContext, code::Int = 1)
    if ctx.is_mpi
        MPI.Abort(ctx.comm0, code)
    else
        error("fatal error (abort code=$code)")
    end
end
```

export 追記: `export bcast!, bcast_scalar, allreduce_sum!, reduce_sum_to_root!, barrier, reduce_counter!, abort_parallel, split_loop, split_range`

- [ ] **Step 4: test pass を確認 → Commit**

```bash
cd MVMCOptimizers.jl && JULIA_NUM_THREADS=1 julia --project -e 'using Pkg; Pkg.test()'
cd .. && git add -A MVMCOptimizers.jl/src MVMCOptimizers.jl/test_unit
git commit -m "feat(mpi-r0): serial no-op communication wrappers + chunked allreduce"
```

---

### Task 5: seed 解決（C parity 表、TDD）

**Files:**
- Modify: `MVMCOptimizers.jl/src/parallel.jl`
- Modify: `MVMCOptimizers.jl/src/run_para_opt_from_namelist.jl:117-124`
- Modify: `MVMCOptimizers.jl/test_unit/test_unit_parallel.jl`

背景（spec §5-1 / A5）: C は missing→11272（**訂正 2026-06-11 review F2**: 当初
「parser が既に 11272 を入れる」と書いたが事実誤認。parser の kwdef default は
12345 で、`constants.jl:119` の `DEFAULT_RND_SEED=11272` は「行が存在するが
parse 不能」の fallback にのみ使われていた。R0 実装で kwdef default を 11272 へ
修正済み: `expert_types.jl`）、
`< 0`→rank0 time + bcast、`== 0`→0、`> 0`→指定値。現行 runner の
`rnd_seed > 0 ? rnd_seed : 11272` は `0` を 11272 に化けさせるので修正する。
既存 fixture は RndSeed 1 / 123456789 のみで影響なし（確認済み）。
parser 側 `parameter_init.jl:313` の legacy fallback は serial 後方互換 API のため
**触らない**。

- [ ] **Step 1: failing test を追記**

```julia
using MVMCOptimizers: resolve_rnd_seed

@testset "resolve_rnd_seed C parity (spec §5-1, A5)" begin
    ctx = serial_context()
    @test resolve_rnd_seed(ctx, 11272, nothing) == 11272   # missing → parser default
    @test resolve_rnd_seed(ctx, 0, nothing) == 0           # RndSeed==0 → seed 0 (C parity)
    @test resolve_rnd_seed(ctx, 123, nothing) == 123       # 正値
    t = resolve_rnd_seed(ctx, -1, nothing)                 # 負値 → 時刻 seed
    @test t isa Int && t > 0
    @test resolve_rnd_seed(ctx, 123, 777) == 777           # 明示 seed kwarg が優先
    # group1 offset (C vmcmain.c:257)
    ctx2 = MVMCOptimizers.ParallelContext(false, nothing, nothing, nothing,
                                          3, 4, 0, 1, 0, 1, 3)   # group1=3 の擬似 ctx
    @test resolve_rnd_seed(ctx2, 100, nothing) == 103
end
```

- [ ] **Step 2: FAIL を確認**（UndefVarError: resolve_rnd_seed）

- [ ] **Step 3: 実装**

`parallel.jl` に追記:

```julia
"""
    resolve_rnd_seed(ctx, modpara_rnd_seed, seed_override) -> Int

C parity の seed 解決（spec §5-1、A5）:
missing/default → 11272（parser が既に格納）、`< 0` → rank0 が時刻 seed を決め
comm0 bcast（C readdef.c:2161-2163）、`== 0` → 0、`> 0` → その値。
最後に `+ ctx.group1`（C vmcmain.c:257 `init_gen_rand(RndSeed+group1)`）。
`seed_override`（runner の `seed` kwarg）は表より優先される。
"""
function resolve_rnd_seed(ctx::ParallelContext, modpara_rnd_seed::Integer,
                          seed_override::Union{Integer,Nothing})
    base = if seed_override !== nothing
        Int(seed_override)
    elseif modpara_rnd_seed < 0
        t = ctx.rank0 == 0 ? Int(floor(time())) : 0
        Int(bcast_scalar(ctx, t))
    else
        Int(modpara_rnd_seed)
    end
    return base + ctx.group1
end
```

export 追記: `resolve_rnd_seed`

- [ ] **Step 4: runner へ配線**

`run_para_opt_from_namelist.jl` で、parse 直後（`data = ...parse_expert_mode_files...`
の次、line 109 付近）に追加:

```julia
    # v0.4 R0: unsupported input の検証は MPI context 構築より前（plan review F4）。
    # 現行は vmc_para_opt!（vmc_para_opt.jl:83）内でのみ呼ばれるが、R0 で
    # build_parallel_context を parse 直後へ移すため、invalid な NSplitSize
    #（< 1、または R0 では > 1 も未 support）で MPI context を作らないよう
    # ここで先に検証する。vmc_para_opt! 側の既存呼び出しは defense-in-depth
    # としてそのまま残す（重複呼び出しは無害）。
    validate_supported_modpara(data.modpara)

    # v0.4 R0: MPI context は seed / init_parameter! より前に作る（spec §4.2, F3）。
    ctx = build_parallel_context(data.modpara.nsplit_size)
```

注意（plan review F4）:

- `build_parallel_context` 内の `nsplit = max(nsplit_size, 1)` は C `vmcmain.c` の挙動
  写しだが、validation を先に通すことで「invalid 値の無言 clamp」には到達しない。
- R3 で `NSplitSize > 1` を解除する際は、`validate_supported_modpara` の gate を
  `NSplitSize > 1 && (NStore != 0 || NSRCG != 0)` 等の制限付き条件へ緩和する task を
  R3 plan に**別項目として**立てる（本 R0 plan では変更しない）。

既存の seed 解決（lines 117-124）:

```julia
    rng = SFMT19937RNG()
    actual_seed = if seed === nothing
        data.modpara.rnd_seed > 0 ? data.modpara.rnd_seed : 11272
    else
        Int(seed)
    end
    Random.seed!(rng, actual_seed)
```

を以下へ置換:

```julia
    rng = SFMT19937RNG()
    # C parity seed 解決 + group1 offset（spec §5-1; C vmcmain.c:257）。
    actual_seed = resolve_rnd_seed(ctx, data.modpara.rnd_seed, seed)
    Random.seed!(rng, actual_seed)
```

- [ ] **Step 5: unit + integration の serial 不変を確認 → Commit**

```bash
cd MVMCOptimizers.jl && JULIA_NUM_THREADS=1 julia --project -e 'using Pkg; Pkg.test()' && cd ..
JULIA_NUM_THREADS=1 OMP_NUM_THREADS=1 julia --project test/integration/runtests.jl
git add -A MVMCOptimizers.jl/src MVMCOptimizers.jl/test_unit
git commit -m "feat(mpi-r0): C-parity RndSeed resolution (+group1) wired into para_opt runner"
```

Expected: integration 全 pass（fixture RndSeed は 1 / 123456789 のため数値不変）。

---

### Task 6: parameter pack/unpack + `sync_modified_parameter!(ctx, data)`（TDD）

**Files:**
- Modify: `MVMCOptimizers.jl/src/stochastic_opt.jl`（`get_parameter_value` を
  `update_parameter_value`（line 40）の直前に追加）
- Modify: `MVMCOptimizers.jl/src/parameter_sync.jl`
- Modify: `MVMCOptimizers.jl/test_unit/test_unit_parallel.jl`

背景（spec §5-2 / F4）: C `SyncModifiedParameter` は冒頭で
`MPI_Bcast(Para, NPara, ...)`（mVMC/src/mVMC/parameter.c:142）。Julia は structured
storage なので、flat index ↔ storage の対応は既存 `update_parameter_value`
（stochastic_opt.jl:40、Proj → RBM → Slater → OptTrans の順）を正とし、その read 版
`get_parameter_value` + 差分適用の `set_parameter_value!` で pack/unpack を組む
（DRY: 書き込み経路は optimizer と共通化）。

- [ ] **Step 1: failing test を追記**

```julia
using MVMCOptimizers: count_total_parameters, get_parameter_value,
                      set_parameter_value!, pack_parameters, unpack_parameters!
using MVMCExpertModeParsers

@testset "parameter pack/unpack roundtrip (spec §5-2, F4)" begin
    fixture = joinpath(@__DIR__, "..", "..", "test", "integration", "reference",
                       "heisenberg_chain_real", "inputs", "namelist.def")
    data = MVMCExpertModeParsers.parse_expert_mode_files(fixture)
    n = count_total_parameters(data)
    @test n > 0

    para = pack_parameters(data)
    @test length(para) == n

    # update_parameter_value との整合: i 番目に δ を足すと pack[i] が δ 増える
    for i in (1, n)
        before = pack_parameters(data)[i]
        MVMCOptimizers.update_parameter_value(data, i, 0.25, -0.5)
        @test pack_parameters(data)[i] ≈ before + ComplexF64(0.25, -0.5)
    end

    # roundtrip: 摂動 → unpack で元に戻る
    original = pack_parameters(data)
    perturbed = original .+ ComplexF64(0.01, 0.02)
    unpack_parameters!(data, perturbed)
    @test pack_parameters(data) ≈ perturbed
    unpack_parameters!(data, original)
    @test pack_parameters(data) ≈ original

    @test_throws ArgumentError unpack_parameters!(data, original[1:max(n - 1, 0)])
end

@testset "duplicate idx invariant (plan review F7 / addendum C1)" begin
    fixture = joinpath(@__DIR__, "..", "..", "test", "integration", "reference",
                       "heisenberg_chain_real", "inputs", "namelist.def")
    data = MVMCExpertModeParsers.parse_expert_mode_files(fixture)

    # unpack 後、同一 idx の duplicate term は全て同値であること。
    para = pack_parameters(data) .+ ComplexF64(0.1, -0.1)
    unpack_parameters!(data, para)
    for idx in unique(t.idx for t in data.orbital_terms)
        vals = [t.value for t in data.orbital_terms if t.idx == idx]
        @test all(v -> v == vals[1], vals)
    end

    # orbital duplicate を人為的に不一致にすると fail-fast すること。
    dup_idx = findfirst(i -> count(t -> t.idx == data.orbital_terms[i].idx,
                                   data.orbital_terms) > 1,
                        eachindex(data.orbital_terms))
    if dup_idx !== nothing
        data.orbital_terms[dup_idx].value += ComplexF64(1.0, 0.0)
        @test_throws ErrorException unpack_parameters!(data, para)
    end

    # RBM duplicate も検査対象であること（addendum C1）。fixture に RBM がないため
    # synthetic に同一 idx・異値の term を 2 つ作る（fresh parse で orbital 側の
    # 不一致と混ざらないようにする）。
    data_rbm = MVMCExpertModeParsers.parse_expert_mode_files(fixture)
    push!(data_rbm.charge_rbm_phys_layer_terms,
          MVMCExpertModeParsers.ChargeRBMPhysLayerTerm(0, ComplexF64(0.1, 0.0), true, 0),
          MVMCExpertModeParsers.ChargeRBMPhysLayerTerm(1, ComplexF64(0.2, 0.0), true, 0))
    @test_throws ErrorException MVMCOptimizers.check_duplicate_consistency(data_rbm)
end

@testset "sync_modified_parameter!(ctx, data) serial == legacy" begin
    fixture = joinpath(@__DIR__, "..", "..", "test", "integration", "reference",
                       "heisenberg_chain_real", "inputs", "namelist.def")
    data_a = MVMCExpertModeParsers.parse_expert_mode_files(fixture)
    data_b = MVMCExpertModeParsers.parse_expert_mode_files(fixture)
    MVMCOptimizers.sync_modified_parameter!(data_a)                       # legacy
    MVMCOptimizers.sync_modified_parameter!(serial_context(), data_b)     # ctx 版
    @test pack_parameters(data_a) ≈ pack_parameters(data_b)
end
```

- [ ] **Step 2: FAIL を確認**（UndefVarError: count_total_parameters）

- [ ] **Step 3: `get_parameter_value` を `stochastic_opt.jl` に実装**

`update_parameter_value`（line 40）の直前に追加。block 構造・境界条件は
`update_parameter_value` と**完全に同一**に保つこと（読み取りのみ）:

```julia
"""
    get_parameter_value(data, para_idx) -> ComplexF64

flat parameter index（1-based、Proj → RBM → Slater → OptTrans）の現在値を返す。
`update_parameter_value` の read 版で、block 境界は同関数と同一。範囲外や
layout の隙間（spin-jastrow 等）は 0 を返す（update 側が no-op になる index と対応）。
"""
function get_parameter_value(data::ExpertModeData, para_idx::Int)::ComplexF64
    layout = MVMCExpertModeParsers.projection_layout(data)
    n_proj = layout.n_proj
    n_rbm = has_rbm_terms(data) ? MVMCExpertModeParsers.count_rbm_parameters(data) : 0
    n_orbital_idx = MVMCExpertModeParsers.count_orbital_parameters(data)
    n_opt_trans = MVMCExpertModeParsers.count_opt_trans_parameters(data)

    if para_idx <= n_proj
        if para_idx <= layout.gutzwiller_offset + layout.n_gutzwiller
            local_idx = para_idx - layout.gutzwiller_offset
            if 1 <= local_idx <= length(data.gutzwiller_terms)
                return ComplexF64(data.gutzwiller_terms[local_idx].value)
            end
        elseif para_idx <= layout.jastrow_offset + layout.n_jastrow
            local_idx = para_idx - layout.jastrow_offset
            if 1 <= local_idx <= length(data.jastrow_terms)
                return ComplexF64(data.jastrow_terms[local_idx].value)
            end
        elseif layout.dh2_offset < para_idx <= layout.dh2_offset + 6 * layout.n_dh2
            local_idx = para_idx - layout.dh2_offset
            if 1 <= local_idx <= length(data.doublon_holon_2site_params)
                return ComplexF64(data.doublon_holon_2site_params[local_idx])
            end
        elseif layout.dh4_offset < para_idx <= layout.dh4_offset + 10 * layout.n_dh4
            local_idx = para_idx - layout.dh4_offset
            if 1 <= local_idx <= length(data.doublon_holon_4site_params)
                return ComplexF64(data.doublon_holon_4site_params[local_idx])
            end
        end
        return ComplexF64(0)
    elseif para_idx <= n_proj + n_rbm
        rbm_idx = para_idx - n_proj - 1
        rbm_sections = (
            data.charge_rbm_phys_layer_terms,
            data.spin_rbm_phys_layer_terms,
            data.general_rbm_phys_layer_terms,
            data.charge_rbm_hidden_layer_terms,
            data.spin_rbm_hidden_layer_terms,
            data.general_rbm_hidden_layer_terms,
            data.charge_rbm_phys_hidden_terms,
            data.spin_rbm_phys_hidden_terms,
            data.general_rbm_phys_hidden_terms,
        )
        section_offset = 0
        for terms in rbm_sections
            n_section = isempty(terms) ? 0 : (maximum(t.idx for t in terms) + 1)
            if rbm_idx < section_offset + n_section
                local_idx = rbm_idx - section_offset
                for term in terms
                    if term.idx == local_idx
                        return ComplexF64(term.value)
                    end
                end
                return ComplexF64(0)
            end
            section_offset += n_section
        end
        return ComplexF64(0)
    else
        slater_start = n_proj + n_rbm + 1
        opt_trans_start = n_proj + n_rbm + n_orbital_idx + 1
        if slater_start <= para_idx < opt_trans_start
            orbital_idx = para_idx - n_proj - n_rbm - 1
            for term in data.orbital_terms
                if term.idx == orbital_idx
                    return ComplexF64(term.value)
                end
            end
        elseif opt_trans_start <= para_idx < opt_trans_start + n_opt_trans
            opt_idx = para_idx - opt_trans_start + 1
            return ComplexF64(data.opt_trans[opt_idx])
        end
        return ComplexF64(0)
    end
end
```

- [ ] **Step 4: pack/unpack + ctx 版 sync を `parameter_sync.jl` に実装**

`parameter_sync.jl` の末尾に追記:

```julia
"""
    count_total_parameters(data) -> Int

C の NPara 相当（Proj + RBM + Slater idx + OptTrans）。
"""
function count_total_parameters(data::ExpertModeData)::Int
    layout = MVMCExpertModeParsers.projection_layout(data)
    n_rbm = has_rbm_terms(data) ? MVMCExpertModeParsers.count_rbm_parameters(data) : 0
    return layout.n_proj + n_rbm +
           MVMCExpertModeParsers.count_orbital_parameters(data) +
           MVMCExpertModeParsers.count_opt_trans_parameters(data)
end

"""
    set_parameter_value!(data, para_idx, value)

flat index の値を `value` にする。書き込みは既存 `update_parameter_value`
（optimizer と同じ経路）への差分適用で行い、対応の二重実装を避ける。

**制限（plan review F7）:** `update_parameter_value` は同一 idx を持つ全
duplicate term（RBM 各 section / `orbital_terms`）へ同じ delta を加算するため、
duplicate 同士が何らかの理由で不一致になっている場合、この差分適用では root 値へ
「代入」修復できない（C の contiguous `Para` bcast より修復力が弱い）。R0 では
rank-local mutation が存在せず divergence は起きないため、delta 方式 + 下の
invariant check（fail-fast）で対応する。R1 以降で rank-local mutation が増える
場合は direct-assignment setter への refactor
（`update_parameter_value` の layout walker 共通化）を再検討する。
"""
function set_parameter_value!(data::ExpertModeData, para_idx::Int, value::ComplexF64)
    delta = value - get_parameter_value(data, para_idx)
    update_parameter_value(data, para_idx, real(delta), imag(delta))
    return data
end

"""
duplicate idx invariant（plan review F7 / addendum C1）: 同一 idx の duplicate term が
全て同値であることを確認する。不一致なら error（silent な部分修復より fail-fast を
選ぶ）。`unpack_parameters!` の冒頭で呼ぶ。

検査対象は `update_parameter_value` が「同一 idx の全 term へ同じ delta」を適用する
全 section: `orbital_terms` + RBM 9 sections（stochastic_opt.jl の `rbm_sections`
tuple と同一順）。RBM を使わない入力では各 section は empty vector なので no-op。
"""
function _duplicate_checked_sections(data::ExpertModeData)
    return (
        (:orbital_terms, data.orbital_terms),
        (:charge_rbm_phys_layer_terms, data.charge_rbm_phys_layer_terms),
        (:spin_rbm_phys_layer_terms, data.spin_rbm_phys_layer_terms),
        (:general_rbm_phys_layer_terms, data.general_rbm_phys_layer_terms),
        (:charge_rbm_hidden_layer_terms, data.charge_rbm_hidden_layer_terms),
        (:spin_rbm_hidden_layer_terms, data.spin_rbm_hidden_layer_terms),
        (:general_rbm_hidden_layer_terms, data.general_rbm_hidden_layer_terms),
        (:charge_rbm_phys_hidden_terms, data.charge_rbm_phys_hidden_terms),
        (:spin_rbm_phys_hidden_terms, data.spin_rbm_phys_hidden_terms),
        (:general_rbm_phys_hidden_terms, data.general_rbm_phys_hidden_terms),
    )
end

function check_duplicate_consistency(data::ExpertModeData)
    for (name, terms) in _duplicate_checked_sections(data)
        by_idx = Dict{Int,ComplexF64}()
        for term in terms
            v = get(by_idx, term.idx, nothing)
            if v === nothing
                by_idx[term.idx] = term.value
            elseif v != term.value
                error("$name idx=$(term.idx) has inconsistent duplicate values " *
                      "($v vs $(term.value)); delta-based unpack cannot repair this. " *
                      "See plan review F7 / addendum C1.")
            end
        end
    end
    return nothing
end

"C の contiguous `Para` 相当へ pack（spec §5-2 の pack/unpack helper）。"
pack_parameters(data::ExpertModeData) =
    ComplexF64[get_parameter_value(data, i) for i in 1:count_total_parameters(data)]

function unpack_parameters!(data::ExpertModeData, para::AbstractVector{ComplexF64})
    n = count_total_parameters(data)
    length(para) == n || throw(ArgumentError(
        "parameter vector length $(length(para)) != NPara $n"))
    check_duplicate_consistency(data)   # F7: 差分適用の前提（duplicate 同値）を保証
    for i in 1:n
        set_parameter_value!(data, i, para[i])
    end
    return data
end

"""
    sync_modified_parameter!(ctx, data; shift_correlations=true)

C `SyncModifiedParameter(comm0)`（mVMC/src/mVMC/parameter.c:134-175）の MPI-aware 版:
rank0 の parameter vector を comm0 で bcast（parameter.c:142）してから、全 rank が
同じ値に対して既存の local sync（DH/GJ shift → Slater rescale → OptTrans normalize）
を実行する。serial（`ctx.is_mpi == false`）では bcast を完全に skip し、既存実装と
bit 同一の経路になる。
"""
function sync_modified_parameter!(ctx::ParallelContext, data::ExpertModeData;
                                  shift_correlations::Bool = true)
    if ctx.is_mpi
        para = pack_parameters(data)
        bcast!(ctx, para; root = 0, which = :comm0)
        unpack_parameters!(data, para)
    end
    return sync_modified_parameter!(data; shift_correlations = shift_correlations)
end
```

export 追記: `count_total_parameters, pack_parameters, unpack_parameters!,
set_parameter_value!, get_parameter_value`（module file）。

- [ ] **Step 5: test pass + serial 回帰 → Commit**

```bash
cd MVMCOptimizers.jl && JULIA_NUM_THREADS=1 julia --project -e 'using Pkg; Pkg.test()' && cd ..
git add -A MVMCOptimizers.jl/src MVMCOptimizers.jl/test_unit
git commit -m "feat(mpi-r0): parameter pack/unpack + ctx-aware sync_modified_parameter! (C Para bcast)"
```

---

### Task 7: runner / `vmc_para_opt!` の ctx 配線 + rank0-only output/readback

**Files:**
- Modify: `MVMCOptimizers.jl/src/vmc_para_opt.jl`（signature、line 296、line 328、line 345）
- Modify: `MVMCOptimizers.jl/src/run_para_opt_from_namelist.jl`（sync 呼び出し、
  CTimer 書き出し、readback 節）

- [ ] **Step 1: `vmc_para_opt!` に ctx kwarg を追加**

signature（vmc_para_opt.jl:76-83 付近）へ kwarg を追加:

```julia
function vmc_para_opt!(
    data::ExpertModeData;
    callback::Union{Nothing,Function} = nothing,
    rng::Union{AbstractRNG,Nothing} = nothing,
    output_dir::Union{String,Nothing} = nothing,
    skip_sr::Bool = false,
    c_timer::Union{CTimer,Nothing} = nothing,
    ctx::ParallelContext = serial_context(),
)::Int
```

- [ ] **Step 2: 出力 2 箇所 + stdout を rank0 gate（C `if(rank==0) outputData()`）**

line 296:

```julia
        # C vmcmain.c:441 `if(rank==0) outputData()` — 出力は rank0 のみ（spec §5-9）
        is_output_rank(ctx) && output_data!(data, state, step; output_dir=output_dir)
```

line 345:

```julia
    is_output_rank(ctx) && output_opt_data!(data; output_dir=output_dir)
```

stdout も rank0 gate する（plan review F5）。file write だけ gate すると stdout が
rank 数分重複し、C の rank0-only な CLI 出力とズレる。対象は `vmc_para_opt!` 内の:

- progress 出力（line 203-206 の `println("Progress of Optimization: ...")`）
- final 出力（line 344-346 の `println("Start: Output opt params.")` /
  `println("End: Output opt params.")`）

```julia
        if is_output_rank(ctx) &&
           (step == 0 || (n_steps < 20) || (step % max(1, n_steps ÷ 20) == 0))
            progress = floor(Int, 100.0 * step / n_steps)
            println("Progress of Optimization: $progress %")
        end
```

```julia
    is_output_rank(ctx) && println("Start: Output opt params.")
    is_output_rank(ctx) && output_opt_data!(data; output_dir=output_dir)
    is_output_rank(ctx) && println("End: Output opt params.")
```

- [ ] **Step 3: step 内 sync を ctx 版へ（C vmcmain.c:509-510 SyncModifiedParameter(comm_parent)）**

line 328:

```julia
        sync_modified_parameter!(ctx, data)
```

- [ ] **Step 4: runner 側の配線**

`run_para_opt_from_namelist.jl`:

(a) 初期化時 sync（line 180）を置換:

```julia
    sync_modified_parameter!(ctx, data)
```

(b) `vmc_para_opt!` 呼び出しに `ctx = ctx` を追加:

```julia
    status = vmc_para_opt!(
        data;
        rng = rng,
        output_dir = String(output_dir),
        c_timer = c_timer,
        ctx = ctx,
    )
```

(c) CTimer 書き出しを rank0 gate:

```julia
    if timer_enabled && is_output_rank(ctx)
        write_ctimer_para_opt(c_timer, String(output_dir))
    end
```

(d) readback 節（「# 5. Read back outputs」の直前）に barrier + 非 rank0 early
return を追加（spec §5-9、F11）:

```julia
    # rank0 の write 完了を待ってから読み返す（spec §5-9、F11）。
    barrier(ctx)
    if !is_output_rank(ctx)
        # 非 rank0 は readback しない。minimal result を返す。
        return (
            status = status,
            output_dir = abspath(output_dir),
            zvo_first_n = String[],
            ctest_values = Float64[],
            final_energy_per_site = NaN,
            effective_nsteps = effective_nsteps,
            effective_nsmp = effective_nsmp,
        )
    end
```

- [ ] **Step 5: serial 回帰（unit + integration 全 pass、bit 不変）**

```bash
cd MVMCOptimizers.jl && JULIA_NUM_THREADS=1 julia --project -e 'using Pkg; Pkg.test()' && cd ..
JULIA_NUM_THREADS=1 OMP_NUM_THREADS=1 julia --project test/integration/runtests.jl
```

Expected: 全 pass。serial では `ctx = serial_context()` のため
`is_output_rank == true`・bcast 不実行・barrier no-op で、v0.3 と同一経路。

- [ ] **Step 6: Commit**

```bash
git add -A MVMCOptimizers.jl/src
git commit -m "feat(mpi-r0): thread ParallelContext through para_opt runner; rank0-only output/readback"
```

---

### Task 8: MPI smoke test（`mpiexec -n 2`、R0 gate）

**Files:**
- Create: `test/mpi/mpi_smoke.jl`（mpiexec 配下で走る worker）
- Create: `test/mpi/run_mpi_smoke.jl`（driver）

- [ ] **Step 1: worker を作成**

`test/mpi/mpi_smoke.jl`:

```julia
# mpiexec 配下（または serial）で run_para_opt_from_namelist を 1 回実行する worker。
# 使い方: julia --project=<workspace> test/mpi/mpi_smoke.jl <output_dir>
using MVMCOptimizers
using MPI

const fixture = joinpath(@__DIR__, "..", "integration", "reference",
                         "heisenberg_chain_real", "inputs", "namelist.def")
const outdir = ARGS[1]

# top-level abort guard（design §6.1、plan review F6）: 片 rank だけが exception で
# 死ぬと他 rank が collective / barrier で hang するため、worker 全体を try/catch で
# 包み、MPI 初期化済みなら MPI.Abort で全 rank を fail-fast させる。
try
    result = MVMCOptimizers.run_para_opt_from_namelist(
        fixture; nsteps = 4, nsmp = 4, mode = :real, output_dir = outdir)

    # rank の役割確認（serial 実行では常に output rank）。
    if isempty(result.zvo_first_n)
        # 非 rank0: minimal result（readback なし）であること。
        @assert isnan(result.final_energy_per_site)
        println("worker: non-root rank ok")
    else
        @assert result.status == 0
        @assert length(result.zvo_first_n) == 4
        println("worker: root rank ok (E/site = $(result.final_energy_per_site))")
    end
catch err
    @error "mpi_smoke worker failed" exception = (err, catch_backtrace())
    if MPI.Initialized() && !MPI.Finalized()
        MPI.Abort(MPI.COMM_WORLD, 1)   # 他 rank を hang させない
    end
    rethrow()
end
```

library 側（`run_para_opt_from_namelist` 本体）には R0 では outer guard を入れない。
rank-divergent exception の abort 責務は MPI worker / driver 側に置く（library が
`MPI.Abort` を呼ぶと serial・対話利用で破壊的なため）。R1 以降で collective が
増えた時点で library 内 guard の要否を再検討する。

- [ ] **Step 2: driver を作成**

`test/mpi/run_mpi_smoke.jl`:

```julia
# R0 smoke gate: serial 実行と `mpiexec -n 2` 実行で rank0 出力が bit 一致すること
# （rank0 の seed は RndSeed + group1 = RndSeed + 0 なので serial と同一 chain）。
# 使い方: JULIA_NUM_THREADS=1 julia --project=. test/mpi/run_mpi_smoke.jl
using MPI: mpiexec
using Test

const worker = joinpath(@__DIR__, "mpi_smoke.jl")
const project = abspath(joinpath(@__DIR__, "..", ".."))

@testset "R0 mpiexec -n 2 smoke (spec §9 R0 gate)" begin
    serial_dir = mktempdir()
    mpi_dir = mktempdir()

    # 1) serial 基準（MPI env なし → SerialContext）。
    run(`$(Base.julia_cmd()) --project=$project $worker $serial_dir`)

    # 2) mpiexec -n 2（両 rank が同じ output_dir を受け取る; rank0 のみ書く）。
    # mpiexec() は Cmd を返す（MPI.jl stable）。do-block 形式は deprecated（plan review F3）。
    run(`$(mpiexec()) -n 2 $(Base.julia_cmd()) --project=$project $worker $mpi_dir`)

    # 3) rank0-only output: ファイル集合が serial と同一（重複 write なし）。
    @test sort(readdir(serial_dir)) == sort(readdir(mpi_dir))

    # 4) bit 一致（rank0 = serial chain）。
    for f in readdir(serial_dir)
        @test read(joinpath(serial_dir, f)) == read(joinpath(mpi_dir, f))
    end
    println("R0 smoke: serial vs mpiexec -n 2 rank0 output is bit-identical")
end
```

- [ ] **Step 3: smoke を実行**

```bash
JULIA_NUM_THREADS=1 OMP_NUM_THREADS=1 julia --project test/mpi/run_mpi_smoke.jl
```

Expected: `Test Summary: ... Pass`、"bit-identical" の出力。
FAIL する場合の典型: (a) 非 rank0 が書いている（gate 漏れ → Task 7 を確認）、
(b) seed 解決差（Task 5 を確認）、(c) MPICH_jll 起動不可（`julia --project -e
'using MPI; println(MPI.versioninfo())'` で backend 確認）。

- [ ] **Step 3.5（任意、plan review Suggested Additional Tests）: 追加 smoke**

時間があれば driver に以下を足す（R0 gate の必須条件ではない）:

- mpiexec 実行の stdout を capture し、`Progress of Optimization` と
  `Start: Output opt params.` が **1 rank 分だけ**出ること（F5 の検証）。
- `JULIA_MVMC_MPI=0` + `mpiexec -n 2`: nonzero exit かつ hang しないこと（F12 経路）。
- `JULIA_MVMC_MPI=1` を MPI env なしで実行: size=1 の MPI context になること。

- [ ] **Step 4: Commit**

```bash
git add test/mpi/
git commit -m "test(mpi-r0): mpiexec -n 2 smoke gate (rank0 output bit-identical to serial)"
```

---

### Task 9: C 側 store 経路 × NSplitSize>1 の実走確認（手動、F2/A1 の結論出し）

**Files:**
- Create: `docs/reports/2026-06-XX-c-mvmc-nsplit-store-verification.md`
  （Shin-mVMC ルート側。実施日で命名、メタ情報ブロック付き）

レビューの実走観察（F2）を正式に再現・記録し、spec §3.1 の「結論が出るまで gate に
入れない」を解消する。**Julia コードには触れない**。

- [ ] **Step 1: C-mVMC を build（既存 build があれば再利用）**

```bash
cd /Users/misawatakahiro/Dropbox/Projects/Shin-mVMC/mVMC
mkdir -p build && cd build && cmake .. && make -j4 vmc.out 2>&1 | tail -3
```

- [ ] **Step 2: heisenberg_chain_real fixture で 4 条件を実走**

一時 dir に `Julia-mVMC/test/integration/reference/heisenberg_chain_real/inputs/` を
コピーし、modpara.def を編集して以下の 4 条件 × (`mpiexec -n 1` split=1 /
`mpiexec -n 2` split=2) を実行（全て `OMP_NUM_THREADS=1`、`NSROptItrStep=2`）:

1. `NStore=0, NSRCG=0`（期待: 一致）
2. `NStore=1, NSRCG=0`（期待: `zvo_SRinfo.dat` 不一致 = F2 再現）
3. `NStore=0, NSRCG=1`（期待: 不一致 = A1 の確認）
4. `NStore=1, NSRCG=1`

比較対象: `zvo_out_001.dat`（step 0/1）、`zvo_SRinfo.dat`。

- [ ] **Step 3: 結果を report に記録**

`docs/reports/` に date コマンド確認のうえメタ情報ブロック付きで記録し、spec §3.1 の
結論（unsupported 維持 or C 再現）と upstream issue 報告の要否判断を書く。LOG.md にも
1 エントリ追記。

---

### Task 10: ohtaka MPIPreferences 疎通 spike（手動、ohtaka 上）

**Files:**
- Create: `docs/reports/2026-06-XX-ohtaka-mpijl-spike.md`

- [ ] **Step 1: ohtaka 側で Julia 環境 + repo を準備**（既存の v0.3 ベンチ環境を流用）

- [ ] **Step 2: system MPI を MPIPreferences で設定**

```bash
# ohtaka ログインノードで（module 名は ohtaka の現行環境に合わせる）
julia --project=Julia-mVMC -e 'using MPIPreferences; MPIPreferences.use_system_binary()'
julia --project=Julia-mVMC -e 'using MPI; println(MPI.versioninfo())'
```

Expected: system MPI（ohtaka の OpenMPI/Intel MPI）が backend として表示される。

- [ ] **Step 3: smoke を 1 node 2 ranks と 2 nodes × 1 rank で実行**

ジョブスクリプトで `srun`/`mpiexec` から `test/mpi/mpi_smoke.jl` を実行し、
rank0-only output と serial bit 一致を確認。結果（module 構成、LocalPreferences.toml、
ジョブスクリプト、判定）を report に記録。LOG.md に 1 エントリ追記。

---

### Task 11: R0 完了 gate + 記録

- [ ] **Step 1: R0 gate を一括確認（spec §9 R0 gate）**

```bash
cd /Users/misawatakahiro/Dropbox/Projects/Shin-mVMC/Julia-mVMC
cd MVMCOptimizers.jl && JULIA_NUM_THREADS=1 julia --project -e 'using Pkg; Pkg.test()' && cd ..
JULIA_NUM_THREADS=1 OMP_NUM_THREADS=1 julia --project test/integration/runtests.jl
JULIA_NUM_THREADS=1 OMP_NUM_THREADS=1 julia --project test/mpi/run_mpi_smoke.jl
```

チェックリスト:
- [ ] serial 既存 integration 全 pass（bit 不変）
- [ ] `split_loop` table（非整除・空 range・loop=0）pass
- [ ] seed parity test（missing / -1 / 0 / positive）pass
- [ ] `mpiexec -n 2` smoke: rank0-only output、serial と bit 一致
- [ ] Task 9 / Task 10 の report 作成済み（未了なら handoff に明記）

- [ ] **Step 2: LOG.md へ R0 完了エントリ**（`date` コマンド確認、逆時系列で追記。
  Manifest-v1.11.toml の扱い・Task 9/10 の結論を summary / handoff に含める）

- [ ] **Step 3: ユーザへ報告**（R0 review 文書の要否と、push / PR はユーザ確認後）

---

## 完了の定義

spec v3.1 §9「R0 gate」の 4 項目 + Task 9（F2/A1 結論）+ Task 10（ohtaka 疎通）。
R1（sample 並列: WeightAverage allreduce / Green reduce-to-root / ReduceCounter 配線 /
PhysCal runner の MPI-aware 化）は本計画完了後に別 plan として作成する。
