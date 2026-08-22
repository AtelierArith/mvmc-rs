---
date: 2026-06-30
datetime: 2026-06-30 13:45 JST
updated_datetime: 2026-06-30 14:11 JST
model: GPT-5 Codex
status: plan
topic: Julia-mVMC V05-3 NSplitSize standard-projection QP split implementation
source_spec: docs/superpowers/specs/2026-06-30-julia-mvmc-v05-3-nsplit-standard-projection-design.md
source_review: docs/reviews/2026-06-30-julia-mvmc-v05-3-nsplit-standard-projection-review.md
target:
  repository: Julia-mVMC
  base: develop
  scope: ParaOpt direct SR, NSplitSize, standard-projection NQPFull
---

# Julia-mVMC V05-3 NSplitSize Standard-Projection QP Split Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Allow ParaOpt direct SR to run `NSplitSize > 1` with standard-projection `NQPFull > 1` while keeping OptTrans, SR-CG split, PhysCal split, and BackFlow out of scope.

**Architecture:** Preserve C-mVMC's split axes: sampling splits QP work inside `comm1`, while main calculation splits samples and keeps full-QP local inner products. Add explicit scalar reduce modes so sampling can reduce IP/logIP over `comm1` and main calculation can remain no-reduce, matching C's `MPI_COMM_SELF` calls.

**Tech Stack:** Julia 1.11, MPI.jl, MVMCOptimizers.jl, MVMCExpertModeParsers.jl, existing `test_unit` and `test/mpi` gates.

---

## Scope Check

This is one implementation plan. The work touches one runtime path:
ParaOpt direct SR sampling under `NSplitSize > 1`.

The following stay rejected by runtime guards:

- `NQPOptTrans > 1` or active `OptTrans` with `NSplitSize > 1`
- `NSRCG != 0` with `NSplitSize > 1`
- PhysCal with `NSplitSize > 1`
- BackFlow

## File Structure

- Modify `Julia-mVMC/MVMCOptimizers.jl/src/unsupported_inputs.jl`
  - Narrow the current `NQPFull > 1` split guard to OptTrans-derived QP only.
- Modify `Julia-mVMC/MVMCOptimizers.jl/src/parallel.jl`
  - Add a 1-based QP split helper and scalar sum/max allreduce helpers.
- Modify `Julia-mVMC/MVMCOptimizers.jl/src/MVMCOptimizers.jl`
  - Export only helpers that tests and MPI workers need directly.
- Modify `Julia-mVMC/MVMCOptimizers.jl/src/vmc_sampling.jl`
  - Add explicit IP reduction mode.
  - Use `ctx` and QP split ranges in all four sampling entry points.
  - Synchronize initial Pfaffian retry and regeneration decisions over `comm1`.
  - Replace FSZ complex hard-coded full-range logIP calls with local QP range calls.
- Modify `Julia-mVMC/MVMCOptimizers.jl/src/vmc_para_opt.jl`
  - Pass the already-built `ParallelContext` into sampling.
- Modify `Julia-mVMC/MVMCOptimizers.jl/src/vmc_main_cal.jl`
  - Make main-calculation IP calls explicitly no-reduce for auditability.
- Modify `Julia-mVMC/MVMCOptimizers.jl/src/green_func_calc.jl`
  - Make full-QP Green-function IP calls explicitly no-reduce for auditability.
- Modify `Julia-mVMC/MVMCOptimizers.jl/test/runtests.jl`
  - Include the new unit test file.
- Modify `Julia-mVMC/MVMCOptimizers.jl/test_unit/test_unit_unsupported_inputs.jl`
  - Replace the old blanket rejection tests.
- Modify `Julia-mVMC/MVMCOptimizers.jl/test_unit/test_unit_parallel.jl`
  - Cover QP split helper and serial scalar reductions.
- Create `Julia-mVMC/MVMCOptimizers.jl/test_unit/test_unit_vmc_sampling_qp_split.jl`
  - Cover IP/logIP default no-reduce, empty ranges, and explicit argument contract.
- Create `Julia-mVMC/test/mpi/mpi_nsplit_standard_projection_smoke.jl`
  - MPI worker for standard-projection `NQPFull > 1` four-case matrix.
- Modify `Julia-mVMC/test/mpi/mpi_failure_modes.jl`
  - Add pre-MPI rejection mode for active OptTrans with split.
- Modify `Julia-mVMC/test/mpi/run_mpi_smoke.jl`
  - Register the new positive and negative MPI smokes.
- Modify docs during the final task:
  - `Julia-mVMC/README.md`
  - `Julia-mVMC/MVMCOptimizers.jl/README.md`
  - `Julia-mVMC/docs/manual/03_optimization.md`
  - `Julia-mVMC/docs/manual/04_physics_calc.md`
  - `Julia-mVMC/docs/manual/05_compatibility.md`

## Task 1: Narrow The Runtime Validation Guard

**Files:**
- Modify: `Julia-mVMC/MVMCOptimizers.jl/src/unsupported_inputs.jl`
- Modify: `Julia-mVMC/MVMCOptimizers.jl/test_unit/test_unit_unsupported_inputs.jl`

- [ ] **Step 1: Write failing validation tests**

Append these testsets inside `@testset "unit/unsupported_inputs: NSplitSize contract"` in `Julia-mVMC/MVMCOptimizers.jl/test_unit/test_unit_unsupported_inputs.jl`, replacing the existing `"NSplitSize > 1 with NQPFull > 1 is rejected for para-opt"` testset.

```julia
    @testset "NSplitSize > 1 accepts standard-projection NQPFull > 1" begin
        spin_data = ExpertModeData()
        spin_data.modpara.nsplit_size = 2
        spin_data.modpara.nsp_gauss_leg = 2
        spin_data.modpara.nmp_trans = 1
        spin_data.n_qp_opt_trans = 1
        @test MVMCOptimizers.validate_supported_para_opt_data(spin_data) === nothing

        momentum_data = ExpertModeData()
        momentum_data.modpara.nsplit_size = 2
        momentum_data.modpara.nsp_gauss_leg = 1
        momentum_data.modpara.nmp_trans = 4
        momentum_data.n_qp_opt_trans = 1
        @test MVMCOptimizers.validate_supported_para_opt_data(momentum_data) === nothing

        trivial_opttrans_data = ExpertModeData()
        trivial_opttrans_data.modpara.nsplit_size = 2
        trivial_opttrans_data.modpara.nsp_gauss_leg = 8
        trivial_opttrans_data.modpara.nmp_trans = -1
        trivial_opttrans_data.n_qp_opt_trans = 1
        trivial_opttrans_data.opt_trans = ComplexF64[1.0 + 0.0im]
        trivial_opttrans_data.qp_opt_trans = [[0]]
        @test MVMCOptimizers.validate_supported_para_opt_data(trivial_opttrans_data) === nothing
    end

    @testset "NSplitSize > 1 rejects OptTrans-derived NQPFull > 1" begin
        data = ExpertModeData()
        data.modpara.nsplit_size = 2
        data.modpara.nsp_gauss_leg = 1
        data.modpara.nmp_trans = 1
        data.n_qp_opt_trans = 2
        data.opt_trans = ComplexF64[1.0 + 0.0im, 0.5 + 0.0im]
        data.qp_opt_trans = [[0], [0]]

        threw, msg = capture_error_message(
            () -> MVMCOptimizers.validate_supported_para_opt_data(data),
        )
        @test threw
        @test occursin("NSplitSize > 1 with NQPOptTrans > 1", msg)
        @test occursin("OptTrans", msg)
    end
```

- [ ] **Step 2: Run the test and verify it fails**

Run from `Julia-mVMC/`:

```bash
julia --project=. MVMCOptimizers.jl/test/runtests.jl
```

Expected: FAIL in `"NSplitSize > 1 accepts standard-projection NQPFull > 1"` because the old guard still rejects all `NQPFull > 1`.

- [ ] **Step 3: Replace the data-level guard**

In `Julia-mVMC/MVMCOptimizers.jl/src/unsupported_inputs.jl`, replace the `validate_supported_para_opt_data` body with:

```julia
function validate_supported_para_opt_data(data::ExpertModeData)
    if data.modpara.nsplit_size > 1
        n_qp_opt_trans = max(1, data.n_qp_opt_trans)
        opt_trans_active = n_qp_opt_trans > 1 ||
                           length(data.opt_trans) > 1 ||
                           length(data.qp_opt_trans) > 1
        if opt_trans_active
            error(
                "NSplitSize > 1 with NQPOptTrans > 1 / OptTrans is not " *
                "supported: grouped QP-split sampling currently supports " *
                "standard-projection NQPFull only (NQPOptTrans = 1), got " *
                "NSplitSize = $(data.modpara.nsplit_size), " *
                "NQPOptTrans = $(data.n_qp_opt_trans). Use NSplitSize = 1 " *
                "for OptTrans-derived QP sectors.",
            )
        end
    end
    return nothing
end
```

Update the docstring above it to say:

```julia
"""
    validate_supported_para_opt_data(data)

Validate parameter-optimization settings that require parsed data, not just
ModPara. `NSplitSize > 1` supports standard-projection `NQPFull > 1` when
`NQPOptTrans == 1`; OptTrans-derived QP sectors stay rejected with split.
"""
```

- [ ] **Step 4: Run the validation tests**

Run from `Julia-mVMC/`:

```bash
julia --project=. MVMCOptimizers.jl/test/runtests.jl
```

Expected: PASS for `unit/unsupported_inputs: NSplitSize contract`.

- [ ] **Step 5: Commit**

```bash
git -C Julia-mVMC add MVMCOptimizers.jl/src/unsupported_inputs.jl MVMCOptimizers.jl/test_unit/test_unit_unsupported_inputs.jl
git -C Julia-mVMC commit -m "Relax NSplitSize guard for standard projection"
```

## Task 2: Add QP Split And Scalar Reduction Helpers

**Files:**
- Modify: `Julia-mVMC/MVMCOptimizers.jl/src/parallel.jl`
- Modify: `Julia-mVMC/MVMCOptimizers.jl/src/MVMCOptimizers.jl`
- Modify: `Julia-mVMC/MVMCOptimizers.jl/test_unit/test_unit_parallel.jl`

- [ ] **Step 1: Write failing helper tests**

Append this to `Julia-mVMC/MVMCOptimizers.jl/test_unit/test_unit_parallel.jl` after the existing `split_range` testset:

```julia
using MVMCOptimizers: qp_split_range, allreduce_sum_scalar, allreduce_max_scalar

@testset "qp_split_range returns 1-based half-open QP ranges" begin
    fake_serial = serial_context()
    @test qp_split_range(4, fake_serial) == (1, 5)

    ctx0 = MVMCOptimizers.ParallelContext(false, nothing, nothing, nothing,
                                          0, 4, 0, 2, 0, 2, 0)
    ctx1 = MVMCOptimizers.ParallelContext(false, nothing, nothing, nothing,
                                          1, 4, 1, 2, 1, 2, 0)
    @test qp_split_range(4, ctx0) == (1, 3)
    @test qp_split_range(4, ctx1) == (3, 5)

    empty_ctx = MVMCOptimizers.ParallelContext(false, nothing, nothing, nothing,
                                               3, 4, 3, 4, 3, 4, 0)
    @test qp_split_range(1, empty_ctx) == (2, 2)
end

@testset "serial scalar reductions are no-ops" begin
    ctx = serial_context()
    @test allreduce_sum_scalar(ctx, 2.0 + 3.0im; which = :comm1) == 2.0 + 3.0im
    @test allreduce_sum_scalar(ctx, 4.5; which = :comm1) == 4.5
    @test allreduce_max_scalar(ctx, 7; which = :comm1) == 7
end
```

- [ ] **Step 2: Run the tests and verify they fail**

Run from `Julia-mVMC/`:

```bash
julia --project=. MVMCOptimizers.jl/test/runtests.jl
```

Expected: FAIL with `UndefVarError: qp_split_range not defined`.

- [ ] **Step 3: Implement helpers**

Add these definitions to `Julia-mVMC/MVMCOptimizers.jl/src/parallel.jl` after `split_range`:

```julia
"""
    qp_split_range(n_qp_full, ctx) -> (qp_start, qp_end)

Return the 1-based half-open QP range used by Julia sampling call sites.
Serial and single-rank comm1 runs get the full range.
"""
function qp_split_range(n_qp_full::Int, ctx::ParallelContext)
    n_qp_full >= 0 || throw(ArgumentError("n_qp_full must be >= 0; got $n_qp_full"))
    if ctx.size1 <= 1
        return (1, n_qp_full + 1)
    end
    ist, ien = split_loop(n_qp_full, ctx.rank1, ctx.size1)
    return (ist + 1, ien + 1)
end

function allreduce_sum_scalar(ctx::ParallelContext, value; which::Symbol = :comm0)
    ctx.is_mpi || return value
    return MPI.Allreduce(value, +, _comm(ctx, which))
end

function allreduce_max_scalar(ctx::ParallelContext, value::Integer; which::Symbol = :comm0)
    ctx.is_mpi || return value
    return MPI.Allreduce(value, max, _comm(ctx, which))
end
```

In `Julia-mVMC/MVMCOptimizers.jl/src/MVMCOptimizers.jl`, extend the parallel export list:

```julia
export bcast!, bcast_scalar, allreduce_sum!, reduce_sum_to_root!, barrier,
       reduce_counter!, abort_parallel, split_loop, split_range, qp_split_range,
       allreduce_sum_scalar, allreduce_max_scalar, resolve_rnd_seed
```

- [ ] **Step 4: Run the helper tests**

Run from `Julia-mVMC/`:

```bash
julia --project=. MVMCOptimizers.jl/test/runtests.jl
```

Expected: PASS for `qp_split_range returns 1-based half-open QP ranges` and `serial scalar reductions are no-ops`.

- [ ] **Step 5: Commit**

```bash
git -C Julia-mVMC add MVMCOptimizers.jl/src/parallel.jl MVMCOptimizers.jl/src/MVMCOptimizers.jl MVMCOptimizers.jl/test_unit/test_unit_parallel.jl
git -C Julia-mVMC commit -m "Add QP split scalar reduction helpers"
```

## Task 3: Make IP And LogIP Reduction Mode Explicit

**Files:**
- Modify: `Julia-mVMC/MVMCOptimizers.jl/src/vmc_sampling.jl`
- Create: `Julia-mVMC/MVMCOptimizers.jl/test_unit/test_unit_vmc_sampling_qp_split.jl`
- Modify: `Julia-mVMC/MVMCOptimizers.jl/test/runtests.jl`

- [ ] **Step 1: Add failing unit tests**

Create `Julia-mVMC/MVMCOptimizers.jl/test_unit/test_unit_vmc_sampling_qp_split.jl`:

```julia
using Test
using MVMCOptimizers
using MVMCOptimizers: calculate_ip_fcmp, calculate_log_ip_fcmp,
                      calculate_ip_real, calculate_log_ip_real
using MVMCExpertModeParsers

function qp_weight_data(weights::Vector{ComplexF64})
    data = MVMCExpertModeParsers.ExpertModeData()
    data.qp_weights = MVMCExpertModeParsers.QuantumProjectionWeights()
    data.qp_weights.qp_full_weight = copy(weights)
    return data
end

@testset "IP helpers default to no reduction and allow empty ranges" begin
    data = qp_weight_data(ComplexF64[2.0 + 0.0im, 3.0 + 0.0im, 5.0 + 0.0im])
    pf = ComplexF64[7.0 + 0.0im, 11.0 + 0.0im, 13.0 + 0.0im]
    @test calculate_ip_fcmp(pf, 1, 4, data) == 112.0 + 0.0im
    @test calculate_ip_fcmp(pf, 2, 3, data) == 33.0 + 0.0im
    @test calculate_ip_fcmp(pf, 2, 2, data) == 0.0 + 0.0im
    @test calculate_log_ip_fcmp(pf, 1, 4, data) ≈ log(112.0 + 0.0im)
end

@testset "real IP helpers default to no reduction and allow empty ranges" begin
    data = qp_weight_data(ComplexF64[2.0 + 9.0im, 3.0 + 0.0im])
    pf = [7.0, 11.0]
    @test calculate_ip_real(pf, 1, 3, data) == 47.0
    @test calculate_ip_real(pf, 2, 2, data) == 0.0
    @test calculate_log_ip_real(pf, 1, 3, data) ≈ log(abs(47.0) + 1e-100)
end

@testset "comm1 reduction requires a context" begin
    data = qp_weight_data(ComplexF64[1.0 + 0.0im])
    pf = ComplexF64[2.0 + 0.0im]
    @test_throws ArgumentError calculate_ip_fcmp(pf, 1, 2, data; reduce = :comm1)
    @test calculate_ip_fcmp(pf, 1, 2, data; ctx = serial_context(), reduce = :comm1) ==
          2.0 + 0.0im
end
```

In `Julia-mVMC/MVMCOptimizers.jl/test/runtests.jl`, include the new test inside the `"Unit Tests"` block:

```julia
    include("../test_unit/test_unit_vmc_sampling_qp_split.jl")
```

- [ ] **Step 2: Run the tests and verify they fail**

Run from `Julia-mVMC/`:

```bash
julia --project=. MVMCOptimizers.jl/test/runtests.jl
```

Expected: FAIL because `calculate_ip_fcmp(...; reduce = :comm1)` is not accepted and empty QP ranges are rejected.

- [ ] **Step 3: Implement reduction-mode helpers**

In `Julia-mVMC/MVMCOptimizers.jl/src/vmc_sampling.jl`, add these helpers before `calculate_log_ip_fcmp`:

```julia
@inline function _check_reduce_mode(ctx::Union{Nothing,ParallelContext}, reduce::Symbol)
    reduce in (:none, :comm1) ||
        throw(ArgumentError("reduce must be :none or :comm1; got $reduce"))
    if reduce == :comm1 && ctx === nothing
        throw(ArgumentError("ctx is required when reduce = :comm1"))
    end
    return nothing
end

@inline function _maybe_reduce_ip(value, ctx::Union{Nothing,ParallelContext}, reduce::Symbol)
    _check_reduce_mode(ctx, reduce)
    reduce == :comm1 || return value
    return allreduce_sum_scalar(ctx::ParallelContext, value; which = :comm1)
end

@inline function _valid_qp_range(qp_start::Int, qp_end::Int, n_qp_full::Int)
    return 1 <= qp_start <= qp_end <= n_qp_full + 1
end
```

Replace each IP/logIP signature with a keyword-compatible form. For example, replace `calculate_ip_fcmp` with:

```julia
function calculate_ip_fcmp(
    pf_m::Vector{ComplexF64},
    qp_start::Int,
    qp_end::Int,
    data::ExpertModeData;
    ctx::Union{Nothing,ParallelContext} = nothing,
    reduce::Symbol = :none,
)::ComplexF64
    if data.qp_weights === nothing
        @error "Quantum projection weights not initialized. Call init_qp_weight!(data) first."
        return 0.0 + 0.0im
    end

    qp_full_weight::Vector{ComplexF64} = data.qp_weights.qp_full_weight
    if !_valid_qp_range(qp_start, qp_end, length(qp_full_weight))
        @error "Invalid qp_start or qp_end: qp_start=$qp_start, qp_end=$qp_end, n_qp_full=$(length(qp_full_weight))"
        return 0.0 + 0.0im
    end
    if length(pf_m) < qp_end - 1
        @error "pf_m array too short: length=$(length(pf_m)), required=$(qp_end - 1)"
        return 0.0 + 0.0im
    end

    ip = 0.0 + 0.0im
    @inbounds for qp_idx = qp_start:(qp_end - 1)
        ip += qp_full_weight[qp_idx] * pf_m[qp_idx]
    end
    return _maybe_reduce_ip(ip, ctx, reduce)
end
```

Apply the same reduction-mode contract to:

```julia
calculate_log_ip_fcmp(...; ctx = nothing, reduce = :none) = log(reduced_ip)
calculate_ip_real(...; ctx = nothing, reduce = :none) = reduced_ip
calculate_log_ip_real(...; ctx = nothing, reduce = :none) = log(abs(reduced_ip) + 1e-100)
```

Keep the real-path logarithm convention unchanged. For the real helpers,
preserve the current invalid-range behavior and add empty-range plus
reduction handling; do not introduce a different real-log sign convention.

- [ ] **Step 4: Run unit tests**

Run from `Julia-mVMC/`:

```bash
julia --project=. MVMCOptimizers.jl/test/runtests.jl
```

Expected: PASS for the new `IP helpers default to no reduction and allow empty ranges` testsets.

- [ ] **Step 5: Commit**

```bash
git -C Julia-mVMC add MVMCOptimizers.jl/src/vmc_sampling.jl MVMCOptimizers.jl/test/runtests.jl MVMCOptimizers.jl/test_unit/test_unit_vmc_sampling_qp_split.jl
git -C Julia-mVMC commit -m "Make QP IP reduction explicit"
```

## Task 4: Wire QP Split Through Sampling

**Files:**
- Modify: `Julia-mVMC/MVMCOptimizers.jl/src/vmc_sampling.jl`
- Modify: `Julia-mVMC/MVMCOptimizers.jl/src/vmc_para_opt.jl`

- [ ] **Step 1: Add the context keyword to sampling entry points**

Change each sampling entry point signature to accept `ctx`:

```julia
function vmc_make_sample!(
    data::ExpertModeData,
    state::VMCOptimizationState,
    rng::AbstractRNG = Random.GLOBAL_RNG,
    c_timer::CTimer = CTIMER_DISABLED;
    ctx::ParallelContext = serial_context(),
)
```

Apply the same keyword to:

```julia
vmc_make_sample_fsz!(...; ctx::ParallelContext = serial_context())
vmc_make_sample_real!(...; ctx::ParallelContext = serial_context())
vmc_make_sample_fsz_real!(...; ctx::ParallelContext = serial_context())
```

- [ ] **Step 2: Replace full QP range initialization**

In each sampling entry point, replace:

```julia
qp_start = 1
qp_end = n_qp_full + 1
```

with:

```julia
qp_start, qp_end = qp_split_range(n_qp_full, ctx)
```

- [ ] **Step 3: Reduce sampling-time IP/logIP over comm1**

In `Julia-mVMC/MVMCOptimizers.jl/src/vmc_sampling.jl`, update every sampling-time IP/logIP call that uses the local QP range. The replacement shape is:

```julia
log_ip_old = calculate_log_ip_fcmp(
    state.slater_matrix.pf_m,
    qp_start,
    qp_end,
    data;
    ctx = ctx,
    reduce = :comm1,
)
```

Apply this shape to complex non-FSZ, complex FSZ, real non-FSZ, and real FSZ sampling calls.

- [ ] **Step 4: Fix the known FSZ complex full-range calls**

Replace the three FSZ complex calls currently shaped like:

```julia
log_ip_new = calculate_log_ip_fcmp(pf_m_new, 1, n_qp_full + 1, data)
```

with:

```julia
log_ip_new = calculate_log_ip_fcmp(
    pf_m_new,
    qp_start,
    qp_end,
    data;
    ctx = ctx,
    reduce = :comm1,
)
```

Verify the search is clean:

```bash
rg -n "calculate_log_ip_fcmp\\([^\\n]*1, n_qp_full \\+ 1" Julia-mVMC/MVMCOptimizers.jl/src/vmc_sampling.jl
```

Expected: no matches in sampling paths.

- [ ] **Step 5: Audit every sampling-time IP/logIP call**

Run from repository root:

```bash
rg -n "calculate_(log_)?ip_(fcmp|real)\\(" Julia-mVMC/MVMCOptimizers.jl/src/vmc_sampling.jl
```

Expected: every call inside `vmc_make_sample!`, `vmc_make_sample_fsz!`,
`vmc_make_sample_real!`, and `vmc_make_sample_fsz_real!` has `ctx = ctx`
and `reduce = :comm1` in its call block. The only exceptions are function
definitions and docstring examples above the helper definitions.

- [ ] **Step 6: Pass ctx from ParaOpt**

In `Julia-mVMC/MVMCOptimizers.jl/src/vmc_para_opt.jl`, pass the existing `ctx` into sampling:

```julia
vmc_make_sample_real!(data, state, rng, timer; ctx = ctx)
vmc_make_sample_fsz_real!(data, state, rng, timer; ctx = ctx)
vmc_make_sample!(data, state, rng, timer; ctx = ctx)
vmc_make_sample_fsz!(data, state, rng, timer; ctx = ctx)
```

- [ ] **Step 7: Run unit tests**

Run from `Julia-mVMC/`:

```bash
julia --project=. MVMCOptimizers.jl/test/runtests.jl
```

Expected: PASS. This confirms serial default behavior still works.

- [ ] **Step 8: Commit**

```bash
git -C Julia-mVMC add MVMCOptimizers.jl/src/vmc_sampling.jl MVMCOptimizers.jl/src/vmc_para_opt.jl
git -C Julia-mVMC commit -m "Wire QP split through sampling"
```

## Task 5: Synchronize Sampling Control Flow Across comm1

**Files:**
- Modify: `Julia-mVMC/MVMCOptimizers.jl/src/vmc_sampling.jl`

- [ ] **Step 1: Add status helpers**

Add this near the IP reduction helpers in `Julia-mVMC/MVMCOptimizers.jl/src/vmc_sampling.jl`:

```julia
@inline function _comm1_max_status(ctx::ParallelContext, status::Integer)::Int
    return Int(allreduce_max_scalar(ctx, Int(status); which = :comm1))
end

@inline function _comm1_any(ctx::ParallelContext, flag::Bool)::Bool
    return _comm1_max_status(ctx, flag ? 1 : 0) != 0
end
```

- [ ] **Step 2: Synchronize initial sample creation failures**

After each initial `make_initial_sample*!` call that can return a status, reduce the status before branching:

```julia
result = make_initial_sample!(
    tmp_ele_idx,
    tmp_ele_cfg,
    tmp_ele_num,
    tmp_ele_proj_cnt,
    data,
    state,
    rng,
)
result = _comm1_max_status(ctx, result)
if result != 0
    @error "Failed to generate initial sample"
    return
end
```

For `make_initial_sample_fsz!` variants that do not return a status, reduce the subsequent `calculate_m_all*` status and base regeneration on that reduced status.

- [ ] **Step 3: Synchronize invalid-sample early returns**

Replace local-only invalid sample checks shaped like:

```julia
if all(x -> x == 0 || x == -1, tmp_ele_idx)
    @error "tmp_ele_idx is invalid after make_initial_sample! (all zeros or -1)"
    return
end
```

with:

```julia
invalid_sample = all(x -> x == 0 || x == -1, tmp_ele_idx)
if _comm1_any(ctx, invalid_sample)
    invalid_sample && @error "tmp_ele_idx is invalid after make_initial_sample! (all zeros or -1)"
    return
end
```

- [ ] **Step 4: Synchronize initial Pfaffian retry status**

Inside each initial Pfaffian retry loop, reduce `calculate_m_all*` status before deciding to regenerate:

```julia
result = calculate_m_all_fcmp!(tmp_ele_idx, qp_start, qp_end, data, state)
result = _comm1_max_status(ctx, result)
if result != 0
    retry_count += 1
    if retry_count <= 3 || retry_count % 10 == 0
        @warn "calculate_m_all_fcmp! failed in comm1 group, retry_count=$retry_count. Regenerating sample..."
    end
    result_init = make_initial_sample!(
        tmp_ele_idx,
        tmp_ele_cfg,
        tmp_ele_num,
        tmp_ele_proj_cnt,
        data,
        state,
        rng,
    )
    result_init = _comm1_max_status(ctx, result_init)
    if result_init != 0
        @error "Failed to regenerate initial sample after Pfaffian calculation failure"
        return
    end
end
```

Apply the same reduced-status rule to real, FSZ, and FSZ-real initial Pfaffian setup.

- [ ] **Step 5: Keep finite-log branches group-consistent**

Because `log_ip_old` is now calculated after `comm1` reduction, this branch is group-consistent:

```julia
if !isfinite(real(log_ip_old)) || !isfinite(imag(log_ip_old))
```

For the real path, use:

```julia
if !isfinite(log_ip_old)
```

After regeneration in these branches, reduce the `calculate_m_all*` status with `_comm1_max_status(ctx, result)` before branching.

- [ ] **Step 6: Run unit tests**

Run from `Julia-mVMC/`:

```bash
julia --project=. MVMCOptimizers.jl/test/runtests.jl
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git -C Julia-mVMC add MVMCOptimizers.jl/src/vmc_sampling.jl
git -C Julia-mVMC commit -m "Synchronize split sampling retry decisions"
```

## Task 6: Keep Main Calculation Explicitly No-Reduce

**Files:**
- Modify: `Julia-mVMC/MVMCOptimizers.jl/src/vmc_main_cal.jl`
- Modify: `Julia-mVMC/MVMCOptimizers.jl/src/green_func_calc.jl`

- [ ] **Step 1: Make main-calculation IP calls explicit**

In `Julia-mVMC/MVMCOptimizers.jl/src/vmc_main_cal.jl`, add `; reduce = :none` to every `calculate_ip_fcmp` and `calculate_ip_real` call. The shape is:

```julia
ip_new = calculate_ip_fcmp(pf_m_new, 1, n_qp_full + 1, data; reduce = :none)
```

For real calls:

```julia
ip_new_real = calculate_ip_real(pf_m_new_real, 1, n_qp_full + 1, data; reduce = :none)
```

In `Julia-mVMC/MVMCOptimizers.jl/src/green_func_calc.jl`, use the same explicit no-reduce keyword for the two full-QP `calculate_ip*` calls. PhysCal still rejects `NSplitSize > 1`, but explicit no-reduce keeps the caller contract clear.

- [ ] **Step 2: Audit no main-calculation comm1 reduction exists**

Run from repository root:

```bash
rg -n "calculate_ip_(fcmp|real)\\([^\\n]*reduce = :comm1" Julia-mVMC/MVMCOptimizers.jl/src/vmc_main_cal.jl Julia-mVMC/MVMCOptimizers.jl/src/green_func_calc.jl
```

Expected: no matches.

- [ ] **Step 3: Run unit tests**

Run from `Julia-mVMC/`:

```bash
julia --project=. MVMCOptimizers.jl/test/runtests.jl
```

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git -C Julia-mVMC add MVMCOptimizers.jl/src/vmc_main_cal.jl MVMCOptimizers.jl/src/green_func_calc.jl
git -C Julia-mVMC commit -m "Keep main calculation IP unreduced"
```

## Task 7: Add MPI Positive And Negative Smokes

**Files:**
- Create: `Julia-mVMC/test/mpi/mpi_nsplit_standard_projection_smoke.jl`
- Modify: `Julia-mVMC/test/mpi/mpi_failure_modes.jl`
- Modify: `Julia-mVMC/test/mpi/run_mpi_smoke.jl`

- [ ] **Step 1: Create the standard-projection worker**

Create `Julia-mVMC/test/mpi/mpi_nsplit_standard_projection_smoke.jl`:

```julia
# mpiexec worker for standard-projection NQPFull > 1 with NSplitSize.
# Usage:
#   julia --project=<workspace> test/mpi/mpi_nsplit_standard_projection_smoke.jl <fixture> <mode> <nsplit> <nstore> <output_dir>
using MVMCOptimizers
using MPI
using Logging

if get(ENV, "JULIA_MVMC_SMOKE_LOG_STDOUT", "0") == "1"
    global_logger(ConsoleLogger(stdout, Logging.Info))
end

length(ARGS) == 5 ||
    error("usage: mpi_nsplit_standard_projection_smoke.jl <fixture> <real|cmp|fsz> <nsplit> <nstore> <output_dir>")

const fixture_name = ARGS[1]
const mode = Symbol(ARGS[2])
const nsplit = parse(Int, ARGS[3])
const nstore = parse(Int, ARGS[4])
const outdir = ARGS[5]
const smoke_nsp_gauss_leg = get(ENV, "JULIA_MVMC_SMOKE_NSPGAUSSLEG", "")

mode in (:real, :cmp, :fsz) || error("mode must be real, cmp, or fsz; got $mode")
nsplit >= 1 || error("nsplit must be >= 1; got $nsplit")
nstore in (0, 1) || error("nstore must be 0 or 1; got $nstore")

const src_inputs = joinpath(@__DIR__, "..", "integration", "reference", fixture_name, "inputs")

function replace_modpara_values!(path::AbstractString, replacements)
    lines = readlines(path)
    open(path, "w") do io
        for line in lines
            stripped = strip(line)
            if isempty(stripped) || startswith(stripped, "-")
                println(io, line)
                continue
            end
            key = first(split(stripped))
            if haskey(replacements, key)
                replacement = replacements[key]
                replacement == "" ? println(io, line) : println(io, rpad(key, 15), replacement)
            else
                println(io, line)
            end
        end
    end
end

function ensure_modpara_value!(path::AbstractString, key::AbstractString, value::AbstractString)
    isempty(value) && return
    text = read(path, String)
    occursin(Regex("(?m)^" * key * "\\s+"), text) && return
    open(path, "a") do io
        println(io, rpad(key, 15), value)
    end
end

function prepare_inputs()
    workdir = mktempdir()
    inputs = joinpath(workdir, "inputs")
    cp(src_inputs, inputs)

    replace_modpara_values!(
        joinpath(inputs, "modpara.def"),
        Dict(
            "NSPGaussLeg" => smoke_nsp_gauss_leg,
            "NSROptItrStep" => "1",
            "NSROptItrSmp" => "1",
            "NVMCWarmUp" => "1",
            "NVMCSample" => "4",
            "NSplitSize" => string(nsplit),
            "NStore" => string(nstore),
            "NSRCG" => "0",
        ),
    )
    ensure_modpara_value!(joinpath(inputs, "modpara.def"), "NSPGaussLeg", smoke_nsp_gauss_leg)
    return joinpath(inputs, "namelist.def")
end

try
    namelist = prepare_inputs()
    result = MVMCOptimizers.run_para_opt_from_namelist(
        namelist;
        nsteps = 1,
        nsmp = 1,
        mode = mode,
        output_dir = outdir,
    )

    label = "nsplit-standard-projection worker: $(fixture_name) nsplit=$(nsplit) nstore=$(nstore)"
    if isempty(result.zvo_first_n)
        @assert isnan(result.final_energy_per_site)
        println("$label non-root rank ok")
    else
        @assert result.status == 0
        @assert length(result.zvo_first_n) == 1
        println("$label root rank ok")
    end
catch err
    @error "mpi_nsplit_standard_projection_smoke worker failed" exception = (err, catch_backtrace())
    if MPI.Initialized() && !MPI.Finalized()
        MPI.Abort(MPI.COMM_WORLD, 1)
    end
    rethrow()
end
```

- [ ] **Step 2: Add OptTrans split failure mode**

In `Julia-mVMC/test/mpi/mpi_failure_modes.jl`, update the usage line and add `nsplit_opttrans` as a mode:

```julia
length(ARGS) == 2 || error("usage: mpi_failure_modes.jl <nsrcg2|nsplit_srcg|nsplit_opttrans> <output_dir>")
```

Add this function:

```julia
function run_nsplit_opttrans_rejection()
    data = MVMCExpertModeParsers.parse_expert_mode_files(fixture)
    data.modpara.nsplit_size = 2
    data.n_qp_opt_trans = 2
    data.opt_trans = ComplexF64[1.0 + 0.0im, 0.5 + 0.0im]
    data.qp_opt_trans = [[0], [0]]

    expect_error_contains(
        () -> MVMCOptimizers.validate_supported_para_opt_data(data),
        ("NSplitSize > 1 with NQPOptTrans > 1", "OptTrans"),
    )
    MPI.Initialized() && error("OptTrans split rejection should happen before MPI.Init()")
    println("failure-mode worker: nsplit_opttrans expected rejection ok")
end
```

Extend dispatch:

```julia
    elseif mode == "nsplit_opttrans"
        run_nsplit_opttrans_rejection()
```

- [ ] **Step 3: Register the MPI matrix in the runner**

In `Julia-mVMC/test/mpi/run_mpi_smoke.jl`, add the worker constant:

```julia
const nsplit_standard_projection_worker = joinpath(@__DIR__, "mpi_nsplit_standard_projection_smoke.jl")
```

Add a runner helper:

```julia
function run_nsplit_standard_projection_case(fixture::AbstractString, mode::AbstractString,
                                             nsplit::Int, nstore::Int, nranks::Int;
                                             nsp_gauss_leg::Union{Nothing,Int} = nothing)
    mpi_dir = mktempdir()
    cmd = `$(mpiexec()) -n $nranks $(Base.julia_cmd()) --project=$project $nsplit_standard_projection_worker $fixture $mode $nsplit $nstore $mpi_dir`
    if nsp_gauss_leg !== nothing
        cmd = addenv(cmd, "JULIA_MVMC_SMOKE_NSPGAUSSLEG" => string(nsp_gauss_leg))
    end
    out = read(
        cmd,
        String,
    )
    assert_files_present(mpi_dir, para_opt_files)
    @test isfile(joinpath(mpi_dir, "zvo_SRinfo.dat"))
    label = "nsplit-standard-projection worker: $fixture nsplit=$nsplit nstore=$nstore"
    @test count("$label root rank ok", out) == 1
    @test count("$label non-root rank ok", out) == nranks - 1
    return (
        zvo = parse_numeric_file(joinpath(mpi_dir, "zvo_out.dat")),
        zqp = parse_numeric_file(joinpath(mpi_dir, "zqp_opt.dat")),
        var = parse_numeric_file(joinpath(mpi_dir, "zvo_var.dat")),
    )
end
```

Add the positive testset:

```julia
@testset "v0.5 NSplitSize standard-projection NQPFull self-consistency" begin
    cases = (
        (fixture = "heisenberg_chain_real", mode = "real"),
        (fixture = "hubbard_tetragonal_momentum_projection_real", mode = "real"),
        (fixture = "heisenberg_chain_cmp", mode = "cmp"),
        (fixture = "heisenberg_chain_fsz", mode = "fsz", nsp_gauss_leg = 8),
    )
    for case in cases
        nsp = haskey(case, :nsp_gauss_leg) ? case.nsp_gauss_leg : nothing
        direct_ref = run_nsplit_standard_projection_case(case.fixture, case.mode, 1, 0, 2; nsp_gauss_leg = nsp)
        store_ref = run_nsplit_standard_projection_case(case.fixture, case.mode, 1, 1, 2; nsp_gauss_leg = nsp)
        direct_split = run_nsplit_standard_projection_case(case.fixture, case.mode, 2, 0, 4; nsp_gauss_leg = nsp)
        store_split = run_nsplit_standard_projection_case(case.fixture, case.mode, 2, 1, 4; nsp_gauss_leg = nsp)

        for (label, result) in (
            ("store_ref", store_ref),
            ("direct_split", direct_split),
            ("store_split", store_split),
        )
            prefix = "$(case.fixture) $label"
            assert_close_vector("$prefix zvo_out", result.zvo, direct_ref.zvo;
                                atol = nsplit_nstore_tol)
            assert_close_vector("$prefix zqp_opt", result.zqp, direct_ref.zqp;
                                atol = nsplit_nstore_tol)
            assert_close_vector("$prefix zvo_var", result.var, direct_ref.var;
                                atol = nsplit_nstore_tol)
        end
    end
end
```

The `heisenberg_chain_fsz` fixture does not declare `NSPGaussLeg`, so the
runner passes `JULIA_MVMC_SMOKE_NSPGAUSSLEG=8` for that case to force a
standard-projection `NQPFull > 1` FSZ path. `zvo_SRinfo.dat` is checked for
presence in the helper. Numeric comparison stays limited to `zvo_out.dat`,
`zqp_opt.dat`, and `zvo_var.dat` because `zvo_SRinfo.dat` is an iteration
diagnostic file and is not currently part of the self-consistency
comparator.

Add the negative testset:

```julia
@testset "v0.5 failure mode: NSplitSize > 1 with OptTrans rejects before MPI.Init" begin
    outdir = mktempdir()
    out = read(`$(mpiexec()) -n 2 $(Base.julia_cmd()) --project=$project $failure_worker nsplit_opttrans $outdir`,
               String)
    @test count("failure-mode worker: nsplit_opttrans expected rejection ok", out) == 2
    @test isempty(readdir(outdir))
end
```

- [ ] **Step 4: Run MPI smoke**

Run from `Julia-mVMC/`:

```bash
JULIA_NUM_THREADS=1 julia --project=. test/mpi/run_mpi_smoke.jl
```

Expected: PASS. The new standard-projection matrix should compare `NSplitSize=1` and `NSplitSize=2` within `1e-8`.

- [ ] **Step 5: Commit**

```bash
git -C Julia-mVMC add test/mpi/mpi_nsplit_standard_projection_smoke.jl test/mpi/mpi_failure_modes.jl test/mpi/run_mpi_smoke.jl
git -C Julia-mVMC commit -m "Add NSplit standard projection MPI smoke"
```

## Task 8: Update Public Documentation

**Files:**
- Modify: `Julia-mVMC/README.md`
- Modify: `Julia-mVMC/MVMCOptimizers.jl/README.md`
- Modify: `Julia-mVMC/docs/manual/03_optimization.md`
- Modify: `Julia-mVMC/docs/manual/04_physics_calc.md`
- Modify: `Julia-mVMC/docs/manual/05_compatibility.md`

- [ ] **Step 1: Find documentation targets**

Run from repository root:

```bash
rg -n "NSplitSize|NQPFull|NQPOptTrans|OptTrans|SR-CG|PhysCal" Julia-mVMC/README.md Julia-mVMC/MVMCOptimizers.jl/README.md Julia-mVMC/docs Julia-mVMC/MVMCOptimizers.jl/docs
```

Expected: list of docs that already describe runtime support.

- [ ] **Step 2: Add the supported/rejected wording**

Use this public wording, without local investigation paths:

```markdown
`NSplitSize > 1` supports ParaOpt direct SR with standard-projection
`NQPFull > 1` when `NQPOptTrans = 1` (`NSPGaussLeg > 1` and/or
`NMPTrans > 1`). OptTrans-derived QP sectors (`NQPOptTrans > 1` or active
`OptTrans`) remain unsupported with `NSplitSize > 1`. SR-CG split and
PhysCal split also remain unsupported.
```

- [ ] **Step 3: Run doc safety checks**

Run from repository root:

```bash
rg -n "($(printf /)Users|Drop""box|Shin-mVMC|Dev_mVMC|docs/(reviews|superpowers))" Julia-mVMC/README.md Julia-mVMC/MVMCOptimizers.jl/README.md Julia-mVMC/docs Julia-mVMC/MVMCOptimizers.jl/docs
```

Expected: no matches.

- [ ] **Step 4: Commit**

```bash
git -C Julia-mVMC add README.md MVMCOptimizers.jl/README.md docs/manual/03_optimization.md docs/manual/04_physics_calc.md docs/manual/05_compatibility.md
git -C Julia-mVMC commit -m "Document NSplit standard projection scope"
```

## Task 9: Final Verification

**Files:**
- No planned source edits in this task.

- [ ] **Step 1: Run package unit tests**

Run from `Julia-mVMC/`:

```bash
julia --project=. MVMCOptimizers.jl/test/runtests.jl
```

Expected: PASS.

- [ ] **Step 2: Run workspace integration tests**

Run from `Julia-mVMC/`:

```bash
julia --project=. test/integration/runtests.jl
```

Expected: PASS.

- [ ] **Step 3: Run MPI smoke**

Run from `Julia-mVMC/`:

```bash
JULIA_NUM_THREADS=1 julia --project=. test/mpi/run_mpi_smoke.jl
```

Expected: PASS.

- [ ] **Step 4: Check public text for local paths**

Run from repository root:

```bash
rg -n "($(printf /)Users|Drop""box|Shin-mVMC|Dev_mVMC|docs/(reviews|superpowers))" Julia-mVMC
```

Expected: no matches in tracked public files. If generated local artifacts are present, inspect them before excluding them from the final diff.

- [ ] **Step 5: Review final diff**

Run from repository root:

```bash
git -C Julia-mVMC status --short
git -C Julia-mVMC diff --stat
git -C Julia-mVMC diff --check
```

Expected: only intended files changed, and `git diff --check` reports no whitespace errors.

- [ ] **Step 6: Commit verification docs if needed**

If Task 8 did not include all docs touched by final wording, commit the remaining documentation-only changes:

```bash
git -C Julia-mVMC add README.md MVMCOptimizers.jl/README.md
git -C Julia-mVMC commit -m "Update NSplit projection support notes"
```

Skip this commit if there are no uncommitted documentation changes.

## Self-Review

- Spec coverage:
  - Runtime scope is Task 1.
  - QP range and scalar reduction helpers are Task 2.
  - IP/logIP reduction mode and empty ranges are Task 3.
  - Sampling QP split and FSZ hard-coded full ranges are Task 4.
  - Initial Pfaffian retry and collective-alignment requirements are Task 5.
  - Main-calculation no-reduce behavior is Task 6.
  - Standard-projection MPI positive matrix and OptTrans negative matrix are Task 7.
  - The `NQPFull == 1 && NSplitSize > 1` regression from design-review Finding 5 remains covered by the existing `mpi_nsplit_nstore_smoke.jl` matrix.
  - Public documentation wording is Task 8.
  - Full verification is Task 9.
- Placeholder scan: no placeholder markers remain.
- Type consistency:
  - `qp_split_range(n_qp_full, ctx)` returns `(qp_start, qp_end)` as 1-based half-open integers.
  - `calculate_ip*` and `calculate_log_ip*` use `ctx::Union{Nothing,ParallelContext}` and `reduce::Symbol`.
  - Sampling entry points keep old positional arguments and add only a keyword `ctx`.
  - Main calculation uses the same IP functions with `reduce = :none`.
