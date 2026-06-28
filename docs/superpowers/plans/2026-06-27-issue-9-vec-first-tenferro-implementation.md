# Issue 9 Vec-First Tenferro Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Resolve GitHub issue #9 by migrating `mvmc-core` toward tenferro through a Vec-first layout normalization, tenferro-backed wrapper internals, dense einsum rewrites, final audit/docs, and one final PR.

**Architecture:** Keep Phase 1 free of tenferro and stabilize wrapper APIs while storage is still `Vec`-backed. Phase 2 swaps wrapper internals to tenferro tensors without broad call-site changes. Phase 3 replaces dense reductions with `tenferro-einsum`; Pfaffian update and sparse/scatter accumulation stay explicit.

**Tech Stack:** Rust 2021, existing `mvmc-core` layout wrappers and parity tests, `tenferro-tensor`, `tenferro-cpu`, `tenferro-einsum`, Julia/C golden fixtures.

---

## Baseline

The implementation worktree is:

```text
/home/shinaoka/.config/superpowers/worktrees/mvmc-rs/issue-9-vec-first-tenferro
```

Baseline commands already passed after initializing submodules:

```bash
git submodule update --init --recursive
cargo build --workspace
cargo test --workspace
```

## File Structure

- Modify `Cargo.toml`: add tenferro workspace dependencies in Phase 2.
- Modify `crates/mvmc-core/Cargo.toml`: add tenferro dependencies in Phase 2; remove direct `ndarray` only if no longer used.
- Modify `crates/mvmc-core/src/state.rs`: own storage wrappers, layout tests, and sample/SR store accessors.
- Modify `crates/mvmc-core/src/slater_update.rs`: remove raw Slater full-slice indexing.
- Modify `crates/mvmc-core/src/sampling/updates.rs`: replace 3D InvM stride math with QP-plane helpers.
- Modify `crates/mvmc-core/src/sampling/driver.rs`: replace sample offsets and InvM full-slice handoff with wrappers/views.
- Modify `crates/mvmc-core/src/observables.rs`: replace storage reach-through and later add dense einsum helpers.
- Modify `crates/mvmc-core/src/run.rs`: replace storage reach-through in initialization/output paths.
- Modify `crates/mvmc-core/src/sr.rs`: integrate tensor-shaped SR store access where needed.
- Modify `docs/PORTING_PLAN.md`: final migration rules in Phase 4.

## Phase 1: Vec-Backed Layout Normalization

### Task 1: Strengthen Layout Wrapper Tests

**Files:**
- Modify: `crates/mvmc-core/src/state.rs`

- [ ] **Step 1: Add explicit Slater layout test**

Add this test in `state.rs` tests next to `slater_elm_flat_row_major_indexing`:

```rust
#[test]
fn slater_elm_vec_layout_preserves_qp_row_major_planes() {
    let mut a = SlaterElmFlat::<f64>::zeros(2, 3);
    a.set(0, 1, 4, 11.0);
    a.set(1, 5, 2, 22.0);

    assert_eq!(a.as_slice()[(0 * 6 + 1) * 6 + 4], 11.0);
    assert_eq!(a.as_slice()[(1 * 6 + 5) * 6 + 2], 22.0);
    assert_eq!(a.qp_slice(0)[1 * 6 + 4], 11.0);
    assert_eq!(a.qp_slice(1)[5 * 6 + 2], 22.0);
    assert_eq!(a.get(0, 1, 4), 11.0);
    assert_eq!(a.get(1, 5, 2), 22.0);
}
```

- [ ] **Step 2: Add explicit InvM layout test**

Add this test in `state.rs` tests next to `inv_m_column_major_indexing_with_pad`:

```rust
#[test]
fn inv_m_vec_layout_preserves_qp_matrix_layout_and_pad_slot() {
    let mut a = InvMColMajor::<f64>::zeros(2, 3);
    a.set(0, 1, 4, 11.0);
    a.set(1, 5, 2, 22.0);
    a.set_pad_slot(0, 7.0);
    a.set_pad_slot(1, 8.0);

    let n_size = 6;
    assert_eq!(a.qp_matrix_slice(0)[1 + 4 * n_size], 11.0);
    assert_eq!(a.qp_matrix_slice(1)[5 + 2 * n_size], 22.0);
    assert_eq!(a.pad_slot(0), 7.0);
    assert_eq!(a.pad_slot(1), 8.0);
    assert_eq!(a.qp_matrix_slice(0).len(), n_size * n_size);
    assert_eq!(a.qp_matrix_slice(1).len(), n_size * n_size);
}
```

- [ ] **Step 3: Run focused tests**

```bash
cargo test -p mvmc-core state::tests::slater_elm_vec_layout_preserves_qp_row_major_planes
cargo test -p mvmc-core state::tests::inv_m_vec_layout_preserves_qp_matrix_layout_and_pad_slot
```

Expected: both pass because the current Vec layouts already satisfy these contracts.

- [ ] **Step 4: Commit**

```bash
git add crates/mvmc-core/src/state.rs
git commit -m "Add explicit Vec layout tests for mvmc state"
```

### Task 2: Add Vec-Backed Plane And Sample Accessors

**Files:**
- Modify: `crates/mvmc-core/src/state.rs`

- [ ] **Step 1: Add immutable/mutable InvM plane helper types**

Add these types below `InvMColMajor<T>`:

```rust
/// Immutable view of one column-major `InvMColMajor` QP matrix plane.
#[derive(Debug, Clone, Copy)]
pub struct InvMPlane<'a, T> {
    data: &'a [T],
    n: usize,
}

impl<T: Copy> InvMPlane<'_, T> {
    #[inline]
    pub fn get(&self, row: usize, col: usize) -> T {
        self.data[row + col * self.n]
    }

    pub fn as_slice(&self) -> &[T] {
        self.data
    }
}

/// Mutable view of one column-major `InvMColMajor` QP matrix plane.
#[derive(Debug)]
pub struct InvMPlaneMut<'a, T> {
    data: &'a mut [T],
    n: usize,
}

impl<T: Copy> InvMPlaneMut<'_, T> {
    #[inline]
    pub fn get(&self, row: usize, col: usize) -> T {
        self.data[row + col * self.n]
    }

    #[inline]
    pub fn set(&mut self, row: usize, col: usize, value: T) {
        self.data[row + col * self.n] = value;
    }

    pub fn as_slice(&self) -> &[T] {
        self.data
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self.data
    }
}
```

- [ ] **Step 2: Add plane constructors to `InvMColMajor<T>`**

Add these methods to `impl<T: Copy> InvMColMajor<T>`:

```rust
pub fn qp_matrix(&self, qp: usize) -> InvMPlane<'_, T> {
    InvMPlane {
        data: self.qp_matrix_slice(qp),
        n: self.n_size,
    }
}

pub fn qp_matrix_mut(&mut self, qp: usize) -> InvMPlaneMut<'_, T> {
    let n = self.n_size;
    InvMPlaneMut {
        data: self.qp_matrix_slice_mut(qp),
        n,
    }
}
```

- [ ] **Step 3: Add `ElectronConfiguration` sample accessors**

Add sample slice accessors for `ele_idx`, `ele_cfg`, `ele_num`, `ele_proj_cnt`, and `ele_spn`. Use existing struct metadata:

```rust
pub fn ele_idx_sample(&self, sample: usize) -> &[i64] {
    let start = sample * self.n_size;
    &self.ele_idx[start..start + self.n_size]
}

pub fn ele_idx_sample_mut(&mut self, sample: usize) -> &mut [i64] {
    let start = sample * self.n_size;
    &mut self.ele_idx[start..start + self.n_size]
}

pub fn ele_cfg_sample(&self, sample: usize) -> &[i64] {
    let start = sample * self.n_site2;
    &self.ele_cfg[start..start + self.n_site2]
}

pub fn ele_cfg_sample_mut(&mut self, sample: usize) -> &mut [i64] {
    let start = sample * self.n_site2;
    &mut self.ele_cfg[start..start + self.n_site2]
}

pub fn ele_num_sample(&self, sample: usize) -> &[i64] {
    let start = sample * self.n_site2;
    &self.ele_num[start..start + self.n_site2]
}

pub fn ele_num_sample_mut(&mut self, sample: usize) -> &mut [i64] {
    let start = sample * self.n_site2;
    &mut self.ele_num[start..start + self.n_site2]
}

pub fn ele_proj_cnt_sample(&self, sample: usize) -> &[i64] {
    let start = sample * self.n_proj;
    &self.ele_proj_cnt[start..start + self.n_proj]
}

pub fn ele_proj_cnt_sample_mut(&mut self, sample: usize) -> &mut [i64] {
    let start = sample * self.n_proj;
    &mut self.ele_proj_cnt[start..start + self.n_proj]
}

pub fn ele_spn_sample(&self, sample: usize) -> &[i64] {
    let start = sample * self.n_size;
    &self.ele_spn[start..start + self.n_size]
}

pub fn ele_spn_sample_mut(&mut self, sample: usize) -> &mut [i64] {
    let start = sample * self.n_size;
    &mut self.ele_spn[start..start + self.n_size]
}
```

- [ ] **Step 4: Add SR sample store accessors**

Add these methods to `SROptData`:

```rust
pub fn sr_opt_o_store_sample(&self, sample: usize) -> &[Complex64] {
    let n = 2 * self.sr_opt_size;
    let start = sample * n;
    &self.sr_opt_o_store[start..start + n]
}

pub fn sr_opt_o_store_sample_mut(&mut self, sample: usize) -> &mut [Complex64] {
    let n = 2 * self.sr_opt_size;
    let start = sample * n;
    &mut self.sr_opt_o_store[start..start + n]
}

pub fn sr_opt_o_store_real_sample(&self, sample: usize) -> &[f64] {
    let n = self.sr_opt_size;
    let start = sample * n;
    &self.sr_opt_o_store_real[start..start + n]
}

pub fn sr_opt_o_store_real_sample_mut(&mut self, sample: usize) -> &mut [f64] {
    let n = self.sr_opt_size;
    let start = sample * n;
    &mut self.sr_opt_o_store_real[start..start + n]
}
```

- [ ] **Step 5: Add accessor tests**

Add tests in `state.rs` that write through sample/store accessors and assert the underlying flat order:

```rust
#[test]
fn electron_config_sample_accessors_preserve_legacy_flat_order() {
    let mut cfg = ElectronConfiguration::zeros(2, 3, 4, 5, false);
    cfg.ele_idx_sample_mut(1)[2] = 12;
    cfg.ele_cfg_sample_mut(1)[4] = 14;
    cfg.ele_num_sample_mut(1)[5] = 15;
    cfg.ele_proj_cnt_sample_mut(1)[3] = 13;

    assert_eq!(cfg.ele_idx[1 * cfg.n_size + 2], 12);
    assert_eq!(cfg.ele_cfg[1 * cfg.n_site2 + 4], 14);
    assert_eq!(cfg.ele_num[1 * cfg.n_site2 + 5], 15);
    assert_eq!(cfg.ele_proj_cnt[1 * cfg.n_proj + 3], 13);
}

#[test]
fn sr_store_sample_accessors_preserve_component_major_order() {
    let mut sr = SROptData::zeros(3, 2, false);
    sr.sr_opt_o_store_sample_mut(1)[4] = Complex64::new(4.0, -1.0);
    sr.sr_opt_o_store_real_sample_mut(1)[2] = 8.0;

    assert_eq!(sr.sr_opt_o_store[1 * (2 * sr.sr_opt_size) + 4], Complex64::new(4.0, -1.0));
    assert_eq!(sr.sr_opt_o_store_real[1 * sr.sr_opt_size + 2], 8.0);
}
```

- [ ] **Step 6: Run focused tests**

```bash
cargo test -p mvmc-core state::tests::electron_config_sample_accessors_preserve_legacy_flat_order
cargo test -p mvmc-core state::tests::sr_store_sample_accessors_preserve_component_major_order
```

Expected: pass.

- [ ] **Step 7: Commit**

```bash
git add crates/mvmc-core/src/state.rs
git commit -m "Add Vec-backed state accessors"
```

### Task 3: Remove Raw Slater/InvM/Sample Storage Reach-Through

**Files:**
- Modify: `crates/mvmc-core/src/slater_update.rs`
- Modify: `crates/mvmc-core/src/sampling/updates.rs`
- Modify: `crates/mvmc-core/src/sampling/driver.rs`
- Modify: `crates/mvmc-core/src/observables.rs`
- Modify: `crates/mvmc-core/src/run.rs`

- [ ] **Step 1: Run pre-audit**

```bash
rg "as_slice\(\)\[|as_mut_slice\(\)\[|inv_base \+|qp_offset \+|sample \*|\* n_size \+|\* n_site2 \+" crates/mvmc-core/src
```

Expected: report current raw reach-through locations.

- [ ] **Step 2: Replace Slater writes and real-copy loops**

In `slater_update.rs` and `sampling/driver.rs`, replace `slater_elm.as_mut_slice()[...]` writes with `set(qp, row, col, value)`. Replace full real-copy by looping over `qp`, `row`, and `col` and using `get` / `set`.

- [ ] **Step 3: Replace InvM reads in ratio/update code**

In `sampling/updates.rs`, change functions that accept `inv_m_flat: &[T]` plus `inv_base` to accept one QP matrix slice or `InvMPlane<'_, T>`. Keep 2D column-major math local as `row + col * n_size`. Do not preserve `qp * (n_size * n_size + 1)` in the call sites.

- [ ] **Step 4: Replace sample history offsets in sampling drivers**

In `sampling/driver.rs`, replace blocks like:

```rust
let off_idx = sample * n_size;
let ele_idx = &mut state.electron_config.ele_idx[off_idx..off_idx + n_size];
```

with accessor calls:

```rust
let ele_idx = state.electron_config.ele_idx_sample_mut(sample);
```

Use equivalent accessors for config, counts, projection counts, and spin history.

- [ ] **Step 5: Replace SR store writes in observables**

In `observables.rs`, replace:

```rust
sr_opt_o_store[i + sample * size_2] = sr_opt_o[i] * sqrtw;
```

with a sample slice:

```rust
let store_sample = sr_data.sr_opt_o_store_sample_mut(sample);
store_sample[i] = sr_opt_o[i] * sqrtw;
```

Use the actual local variable names in the current function.

- [ ] **Step 6: Run post-audit and focused tests**

```bash
rg "as_slice\(\)\[|as_mut_slice\(\)\[|inv_base \+|qp_offset \+|sample \*|\* n_size \+|\* n_site2 \+" crates/mvmc-core/src
cargo test -p mvmc-core calc_m_all_vs_julia
cargo test -p mvmc-core candidate_vs_julia one_move_vs_julia metropolis_vs_julia pf_update_vs_julia
cargo test -p mvmc-core run_smoke phase4_zvo_gate phase4_zvo_gate_cmp phase4_zvo_gate_fsz phase5_zvo_gate_hubbard
```

Expected: tests pass. Remaining audit hits must be local 2D matrix helpers, config/index formulas, or documented exceptions.

- [ ] **Step 7: Commit**

```bash
git add crates/mvmc-core/src/slater_update.rs crates/mvmc-core/src/sampling/updates.rs crates/mvmc-core/src/sampling/driver.rs crates/mvmc-core/src/observables.rs crates/mvmc-core/src/run.rs crates/mvmc-core/src/state.rs
git commit -m "Remove raw multidimensional storage reach-through"
```

## Phase 2: Tenferro Storage Swap

### Task 4: Add Tenferro Dependencies

**Files:**
- Modify: `Cargo.toml`
- Modify: `crates/mvmc-core/Cargo.toml`
- Modify: `Cargo.lock`

- [ ] **Step 1: Add workspace dependencies**

Add these dependencies to `[workspace.dependencies]`:

```toml
tenferro-tensor = { git = "https://github.com/tensor4all/tenferro-rs.git" }
tenferro-cpu = { git = "https://github.com/tensor4all/tenferro-rs.git", default-features = false, features = ["cpu-faer"] }
tenferro-einsum = { git = "https://github.com/tensor4all/tenferro-rs.git", default-features = false, features = ["cpu-faer"] }
```

- [ ] **Step 2: Add `mvmc-core` dependencies**

Add to `crates/mvmc-core/Cargo.toml`:

```toml
tenferro-tensor = { workspace = true }
tenferro-cpu = { workspace = true }
tenferro-einsum = { workspace = true }
```

Keep `ndarray` until `cargo tree -p mvmc-core -i ndarray` proves it can be removed.

- [ ] **Step 3: Resolve lockfile**

```bash
cargo check -p mvmc-core
```

Expected: dependencies resolve and `mvmc-core` still checks.

- [ ] **Step 4: Commit**

```bash
git add Cargo.toml crates/mvmc-core/Cargo.toml Cargo.lock
git commit -m "Add tenferro dependencies"
```

### Task 5: Replace Wrapper Internals With Tenferro Tensors

**Files:**
- Modify: `crates/mvmc-core/src/state.rs`

- [ ] **Step 1: Change `SlaterElmFlat<T>` internals**

Replace `data: Vec<T>` with:

```rust
data: tenferro_tensor::TypedTensor<T>,
```

Construct with:

```rust
let data = tenferro_tensor::TypedTensor::<T>::from_vec_col_major(
    vec![n_site2, n_site2, n_qp_full],
    vec![T::default(); n_qp_full * n_site2 * n_site2],
)
.expect("SlaterElmFlat shape and data length must match");
```

Keep public indexing order unchanged. Because tenferro is column-major with axes `[col, row, qp]`, implement:

```rust
pub fn get(&self, qp: usize, row: usize, col: usize) -> T {
    *self.data.get(&[col, row, qp]).expect("SlaterElmFlat index in bounds")
}

pub fn set(&mut self, qp: usize, row: usize, col: usize, value: T) {
    *self
        .data
        .get_mut(&[col, row, qp])
        .expect("SlaterElmFlat index in bounds") = value;
}
```

Implement `as_slice`, `as_mut_slice`, `qp_slice`, and `qp_slice_mut` using `host_data()` / `host_data_mut()` and the same legacy flat ranges.

- [ ] **Step 2: Change `InvMColMajor<T>` internals**

Use compact matrix tensor plus explicit pad vector:

```rust
matrix: tenferro_tensor::TypedTensor<T>,
pad: tenferro_tensor::TypedTensor<T>,
```

Construct with:

```rust
let matrix = tenferro_tensor::TypedTensor::<T>::from_vec_col_major(
    vec![n_size, n_size, n_qp_full],
    vec![T::default(); n_qp_full * n_size * n_size],
)
.expect("InvM matrix shape and data length must match");
let pad = tenferro_tensor::TypedTensor::<T>::from_vec_col_major(
    vec![n_qp_full],
    vec![T::default(); n_qp_full],
)
.expect("InvM pad shape and data length must match");
```

Implement `get(qp, row, col)` as `matrix.get(&[row, col, qp])`. Implement `qp_matrix_slice(qp)` from the compact matrix host data with stride `n_size * n_size`.

If any current caller still needs whole padded storage, replace that caller with plane/pad APIs instead of recreating padded storage.

- [ ] **Step 3: Keep sample/SR stores Vec-backed in this PR slice**

Do not force sample history and SR stores into tenferro in this task unless the Phase 1 accessor API makes the change mechanical and all tests stay green. The issue #9 requirement is to migrate toward tenferro safely; wrapper internals for Slater and InvM are the required Phase 2 storage swap.

- [ ] **Step 4: Run layout and parity tests**

```bash
cargo test -p mvmc-core state::tests::slater_elm_vec_layout_preserves_qp_row_major_planes
cargo test -p mvmc-core state::tests::inv_m_vec_layout_preserves_qp_matrix_layout_and_pad_slot
cargo test -p mvmc-core calc_m_all_vs_julia pf_update_vs_julia
cargo test --workspace
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add crates/mvmc-core/src/state.rs
git commit -m "Back core layout wrappers with tenferro tensors"
```

## Phase 3: Dense Einsum Rewrites

### Task 6: Add SR Gram Einsum Helper

**Files:**
- Modify: `crates/mvmc-core/src/observables.rs`

- [ ] **Step 1: Add helper**

Use the concrete typed API:

```rust
fn sr_store_gram_einsum(
    backend: &mut tenferro_cpu::CpuBackend,
    store: &tenferro_tensor::TypedTensor<Complex64>,
) -> tenferro_tensor::Result<tenferro_tensor::TypedTensor<Complex64>> {
    use tenferro_einsum::TypedTensorEinsumExt;

    let shape = store.shape();
    assert_eq!(shape.len(), 2, "SR store must be [component, sample]");
    let raw = store.host_data()?;
    let mut conj_data = Vec::with_capacity(raw.len());
    conj_data.extend(raw.iter().map(|z| z.conj()));
    let conj = tenferro_tensor::TypedTensor::<Complex64>::from_vec_col_major(
        shape.to_vec(),
        conj_data,
    )?;
    [&store, &conj].einsum("is,js->ij", backend)
}
```

- [ ] **Step 2: Add manual-reference helper test**

Add a unit test with a `[2, 3]` `TypedTensor<Complex64>` and compare the result against a manual loop using column-major indexing.

- [ ] **Step 3: Use helper in SR finalization only if data is already tensor-shaped**

If the current SR store remains `Vec`, add a small conversion at the explicit boundary:

```rust
let store_tensor = tenferro_tensor::TypedTensor::<Complex64>::from_vec_col_major(
    vec![size_2, sample_size],
    sr_opt_o_store.to_vec(),
)
.expect("SR store shape must match");
```

Keep this conversion outside hot inner loops. If the existing loop is performance-critical, leave loop replacement to a follow-up only if a tensor-backed SR store exists; still commit the helper and tests.

- [ ] **Step 4: Verify**

```bash
cargo test -p mvmc-core observables::tests::sr_store_gram_einsum_matches_manual_complex_reference
cargo test -p mvmc-core sr::tests::cholesky_solve_matches_direct_inverse
cargo test -p mvmc-core phase5_regression_50step
```

- [ ] **Step 5: Commit**

```bash
git add crates/mvmc-core/src/observables.rs
git commit -m "Add tenferro einsum helper for SR Gram contractions"
```

### Task 7: Add QP Weighted Reduction Einsum Helper

**Files:**
- Modify: `crates/mvmc-core/src/observables.rs`

- [ ] **Step 1: Add helper**

```rust
fn qp_weighted_orbital_sum_einsum(
    backend: &mut tenferro_cpu::CpuBackend,
    weights: &tenferro_tensor::TypedTensor<Complex64>,
    buffer: &tenferro_tensor::TypedTensor<Complex64>,
) -> tenferro_tensor::Result<tenferro_tensor::TypedTensor<Complex64>> {
    use tenferro_einsum::TypedTensorEinsumExt;

    [&buffer, &weights].einsum("oq,q->o", backend)
}
```

- [ ] **Step 2: Add manual-reference test**

Add a test that constructs a `[2, 3]` buffer and `[3]` weights, then compares the einsum output to a manual column-major reference loop.

- [ ] **Step 3: Replace dense reduction only at explicit tensor boundary**

If the current reduction buffer is still `Vec`, convert it to a `TypedTensor<Complex64>` at the boundary and keep scatter accumulation explicit. Do not move sparse/scatter accumulation to einsum.

- [ ] **Step 4: Verify**

```bash
cargo test -p mvmc-core observables::tests::qp_weighted_orbital_sum_einsum_matches_manual_complex_reference
cargo test -p mvmc-core projection_vs_julia rbm_vs_julia
```

- [ ] **Step 5: Commit**

```bash
git add crates/mvmc-core/src/observables.rs
git commit -m "Add tenferro einsum helper for QP weighted reductions"
```

## Phase 4: Final Audit, Docs, And Single PR

### Task 8: Audit And Document Migration Rules

**Files:**
- Modify: `docs/PORTING_PLAN.md`

- [ ] **Step 1: Run audit**

```bash
rg "\[[^\]]+\*[^\]]+\]|\+ .*\* n_|\* n_size \+|\* n_site2 \+|sample \*|qp \*" crates/mvmc-core/src
```

Classify remaining matches in the PR body as one of:

- 1D domain offset,
- local 2D matrix-plane helper,
- parser/config/index formula,
- sparse/scatter accumulation,
- bug fixed before PR.

- [ ] **Step 2: Add docs section**

Add this section to `docs/PORTING_PLAN.md`:

```markdown
## Tenferro Tensor Migration Rules

- Layout-sensitive state access goes through wrapper APIs before storage changes.
- 3D+ state arrays use layout wrappers backed by tenferro tensors or explicit tensor views once the Vec-backed parity gate has passed.
- Dense contractions use `tenferro-einsum` where tensor-shaped data exists.
- Pfaffian local update kernels may keep explicit 2D column-major matrix indexing inside small helper/view types because they operate on one QP matrix plane at a time and must preserve Julia/PfaPack operation order.
- Sparse/scatter-style orbital accumulation remains explicit until tenferro exposes a suitable sparse/scatter abstraction.
- Layout conversions must be explicit and covered by parity tests.
```

- [ ] **Step 3: Run final verification**

```bash
cargo fmt --all --check
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo test -p mvmc-core phase5_regression_50step
```

Run the benchmark only if the implementation makes a performance claim:

```bash
cargo run -p xtask -- bench-julia --steps 50 --reps 3 --warmups 1 --threads 1
```

- [ ] **Step 4: Commit docs/audit changes**

```bash
git add docs/PORTING_PLAN.md
git commit -m "Document tenferro migration boundaries"
```

- [ ] **Step 5: Open one PR**

Push only `issue-9-vec-first-tenferro` and open one draft PR that closes #9. The PR body must include:

- summary of Vec-first phase,
- summary of tenferro dependency/revision,
- audit classification,
- verification commands and outcomes,
- `Closes #9`.

## Self-Review

- Spec coverage: Phase 1 covers Vec layout normalization, call-site cleanup, sample/SR accessors, and parity gate. Phase 2 covers tenferro wrapper internals. Phase 3 covers dense einsum helpers. Phase 4 covers final audit/docs and single PR.
- Placeholder scan: no incomplete implementation markers remain.
- Type consistency: tenferro snippets use current public APIs verified from `tensor4all/tenferro-rs`: `TypedTensor::<T>::from_vec_col_major`, `host_data`, `host_data_mut`, `get`, `get_mut`, `shape`, `tenferro_cpu::CpuBackend`, and `tenferro_einsum::TypedTensorEinsumExt`.
