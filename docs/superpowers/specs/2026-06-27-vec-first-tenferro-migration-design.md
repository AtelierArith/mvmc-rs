# Vec-First Tenferro Migration Design

## Goal

Reduce layout bugs while migrating `mvmc-core` storage and dense reductions
toward tenferro. The migration must preserve the existing Julia/C parity gates
and keep numerical operation order explicit where that order matters.

## Core Decision

Use a staged migration:

1. Normalize the existing `Vec` / slice-backed layouts first.
2. Replace wrapper internals with tenferro tensors only after the `Vec` layout
   boundaries are stable and parity-tested.
3. Introduce `tenferro-einsum` only for dense contractions after storage
   wrappers already expose clear tensor-shaped data.

This separates two high-risk changes that should not land together:

- changing physical layout and call-site access patterns,
- changing the backing storage library.

## Architecture

### Phase 1: Vec-Backed Layout Normalization

No tenferro dependency is introduced in this phase.

`SlaterElmFlat`, `InvMColMajor`, sample history stores, and SR sample stores
remain backed by `Vec<T>`. Their APIs become stricter: call sites use domain
accessors, QP plane views, or sample views instead of full-slice indexing and
duplicated stride math.

`InvMColMajor` should expose matrix planes and pad slots as separate concepts.
The matrix plane remains contiguous and column-major for each QP because
`pfapack::SqMat` depends on that boundary. Pad slots stay explicit and must not
leak into ordinary matrix access.

### Phase 2: Tenferro Storage Swap

After Phase 1 passes layout and parity tests, replace wrapper internals with
tenferro tensors while preserving the Phase 1 public API. Most call sites should
not change in this phase. Failures in this phase should point to tenferro
storage, host-data access, or layout mapping rather than domain algorithm
changes.

### Phase 3: Tenferro Einsum For Dense Reductions

Use `tenferro-einsum` for dense contraction code such as SR Gram construction
and QP weighted reductions. Pfaffian local update kernels and sparse/scatter
orbital accumulation stay as explicit loops unless tenferro provides a suitable
sparse/scatter abstraction and parity remains defensible.

## Components

### Layout Wrappers

These types are the only places that know physical storage layout:

- `SlaterElmFlat`
- `InvMColMajor`
- sample history store wrappers
- SR sample store wrappers

Phase 1 keeps them `Vec`-backed. Phase 2 changes their internals to tenferro.

### Plane And Sample Views

Add thin view APIs that remove multidimensional offset arithmetic from callers:

- Slater QP plane access for row-major Slater tables.
- InvM QP matrix plane access for column-major matrices.
- Mutable InvM plane helpers when update kernels need repeated reads/writes.
- Sample history accessors such as `ele_idx_sample_mut(sample)`.
- SR store accessors using `[component, sample]` semantics.

Local 2D matrix indexing is allowed inside small helpers or views when the
helper name documents the column-major semantics.

### Kernel Call Sites

`pfaffian.rs`, `sampling/updates.rs`, `sampling/driver.rs`, and
`observables.rs` should consume wrappers/views instead of calculating 3D
offsets directly. `pfaffian.rs` continues to receive contiguous QP matrix
slices for `pfapack::SqMat`.

### Dense Contraction Helpers

Place small tenferro-einsum helpers near their domain users in
`observables.rs` or `sr.rs`. Each helper should have a focused unit test against
a manual complex reference.

## Data Flow

1. State allocation constructs layout wrappers.
2. Sampling and Pfaffian code obtain QP plane views from wrappers.
3. Observables and SR code obtain sample or component/sample views.
4. Dense reductions call tenferro-einsum helpers only after tensor-shaped data
   is already available.
5. Output buffers keep the existing layout expected by parity tests.

## Testing Strategy

### Phase 1 Tests

Add layout tests that prove the `Vec` wrappers still match legacy storage:

- `SlaterElmFlat` row-major legacy order.
- `InvMColMajor` contiguous column-major matrix planes plus explicit pad slots.
- Sample history `[local_index, sample]` view access.
- SR store `[component, sample]` view access.

Run focused parity gates after call-site cleanup:

```bash
cargo test -p mvmc-core calc_m_all_vs_julia
cargo test -p mvmc-core candidate_vs_julia one_move_vs_julia metropolis_vs_julia pf_update_vs_julia
cargo test -p mvmc-core run_smoke phase4_zvo_gate phase4_zvo_gate_cmp phase4_zvo_gate_fsz phase5_zvo_gate_hubbard
```

### Phase 2 Tests

Re-run the same layout tests and focused parity gates after swapping internals
to tenferro tensors:

```bash
cargo test -p mvmc-core calc_m_all_vs_julia pf_update_vs_julia
cargo test --workspace
```

### Phase 3 Tests

Add small manual-reference tests for each einsum helper:

- SR Gram: `is,js->ij` with conjugation on the second operand.
- QP weighted reductions: `oq,q->o`.

Then run:

```bash
cargo test -p mvmc-core phase5_regression_50step
```

The phase 5 test may skip only under the repository's existing behavior when
the Julia-mVMC checkout is absent.

### Final Gates

Finish the migration with:

```bash
cargo fmt --all --check
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo test -p mvmc-core phase5_regression_50step
```

For performance claims, run and record:

```bash
cargo run -p xtask -- bench-julia --steps 50 --reps 3 --warmups 1 --threads 1
```

## GitHub Issue Restructure

Reuse the existing task issue numbers instead of creating a second parallel
set:

- `#9`: umbrella updated to the vec-first staged migration.
- `#10`: normalize `Vec`-backed layout wrappers.
- `#11`: remove raw full-slice access from Slater/InvM call sites.
- `#12`: wrap sample history and SR stores while keeping `Vec` backing.
- `#13`: run the Vec-backed layout parity gate.
- `#14`: swap wrapper internals to tenferro tensors.
- `#15`: use `tenferro-einsum` for SR Gram contractions.
- `#16`: use `tenferro-einsum` for QP weighted reductions.
- `#17`: final indexing audit and documentation.

Issue bodies should make clear which phase they belong to and which tests are
required before moving to the next phase.

## Non-Goals

- Do not introduce tenferro in Phase 1.
- Do not change numerical algorithms while normalizing layout access.
- Do not move sparse/scatter-style accumulation to dense einsum just to use
  tenferro.
- Do not make performance claims without benchmark evidence.

## Risks And Mitigations

- **Layout drift:** covered by explicit wrapper layout tests and existing Julia
  parity tests.
- **Hidden raw access remains:** use `rg` audits for `as_slice()`, `as_mut_slice()`,
  `sample *`, `qp *`, `inv_base`, and `qp_offset` patterns.
- **PfaPack boundary regression:** keep `SqMat` construction on named QP matrix
  slices.
- **Tenferro host layout mismatch:** Phase 2 preserves the Phase 1 API and
  reruns the same tests, isolating the failure source.
- **Einsum conjugation/order mistakes:** use manual complex reference tests with
  non-zero imaginary values.
