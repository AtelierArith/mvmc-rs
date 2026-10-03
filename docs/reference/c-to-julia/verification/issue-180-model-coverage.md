# Issue #180 model coverage

This milestone is **incomplete**. An input inventory, accepted orbital reader
section, or successful one-step execution does not establish deterministic
parity or a long ctest summary. Missing evidence remains unverified.
Lanczos/InterAll, runner changes, MPI, and the portable numerical policy (#190)
are owned separately. No commits were made for this work.

Audit date: 2026-10-03. Shared Rust checkout initially
`30d8d69ffc6d2700c56fda81841827a49e72e57c` with pre-existing changes; Julia
reference checkout `8bb1b9e8ae47b1512c00b321be05664ddcac0fd1`. Native Linux
x86_64, Rust 1.99.0 (`b940084d7`, LLVM 23.1.1), system OpenBLAS package 0.3.26.
Package metadata alone does not establish the backend linked by every test.
Julia 1.13.1 is not installed here; no new Julia verification is claimed.

The following counts and seeds come from each pinned ctest input, not from
example inputs. AP/P acceptance below is an actual standalone native C reader
result; it does not establish acceptance of all other input sections. See
[probe provenance](../../../../c_toolbox/ctest_orbital_contracts.md).

| Model | Mode | Steps/window | Seed | Native C orbital sections | Independent prefix evidence before this work | First missing coverage |
| --- | --- | --- | --- | --- | --- | --- |
| heisenberg_chain_real | real | 1000/100 | 1 | AP accepted | Existing 1/2/3/50 SR fixtures | Native long summary in this session |
| hubbard_chain_real | real | 500/50 | 1 | AP accepted | Existing 1–10/50 SR fixtures | Native long summary in this session |
| heisenberg_chain_cmp | cmp | 1000/100 | 1 | AP accepted | Existing 1/2/3/50 SR fixtures | Native long summary in this session |
| heisenberg_chain_fsz | fsz | 5000/500 | 1 | AP/P accepted | Existing 1/2/3/50 SR fixtures; label historical/mixed-C provenance | Native long summary and current numerical policy |
| hubbard_chain_cmp | cmp | 500/50 | 1 | AP accepted | No canonical model prefix fixture located | Independent configs/RNG/SR/output prefix oracle |
| hubbard_chain_fsz | fsz | 200/20 | 1 | AP/P accepted | No canonical model prefix fixture located | Independent FSZ configs/spins/RNG/SR/output oracle |
| kondo_chain_real | real | 200/20 | 1 | AP accepted | PhysCal evidence belongs to #181, not this optimization gate | Optimization local-spin trajectory and SR oracle |
| kondo_chain_cmp | cmp | 200/20 | 1 | AP accepted | No canonical model prefix fixture located | Complex local-spin trajectory and SR oracle |
| kondo_chain_stot1_cmp | cmp | 200/20 | 123456789 | AP accepted | No canonical model prefix fixture located | Nonzero total-spin projection trajectory/SR oracle |
| general_rbm_cmp | cmp | 1500/100 | 12395 | AP accepted | Canonical mixed C-counter/Julia SR 1/2/3/50 fixtures already exist | Current native discrete gate and long summary |
| hubbard_tetragonal_real | real | 200/20 | 1 | AP accepted | No canonical model prefix fixture located | Lattice-specific trajectory/SR oracle |
| hubbard_tetragonal_momentum_projection_real | real | 200/20 | 1 | AP accepted | No canonical model prefix fixture located | Momentum-sector weights/signs and trajectory/SR oracle |
| kondo_chain_fsz | fsz | 200/20 | 1 | AP/P accepted | No canonical model prefix fixture located | FSZ local-spin trajectory/SR oracle |

“Missing oracle” does not mean an algorithm is absent. In particular, the
samplers already consume local-spin mappings, and Hamiltonian evaluation
already supports Exchange/Hund/CoulombInter. The old blanket claim that Kondo
Hamiltonian parity is unimplemented requires a first-divergence investigation;
it is not a diagnosis. Never repair these coverage gaps by enabling flag pairs,
filling incomplete C inputs implicitly, or recycling another model's goldens.
The historical malformed inputs and explicit replacements documented in
`tests/fixtures/c_orbital_inputs/README.md` remain untouched.

## Gates and interpretation

The ctest long gate is ignored by default and uses `support::require_gate`.
It preserves the input's seed and optimization/window counts. Its final two
summary columns use the upstream failure rule: a difference fails when it is
both at least three reference standard deviations and at least `1e-8`.
Nonfinite summaries/references and negative standard deviations fail. This
statistical gate cannot substitute for exact trajectory comparisons.

The new `ctest_model_prefixes` gate independently restarts the canonical
GeneralRBM input at seed 12395 for prefixes 1, 2, 3, and 50, for direct stored
SR and CG. It checks every saved configuration, burn-in state, counters, and
the next full 624-word SFMT block exactly, before any numerical comparison.
It reads only existing checked-in fixtures. It does not assert new energy,
parameter, SR-buffer or output coverage: those remain in the existing
`canonical_general_rbm_complex_reference_uses_native_c_counter_order` runner
gate and must be verified under #190's numerical policy. These fixtures are
mixed references, not full C executable validation.

```sh
cargo nextest run -p mvmc-core --cargo-profile test-fast \
  --test ctest_equivalent --run-ignored only \
  -E 'test(inventory_all_thirteen_ctest_input_contracts)' --success-output immediate
cargo nextest run -p mvmc-core --cargo-profile test-fast \
  --test ctest_equivalent --run-ignored only \
  -E 'test(audit_thirteen_ctest_one_step_execution_paths)' --success-output immediate
MVMC_RS_CTEST_PREFIXES=1 cargo nextest run -p mvmc-core \
  --cargo-profile test-fast --test ctest_model_prefixes --run-ignored only \
  --no-fail-fast --retries 0 --success-output immediate
MVMC_RS_CTEST_MODELS=heisenberg_chain_real cargo nextest run -p mvmc-core \
  --cargo-profile test-fast --test ctest_equivalent --run-ignored only \
  -E 'test(rust_ctest_equivalent_selected_models)' --success-output immediate
```

The inventory and execution-audit commands are diagnostics, not #180 parity
execution. Explicit invocation of the long or prefix gate with absent, empty,
or `skip` selection fails; normal runs show the gate as ignored.

## Actual execution record

- Executed: standalone native C AP/P section audit, all 13 AP and three P
  sections accepted after correcting probe storage.
- Executed: `uv run --no-project python scripts/check_orbital_contracts_c_parity.py`,
  all 92 existing C contract cases passed and the excerpt matched upstream.
- Rust inventory, one-step audit, and GeneralRBM discrete gates: queued behind
  the shared Cargo build lock at the time this initial record was written.
- Unrun: long ctest summaries, new independent oracles for the remaining model
  families, native macOS checks, and new Julia 1.13.1 reference generation.

Update this execution record with observed outcomes; do not infer success from
test registration, fixture presence, or compilation.
