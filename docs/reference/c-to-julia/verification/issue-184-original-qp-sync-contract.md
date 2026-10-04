# Original S128/S129 synchronization contracts

Related to #184 and #185; neither issue is completed by this packet.

The focused Rust test is
`crates/mvmc-core/tests/issue184_qp_sync_contract.rs`. It preserves the original
two synthetic constructors: ParaQPTrans `[1, -2+2i]` remains unchanged, while
OptTrans `[3+4i, 1]` is independently normalized by five to `[0.6+0.8i, 0.2]`.
The original computed-value absolute budget `1e-14` is unchanged. Each case
exercises local synchronization and the `SingleProcessReducer` broadcast entry.
Same-execution copies additionally check declared ParaQPOptTrans and flags.

## Reference pins and inspected boundaries

- [Julia original test, lines 119–161](https://github.com/tmisawa/Julia-mVMC/blob/8bb1b9e8ae47b1512c00b321be05664ddcac0fd1/MVMCOptimizers.jl/test_unit/test_unit_parameter_sync.jl#L119):
  complete-file SHA-256
  `49937cd1879e850aea8992d5ddafba96c5d1b1fcfe56261b94ff230bb8e0b171`.
  These are the original S128/S129 assertion bodies; no Julia rerun was acquired.
- [C SyncModifiedParameter, lines 134–178](https://github.com/issp-center-dev/mVMC/blob/d73d06bd529d3b2573f38eb5817c4a5f52971006/src/mVMC/parameter.c#L134):
  complete-file SHA-256
  `46ad04622f4475337028cee633bd76ce318a6d5058d03f202b55204500399fb0`.
  The inspected function normalizes the dedicated OptTrans array when enabled,
  not ParaQPTrans. These are source-inspection boundaries, not a newly compiled
  C extraction or model-run acquisition. The C function also accesses Slater;
  the empty-Slater Julia synthetic constructors are not claimed as runnable C inputs.

## Current publication verification

Publication base: merged main `12c5bd83967c12b8d246fa81a102ada360bc19e1`.
The new Rust test is byte-identical to the previously reviewed 92a9 version.
All 59 tracked paths changed between those baselines were transferred and their
SHA-256 values checked against the official 12c5 worktree before recording the
final receipt. The retained container source directory still has `main92a9` in
its historical name; the runtime receipt explicitly records the actual 12c5 base.

Owner session `45633` completed with pipeline status zero. Nextest UUID
`b1154204-5566-466e-84a9-723e625b540d`: **4 passed**, 225 filtered, 0.064 seconds.
The commands below, targeted strict Clippy, formatting, and complete source/tool
before-after checks all returned zero. Final receipt:
`happy_jackson:/tmp/issue184-qp-sync-four-main12c5-proof`.
The existing dependency deprecation warning from tenferro-runtime was retained;
no warning suppression or dependency change was made.

## Historical 92a9 verification

Baseline: `92a9de4fae7d1eb8cafc97b351550d26e2e6187a`, plus the sole new Rust test.
Nextest UUID `42d4ba59-747a-45c5-8753-4c13c8da1c40`: **4 passed**, 223 filtered,
0.056 seconds. This combines the two new original cases with the existing
`broadcasts_and_applies_canonical_variational_parameters` and
`opttrans_sync_matches_julia_with_declared_slater_normalization_from_c` tests.

```sh
cargo nextest run --locked --cargo-profile test-fast -p mvmc-core \
  --lib --test issue184_qp_sync_contract \
  -E 'test(original_s128_para_qp_trans_is_not_normalized) | test(original_s129_opttrans_scales_by_five_without_changing_para_qp_trans) | test(opttrans_sync_matches_julia_with_declared_slater_normalization_from_c) | test(broadcasts_and_applies_canonical_variational_parameters)' \
  --no-fail-fast --retries 0
cargo clippy --locked -p mvmc-core --test issue184_qp_sync_contract -- -D warnings
rustfmt --edition 2021 --config skip_children=true,max_width=100 --check \
  crates/mvmc-core/tests/issue184_qp_sync_contract.rs
```

All three commands and source/tool before-after checks returned zero. Acquisition
used Linux x86_64 DevContainer `happy_jackson`, Rust 1.99.0, nextest 0.9.146,
OPENBLAS/OMP/MKL thread settings one. Owner receipt location:
`happy_jackson:/tmp/issue184-qp-sync-four-main92a9-proof` (local acquisition path,
not an attached or publicly accessible artifact). Test SHA-256:
`5adbb124596392ba3406a82bcf65fd029718c87b094ad69f92b899705517a6e1`.

The README was updated after each test acquisition; it does not change tested code.
There was no fixture generation, C-model run or Julia 1.13.1 rerun. The broadcast
entry uses a single-process reducer, not native MPI. Sync takes no RNG argument
and uses no global RNG: this packet makes no RNG, sampling-trajectory or full-model
claim. Those require runner evidence. Generic QPTrans API decisions are separate;
no production API change or broad ledger promotion is included.
