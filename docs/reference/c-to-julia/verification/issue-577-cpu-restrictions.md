# CPU input and grouped SR-CG restrictions (issue #577)

This audit concerns the vendored mVMC 1.3.0 CPU source, not the separate GPU
runner. The Rust runner rejects the inputs below before calculation. No C
supported input was identified among these three restrictions.

| Input | C 1.3.0 evidence | Rust decision and coverage |
| --- | --- | --- |
| `BF`, `BFRange` | `src/mVMC/readdef.c` defines `_NOTBACKFLOW` at line 35. Its `KWBFRange` and `KWBF` cases at lines 539–558 report that Back Flow is unsupported and fail reading. BackFlow code elsewhere does not make these input sections available in the shipped source. | Reject in both optimization and PhysCal. `crates/mvmc-cli/tests/issue366_physcal_admission.rs` and `crates/mvmc-core/tests/runtime_contract.rs` exercise admission. |
| `SpinJastrow` | The C NameList keyword table and `NProj` construction (`readdef.c`, around lines 760–761 and the keyword declarations) contain Jastrow and DH terms but no SpinJastrow input or projection block. `projection.c:289` describes spin Jastrow as future work. | Reject as unsupported; do not treat it as ordinary Jastrow. `crates/mvmc-core/tests/runtime_contract.rs` checks optimization and PhysCal admission. |
| `NSplitSize > 1`, `NSRCG != 0` | `VMCMainCal` partitions samples with `SplitLoop` (`vmccal.c:108`), writes stored derivatives using the global `sample` index (`vmccal.c:241,248`), then passes the allocation base and local `sampleEnd - sampleStart` to `calculateOO_Store` (`vmccal.c:311–318`). On ranks with `sampleStart > 0`, that reads unwritten malloc storage (`setmemory.c:417–421`); the corresponding grouped CG calculation has no defined reference result. | Reject grouped CG rather than inventing a C numerical contract. `crates/mvmc-cli/tests/issue178_support_matrix.rs` covers this rejection. Grouped direct SR remains independently supported. |

For ungrouped runs, C selects CG for nonzero `NSRCG` but allocates sample
storage only for `NSRCG == 1` or `NStore != 0` (`setmemory.c:417–421`).
Rust therefore rejects `NSRCG >= 2` with `NStore = 0`; this is separate from
the grouped restriction. See `crates/mvmc-core/src/validation.rs` and its
storage-contract tests. The user-facing matrix is in
[`manual/07-input-files.md`](../../../manual/07-input-files.md#75-supported-and-rejected-inputs).
