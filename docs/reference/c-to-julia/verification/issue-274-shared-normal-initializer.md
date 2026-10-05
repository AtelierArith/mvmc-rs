# Shared normal initializer milestone (#274)

Related to #274, #180, #182 and #185. This milestone restores a C-authoritative
normal initialization lifecycle; it does not close their remaining native retry,
full-model sampling, threading or reference-adoption acceptance work.

## Production contract

C's shared `makeInitialSample` validates placement using the complex master even
for a normal-real sampler. Rust now follows that order, then performs the
separate real or complex caller setup. Signed comm1 `MPI_MAX` preserves native
negative INFO: only a positive result retries. The completed 101st call exhausts
even when that call succeeds. Exhaustion is a typed error, not an empty success.

Unsupported input/kernel errors are coordinated before overlap collectives and
propagated through public normal consumers. Their separate preflight accepts only
zero or the reserved unsupported sentinel; reduced -3/1 are inconsistent typed
statuses, not native INFO. Native negative INFO inside the retry loop remains
valid. MPI collectives stay on the initializing thread, outside Rayon.

The native complex path preserves C scalar publication and factor status rather
than Julia's whole-result atomicity or MINABS2 rejection. No numerical tolerance,
fixture value, RNG algorithm, seed or rejection cap is changed. Raw LocSpin==1
selects localized spins; other definition integers project to zero. Checked
dimensions/slot conditions do not claim every impossible placement terminates.

LocSpin workspace is written before caller preflight. Rejection tests assert
configuration/RNG preservation, not that every state byte is unchanged. Ne=0
normal child INFO-5 is justified by the authoritative SKTRF LDA checks; actual
XERBLA abort/return and full native Ne=0 model behavior remain unmeasured.

## Independent reference and regression scope

The optional toolbox records original `vmcmake.c` SHA-256
`8431b58eaec53e5325c801b69aa8a56306ec6a154b54e099266f4e21e5dd42ed`,
complete extraction boundaries and compiler flags. Six checked-in C-derived
caller-control fixtures cover first success, retry, negative INFO, peer MAX,
call101-success and exhaustion. Placement/kernel/MPI are explicit stubs:
**CALLER_CONTROL_ONLY**, not actual SFMT/SKTRF/MPI model evidence. Cargo reads
only the fixtures, never toolbox/reference programs. See their PROVENANCE.txt
and the toolbox README for optional reproduction.

Consumer doubles implement signed MAX without consuming existing failure/SR
collective ordinals. PhysCal uses its supported nonzero fixed-parameter input.
Merged PR286's fallible declaration APIs and callback ordinals 7/[2,6], collective
count7 and callback count1 are retained, not replaced by the pre-merge layout.

## Actual Linux verification

Frozen production base: `12c5bd83967c12b8d246fa81a102ada360bc19e1` plus this
implementation, Linux x86_64, Rust1.99.0, MPICH4.2.0. Separate target, jobs2,
BLAS/OMP/MKL/Rayon1, locked test-fast with no-fail-fast/retries0. Long baselines
remain20 steps; historical50-step artifacts are not new runtime gates.

| Gate | Actual result |
| --- | --- |
| Initializer/native-status/projection units | 16 PASS |
| Repaired consumers/ordinal control | 9 PASS |
| Default workspace | 1088 PASS,41 ignored,224.307s |
| All-features MPI+SIMD workspace | 1090 PASS,60 ignored,250.228s |
| Explicit MPI ignored protocol | worlds2/4 PASS;80/160 participant markers;2/4 rank summaries |
| Formatting / strict ordinary all-target Clippy / doctests | all status0 |

Default UUID is `632cc3ae-c487-4011-a69c-b03a917b97a1`; all-feature
UUID is `20462196-e573-4db6-b0cb-c62bd0b17bd4`. Default receipt `xCa1Ox`, quality
receipt `VkCvwa`, all-feature/MPI receipt `MzoOSM` are under container73c's
`/tmp/mvmc-274-validation-proof.*`. These are evidence locations, not dependencies
of normal Cargo tests. Source/fixture membership and bytes, tools/runtime,
selected145 all-feature ELFs/providers, LIST post association and cleanup were
independently replayed successfully. MPI protocol uses synthetic invalid storage
and signed status operands, not a native C full-model trajectory.

The earlier all-feature receipt `ViZezj` remains a failure: selector/child/outer255,
top execution1, compile0. A canonical-source-path association correction enabled
the independently authorized successor; no kernel/test expectations changed.
Earlier owner-guard failures likewise remain failures, not numerical passes.

Publication incorporates main6689cd3e's test/doc-only PR287/288/289 delta without
relabelling the frozen12c5 receipts. No performance/speedup claim is made.
(Historical: native boundary operands were added later by PR #314; see the
refreshed mapping below. Broader full20 sampling remains under #180.)

## Acceptance mapping (refreshed after PR #314)

Status refresh: PR #291 (initializer lifecycle), PR #304 (real zero-IP log, see
`issue-274-real-zero-ip.md`) and PR #314 (ten native C-derived boundary fixtures)
are merged. The earlier statement above that native first-failure/retry/exhaustion
operands are missing is superseded by the table below. The frozen12c5 receipts
earlier in this document remain historical and were not re-labelled.

Native boundary fixtures: `tests/fixtures/issue274_native_boundaries/{first,retry,
burn,recover,exhaust}.{real,complex}.json`, provenance in
`tests/fixtures/issue274_native_boundaries.PROVENANCE.md`, consumed by
`crates/mvmc-core/tests/native274_boundaries.rs` (`native274_{first,retry,burn,
recover,exhaust}_{real,complex}`). Each case checks consumed primitive words,
counts/cursor, defined configuration, final public caller state, INFO and
collective ordinals against C checkpoint serialization (world1 caller prefixes,
Linux x86_64, original `vmcmake.c`/`vmcmake_real.c`/`matrix.c`/`SFMT.c`). Caller-control
fixtures: `tests/fixtures/issue274_caller_control/*.stdout`, consumed by the
`sampling::normal_initial` unit tests. MPI: `mpi_issue274_normal_initialization.rs`
(explicit ignored 2/4-rank gate).

| Requirement | Coverage on main | Still absent |
| --- | --- | --- |
| First success | `native274_first_{real,complex}`; `shared_loop_matches_independent_c_caller_controls` (success.stdout) | none for the initializer prefix |
| Rejected layout / retry | `native274_retry_{real,complex}` (primitive count 8); retry.stdout, peer-retry.stdout, `maximum_status_controls_retry_not_local_status` | none for world1 |
| Negative native INFO | negative.stdout, `negative_info_is_not_a_boolean_retry`; MPI signed negative MAX | None: native kernel negative INFO (Ne=0, INFO -5) measured and fixture-covered by #342, see `issue-342-negative-info.md` |
| Subsequent / nonfinite-IP recovery | `native274_recover_{real,complex}` (single C recovery, count 4); `native274_real_log_ip.rs` (4 analytic zero/negative/subnormal/reduction-order tests, `clog` semantics) | none for the prefix |
| Burn restore | `native274_burn_{real,complex}` (count 0 restore); storage/preflight units; MPI burn-shape/burn-kernel peer errors | none for the prefix |
| Exhaustion | `native274_exhaust_{real,complex}` (101 placement/factor attempts, count 202); exhaustion.stdout, `call101_exhausts_even_if_its_status_succeeds` | none |
| MPI ordering / propagation | Live worlds 2/4 signed MAX and typed peer failures (80/160 markers) | Multi-rank native C trajectory; MPI protocol uses synthetic storage/status operands, and the native fixtures are world1 only |

Scope notes. Intermediate configuration in the native fixtures is independently
replayed placement, not live working-buffer observation; the full future624 tail
comes from the common passive C observer pair. The ten fixtures are C
initializer-prefix evidence only (zero proposal iterations), not a full sampling,
FSZ, 13-model, thread or MPI trajectory. Broader full-model sampling trajectory
parity is tracked by #180 (reference adoption by #185) and is not a #274
deliverable. This refresh changes documentation only: no production code, fixture
or tolerance changed. Verified on main36b1eb5a, Linux x86_64, `test-fast`:
`native274_boundaries` 10 PASS, `native274_real_log_ip` 4 PASS,
`sampling::normal_initial` unit tests and real FSZ regression binaries PASS
(23 selected, 23 PASS); the MPI gate is explicit/ignored and was not rerun here.
Related to #274, #180 and #185.
