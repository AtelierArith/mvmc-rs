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
Independent native first-failure/retry/exhaustion operands, exact primitive
trajectory at those boundaries and broader full20 sampling remain acceptance
work under the related open issues.

## Acceptance mapping (keep #274 open)

| Requirement | Actual coverage | Remaining independent evidence |
| --- | --- | --- |
| First success | C `success.stdout`, `shared_loop_matches_independent_c_caller_controls`; public zero-preflight first-attempt test | Native C placement/kernel primitive/configuration fixture at this lifecycle boundary |
| Rejected layout / retry | C `retry.stdout` and `peer-retry.stdout`, signed-MAX ordinal checks | Actual failing kernel operands and resulting native C/Rust primitive/configuration sequence |
| Negative native INFO | C `negative.stdout`, `negative_info_is_not_a_boolean_retry`; MPI signed negative MAX | Native negative-kernel acquisition where supported (not synthetic status alone) |
| Subsequent/nonfinite-IP recovery | Both caller paths use the shared initializer and distinct setup; ordinary full regressions pass | Source-pinned native recovery fixture, exact words/count/configuration at its first failure |
| Burn restore | Storage/preflight regressions; MPI burn-shape/burn-kernel peer errors with raw624/cursor/count/future preserved | Native successful restore/recovery trajectory fixture at the relevant boundary |
| Exhaustion | C `exhaustion.stdout` and `call101-success.stdout`, exact101 control calls and typed last INFO | Actual native placement/kernel/RNG/configuration exhaustion operands |
| MPI ordering/propagation | Live worlds2/4, widths1/2, signed MAX and real/complex typed peer failures;80/160 markers | Native numeric failure/recovery model collective trajectory, distinct from synthetic preconditions |

The six C fixtures establish independent caller order with stubs, not all native
operands or primitive words. Passing full Rust suites does not fill those missing
oracle requirements. This PR therefore uses Related to #274, not Closes #274.
