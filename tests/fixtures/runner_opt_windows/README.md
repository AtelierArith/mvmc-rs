# Historical workload C output windows

These fixtures preserve the historical independently observed Julia numerical
workloads while replacing incomplete Julia output expectations with complete
C declared-slot schemas and verbatim C aggregation. They are not newly
verified Julia-writer behavior and not full native C sampler/SR/MPI parity.
No pinned Julia source or writer was edited. Julia writer defects belong to
the separate upstream patch worktree/PR54.

Initial captured source: `/tmp/mvmc-history-source.TUyDyl`, regular-file
digest `943601528b575c63cab6d66938f659a74398b1ec6fa315829a88a93dba2c90a5`.
Observer reproduction and boundaries: `c_toolbox/runner_opt_windows.md`.
C extraction and mathematical/schema contract: `c_toolbox/ctest_opt_window.md`.
Actual C adapter/executable/compiler hashes are recorded in
`dh24_real/direct-store0/README.md`; per-prefix provenance records original
overlay input hashes, Julia/BLAS settings, modpara and complete history hash.

Imported groups by acquisition snapshot (explicit steps=window=prefix, seed=1):

| Snapshot | Models | Solver/store | Prefixes | Generation |
| --- | --- | --- | --- | --- |
| TUyDyl / 943601… | dh2_real, dh2_cmp, dh4_real, dh4_cmp, dh24_real, dh24_cmp | direct/0 and CG/0 | 1/2/3/50 | 6833, 9527, 72623; completed original status assertions |
| HHdjE0 / 5ced54… | same six normal DH models | direct/1 | 1/2/3/50 | 62067 exit0; 12 assertions/model |
| HHdjE0 / 5ced54… | rbm_real, rbm_cmp, rbm_general_cmp, rbm_dh24_cmp | direct/0 and CG/0 | 1/2/3/50 | 80191 exit0; 16 assertions/model/solver |
| HHdjE0 / 5ced54… | rbm_fsz | CG/0 | 1/2/3/50 | 9260 exit0; 16 assertions |
| HHdjE0 / 5ced54… | opt_real | direct/0 and CG/0 | 1/2/3/27/28/29/50 | 90440 exit0; 28 assertions/solver |
| HHdjE0 / 5ced54… | opt_cmp, opt_dh24_rbm_cmp | direct/0 and CG/0 | 1/2/3/50 | 89926 exit0; 16 assertions/model/solver |
| HHdjE0 / 5ced54… | rbm_real, rbm_cmp, rbm_general_cmp, rbm_dh24_cmp, rbm_fsz, opt_real, opt_cmp, opt_dh24_rbm_cmp | direct/1 | 1/2/3/50 | 42564 exit0; original status assertions retained |
| HHdjE0 / 5ced54… | rbm_fsz; rbm_reference_cmp | direct/0; direct/1 and CG/0 | 1/2/3/50 | 89526 exit0; 20 assertions/model/solver |

These imported groups comprise 44 model/solver/store axes and 182 prefix
observations, including explicitly failed SR histories without final output.
Canonical rbm_reference_cmp uses the original C-kernel-order translation and
seed12395 rather than seed1. No all-axis Rust acceptance is implied by this
acquisition inventory; focused test outcomes are in `failure-ledger.md`.

Additional acquisition KXznOf digest
`71304a6d333eb061601af92a8c0c875eb631d2c78de2b87bf60065f66af7506b`
imports DH2/DH4/DH24 FSZ direct/0, direct/1 and CG/0, plus OptTrans-FSZ
direct/0 and direct/1: 11 axes, 44 prefix observations (1/2/3/50), bringing
the inventory to 55 axes and 226 observations. All accepted axes passed exact
saved-configuration/next624 checks against SHA-verified native fixture
inheritance before C window aggregation. This is acquisition, not 55 passing
Rust tests. It combines unchanged original Julia initialization/sampling/SR
with the native C energy bridge and C projection-counter checks, then actual
C output aggregation; it is not full C executable or MPI verification.
The bridge's 108 checks are 72 complex plus 36 scalar kernel/ABI checks,
not 108 full pipeline runs. Observer SHA256 is
`305934493f2e724cddfdd29274fad6c59eaa6e09b4bc37103d2dda8682286eed`;
bridge library SHA256 is
`5434e44ab1bb1eab95acae35a5d54e8c093f0b9ee7f1a0519dab947e3e74b235`.
Binding-world-age warnings are retained in the external generation logs.
The completed generator STORE1 log is
`/tmp/mvmc-runner-native-final-opt_fsz-direct-store1.log` in the container.

OptTrans-FSZ CG/0 is deliberately **not imported**: prefix50's regenerated
saved configuration/RNG does not match its archived checkpoint. The original
unobserved regeneration matches the observed one, ruling out the history
observer as the cause but not resolving the first discrete divergence.
The new Rust C-faithful CG recurrence must also be reclassified against these
historical original-Julia trajectories; successful acquisition is not a claim
that a different recurrence has identical trajectories. No references may be
replaced with Rust values or accepted by relaxing discrete assertions.

The later immutable snapshot `/tmp/mvmc-history-retained.HHdjE0` has regular-file
digest `5ced54f40b7fc3be54aa29b15fb059e961650e2d2f8d2efc985488bb0d498cba`.
Its observer SHA256 is
`e13f5e2fbbd2ab7250e64fdd4041444c89e18152968efd2f91a561c32e226037`.
Unlike the initial mapped-only packing observer, this acquisition bridge
captures the original dense RBM initializer, indexed overlays and actual
solver deltas in a sidecar for unmapped slots. It does not repair Julia
production or change original sampling/solver arrays. The old zero-filled
RBM-FSZ CG expectations were replaced only after independent retained values
and actual C aggregation, never from Rust output.

For every imported method/prefix, newly captured next-624 RNG and saved
configuration records matched the unchanged archived `sr_{direct,cg}` records
exactly by cmp for prefixes 1/2/3/50; opt_real direct additionally checks
27/28/29. No archived opt_real CG checkpoints exist for 27/28/29, so those
additional generated windows do not establish archived discrete parity.
The original checkpoints remain the Rust exact-discrete
expectations; they are not regenerated from Rust or replaced here.

`c-window-input.txt` contains measured pre-SR Etot/Etot2 plus synchronized
post-SR declared parameters, including DH/RBM/reserved slots where present.
`zvo_var.dat` is the separate complete pre-SR declared-slot observation.
`zqp*.dat` are actual C StoreOptData/OutputOptData results on that independent
history. Preserve C trailing spaces and filenames. Window=1 has only the main
paired row; larger windows include active block files with sample deviations.

Rust must assert the full C layout/manifest as well as the existing discrete
checkpoints and parameter/energy gates. Historical Julia omission checks, if
retained, must explicitly project the named subset; they cannot replace full
declared-slot assertions. Numerical comparisons retain abs=rel=1e-11, as
justified in the C-window documentation, without relaxing discrete contracts.
