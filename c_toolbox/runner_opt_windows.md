# Historical runner window observer

`runner_opt_windows.jl` observes the same historical workloads as
`scripts/check_sr_direct_runner_parity.jl` and `check_sr_cg_runner_parity.jl`.
It does not run Rust or read Rust outputs. It redirects the original script's
fixture writes to an explicitly supplied external staging directory, retaining
the original sampler, solver, status assertions and RNG checkpoint generation.
Read-only hooks additionally record:

- Every successful post-SR synchronized declared parameter vector with the
  measured pre-SR Etot and Etot2, in chronological order.
- Every pre-SR output row with all declared parameter slots, including DH/RBM
  slots omitted from historical Julia output.
- Source, input-directory, project/manifest and history SHA-256 hashes, actual
  Julia/BLAS information, solver options, prefix and effective modpara.

Run from an unchanged captured repository snapshot, with Julia 1.13.1 and its
`Manifest-v1.13.toml`. The stage must be outside that snapshot:

```sh
julia +1.13.1 --project=extern/Julia-mVMC c_toolbox/runner_opt_windows.jl \
  /tmp/mvmc-independent-dh24-real direct --case=dh24_real --steps=1,2,3,50
```

`--store=1` and `--c-kernel-order` are the original observer's options. Select
them only for workloads whose archived provenance used them. The unmodified
FSZ script is not an authoritative native-C-energy reference; matching FSZ
kernel installation, bridge provenance and input routing must be established
before accepting an FSZ fixture. No blanket Julia numerical behavior is
declared C-compatible by this observer.

Build and run the standalone C aggregator documented in `ctest_opt_window.md`
on each `step-N/c-window-input.txt`. Its numerical bodies are verbatim C
`StoreOptData`/`OutputOptData`; history remains independently generated Julia
or C-contract-shim data, not a full C sampler/SR run. Record compiler options,
source/extraction/adapter/executable/input/output hashes separately. FSZ
auxiliary-file flags remain an explicit probe limitation; a generic Slater
auxiliary file is not proof of the FSZ filename/schema contract.

For an SR failure, the stage records completed successful history only and the
failure marker. Production returns before final output; do not use aggregation
of that partial history to assert that an output file should exist. No C output
window is defined by uninitialized missing rows. Windows here are positive
prefix-length windows, not a silently clamped oversized caller window.

The initial mutable-draft pilot (handle 6274) completed direct DH24-real
prefixes 1/2/3 and passed nine original script status assertions. It is only
observer-machinery validation: its source was edited during dependency
compilation, so it is not an accepted fixture/provenance generation run.
Normal Cargo tests must only read subsequently checked-in independent fixtures;
they must never compile/invoke this observer or C toolbox.

## Dense-slot retention bridge (not a production repair)

The pinned Julia `_add_parameter_delta_direct!` in
`MVMCOptimizers.jl/src/stochastic_opt.jl:259` updates mapped term locations;
`parameter_sync.jl:226` packs those locations into an initially zero vector.
Unmapped declared RBM slots consequently lose solver updates. C instead updates
the dense `para` vector via `smatToParaIdx` in `stcopt_cg_impl.c:229–234`.
The first RBM-FSZ CG prefix1 difference is ChargeRBM_PhysLayer index1 (zero
based), global Para index22, output column48: a retained value of
`-0.0569751086369507942` versus the historical mapped-only zero.

This observer captures the original initializer's complete RBM values,
independently decodes indexed input overlays, and records actual original
solver deltas. It fills only unmapped slots of the output/history sidecar.
It does not alter the production term arrays, solver, sampling, or RNG, and
must not be described as fixing Julia production. That repair belongs to the
separate Julia fork and existing upstream PR54.

Accepted retained-slot generation used immutable snapshot
`/tmp/mvmc-history-retained.HHdjE0`, regular-file inventory digest
`5ced54f40b7fc3be54aa29b15fb059e961650e2d2f8d2efc985488bb0d498cba`;
observer SHA256
`e13f5e2fbbd2ab7250e64fdd4041444c89e18152968efd2f91a561c32e226037`.
RBM-FSZ CG prefixes 1/2/3/50 matched independent archived next624 RNG and saved
configurations exactly before actual C aggregation. The adapter executable
SHA256 is `4047c586860f319fd7d103f99db981da1ffc68d5f24d1a86ecced9156a65abcc`,
compiled with GCC 13.3.0, `-O0 -ffp-contract=off`, and `-lm`.
These are mixed independent Julia-history/C-output expectations, not full
native C sampling/SR execution or verification of the repaired Julia head.

## Native FSZ local-energy acquisition

For DH2/DH4/DH24/OptTrans FSZ, pass
`--native-fsz-bridge=/tmp/mvmc-runner-native-fsz-bridge`. This installs the
existing `scripts/reference_native_fsz_energy.jl` native C local-energy bridge
before executing the original runner script. The initializer, RNG, sampler
and SR solver remain the original Julia implementations. No imaginary output
fields are stripped. The bridge also checks saved projection counters via C
MakeProjCnt. This is not full native C executable or MPI sampling parity.

Bridge build/validation used the unchanged snapshot and the optional command
`uv run --no-project python scripts/check_native_fsz_runner_bridge.py --build-dir /tmp/mvmc-runner-native-fsz-bridge`:
72 complex plus 36 scalar FSZ energy gates passed with unchanged borrowed
operands. Library SHA256:
`5434e44ab1bb1eab95acae35a5d54e8c093f0b9ee7f1a0519dab947e3e74b235`.
Per-prefix provenance includes the library, bridge helper and underlying C
source/compiler metadata. Native archived checkpoint inheritance is resolved
with `reference_native_fsz_fixture_inheritance.jl`, including SHA checks.

Snapshots PpHwlb and LlrTNE failed during Julia world-age installation; no
expectations from those attempts are accepted. Snapshot KXznOf uses
`invokelatest` only for bridge installation (not solver/sample arithmetic).
Its regular-file digest is
`71304a6d333eb061601af92a8c0c875eb631d2c78de2b87bf60065f66af7506b`;
observer SHA256
`305934493f2e724cddfdd29274fad6c59eaa6e09b4bc37103d2dda8682286eed`.
Julia emits binding-world-age warnings during installation; retain them in
the generation logs. Pilot49603 passed DH2-FSZ prefix1 with three original
status assertions and exact next624/configuration checkpoint comparisons.
