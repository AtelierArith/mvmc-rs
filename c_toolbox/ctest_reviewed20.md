# Reviewed canonical 20-step reference generation

The user changed the long runner baseline from 50 to 20 steps. Historical
50-step artifacts retain their original meaning; their final parameters must
not be relabeled as 20-step expectations.

`ctest_prefix_oracle.jl` now requires the reviewed Julia PR54 commit
`62b0f97f076fb55c71c3ab0caa041a9adff94e04` for 20-step generation. It verifies
all 63 source/project/Manifest identities in the pinned `reviewed-source.sha256`
(SHA-256 `4a22c8b21d2901bd3d88e08b65dcefbb349c5ec259583fd8d3c0482b73cec0b4`),
and checks the actually loaded Optimizers, ExpertModeParsers and SFMT paths.
It installs no historical initialization, retained-coefficient or CG replacement.
The runner extraction retains the original operations and adds observation hooks.

The optional native FSZ energy bridge is a standalone C local-energy reference,
not a full native C sampler, runner or SR oracle. Other energy paths are explicitly
labeled `reviewed_Julia_kernel`. Parameter initialization, sampling, optimization,
and captured window history use the reviewed Julia implementation. C-window
aggregation of that history is a separate offline step, not implicit native-C
full-run validation.

Actual acquisition completed: handle `58651`, terminal 0, 52/52 cases,
zero unverified cases. Frozen producer SHA-256:
`9355984f3b7762f82a82e49455d841ddc7ca93c9a69d5e3d0f54f9384c2a6a8e`.
Full producer stdout SHA-256:
`915437efc0b6637fcc53ab570680413353af0e1e231a87ee6cfde57c0e3e23c6`.
Root provenance SHA-256:
`57462e37a973d850ccfd7a9592067cad9e679eebf18e426b660df50c27ee8636`.

`ctest_audit_reviewed20.jl` checks input closure and artifacts before import.
Handle `52008` completed terminal 0, all 52 cases passed. Window headers contain
17 integers: steps, NPara and 15 family widths. Dense history rows contain
`2*(NPara+2)` finite components; final parameters contain `2*NPara`, energy two,
and each pre-SR var row `6+3*NPara` fields. Raw SFMT capture contains 624 words,
index and draw count; next-624 is separately labeled. These schema checks are not
model numerical acceptance.

Native window aggregation handle `37326` completed terminal 0, 52/52 windows.
Actual compiler: GCC Ubuntu 13.3.0-6ubuntu2~24.04.1, options
`-O0 -ffp-contract=off -lm`, no BLAS. Probe SHA-256:
`7a0434c1c7a377e8a84d31fa2a02499cd5cbb1007a83ff388b7e8ef932678d75`;
adapter source `4069961fb468957de87760db9e1a25a1587351f4c92b17c3d3e925698904e652`;
upstream extraction `47c801ea68d9bf40af967a3a0db6189125fd54ad146820b6470d63f9317a69da`.
Actual aggregation stdout SHA-256:
`18108b58120b9ef87d587b4263e23543a26d4de31cc1e3a07698c1a19dff41a9`.
See `ctest_opt_window.md` for upstream source/license/extraction boundaries.

The validated SFMT diagnostic library (`7e77954acae2073591edf17b5c9825f0c020a2525026ae450ac31b780d7a7c86`)
and observer (`91102d67e9674c3bbc3fb1ceca8b4dbfa32c2440219a8b4e48e2a70313ec2591`)
capture raw 624-word state, index and primitive draw count without consuming draws.
The separate next-624 artifact is also a non-consuming peek. See
`reviewed_sfmt_state.md` for original C translation units, compiler and invariance
checks; it does not establish model trajectory parity by itself.

Each case sets both effective optimization steps and window length explicitly,
records the canonical settings, seed and actual thread/backend configuration,
and validates all 15 declared family widths against NPara and dense finite window
rows. Output stages and case directories are exclusively created, never reused.

Input hashes cover every regular file in the canonical input directory, including
an implicit neighboring `initial.def`. The Rust fixture preflight checks every
listed hash, all parser-metadata namelist references, and a present `initial.def`
because these canonical gates use `InitialDef::Auto`. This is a flat input-bundle
contract; it does not cover arbitrary externally specified initial paths.
Overlay order remains the runtime's authoritative fixed keyword order, not hash
manifest order. Normal Rust tests invoke no C or Julia program.

Current acquisition is in container `73c57e563c61`, immutable source clone
`/tmp/mvmc-ctest20-reviewed62b.D3rDdz`. Hardened one-case smoke output is
`/tmp/mvmc-ctest20-reviewed62b.D3rDdz-smoke6` (terminal 0). Batch output is
`/tmp/mvmc-ctest20-reviewed62b.D3rDdz-matrix`, with full producer stdout at that
path plus `.stdout.txt`; C aggregation log is
`/tmp/mvmc-ctest20-reviewed62b.D3rDdz-c-windows.stdout.txt`.
Fixture import and integrated Rust gates remain pending; no native C sampler or
new macOS validation is claimed.
