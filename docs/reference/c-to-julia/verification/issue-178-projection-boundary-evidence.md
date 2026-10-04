# Issue 178: consistent zero-translation rejection

Related to [#178](https://github.com/AtelierArith/mvmc-rs/issues/178).
This records one public-library/CLI boundary repair, not completion of #178.

## Authority and minimal repair

The C Expert manual (`extern/mVMC-1.3.0/doc/en/source/expert.rst`,
504–513) specifies absolute NMPTrans as the number of translation sectors,
negative values for anti-periodic boundaries, and **1** for no projection.
`src/mVMC/readdef.c` 1904–1905 reads the integer; 742–748 converts negative
counts to their absolute value and sets APFlag; 778–779 derives NQPFix using
that count. Zero is not an identity sector. These sources do **not** establish
a native-C explicit zero-input error, and no such executable claim is made.

Rust previously rejected zero only in `validate_para_opt`, while PhysCal
accepted it and subsequently changed it to one. The repair moves the existing
check and exact diagnostic into `validate_supported_modpara`. Both entry
points now reject before initialization/output. It changes no accepted
parameter, numerical algorithm, RNG algorithm, or runner implementation.

## Actual old failure

The regression uses checked-in Heisenberg inputs/fixed records from
`tests/fixtures/physcal_181/heisenberg_chain_real`, sample3/warmup1/interval1,
one optimization step/window, seed11272. OPT excludes measurement-only
TwoBodyGEx so another unsupported setting cannot mask the intended boundary.
Library calls use the actual `vmc_para_opt` and `vmc_phys_cal_in_place` APIs.

Old library handle9589 terminated100, nextest
`c9fa8e55-1861-46a4-8ca7-95241970e621`: 2 PASS / 1 FAIL.
OPT rejected with words1→1, cursor1→1, unchanged state, and no output.
PhysCal returned Ok, changed NMPTrans0→1, consumed words1→137 with
cursor1→137, changed the published caller state, and created the output child.
Old CLI handle28680 terminated100: PhysCal exited0, reported completion of
one sample, and created output; the paired OPT case rejected first. The old
failure stops the test before the later signed controls; it proves nothing
about their execution.

## Fixed public boundaries and valid controls

`crates/mvmc-core/tests/issue178_projection_boundary.rs` checks both rejected
calls against their own pre-call snapshots: all data (including flags), state,
materialized complex/real Slater and inverse planes, raw624 RNG words/cursor,
draw count, and nonconsuming next624 remain unchanged. The exclusively owned
parent's requested child stays absent and its inventory stays empty.
The sentinel state has nonempty planes; this is not an empty-state comparison.
The same file exercises actual PhysCal calls with counts+1/-1.

The new paired CLI test in `crates/mvmc-cli/tests/runtime_contract.rs` runs
six actual subprocesses: OPT/PhysCal × zero/+1/-1. Zero exits1 with the exact
diagnostic and no output child. Both signed controls exit0 and create their
respective ordinary/indexed output files. Every copied input and fixed-record
byte remains unchanged. Output directory creation is exclusive, without
deleting pre-existing paths. This is accepted-entry evidence, not numerical
golden-output or anti-periodic-physics validation.

For negative CLI controls, all36 Orbital mapping rows explicitly carry a
fourth +1 sign; all12 TransSym rows already carry explicit +1 signs. Thus the
control does not rely on missing-column defaults. C GetInfoOrbitalAntiParallel
(readdef.c 2463 onward) and GetInfoTransSym (2233 onward) read these sign
columns. This all-positive-sign input is bounded; it does not prove arbitrary
anti-periodic sign patterns. No original fixture/vendor file is modified.

Final owner handle93542 terminated0, nextest
`916e3128-bb4f-4868-9ad4-011451fdf519`: 4/4 PASS, 0.058s.
Handle86133 focused strict clippy terminated0, 2.68s; owned-file rustfmt and
diff checks terminated0. Parent independently reports
`e837f396-1b5e-4806-9555-09b3fe04bf63`: 5 PASS, 0.107s, including the older
zero-translation CLI regression. Parent strict clippy handle29115 terminated0,
1.14s. The subsequent common-validator regression run
`6b839209-e2ca-47aa-9a6a-d91aee6c227d` at the parent's 084 checkpoint ran
`issue184_public_rejection_boundaries` and `runner_config`: 16/16 PASS,
0 skipped, 0.102s, terminal0. These focused checks are not a full-workspace
or complete MPI failure-matrix validation.

## Reproduction and provenance scope

Owner tests ran in frozen container73c57e563c61, checkout
`/tmp/mvmc-cli174-gmUVyr`, target
`/home/vscode/.cache/mvmc/target/cli174-gmUVyr`, Linux x86_64,
Rust1.99.0, OpenBLAS with OPENBLAS_NUM_THREADS=1 / OMP_NUM_THREADS=1.
This reuses a dirty copied snapshot with the three reviewed paths overlaid,
not a clean-current-main whole-workspace proof. Host/container source hashes
matched after validation; no complete compiler-input closure digest or
separately retained old executable is claimed.

```sh
cargo nextest run --locked --cargo-profile test-fast -p mvmc-core -p mvmc-cli \
  -E 'binary(issue178_projection_boundary) | test(paired_zero_translation_cli)' \
  --success-output immediate --no-fail-fast --retries 0
cargo clippy --locked --profile test-fast -p mvmc-core \
  --test issue178_projection_boundary -p mvmc-cli --test runtime_contract \
  -- -D warnings
```

Frozen source SHA-256:

```text
validation.rs c3a190bc054258b3ad803a7c5583fa5f1e28b083a873049b11912006d886fd79
issue178_projection_boundary.rs 074ae23d3358896ddcb83afe8a70f30fc5839ab9f4a36c997637758b3659719f
runtime_contract.rs 89a5308525876d7225b8a425978eb32116d34274cc619d53d486d44d0bff37f3
```

Retained host logs under `/tmp/mvmc-cli174-gmUVyr`:

```text
178-projection-old-library.log a73b4b54d115842e92347921bc9833fef42f98eacb733b348946a56ca4fa2b3d
178-projection-old-cli-built.log eefe6dd7da5bba726b57266d5f2d030f49622d2ce8c3bfbbc4d70053e3852df9
178-projection-final.log 6f3d5f68cded6813ca1f5cc4c51b8efe4840029b865050d8df66cbc566382b7f
178-projection-clippy.log 81c8741171bed98202e3499d8f3bb6dd2751d3d4dcfacd40e19485c6e0fa5ce6
```

The CLI exposes no raw RNG/state on error: process exit/input/output checks
are not subprocess RNG observations. Exact caller RNG/state evidence belongs
to the separate in-place library calls. Normal Rust tests use existing offline
fixtures only, without C/Julia oracle execution. Actual MPI sampling failures
and the remaining #178 collective matrix remain separate, open work.
