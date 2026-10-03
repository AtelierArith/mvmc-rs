# Focused parallel range / group / seed literals

Related to #179 and #184, not complete MPI model verification. New independent
test: `crates/mvmc-core/tests/mpi_issue179_parallel_literals.rs`.
No production API/kernel changes, ledger updates, C/Julia runtime calls or
toolbox dependency. Twelve original IDs are represented by eleven named tests.

## Assertion correspondence

Original source: Julia gitlink `8bb1b9e8ae47b1512c00b321be05664ddcac0fd1`,
`MVMCOptimizers.jl/test_unit/test_unit_parallel.jl`, SHA
`a2b582f61dc981c318f817fc0575cfadfdb8c93e34c5679aeaf7fdae48be5050`.

| IDs | Exact Rust named test and contract |
|---|---|
| M0518 | `m0518_serial_qp4_is_full`: SingleProcessReducer QP4 `0..4` (Julia `(1,5)`) |
| M0519 | `m0519_qp4_local_rank0_width2`: GroupAssignment local0/size2 QP4 `0..2` (Julia `(1,3)`) |
| M0520 | `m0520_qp4_local_rank1_width2`: local1/size2 QP4 `2..4` (Julia `(3,5)`) |
| M0521 | `m0521_qp1_local_rank3_width4_is_empty`: local3/size4 QP1 `1..1` (Julia `(2,2)`) |
| M0522 | `m0522_world3_width2_has_two_groups_with_short_final_group`: actual public assignment function for ranks0/1/2 gives group0/0/1, local0/1/0, actual sizes2/2/1 |
| M0562, M0578 | `m0562_m0578_missing_seed_uses_parser_default11272`: removes RndSeed line, parsed default11272 and actual prepared RNG |
| M0563 | `m0563_zero_seed_is_not_default_or_clock`: declared0, actual RNG seeded0 |
| M0564 | `m0564_positive_seed123`: declared123, actual RNG seeded123 |
| M0565 | `m0565_negative_seed_uses_positive_current_unix_seconds`: declared-1, prepared full block matches a positive Unix second observed before/after the public call |
| M0566 | `m0566_explicit777_overrides_input123`: declared123 remains parsed123; explicit API seed777 chooses actual RNG |
| M0567 | `m0567_actual_mpi_group3_base100_is103`: ignored live world4/width1 test, production MpiGroupContext + public preparation, rank3/group3 actual RNG seed103; other ranks verify100/101/102 |

The range tests intentionally use Rust zero-based half-open APIs, not Julia
object identity or fake nullable communicators. World3/width2 is arithmetic
coverage, **not a three-rank MPI launch**. Group3 seed coverage is an actual
four-rank launch, not a handmade offset reducer.

All seed cases exercise public `prepare_phys_cal_from_namelist` (or its actual
MPI reducer counterpart), which parses/loads/synchronizes fixed parameters
before producing the RNG. The preparation intentionally has not drawn any
words yet. Each test compares624 primitive outputs against the existing SFMT
primitive initialized with the independently specified expected seed, with
word consumption0 on the original before/after clone peeking. This tests seed
selection and lifecycle; it is not a new independent C SFMT golden vector.
Existing primitive C golden tests remain separate. No MCMC trajectory or
floating-point model parity follows from these literals.

Offline input: `tests/fixtures/physcal_181/heisenberg_chain_real/inputs` plus
its `zqp_opt.dat`, already recorded in that fixture's provenance. Each test
copies files into an exclusively created process/counter temporary directory;
only RndSeed is removed/replaced. Original input and shared source are not
modified. Concurrent rank/tests cannot share those directories.

C authority (source-only provenance, no new native execution claim):

- `splitloop.c:33–68`, SHA
  `3734f9deadd12f1c5af0fbf4927dabc3e77c80bdb512c3d0160a809ff0117ff8`:
  zero-based partitioning and empty-work branch.
- `vmcmain.c:239–257`, SHA
  `fdcd661c4eb028786fc58ae5e1f5232426c943b547f95a0d74e888e602256d63`:
  `group1=rank0/NSplitSize`, short final communicator allowed, SFMT
  initialization with `RndSeed+group1`.
- `readdef.c:1773,1954–1955`, SHA
  `6c53cb832f93d6cbfd7cea955fbb693738af5536b913d36af32b98eed38c32d9`:
  default11272; negative input uses `time(NULL)`. Explicit seed override is
  Julia/Rust architecture, not a C input extension claim.

## Actual bounded proof

Snapshot `/home/vscode/.cache/mvmc/issue179-parallel-literals.3s00cW/repo`:
committed base `70e525ac2a6150820d679b0ae269d1804dac1ed9` plus **only this
new test**. No shared production drafts were overlaid. Full compiled input
manifest SHA `85842997498100be93e5c25b865ea70b6a071bbb66f6ff995dcd763d31e63681`;
environment SHA `94fea8fabc717e11af33b4d2414c2064989b0c26193d4339dfdf7739fe0a9e37`.
Test source SHA `fc0655a88fea7f6ed23f0547a0a322c38d4fd6fc62545e2da61ff7094751c2ae`.
Binary `mpi_issue179_parallel_literals-a8ff788ff2270312` SHA
`3e4f0b0b5eea9d2b97db3ad9b45ba715d2809a6d219c66283e7f80c384ce1892`.

Actual repaired container `73c57e563c61`, MPICH4.2.0 internal Hydra PMI,
Linux x86_64, Rust1.99, system LP64 OpenBLAS.26, BLAS/OMP threads1.
Named target: `/home/vscode/.cache/mvmc/target/issue179-parallel-literals`.
This is not the old Open MPI image nor historical `28851` proof.

Build/nextest/clippy handle7821 terminal0: nextest10/10 passed0.052s,
one live MPI test deliberately ignored; all-features focused clippy exit0
13.78s (transitive tenferro-runtime deprecation warning remains recorded).
Then actual `mpiexec -n4` selected exactly the ignored group3 test, bounded120s,
launcher exit0: each of four ranks reported1 passed0 failed0 ignored,0.07s.
Full source/test/binary SHA postchecks passed. Evidence files at snapshot parent:
`nextest.log`, `nextest.exit`, `clippy.log`, `clippy.exit`, `ignored-list.txt`,
`mpi4.log`, `mpi4.exit`, `mpi-binary.sha256`, `mpi-binary-check.txt`,
`source.sha256`, `source-check.txt`, `environment.txt`, `provenance.sha256`.

Commands (with the recorded named target/environment):

```sh
timeout --kill-after=10s 600s cargo nextest run --locked --cargo-profile test-fast -p mvmc-core --features mpi --test mpi_issue179_parallel_literals --no-fail-fast --retries 0
timeout --kill-after=10s 600s cargo clippy --locked -p mvmc-core --all-features --test mpi_issue179_parallel_literals -- -D warnings
timeout --kill-after=5s 120s mpiexec -n 4 "$BINARY" --ignored --exact m0567_actual_mpi_group3_base100_is103 --nocapture
```

This closes only the listed literal/seed assertions subject to independent
review. Current full grouped20/model/expanded callback accuracy and the
remaining parallel74/MPI87 semantic gaps stay open.
