# Actual two-rank literal sampled CG operator regression

Owned ignored test: `crates/mvmc-core/tests/mpi_issue179_literal_operator.rs`,
`issue179_literal_two_rank_cg_products`. Cargo's ordinary test execution does not
launch MPI or a C/Julia reference. The explicit gate must compile with feature
`mpi` and verify this exact test is listed before launch (zero tests cannot pass).

Operands and final expected products are inherited from pinned Julia
`test/mpi/mpi_srcg_operate_smoke.jl`, SHA-256
`c883547ddb30e826c074a1fcaff70129ff4119feec97f1137c98b1830d99f861`.
Numerical authority is C `stcopt_cg_impl.c` lines 356–426, SHA-256
`41452de5fe766409431c6e1cf73cd2b12485faaeb4bfbcdec1e53444e9af1cf9`:
root broadcast, local real/imaginary sampled Gram products, barrier/global sum,
then `invW*z - dot(mean,x)*mean + diagonal*shift*x`.
No reference runtime is called from the Rust test.

Actual operands: rank0 real matrix [1 2;3 4], rank1 [5 6;7 8]; imaginary matrices
[.5 -1;1.5 .25] and [-.5 1.25;.75 -1.5]. Mean [.25,-.5], diagonal [2,3],
invW=.25, shift=.1. Root search [.5,-1] replaces non-root [999,999].
Real local products are [-8.5,-19.5] and [-52.5,-71.5], global [-61,-91].
Imaginary contributions [.125,-2.0625] and [3.15625,-3.9375] give complex global
[-57.71875,-97]. Dot(mean,x)=.625. C's final real product is
[-15.30625,-22.7375], complex [-14.4859375,-24.2375]. These literal calculations
are independent of Rust-generated fixtures.

Assertions inspect all four actual read-only product phases on both ranks and
the returned vector. Root search is an exact input/broadcast contract. Products
use absolute 1e-12, relative 0, for this short two-component/two-sample/two-rank
path (raw magnitude ≤97; approximately 45 binary64 unit roundoffs at that scale).
This conservative elementary-operation budget is not a CG forward tolerance,
conditioning allowance, or a sampling-trajectory gate.

## Actual bounded execution

Frozen snapshot `issue179-62b.qAZUvg` in container `73c57e563c61`; production
kernel SHA-256 `7df70d1074424dbe202276e54e2962574c174bf571b7d621446176f65254caac`.
New test copied separately without modifying existing frozen production sources.
Build handle **21763**, terminal **0**, using the named-volume target:

```bash
CARGO_TARGET_DIR=/home/vscode/.cache/mvmc/target/issue179-62b \
LIBCLANG_PATH=/usr/lib/llvm-18/lib OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
timeout --kill-after=5s 600s cargo test --locked --profile test-fast \
  -p mvmc-core --features mpi --test mpi_issue179_literal_operator \
  --no-run --message-format=json
```

Actual binary:
`/home/vscode/.cache/mvmc/target/issue179-62b/test-fast/deps/mpi_issue179_literal_operator-8b3ce31a4270b5cf`.
After exact-name listing preflight, launch:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 timeout --kill-after=5s 30s \
  mpiexec -n 2 "$binary" --ignored --exact \
  issue179_literal_two_rank_cg_products --nocapture </dev/null
```

Launch tool chunk **21c15c**, terminal **0**, one test passed on each rank.
All four `LITERAL_CG` records contain the specified root vector and expected
products. Evidence root:
`/home/vscode/.cache/mvmc/issue179-literal-operator.v11zzu`.
Files: `launch.log`, `source-binary-sha256.txt`, `source-binary-check.txt`,
`mpi-version.txt`, `binary-ldd.txt`. All four source/binary postchecks passed.
Actual runtime MPICH 4.2.0, repaired internal-Hydra container; BLAS LP64 OpenBLAS.

Test SHA-256 `40c5f6c53383412ed00f47a08ac014ae72c7e61094132c51e1cb071d3b21a831`;
binary `804de6442d9e7de5146f3935dd8621c9bb13695842810be961507bc3277760b4`;
launch log `24ffdaaf8b58ec696bc11405279e30c45666753aef33879cb6adfce712a94d00`.
Owned Rust file was rustfmt-formatted and its format check passed.

Parent independent review/rerun: exact listing identified one test (review
67f131); actual same binary under `mpiexec -n 2` rerun **39cd1b** terminated
**0**, each rank one test passed in .08 seconds, both real/complex products
and root input matched. This independently confirms only the literal operator
case, not broader MPI semantics or solver convergence. Expected retained hashes
remain binary `804de644...`, test `40c5f6c5...`, kernel `7df70d10...`.

Launch-policy and public root-output/message assertions remain separate follow-up
gates. Rust currently selects MPI by launcher detection plus the compile-time
feature, not Julia's `JULIA_MVMC_MPI` switches. Also, the current small-weight
average helper silently returns: the Julia root-only warning is not present and
cannot be claimed from this operator test or manufactured by a harness message.
