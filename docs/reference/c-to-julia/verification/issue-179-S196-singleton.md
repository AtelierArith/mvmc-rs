# S196/M0998 accepted singleton: actual grouped public PhysCal

Owned test `mpi_issue179_physcal_singleton.rs` sends singleton complex storage
`opt_trans=[1+0i]`, NQPOptTrans1, split2/Lanczos0/general0 through actual
`vmc_phys_cal_with_reducer`. Pinned Julia S196/M0998's `[[0]]` is one-site
validator metadata; the independently fixtured six-site normal spin-chain runtime
uses the equivalent complete single identity map `[0,1,2,3,4,5]`. This adaptation
is explicit, not a claim that a one-element map can sample a six-site model.

Fixture: checked-in `physcal_181/heisenberg_chain_real` inputs and fixed
`zqp_opt.dat`, no reference executable invoked by Cargo. Overrides: NSP1/NMP1,
NVMCSample3, warmup1, one PhysCal iteration, seed1 plus actual group offset,
world2/4 width2, requested workers1/2/4, inner threshold1, BLAS/OMP threads1.
The two public runs in each configuration verify exact initial primitive SFMT
stream versus the corresponding seed, exact final next624/draw count/configuration/
counters, finite energy, and numerical repeat fields at abs1e-12+rel1e-12.
Green index/layout fields remain exact; computed fields are not compared bitwise.
Independent local-spin occupancy sum n(0,up)+n(0,down)=1 uses abs1e-12, as does
its zero imaginary component. Global weight is exactly 3 samples per chain.
These are focused repeatability/kinematic checks, not full independent energy
oracles or all #179/#184 acceptance criteria.

## Corrected terminal evidence

Build handle **90310**, terminal **0**; local output-checker regression passed.
Actual bounded MPI tool chunk **98d22d**, terminal **0**, six rows all launcher0:
world2/4 × workers1/2/4, each executing two public PhysCal runs. No zero-test gate:
the exact ignored name was checked in the executable listing before launch.
Each MPI launch uses `timeout --kill-after=5s 60s`.
Evidence container `73c57e563c61`, root
`/home/vscode/.cache/mvmc/issue179-S196-indexed.jIJsuw`.
Frozen production remains `issue179-62b.qAZUvg`, NOT post-integration main.

Retained `binary`, `test.rs`, `source-binary-sha256.txt` and postcheck,
`fixture-sha256.txt` and postcheck, `results.tsv`, `w<world>-t<workers>.log`, and
both repeat output directories. All four binary/test/production postchecks and
fixture checks passed. Actual binary SHA-256:
`27bb4706fd90ad217d6ebd2daa722de1172ce39506bea537d79469785a221551`;
test `7dc64dab3d35dd35a7b70dc7eff6af5c61c43025f2c87c8454a48046c1217dd2`;
results TSV `aeadd751eb2ba2afc7770d724132d31be2b959613922db2387c9672ff06dd2f7`.
Production run.rs SHA-256 `d2bbeb6ee0579dc886e5949f4021b47b85c662c3ac4f75fb937d00d7142faf05`.

World4/worker4 actual observation shows nonzero parallel calls/entries on all
ranks, observed IDs [0,1,2,3], not merely requested pool capacity. Worker1 records
the serial path. World2 density=1 and energy=-2.567340080161576; world4 density=1
and energy=-2.3412032957346662, with independent chain seeds1/2. Energies are
reported observations, not newly authored reference expectations.

Reproduction in the frozen snapshot after compiling the named ignored test:

```bash
MPI179_SINGLETON_OUTPUT="$new_output" MVMC_RS_INNER_THREADS="$workers" \
MVMC_RS_INNER_THRESHOLD=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
timeout --kill-after=5s 60s mpiexec -n "$world" "$retained_binary" \
 --ignored --exact issue179_grouped_physcal_singleton_is_accepted_and_repeatable \
 --nocapture </dev/null
```

## Failed attempts remain failed

Original handle **4573**, terminal **1**, root `issue179-S196-singleton.7eebgQ`:
six launcher124 timeouts after the test's root asserted unindexed zvo_out/var
filenames, while production correctly wrote indexed filenames. The failed test
and binary were retained before correction (`failed-test.rs`, `failed-binary`).
This is a harness defect, not evidence of production deadlock or a successful gate.
Second root `issue179-S196-corrected.gdKdE9` also exited1: six launcher101 results
after coordinated output validation wrongly rejected the writer's blank separator
rows. The corrected checker now preserves exact blank-row layout and rejects
missing rows, wrong indices, empty-only output, excess errors and nonfinite values.
Neither failed capture is retroactively relabelled PASS.
