# Grouped (`NSplitSize > 1`) native-C references (issue #349)

Native `vmc.out` (stock `extern/mVMC-1.3.0`, see `PROVENANCE.txt`) outputs at world
sizes 1/2/4 and `NSplitSize` 1/2/4 on the `physcal_181` inputs. Regenerate with
`c_toolbox/grouped_nsplit_349/generate.sh` (explicit developer command, Dev Container).
Layout: `<case>/c/r<world>s<split>/<file>`; `<case>/overrides.txt` holds the modpara
overrides applied by both the generator and the Rust test; `cases.txt` lists the cases.
Rust tests read only these files; they never run C.

C seeding rule: group `rank/NSplitSize` uses seed `RndSeed + group` (`vmcmain.c:239,257`).
Hence `r2s2`, `r4s4` equal the serial chain `r1s1`, and `r4s2` equals `r2s1` (seeds 0 and 1).

## Per-combination findings (C, 2 and 4 ranks)

| Combination | C result | Rust |
|---|---|---|
| FSZ PhysCal, `NQPFull` 1 and 2 | defined; grouped == ungrouped to <= 3.5e-15 (scaled) | accepted; matches C to <= 2.4e-11 |
| FSZ ParaOpt, `NQPFull` 2 | defined; grouped == ungrouped to <= 2.2e-10 (SR solve) | accepted; matches C to <= 7e-10 |
| Lanczos 1 and 2 PhysCal | defined; <= 7.9e-14 | accepted; matches C to <= 2.3e-13 |
| OptTrans `NQPOptTrans = 3` PhysCal (`-o`) | defined; <= 2.2e-15 | accepted; matches C to <= 1e-14 |
| OptTrans ParaOpt (`-o`) | defined; <= 9.2e-14 | accepted; grouped == serial Rust to <= 3e-13. Rust vs C differs at 1e-3 even ungrouped (OptTrans derivative layout, `docs/manual/11-compatibility.md` item 8) so only grouped-vs-serial Rust is asserted |
| SR-CG (`NSRCG != 0`) | **undefined**: step-1 parameters of `r2s2`/`r4s4` differ from `r1s1` by O(1e-3) in energy | rejected, message cites the C reason |

### The C defect behind the SR-CG rejection

`vmccal.c:241,248` writes `SROptO_Store[i + sample*SROptSize]` at the *global* sample index
`sample` in `[sampleStart, sampleEnd)`, but `calculateOO_Store[_real]` (`vmccal.c:314-318`,
`673-721`) reads the first `sampleEnd - sampleStart` columns from the base pointer. Ranks after
the first of a group therefore read unwritten `malloc` memory (`setmemory.c:419-421`) for
`OO` while `HO`/`O` include their samples. It affects every `NStore != 0` or `NSRCG != 0` run
with `NSplitSize > 1` (also direct SR with `NStore = 1`: `heisenberg_chain_real`, 4 steps, 40
samples, step-1 energy differs by 0.26 between `r1s1` and `r2s2`; with `NStore = 0` they agree to
4e-15). Rust computes the `NStore != 0` direct-SR grouped case correctly (grouped == serial to
4e-15) and rejects only SR-CG. See also `c_toolbox/mpi_issue179_store.md`.

## Tolerances (docs/NUMERICAL_COMPARISONS.md policy: no bitwise float comparison)

Scaled difference `|a-b|/(1+|b|)`, per file entry. Bounds are justified by the observed first
divergence (reduction-order roundoff, amplified by the SR solve for ParaOpt) and sit far below the
O(1e-2) a lost QP or sample slice would produce:

* C grouped vs C ungrouped (`c_grouped_fixtures_equal_c_ungrouped_chains`): PhysCal 1e-11, ParaOpt 1e-8.
* Rust grouped vs C, same world/split: PhysCal 1e-9, ParaOpt 1e-8.
* Rust one group vs serial Rust chain: PhysCal 1e-10, ParaOpt 1e-9.

RNG state and configurations are not compared across grouping: the group communicator changes
only reduction order, and the RNG stream per group is fixed by the seeding rule above.

## Running the MPI test

```sh
cargo test --locked -p mvmc-core --features mpi --profile test-fast --test grouped_nsplit_349 --no-run
MPI_ISSUE349_OUTPUT=/tmp/new-dir mpiexec -n 2 <binary> --ignored --exact \
  mpi_run::grouped_runs_match_c_fixtures_and_serial_chain --nocapture   # and -n 4
```
