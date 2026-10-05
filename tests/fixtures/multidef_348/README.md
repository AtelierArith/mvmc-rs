# C MultiDef mode (`vmc.out -m N`) references (issue #348)

* `split_c_probe.txt`: the colour of `initMultiDefMode` (`vmcmain.c:752-759`, statements
  copied verbatim) for every `1 <= N <= size <= 48` and rank. Standalone kernel check
  (no MPI). Provenance: `split_PROVENANCE.txt`; regenerate with
  `c_toolbox/multidef_348/generate_split.sh`.
* `c/r<W>/{a,b}/`: numerical outputs of the stock native C `vmc.out`
  (`mpiexec -n W vmc.out -m 2 dirs.txt namelist.def zqp_opt.dat`, W = 2, 3, 4; full
  executable + MPI run). `a` = `physcal_181/heisenberg_chain_real`, `b` =
  `physcal_181/hubbard_chain_real`, both with `NSplitSize 1`, `NVMCCalMode 1`.
  Provenance: `PROVENANCE.txt`; regenerate with `c_toolbox/multidef_348/generate_c_runs.sh`
  (Dev Container).
* `c_messages/`: C stderr of the successful runs (`r3` holds the load-imbalance warning)
  and the first stderr line plus exit status of the error cases.

Rust tests read only these files (`crates/mvmc-core/tests/multidef_348_split.rs`,
`crates/mvmc-cli/tests/issue348_multidef.rs`); they never run C.

## C defects found (not reproduced in Rust)

1. `initMultiDefMode` divides by `nMultiDef` unchecked (`vmcmain.c:752-753`): `-m 0` raises
   `SIGFPE`, and a negative `N` gives an invalid `MPI_Comm_split` colour. Rust rejects
   `N <= 0` (`error: -m: N should be a positive integer.`).
2. A rank-0 failure is not collective. `info` is set only on rank 0 (`:769-782`); other ranks
   continue into `MPI_Scatter` of the uninitialized `dirNameList` (`malloc`, `:767`) and
   `chdir()` to garbage (observed: `error: chdir(): <junk bytes>: No such file or directory`)
   because `MPI_Abort` is asynchronous in MPICH; the aborting run also segfaults in
   follow-up code. The stderr beyond the first line is therefore not a fixture. `dirNameList`
   is never freed (`:767`) either.
3. Only the group's rank 0 (`group2 == 0`, `:788`) changes directory; the other ranks of the
   group stay in the launch directory. This is harmless in C because only rank 0 reads files
   and writes output, but any per-rank file access (all Rust ranks parse the definition
   files) would use the wrong directory. Rust changes directory on every rank.
4. `-e`/`-s` after `-m` silently disable MultiDef (`:151-157`), so `-m 2 -e dirs namelist`
   reads `dirs` as the namelist. Ported as is.
5. The seed is `RndSeed + group1` with `group1 = rank0 / NSplitSize` inside the group
   communicator (`:239,257`), not the MultiDef group index: two groups running the same
   input at the same width produce identical chains. Ported as is.
