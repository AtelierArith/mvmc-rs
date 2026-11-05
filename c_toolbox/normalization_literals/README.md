# PhysCal normalization literals (#181, regenerated under 1.4.0 in #695)

Standalone kernel check, not a full `vmc.out` run. `driver.c` includes, verbatim,
`weightAverageReduce_fcmp` and `weightAverageReduce_real` from
`extern/mVMC-1.4.0/src/mVMC/average.c` lines 277-333 (SHA-256
`ccdaaa27bd0393e9bd73341459227555da798928fd1ee25b3e1151b6aecc54be`; the same text
as 1.3.0 lines 276-332) with size-one MPI/workspace stubs, and prints the
single-process result for weights 3, 10, 100 and inputs `.3-.7i`, `-.7+.3i`,
`13.486979846002933` as binary64 hex.

```sh
c_toolbox/normalization_literals/generate.sh   # rewrites the fixture
```

Linux x86_64, GCC 13.3.0 (Ubuntu 13.3.0-6ubuntu2~24.04.1), `-std=c11 -O3`. The
1.4.0 output is byte-identical to the former 1.3.0 acquisition. Rust tests read
only `crates/mvmc-core/tests/fixtures/normalization_c_literals.tsv`.
