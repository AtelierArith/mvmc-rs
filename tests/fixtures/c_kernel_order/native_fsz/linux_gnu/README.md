# Linux native C energy / Julia runner references

These independent Linux x86_64 GNU references retain Julia 1.13.1's
initialization, SFMT draws, sampler, SR and formatted output, while the local
FSZ energy executes the extracted serial C kernel. Every borrowed projection
counter is independently checked by C `MakeProjCnt`. This is not a complete
native C executable, native C sampler/SR or MPI trajectory comparison.

The seven no-RBM families (`fsz`, `interall`, `pairhop_fsz`, `opt_fsz`,
`dh2_fsz`, `dh4_fsz`, `dh24_fsz`) cover direct SR, stored-O direct SR and CG
at prefixes 1, 2, 3 and 50. DH post-sync history boundaries are included.
Rust verifies configurations, spins, projection counts, burn-in, counters and
the full 624-word RNG block before numerical values; parameters, energy, CG
iterations/status/residual output and formatted output remain exact.

The native C energy changes the OptTrans FSZ CG trajectory. Both the historical
and C-adapted runs complete 50 steps successfully, but their final configuration
and RNG block differ. Both changed control files are stored explicitly, and
Rust matches this independently generated C-adapted trajectory exactly.

`SHA256.json` records all 714 full independently generated outputs. The parent
`inheritance.tsv` explicitly maps 195 byte-identical Linux files to preserved
archives, with verified SHA-256 values. The parent `unused-output-sha256.tsv`
represents only 18 large fixed-input/Gram snapshots which Cargo never reads;
read-only Julia reproduction computes and compares their complete SHA-256.
OptTrans direct fixed inputs and stored Gram remain full files. There is no
alternate for missing numerical or RNG expectations. Mac mappings remain
separate and are preserved. `provenance.json` records the actual compiler,
bridge, Julia/BLAS providers, pinned manifest and source hashes.

From a Linux Dev Container, build the independent bridge and reproduce the
sparse expectations without writing them:

```sh
uv run --no-project python scripts/check_native_fsz_runner_bridge.py --build-dir /tmp/native-fsz-bridge
julia +1.13.1 --project=extern/Julia-mVMC scripts/regenerate_native_fsz_runner_fixtures.jl --bridge-dir=/tmp/native-fsz-bridge --output-root=tests/fixtures/c_kernel_order/native_fsz/linux_gnu --steps=1,2,3,50
julia +1.13.1 --project=extern/Julia-mVMC scripts/regenerate_native_fsz_dh_boundaries.jl --bridge-dir=/tmp/native-fsz-bridge --output-root=tests/fixtures/c_kernel_order/native_fsz/linux_gnu
```

Use `--write` only in an independent reviewable staging checkout to materialize
full generated outputs. Ordinary Cargo builds and tests invoke no bridge,
Julia runtime or toolbox.
