# Independent complete DH24-real windows

Generated from captured unchanged source `/tmp/mvmc-history-source.TUyDyl`,
regular-file inventory digest
`943601528b575c63cab6d66938f659a74398b1ec6fa315829a88a93dba2c90a5`.
This digest names relative paths and file SHA-256 values; it excludes symlinks.
Observer process 6833 exited 0 with 12 original status assertions passing.

```sh
julia --project=extern/Julia-mVMC c_toolbox/runner_opt_windows.jl \
  /tmp/mvmc-runner-independent-dh24-real-frozen direct \
  --case=dh24_real --steps=1,2,3,50
cc -O0 -ffp-contract=off c_toolbox/ctest_opt_window.c -lm -o /tmp/probe
for prefix in 1 2 3 50; do
  /tmp/probe /tmp/mvmc-runner-independent-dh24-real-frozen/step-$prefix/c-window-input.txt \
    /tmp/mvmc-runner-independent-dh24-real-frozen/step-$prefix/zqp
done
```

Actual runtime: Julia 1.13.1, Linux x86_64, one-thread ILP64 OpenBLAS via
libblastrampoline; compiler GCC Ubuntu 13.3.0, `-O0 -ffp-contract=off -lm`.
C numerical bodies are verbatim `avevar.c` extraction; no C sampler/SR/MPI
execution is claimed. See `c_toolbox/ctest_opt_window.md` extraction boundaries.
All four newly observed next-624 RNG records and saved-configuration records
matched the archived `sr_direct/dh24_real_runner/step-{N}-{rng,configs}.txt`
exactly with `cmp`; original archives are not replaced.

Each prefix explicitly sets BOTH steps and window to N, seed=1, direct SR,
NStore=0, with the original nonzero DH2/DH4 overlay workload. `provenance.txt`
records the archived overlay-directory hashes, modpara and source hashes.
Each `c-window-input.txt` contains every successful synchronized post-SR
declared parameter vector and actual pre-SR Julia Etot/Etot2. NPara=35:
NProj=23 plus twelve Slater slots, including complete declared DH slots.
`zvo_var.dat` independently records PRE-SR declared slots, not Rust values.

Aggregator executable SHA-256:
`7a0434c1c7a377e8a84d31fa2a02499cd5cbb1007a83ff388b7e8ef932678d75`.
Adapter SHA-256:
`4069961fb468957de87760db9e1a25a1587351f4c92b17c3d3e925698904e652`.
Extraction SHA-256:
`47c801ea68d9bf40af967a3a0db6189125fd54ad146820b6470d63f9317a69da`.

Window=1 emits only the main contiguous paired row; larger windows emit
means/sample deviations and the active block files, including both DH sections.
Rust asserts the exact file manifest and row/column layout. Computed values
use the existing justified abs=rel=1e-11 policy; RNG/configuration contracts
remain exact. This fixture does not prove the other real/complex/FSZ workloads.
