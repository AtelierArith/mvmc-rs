# Native C OptTrans derivative fixture (#492)

`opttrans_derivative.txt` is the stdout of the optional standalone C probe
`c_toolbox/native_comparison_492/opttrans_derivative.c`. That program contains
verbatim `calculateOptTransDiff` from authoritative `vmccal.c:639-655`, with
explicit fixed weights, Pfaffians, overlap and sentinel output values. The
upstream source hash, probe hash, submodule commit, compiler and architecture
are in `provenance.txt`. Generated on Linux x86_64 with GCC 13.3, `-std=c11 -O3`;
no BLAS or Julia runtime is used in this standalone scalar probe.

Reproduction inside the Dev Container:

```sh
gcc -std=c11 -O3 c_toolbox/native_comparison_492/opttrans_derivative.c -o /tmp/opttrans492
/tmp/opttrans492 > tests/fixtures/native_comparison_492/opttrans_derivative.txt
```

The Rust test reads only this checked-in text. It does not compile or invoke C.
The six output rows each contain real and imaginary components. The first two
are sector derivatives in consecutive slots, rather than real/imaginary pairs;
the other four are untouched sentinel values. Computed components use explicit
absolute and relative `1e-14` bounds for the short complex arithmetic sequence.

`rbm_sr_system.json` and `rbm_state.txt` come from the optional full native
executable probe, not Rust. They capture the first pre-SR boundary on the
`physcal_181/hubbard_chain_dh_rbm_opttrans` inputs with `NVMCCalMode=0`,
`NLanczosMode=0`, `NVMCSample=1000`, one optimization step, one final averaging
step, no initial parameter file, `-o`, one MPI rank and one OpenMP/BLAS thread.
The JSON preserves C's column-major unfactored stabilized SR matrix, RHS and
57 active indices. The state includes exact RNG draw count and next 624 words.
The Rust regression independently runs those inputs and compares operands
with `1e-13` absolute/relative bounds; measured Linux operand divergence is
at most `2.1e-15`. Component indices and RNG are exact. This tests the first
sampling boundary, not an entire optimization trajectory.

Regenerate with `scripts/capture_native_c.py` using the probe described in
`c_toolbox/native_comparison_492/README.md`; copy its C-system JSON and C state
dump. Record probe/source/patch/binary hashes and the environment alongside
each new generation. Cargo never invokes this optional command.

`full_probe_provenance.json` identifies the C observation binary and original
sources; `patch_hashes.txt` identifies the applied instrumentation patches.
`environment.json` records the compiler, CPU, MPI and linked BLAS environment
from a neighboring benchmark run in the same container. Its benchmark binary
arguments describe that run; the full-probe binary hash is the one in
`full_probe_provenance.json`, not the benchmark binary in `environment.json`.
