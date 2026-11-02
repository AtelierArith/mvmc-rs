# Issue #605: native C counter and SR diagnostics

This optional probe instruments a copy of the pinned mVMC C source; it is not a Rust build or test dependency. `probe.patch` adds state, SFMT, direct-SR matrix, CG iterate and post-sync parameter dumps. It preserves the sampler, its RNG state and solver arithmetic. The trace file contains potentially large records and is for these one-rank diagnostic inputs, not production use.

The recorded run used Linux x86_64, GCC 13.3.0 (`-O3 -DNDEBUG`), MPICH 4.2.0, OpenBLAS one thread, mVMC `d73d06bd529d3b2573f38eb5817c4a5f52971006`, and Julia reference `22bc829d3607e4aff860d5b0d1244932fac585fe`. The instrumented executable SHA-256 was `b99690af1a8d6afa57cc6e561f2059554f4035a11f7870bcc75cdf676b94c45c`; builds can differ by compiler and paths. The probe patch SHA-256 is `2231b1b73e425feb4439f5c946d28d7fed041dd117c904020c0e05da53137ba3`.

Unmodified source SHA-256:

- `src/sfmt/SFMT.c`: `b61f0f0239193fd4c06f2879e6e8452225bec45b3bca426902c244fd31ad4e3a`
- `src/mVMC/vmcmain.c`: `fdcd661c4eb028786fc58ae5e1f5232426c943b547f95a0d74e888e602256d63`
- `src/mVMC/stcopt_cg_impl.c`: `41452de5fe766409431c6e1cf73cd2b12485faaeb4bfbcdec1e53444e9af1cf9`
- `src/mVMC/stcopt_dposv.c`: `2bd48d880dcbd95ea1b1b92931178c07fd08f981b8ebf04e7db1709e909e57ca`

Run inside the Linux development container with the reference submodules initialized:

```sh
repo="$PWD"
probe="$(mktemp -d)"
cp -a "$repo/extern/mVMC-1.3.0" "$probe/src"
patch -d "$probe/src" -p1 < "$repo/c_toolbox/issue_605/probe.patch"
mkdir "$probe/build"
cd "$probe/build"
CC=gcc CXX=g++ FC=gfortran cmake ../src -DCMAKE_BUILD_TYPE=Release \
  -DGIT_SUBMODULE_UPDATE=OFF -DTesting=OFF -DMPI_C_COMPILER=/opt/mpich/bin/mpicc
make -j1 vmc.out
for model in heisenberg_chain_real heisenberg_chain_cmp heisenberg_chain_fsz hubbard_chain_real general_rbm_cmp; do
  mkdir "$probe/$model"
  cp -a "$repo/extern/Julia-mVMC/test/integration/reference/$model/inputs/." "$probe/$model/"
  sed -i -E 's/^(NSROptItrStep|NSROptItrSmp)[[:space:]]+[0-9]+/\1  1/' "$probe/$model/modpara.def"
  if [ "$model" = general_rbm_cmp ]; then
    sed -i -E 's/^NSRCG[[:space:]]+[0-9]+/NSRCG 1/; s/^NStore[[:space:]]+[0-9]+/NStore 0/' "$probe/$model/modpara.def"
  fi
  (
    cd "$probe/$model"
    if [ -f initial.def ]; then
      OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /opt/mpich/bin/mpirun -n 1 "$probe/build/src/mVMC/vmc.out" -e namelist.def initial.def > run.log 2>&1
    else
      OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /opt/mpich/bin/mpirun -n 1 "$probe/build/src/mVMC/vmc.out" -e namelist.def > run.log 2>&1
    fi
  )
done
printf 'probe outputs: %s\n' "$probe"
```

The `initial.def` argument matters: C does not load that file merely because it exists. `state_dump_000.txt` contains the six native C counters and a non-mutating snapshot of the next 624 SFMT outputs. Compare them to the checked-in historical reference only when the initialization and inputs match. `sr_dump.txt`, `solver_dump.txt`, `cg_dump.txt`, `cg_trace.txt` and `para_dump.txt` expose the numerical boundaries. SFMT draw count is an instrumentation count of `gen_rand32` calls, not a replacement for comparing the state and control path.

For the real Heisenberg direct solve, the 10x10 matrix condition number was `1.56498e7`, the C normalized backward residual `3.30e-17`, and a solve of the independently captured Julia matrix differed from C by up to `5.33e-10`. For general-RBM CG, the 194x194 operator condition number was `4.40e5`; C, Rust and reconstructed Julia relative L2 residuals after the C default 194-iteration limit were respectively `6.55e-5`, `5.57e-5` and `3.56e-5`. The maximum observed Rust/Julia final parameter difference was `5.44e-5`. The first five Rust/C iterate differences were at most `5.53e-17`. These measurements motivate case-specific bounds; they do not justify changing the C iteration limit or a general tolerance.

## FSZ platform diagnostic

The [additional FSZ probe](fsz-diagnostic.patch) instruments `src/mVMC/vmccal_fsz.c` (upstream SHA-256 `9ac529ef2a18d80f61c5aca92aacce10391e2794ecc56aa8ca947691dbdd974e`) after energy and derivative calculation. Its SHA-256 is `31de414b7cc5c4553b2b4e1fb9e910da34467c10077c9960e9b1d160359fbe19`. Apply it to the already patched C copy, rebuild, then run the `heisenberg_chain_fsz` command above with `initial.def`; the resulting instrumented binary used for the recorded C values had SHA-256 `c0d14cfcaa41480b6b32bcb6639d31bc8d934364d86ccb824a69c34b9b118cf3`. The [C Linux records](fsz-c-linux.log.gz), [Rust Linux diagnostic records](fsz-rust-linux.log.gz), and [Rust macOS ARM64 CI diagnostic records](fsz-rust-macos.log.gz) contain the 200 sample indices and six real values (overlap, local energy, derivative component 8), not golden test fixtures. The macOS records are from [CI job 125160132656](https://github.com/AtelierArith/mvmc-rs/actions/runs/41409727547/job/125160132656), commit `7c2281e1e727dbdfa1f212e37cc25a0776bb7f7d`; its actual saved configuration/counter rows and next 624 SFMT outputs matched the historical reference exactly. To recalculate the recorded differences, run:

```sh
uv run --no-project python c_toolbox/issue_605/analyze_fsz_diagnostic.py
```

Mac/Linux overlaps agree in every printed digit; the first local-energy divergence is sample 0, `4.44e-16 - 2.22e-15i`. The largest occurs at sample 59, `7.91e-11 + 3.34e-11i`, where the derivative is approximately `-216.256 - 19.190i`. Summing the observed differences of energy times derivative over the 200 samples reproduces the observed normalized HO component-8 real difference (`-1.091067e-10` reconstructed versus `-1.091109e-10` observed); sample 59 accounts for `-8.236043e-11`. The complex component-8 real part is flattened index 16. A subsequent macOS CI diagnostic recorded all 100 flattened HO components: 32 exceeded the previous bound and the maximum absolute difference was `1.28653e-10` at indices 57 and 58. Sample 59 has large derivatives in several components (up to about 242 in magnitude), consistent with the same amplification. The recorded full row, derivatives and parameter/energy comparisons are in `fsz-rust-macos-full.log.gz` from job 125164009488 at commit `8d17b0496bf56ae42a5b9cb4caccecb20bbf15ce`. An absolute bound of `3e-10` for this FSZ HO row is about 2.3 times the largest observed difference; FSZ OO and other model rows retain the existing bound. All FSZ post-SR parameter and energy components remained within their earlier bounds. The independent C/Linux sample energies differ from Linux Rust by up to `6.87e-11` real and `4.23e-11` imaginary, corroborating this sensitive path rather than supplying Rust-generated expected values.

## CG platform diagnostic

For the same general-RBM input and C default 194-iteration limit, macOS ARM64's saved configurations, counters and next 624 SFMT outputs matched the historical reference exactly; its full normalized pre-SR OO/HO arrays and first five CG solution vectors passed their unchanged independent checks. Across the 194 recorded active components, maximum Linux/macOS absolute differences were `3.20e-17` for the gradient, `2.23e-16` for the mean, `4.45e-16` for the variance and `1.597e-5` for the iteration-limited solution. The largest final parameter difference against the reviewed Julia reference was `9.914e-6`, below that test's case-specific `1e-4` absolute bound.

The test's independently recomputed sampled-operator relative L2 residuals were `5.57e-5` on Linux and `1.37642e-4` on macOS, compared with `6.55e-5` for C. Using the independently captured C sample matrices with the observed Rust gradient, mean, variance and final solution reconstructs `5.571449e-5` and `1.376413e-4`, respectively; these mixed-input checks closely reproduce, but are not identical to, each platform's own-operator residual. The `3e-4` residual ceiling for this one fixed input is about 2.18 times the largest observed residual. It is not a convergence guarantee or a general numerical tolerance, and the early C-iteration, exact discrete and independent pre-SR checks remain in place.

[C sampled matrices](cg-c-linux.log.gz), [Linux Rust vectors](cg-rust-linux.log.gz), and [macOS Rust vectors](cg-rust-macos.log.gz) are diagnostic records rather than test-generated fixtures. The macOS records came from [CI job 125171250046](https://github.com/AtelierArith/mvmc-rs/actions/runs/41413354732/job/125171250046), commit `7ce910958a5681c2224174fd91d947afeeedd819`. Recompute the vector differences and C-matrix cross-checks using:

```sh
uv run --no-project python c_toolbox/issue_605/analyze_cg_diagnostic.py
```
