# Issue #347: C `vmc.out -b` varbin layout probe

Full-executable validation (not a standalone kernel): builds the unmodified `extern/mVMC-1.4.0`
`vmc.out` and runs it with `-b` on the tiny Heisenberg-chain inputs to produce
`tests/fixtures/issue347_varbin/c_expected/`. The Rust build and tests never read this directory.

Origin: `extern/mVMC-1.4.0` (`e83aa6cb`, no patch), `src/mVMC/vmcmain.c` SHA-256
`da14885da14fd5b98feb27f7c53efd6fa229a4a33d27c30fe44848e6514394f0`, `src/mVMC/initfile.c`
`def6d4aa57f1ccdb238df0ea64b93ea7037681bc50da445c303b4b9c2b13f87e`. Extraction boundary: none
(whole executable). Compiler: gcc 13.3.0, MPICH, CMake Release,
`-DGIT_SUBMODULE_UPDATE=OFF -DTesting=OFF`.

Reproduction (Linux x86_64 Dev Container image, scratch directory `$S`; the shell scripts use
`/repo` for the repository and `/work` for `$S`):

```sh
S=/path/to/scratch; mkdir -p "$S"
cp c_toolbox/issue347_varbin/*.sh "$S"/ ; mv "$S/build_in_container.sh" "$S/build.sh"
docker run --rm -v "$PWD":/repo:ro -v "$S":/work <dev-container-image> bash /work/build.sh
# setup_inputs.sh prepares /work/opt and /work/phys from the physcal_181 fixture
# (including removing the Green-function keywords from opt/namelist.def):
S="$S" M="$PWD/tests/fixtures/physcal_181/two-samples/heisenberg_chain_real" bash "$S/setup_inputs.sh"
docker run --rm -v "$S":/work <dev-container-image> bash /work/run_in_container.sh
```

`run_in_container.sh` writes `/work/run/<case>/output/zvo_varbin_*.dat`; the copies checked into
the fixture directory are described in the fixture's `PROVENANCE.md`. The `opt` inputs have the
`OneBodyG`/`TwoBodyG`/`TwoBodyGEx` entries removed; `phys` keeps them.
