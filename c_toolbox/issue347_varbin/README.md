# Issue #347: C `vmc.out -b` varbin layout probe

Full-executable validation (not a standalone kernel): builds the unmodified `extern/mVMC-1.3.0`
`vmc.out` and runs it with `-b` on the tiny Heisenberg-chain inputs to produce
`tests/fixtures/issue347_varbin/c_expected/`. The Rust build and tests never read this directory.

Origin: `extern/mVMC-1.3.0` (no patch), `src/mVMC/vmcmain.c` SHA-256
`fdcd661c4eb028786fc58ae5e1f5232426c943b547f95a0d74e888e602256d63`, `src/mVMC/initfile.c`
`8a7b20d54ab495cfac4df42646a30102aebaa228311ca3b17c1e29af5d0581f8`. Extraction boundary: none
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
