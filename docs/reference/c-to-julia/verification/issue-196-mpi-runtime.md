# Issue #196: internal-PMI/Hydra runtime repair

Related to #179, #186 and #190. This optional developer check proves a real MPI
world, not numerical/sampling parity. Normal Cargo tests do not invoke C/Julia
or the C toolbox. The Rust diagnostic is pure Rust, ignored unless explicitly
launched with MPI; the verifier fails if the requested test is absent.

## Diagnosis and pinned build

The Ubuntu MPICH 4.2.0 package used external PMIx with Hydra: `mpiexec -n 2`
set `PMI_SIZE=2`, but independent C and Rust MPI initialization reported two
singleton rank-0 worlds. This was not a mixed Open MPI launcher. Upstream
[MPICH issue 7064](https://github.com/pmodels/mpich/issues/7064) identifies
Hydra/external-PMIx incompatibility and recommends internal PMI for Hydra.

The Dockerfile retains MPICH 4.2.0 and `ch4:ucx`, building the official source:

- Origin: https://www.mpich.org/static/downloads/4.2.0/mpich-4.2.0.tar.gz
- SHA-256: `a64a66781b9e5312ad052d32689e23252f745b27ee8818ac2ac0c8209bc0b90e`
- Compilers: `CC=gcc CXX=g++ FC=gfortran`; `make -j4`.
- Configure: `--prefix=/opt/mpich --with-device=ch4:ucx --with-ucx=/usr --with-pm=hydra --with-pmi=pmi1 --without-pmix`.

PATH/MPICC/library/pkg-config paths select `/opt/mpich` explicitly. Existing
containers and their named volumes must not be deleted while checks are live.
Julia's MPICH_jll launcher/ABI is separate; do not launch it with this system MPI.

## Reproduce in the current devcontainer

Follow [DEV_CONTAINER.md](../../../DEV_CONTAINER.md), using a unique instance
label so an older live environment is preserved:

```sh
devcontainer up --workspace-folder "$PWD" --id-label mvmc.issue196=internal-pmi-reference
docker inspect <new-container-id> --format '{{.Image}}'
```

Inside that container, at its workspace root:

```sh
MPI196_IMAGE_ID=sha256:<actual-inspected-image> \
MPI196_TARGET_DIR=/home/vscode/.cache/mvmc/target/issue196-world \
bash scripts/verify_mpi_issue196_runtime.sh
```

The C compiler command is `mpicc -std=c11 -O2 -Wall -Wextra -Werror` on the
original diagnostic `c_toolbox/mpi_issue196_world.c` (not extracted upstream
numerics). Both C and Rust must report every unique rank in worlds 2/4, sums
3/10 and payload 196 broadcast from the last global rank. Launches have a
30-second timeout with a 5-second kill grace; the cold Rust build is bounded at
1800 seconds. The evidence directory records actual image/runtime/compiler/
OpenBLAS versions, loaded libraries, pinned submodules, executable hashes and
tracked plus uncommitted source hashes, checked again after execution.

## Coverage and acceptance

Isolated current-main validation completed on base
`56c191dadc69f0e2c1dfc4627d0ae4271e7a09db`, with the five scoped runtime-repair
draft paths and pinned submodules, in container `be641a739469`, actual image
`sha256:62853adfe3779a3239c43af5fd261b5ea2d73dcde2fe05c462b4b2d60064cd8f`.
Cargo targets and caches used named volumes, not host targets.

- Latest runtime verifier: terminal 0, evidence
  `/home/vscode/.cache/mvmc/issue196-world.V6XPLH`; C and Rust each passed worlds
  2/4, integer sums and last-rank broadcast. `source-check.txt` confirms the
  captured uncommitted code did not change during execution.
  Source-manifest SHA-256:
  `fbe12611b14a45df272a9091544a47b8aac7befc85fbc013699f757970df41c8`.
- Full workspace nextest, `--cargo-profile test-fast --locked --no-fail-fast --retries 0`:
  terminal 0, **549 passed, 19 skipped**, 109.728 seconds. Skipped tests are not
  counted as runtime coverage; the ignored world diagnostic was explicitly run
  by the verifier. Log: `/home/vscode/.cache/mvmc/issue196-workspace-final-base.log`.
- Workspace all-target/all-feature strict Clippy and `cargo fmt --all --check`:
  terminal 0. All-feature locked workspace doctests: terminal 0, zero doctests
  present; log `/home/vscode/.cache/mvmc/issue196-doctests-final-base.log`.

The actual image has `/usr/bin/jq` (Ubuntu package
`1.7.1-3ubuntu0.24.04.1`). The base image's
[common-utils dependency list](https://github.com/devcontainers/features/blob/main/src/common-utils/main.sh)
includes jq. The verifier additionally checks its presence before building and
records its path/version; an absent dependency fails explicitly. No additional
apt installation or image rebuild was needed for jq.

Separately, repaired image
`sha256:28023cf1c31b9854391c41a4af428fada275f9d4883f85120506ffb25b1c51c9`
passed C/Rust 2/4-rank mapping in a #179 draft snapshot; that result must not be
presented as current-main numerical validation. #196 remains open pending its
#179 independent-state follow-on acceptance. #179 requires additional grouped
sampling, exact discrete trajectories and justified portable numerical checks;
runtime smoke cannot satisfy those criteria.
