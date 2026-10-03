# Apple Silicon reviewed-CG references (BLAS bridge + archived AVX2)

Optional developer procedure. Cargo neither builds nor runs any of it. It
regenerates the `reviewed_cg_62b` CG runner references for macOS aarch64 so the
Rust CG prefix comparisons resolve through the platform overlay instead of the
archived Linux lineage. It writes only into an external stage directory; the
operator imports the results explicitly afterwards.

## Why the bridge and the archived AVX2 replay are required

The reviewed references are generated from the published PR54 fork commit
`62b0f97f076fb55c71c3ab0caa041a9adff94e04` (C-faithful CG recurrence). The SR
normal equations are ill-conditioned (`sDiagMin=0.0` for the canonical case),
so the CG solve amplifies last-bit GEMV/reduction rounding.

A plain native Julia aarch64 run is not a portable ARM expectation by itself:
its OpenBLAS kernels and vectorized reductions differ from the ones Rust uses,
and the amplified difference reaches ~`1e-6`, far beyond the `1e-11` prefix
budget. Measured for `real` CG step 1:

| Reference generation | step 1 `parameters[4]` | Rust (`-1.28155540745630714e0`) |
| --- | --- | --- |
| Archived Linux reviewed | `-1.28155444378483474e0` | error `9.64e-7` (fails) |
| Native Julia aarch64 | `-1.28155788902414236e0` | error `2.48e-6` (fails) |
| Julia aarch64 + bridge + archived AVX2 | matches | passes at `1e-11` |

Two independent fixes make the ARM reference agree with Rust:

1. **ABI-only LP64 system OpenBLAS bridge** (`scripts/reference_lp64_blas.jl`,
   `c_toolbox/blas_lp64_reference.c`). It forwards Julia's `dgemv`, `dpotrf`
   and related routines to the same system OpenBLAS library that Rust links.
   It changes only the integer ABI (ILP64 → checked LP64); no numerical routine
   is replaced.
2. **Archived Julia AVX2 two-hop bilinear replay**
   (`scripts/reference_archived_avx2.jl`). It restores the reduction tree that
   the Rust sampling kernels deliberately retain, instead of letting LLVM
   vectorize the aarch64 reduction differently.

`c_toolbox/runner_opt_windows_reviewed_arm.jl` installs both and then runs the
unchanged reviewed-fork generator.

## Prerequisites

- macOS aarch64 with Julia `1.13.1` (`julia +1.13.1`).
- Homebrew OpenBLAS; `PKG_CONFIG_PATH="$(brew --prefix openblas)/lib/pkgconfig"`
  and, when running the Rust tests, `OPENBLAS_CORETYPE=NEOVERSEN1`
  (`openblas_get_corename()` must match the overlay directory name).
- The reviewed fork fetched into `extern/Julia-mVMC`:
  ```sh
  git -C extern/Julia-mVMC fetch origin refs/pull/54/head:refs/remotes/origin/pr54
  git -C extern/Julia-mVMC checkout 62b0f97f076fb55c71c3ab0caa041a9adff94e04
  ```
- `Manifest-v1.13.toml`. Commit `62b0f97f` predates it; the recorded reviewed
  acquisition used the content added later at `8bb1b9e`, SHA256
  `09ebd06dab244510094b99fe7c6efa2fe7a3d22221d1336b951123a5a6e8befc`:
  ```sh
  git -C extern/Julia-mVMC show 8bb1b9e:Manifest-v1.13.toml \
    > extern/Julia-mVMC/Manifest-v1.13.toml
  ```
- A scratch directory (never inside the repository for the generated stages):
  ```sh
  export WORK=/tmp/mvmc-arm-reviewed
  mkdir -p "$WORK"
  ```
- `jq`/`shasum` (macOS), and the checked-in probe fixtures used by
  `scripts/reference_archived_avx2.jl` (`tests/fixtures/pfaffian_cg/two_hop_bilinear.txt`).

Restore `extern/Julia-mVMC` to its recorded submodule HEAD
(`8bb1b9e8ae47b1512c00b321be05664ddcac0fd1`) and delete the temporary
`Manifest-v1.13.toml` after generation.

## Build the two host libraries

Observed-SFMT-state library (see `reviewed_sfmt_state.md` for the contract):

```sh
clang -std=gnu11 -O3 -dynamiclib -fPIC -DMEXP=19937 \
  -I extern/Julia-mVMC/SFMT.jl/deps/sfmt \
  c_toolbox/reviewed_sfmt_state.c \
  extern/Julia-mVMC/SFMT.jl/deps/sfmt/SFMT-real.c \
  -o "$WORK/libsfmt-observed.dylib"
```

LP64 system OpenBLAS bridge:

```sh
clang -O0 -ffp-contract=off -dynamiclib \
  -L"$(brew --prefix openblas)/lib" -lopenblas \
  c_toolbox/blas_lp64_reference.c -o "$WORK/libblas.dylib"
```

Reference hashes used for the first validated ARM overlay
(`NEOVERSEN1`, macOS 27, Rust `1.98.1`):

| Artifact | SHA256 |
| --- | --- |
| `c_toolbox/reviewed_sfmt_state.jl` (observer API) | `91102d67e9674c3bbc3fb1ceca8b4dbfa32c2440219a8b4e48e2a70313ec2591` |
| `libsfmt-observed.dylib` | `8607569ed79ddf971acb6106c71f52421cd07d494765facb6f7fc34a89308ceb` |
| `libblas.dylib` | `5bf96a5ae6e72aeeb250db97ea46c46315ecca02a91bd4eb02a3bd5570adcc76` |

These hash the locally built binaries; they are provenance, not fixed review
inputs. Record the actual hashes in the generated `provenance.txt`.

## Build the reviewed source manifest

The reviewed generator requires a SHA256 manifest of every production `.jl`
and environment file, rooted at `extern/Julia-mVMC`:

```sh
R=extern/Julia-mVMC
{
  for d in MVMCOptimizers.jl/src MVMCExpertModeParsers.jl/src SFMT.jl/src PfaPack.jl/src; do
    find "$R/$d" -name '*.jl' | sort | while read -r f; do
      printf '%s %s\n' "$(shasum -a 256 "$f" | cut -d' ' -f1)" "${f#$R/}"
    done
  done
  for f in Project.toml Manifest-v1.13.toml MVMCOptimizers.jl/Project.toml \
           MVMCExpertModeParsers.jl/Project.toml SFMT.jl/Project.toml PfaPack.jl/Project.toml; do
    printf '%s %s\n' "$(shasum -a 256 "$R/$f" | cut -d' ' -f1)" "$f"
  done
} > "$WORK/reviewed-manifest.txt"
```

## Generate a case

The driver installs the bridge and archived AVX2, then forwards every other
argument to `runner_opt_windows_reviewed.jl`:

```sh
export OPENBLAS_CORETYPE=NEOVERSEN1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1

julia +1.13.1 --project=extern/Julia-mVMC \
  c_toolbox/runner_opt_windows_reviewed_arm.jl \
  "$WORK/stage-real" cg \
  --blas-bridge="$WORK/libblas.dylib" \
  --reviewed-commit=62b0f97f076fb55c71c3ab0caa041a9adff94e04 \
  --reviewed-manifest="$WORK/reviewed-manifest.txt" \
  --sfmt-state-library="$WORK/libsfmt-observed.dylib" \
  --sfmt-state-library-sha256="$(shasum -a 256 "$WORK/libsfmt-observed.dylib" | cut -d' ' -f1)" \
  --sfmt-state-observer-sha256=91102d67e9674c3bbc3fb1ceca8b4dbfa32c2440219a8b4e48e2a70313ec2591 \
  --case=real --steps=1,2,3,20
```

Case names and prefix lists follow the reviewed generator
(`scripts/check_sr_cg_runner_parity.jl`). The canonical case additionally needs
`--c-kernel-order`:

```sh
julia +1.13.1 --project=extern/Julia-mVMC \
  c_toolbox/runner_opt_windows_reviewed_arm.jl \
  "$WORK/stage-canonical" cg \
  --blas-bridge="$WORK/libblas.dylib" \
  --reviewed-commit=62b0f97f076fb55c71c3ab0caa041a9adff94e04 \
  --reviewed-manifest="$WORK/reviewed-manifest.txt" \
  --sfmt-state-library="$WORK/libsfmt-observed.dylib" \
  --sfmt-state-library-sha256="$(shasum -a 256 "$WORK/libsfmt-observed.dylib" | cut -d' ' -f1)" \
  --sfmt-state-observer-sha256=91102d67e9674c3bbc3fb1ceca8b4dbfa32c2440219a8b4e48e2a70313ec2591 \
  --case=rbm_reference_cmp --c-kernel-order --steps=1,2,3,20
```

Each stage directory receives `step-N-{parameters,energy,configs,rng,SRinfo}.txt`,
`step-N/` (`c-window-input.txt`, `provenance.txt`, `zvo_var.dat`) and, for the
load-bearing DH/RBM/OptTrans cases, top-level `step-N-zqp_*.dat` and
`step-N-zvo_out.dat`.

FSZ-family cases (`fsz`, `pairhop_fsz`, `dh*_fsz`, `rbm_fsz`, `opt_fsz`,
`interall`) additionally need `--native-fsz-bridge=DIR` from
`scripts/check_native_fsz_runner_bridge.py`; see `runner_opt_windows_reviewed.md`.

## Import into the platform overlay

Rust resolves reviewed CG reads through `julia_fixture::fixture_path`, so an
Apple Silicon overlay under
`tests/fixtures/macos_arm_julia/<openblas-core>/reviewed_cg_62b/<case>/`
overrides the base `reviewed_cg_62b/<case>/` per file. Copy the numeric files
(`configs`/`rng` may fall back, but keeping them makes the overlay
self-contained):

```sh
CORE=neoversen1
SRC="$WORK/stage-real"
DST="tests/fixtures/macos_arm_julia/$CORE/reviewed_cg_62b/real"
mkdir -p "$DST"
for n in 1 2 3 20; do
  cp "$SRC/step-$n-parameters.txt" "$SRC/step-$n-energy.txt" \
     "$SRC/step-$n-configs.txt" "$SRC/step-$n-rng.txt" \
     "$SRC/step-$n-SRinfo.txt" "$DST/"
  mkdir -p "$DST/step-$n"
  cp "$SRC/step-$n/c-window-input.txt" "$SRC/step-$n/provenance.txt" \
     "$SRC/step-$n/zvo_var.dat" "$DST/step-$n/"
done
```

The declared-window `step-N/zqp_*.dat` files are not produced by the reviewed
generator. Generate them from the ARM `step-N/c-window-input.txt` with the
checked-in C adapter (`ctest_opt_window.c`; see `ctest_opt_window.md`), which
runs the verbatim upstream `OutputOptData` bodies:

```sh
cc -O0 -ffp-contract=off c_toolbox/ctest_opt_window.c -lm -o "$WORK/probe"
for n in 1 2 3 20; do
  d="$DST/step-$n"; [ -f "$d/c-window-input.txt" ] || continue
  tmp=$(mktemp -d); cp "$d/c-window-input.txt" "$tmp/in.txt"
  # Optional third..fifth args select the verbatim Slater branch:
  #   dh2_fsz/dh4_fsz/dh24_fsz/rbm_fsz/opt_fsz : 1 36 15
  #   interall : 1 0 0   (not window-compared)
  #   everything else (plain Slater) : omit
  ( cd "$tmp" && "$WORK/probe" in.txt c-window 1 36 15 )
  for f in "$tmp"/c-window_*.dat; do
    b=$(basename "$f"); b=${b/c-window_/zqp_}; cp "$f" "$d/$b"
  done
  rm -rf "$tmp"
done
```

The triple only selects the verbatim auxiliary Slater filenames
(`_orbital_opt.dat` / `_orbitalAntiParallel`+`_orbitalParallel` /
`_orbital_general`); the main `zqp_opt.dat` row is unchanged. `zvo_out.dat`,
`zvo_var.dat` and the top-level `step-N-*.txt` come from the generator stage.

Do not route the base Linux lineage through `read_fixture`'s direct
`reviewed_case_dir.join(name)` branch; that bypasses the overlay. The checked-in
implementation uses `fixture_path` for reviewed prefixes and prefers the ARM
`step-N` directory (whole-directory) for the declared window.

## Verify

```sh
export PKG_CONFIG_PATH="$(brew --prefix openblas)/lib/pkgconfig"
export OPENBLAS_CORETYPE=NEOVERSEN1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
cargo nextest run -p mvmc-core --locked --cargo-profile ci --no-fail-fast --retries 0 \
  -E 'test(run::callback_tests::real_cg_prefixes_match_julia_parameters_samples_energy_and_full_rng_blocks)'
```

`real_cg_prefixes_match_julia_parameters_samples_energy_and_full_rng_blocks`
passes at the existing `1e-11` budgets with the bridge reference; the native
aarch64 reference fails at `2.48e-6`. Regenerate one case, verify it, then
repeat for the remaining consumer cases in `reviewed_cg_62b/README.md`.

## Scope and limitations

- This is an independent mixed reference: Julia runner/solver with an ABI-only
  system-OpenBLAS adapter and the archived Julia AVX2 bilinear reduction. It is
  not unmodified Julia and not a full C executable/MPI parity claim.
- The archived AVX2 replay and BLAS bridge are host-neutral scripts already used
  for the earlier `macos_arm_julia` lineage (`docs/APPLE_SILICON_PARITY.md`).
- The DH/RBM/OptTrans window comparisons read `step-N/zqp_*.dat`. The ephemeral
  reviewed-import probe is not checked in; `ctest_opt_window.c` with the layout
  triple above reproduces it from the ARM `c-window-input.txt`.
- The related Apple Silicon overlays generated by the same procedure are:
  `macos_arm_julia/<core>/sr_cg/c_refresh/` (compile `ctest_cg_refresh.c` with
  the system OpenBLAS on the host and run it on `tests/fixtures/sr_cg/*.txt`)
  for the fixed-input CG test, and
  `macos_arm_julia/<core>/physcal_181/two-samples/` plus
  `.../native-c-weighted-green/` (run `generate_physcal_181.jl` with
  `PHYSCAL181_TWO_SAMPLES=1`, then `generate_physcal_181_weighted_green.jl`
  against those histories) for the PhysCal two-sample test.
- Ordinary Rust tests never read the external stages or invoke Julia; only the
  checked-in overlay files are consumed.
