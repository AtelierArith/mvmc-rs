# Optional fixed-operand SR factorization-failure probes

Owned new directory for issues 179/190. These original standalone C drivers
exercise LAPACK's DPOSV failure contract independently of the Rust implementation.
They are not extracted upstream helpers, full native C executable/MPI/sampler
validation, a Julia repair, a successful optimization oracle, or a tolerance
justification. Cargo build scripts and Rust tests must not compile, invoke or
read this directory. No runtime FFI is introduced into Rust.

## Numerical authority and source boundaries

Authoritative reference: `extern/mVMC-1.3.0`, gitlink
`d73d06bd529d3b2573f38eb5817c4a5f52971006`.

- `src/mVMC/stcopt_dposv.c`, SHA-256
  `2bd48d880dcbd95ea1b1b92931178c07fd08f981b8ebf04e7db1709e909e57ca`:
  lines 33–49 (`stcOptMain`) call DPOSV with upper triangle, one RHS and column-major
  storage, and return LAPACK INFO. Lines 53–84 construct the SR matrix/gradient.
- `src/mVMC/stcopt.c`, SHA-256
  `43ed8790cff2715284849f0f8906e4645179dcbb100b51f819918b279d6a36f2`:
  lines 144–147 retain the solver status; the parameter-update block at line 185
  is conditional on `info == 0`.

These are reviewed reference boundaries, **not extraction boundaries**: no C
function bodies were copied. The upstream files carry The University of Tokyo's
2016 copyright and GPL-3.0-or-later notice. Their pristine vendored contents are
unchanged. The drivers call the system LP64 LAPACK interface with fixed operands;
they do not recreate SR assembly or parameter updates.

## Fixtures and retained lineage

`zero_sr_boundary.c` uses a literal ten-dimensional all-zero matrix/RHS. DPOSV
must return INFO 1 without solving or changing the finite zero RHS. A separately
labelled unsafe diagnostic deliberately ignores positive POTRF INFO and calls
POTRS; the validated OpenBLAS environment then produces ten nonfinite entries.
The literal uses positive-zero RHS; the frozen step-15 observation used negative
zeros. This is a zero-factorization boundary check, not a claim of byte-identical
native sampled operands or signed-zero output parity.

`step14_sr_boundary.c` stores the actual observed pre-factorization operands:

```text
S = [8.32675595141552149e-17  -2.77555756156289135e-17;
     -2.77555756156289135e-17  6.93896329284626790e-18]
b = [8.88178419700125251e-18, -3.33066907387546950e-18]
```

They are fixed inputs to an independent C/LAPACK replay, not Rust-generated
expectations for a successful numerical solve. The observed source scenario is
six-site `heisenberg_chain_real`, real parameters, identity QP count one,
world two, group width one, direct SR, NStore one, requested seed one (rank seeds
one/two), three samples, warm-up one and requested steps/window 20.
Active component indices are `[16, 20]` (zero-based pair-layout component indices),
recorded in both retained rank files. Step numbers are zero-based. DPOSV returns
INFO 2 and leaves RHS unchanged. The original input file declares steps/window
one; the separately hashed external recorder explicitly overrides **both** to 20
before running. No final parameters are obtained by truncating a longer run.
The generated input `modpara.def` SHA-256 is
`7f6ab3912d707100a4962cdf5cc0f784a250f272fb116d59fa05cae5955191f4`,
`namelist.def` is `9a72042e4587c41249c052ec469eebee10864dedc3e1d7fdb5d8e9e6a3db2c81`.
The complete per-file manifests remain in each `failure14/w*/inputs.sha256`.

Container `73c57e563c61` retains these distinct lineages:

- Original committed `02c83f31e0feec8c97105727a4975f48afaa5687`:
  `/home/vscode/.cache/mvmc/issue179-02c83f31.D2ZuYh/long20-width1/`.
  The full 72-row width-one matrix ended 1 (69 successful comparisons, three
  real/world2/direct/store1 failures across workers 1/2/4). The old solver ignored
  positive factorization status at step 14, applied an invalid finite update,
  then failed at step 15 with a zero 10×10 matrix and NaN increments. These
  failures remain unchanged; none is retrospectively relabelled PASS.
- Corrected source overlay:
  `/home/vscode/.cache/mvmc/issue179-02-srfactorfix.S9r6tM/`, based on the above
  production snapshot plus Ramanujan's SR file SHA-256
  `e7bfd4852d274b3b5dabef393733fdff56d3d0a95720ff051a99c6849293da33`.
  `failure14/w1/repeat1/rank-0.txt` and rank 1 record
  `n:sr-system-000014-matrix`, `-rhs`, `-increment` and actual factor/solve status.
  The corrected runner stops at step 14, INFO 2, without calling POTRS.
  `n:parameters` remains the last valid `n:sr-step-13` checkpoint.
- Optional initial external probes remain in `C-zero-SR-boundary` (old root) and
  `C-step14-boundary` (corrected root). Their source/binary hashes differ from
  these packaged drivers, which add LP64 compile-time checks and provenance.
  Do not relabel those historical binaries as the packaged versions.

## Explicit developer reproduction

Requirements: Linux, a C11 compiler, GNU `timeout`, `pkg-config`, and matching
OpenBLAS **LP64** headers/library providing LAPACK. Both drivers compile-time
assert that BLAS integer width matches `int`; ILP64 is unsupported. The captured
reference environment was Linux x86_64, GCC 13.3.0, OpenBLAS 0.3.26 LP64, BLAS/OMP
one. This is not the old Open MPI devcontainer or Julia's ILP64 OpenBLAS.

From the repository root, choose a new evidence directory:

```sh
bash c_toolbox/sr_factor_failure/run.sh /tmp/sr-factor-failure-new-evidence
```

The script uses `-std=c11 -O0 -ffp-contract=off -Wall -Wextra -Werror`,
`pkg-config --cflags --libs openblas` and `-lm`. Each launch is bounded to ten
seconds plus a two-second kill grace. It records exact compile commands,
compiler/platform/backend/thread settings, source/upstream/binary SHA-256,
linkage, launch statuses, stdout and post-checks. Existing evidence is never
overwritten. A missing tool, compile failure, timeout, unexpected outcome or hash
change exits nonzero. Expected stdout includes:

```text
C_DPOSV info=1 finite_rhs=10 dimension=10 rhs_unchanged=1
OLD_POSITIVE_INFO_IGNORED factor=1 solve=0 nonfinite_rhs=10
C_STEP14_DPOSV dimension=2 info=2 rhs_unchanged=1 finite=1
```

Successful execution proves only this fixed-operand failure boundary. It does
not change the model's acceptance scope or prove native C sampling, all MPI
collective domains, solver convergence or numerical accuracy for other cases.

## Parent read-only audit of the unchanged frozen MPI checker

This does not launch or regenerate MPI captures. It checks the six retained
launch statuses, the frozen source/binary/checker hashes, and runs the exact
frozen validator on each pair. Use a non-login shell: the retained login-shell
wrapper returned one because `set -e` propagated `.bash_logout`'s failed
`clear_console -q`; preserve that status separately from the read-only audit.

```sh
docker exec 73c57e563c61 bash -c '
set -euo pipefail
root=/home/vscode/.cache/mvmc/issue179-02-srfactorfix.S9r6tM
old=/home/vscode/.cache/mvmc/issue179-02c83f31.D2ZuYh/long20-width1/r2-real-s1-cg0-store1
cd "$root/repo"
sha256sum -c "$root/failure14/binary-checkers-sources.sha256"
awk -F "\t" '\''NR==1 {if($0!="workers\tfirst_exit\tsecond_exit\tvalidation_exit") exit 1; next}
{if(NF!=4 || ($1!=1 && $1!=2 && $1!=4) || seen[$1]++ || $2!=0 || $3!=0 || $4!=0) exit 1; n++}
END {if(n!=3) exit 1}'\'' "$root/failure14/results.tsv"
for w in 1 2 4; do
  sha256sum -c "$root/failure14/w$w/inputs.sha256"
  uv run --no-project python "$root/check_factor_failure14.py" \
    "$root/failure14/w$w" "$old/w$w"
done
'
```

The frozen checker SHA-256 is
`c5cb8a0cb745dc8ed441bd633359aeffacce20fc6dbb440e2b9c989b8b6e96bb`;
binary `f159d0fecea59699f6b3c13b2ca792a7a014b0474da29e433c0608a67a419ca1`.
It requires explicit failed status, factor info two/no solve, finite unchanged
RHS/increments and last valid parameters; exact repeated configurations/RNG and
the old sampling trace prefix before the first failure; two 15-row output files
and no final optimized parameter file. Computed repeat values use explicit
absolute/relative `1e-12` bounds. It does not require reproducing the old invalid
post-factorization continuation or call a failed 20-step optimization successful.
