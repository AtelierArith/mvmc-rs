# Optional ctest orbital input audit (#180)

`ctest_orbital_contracts.c` supplies storage to the actual C AP/P readers in
`orbital_contracts_upstream.inc`. It expands the older probe's storage to cover
the ten-site GeneralRBM input and both spin blocks. This is an input-section
check, not a full C executable, sampling, SR, MPI, or numerical parity check.
Neither Cargo nor Rust tests compile, invoke, or read this probe.

Upstream origin: `extern/mVMC-1.3.0/src/mVMC/readdef.c`, SHA-256
`6c53cb832f93d6cbfd7cea955fbb693738af5536b913d36af32b98eed38c32d9`.
The existing excerpt retains the upstream copyright and license. Extraction
boundaries are complete balanced-brace function bodies, selected by
`scripts/check_orbital_contracts_c_parity.py`: `ReadDefFileError`, `CheckSite`,
`CheckPairSite`, `GetInfoOpt`, `GetInfoOptOrbitalParalell`, `ReadBuffIntCmpFlg`,
`GetInfoOrbitalAntiParallel`, and `GetInfoOrbitalParallel`. No upstream source
was edited. The script verifies the existing excerpt before compiling its
own cases; it does not generate expectations from Rust.

Executed on 2026-10-03, native Linux x86_64, Ubuntu GCC
`13.3.0-6ubuntu2~24.04.1`, with `cc -O0`; this integer-reader probe has no BLAS
dependency. The Julia fixture checkout was
`8bb1b9e8ae47b1512c00b321be05664ddcac0fd1`. No Julia runtime was used.
All thirteen standard model AP sections returned status zero. The P sections
of Heisenberg FSZ, Hubbard FSZ, and Kondo FSZ also returned status zero.
The initial probe allocated only spatial row pointers and crashed on P's
spin-block writes; the final probe allocates and initializes both blocks.
Only the corrected rerun is acceptance evidence.

Reproduce from the repository root (temporary binaries remain outside it):

```sh
uv run --no-project python scripts/check_orbital_contracts_c_parity.py
ctest_probe_dir=$(mktemp -d /tmp/mvmc-ctest-contracts.XXXXXX)
cc -O0 c_toolbox/ctest_orbital_contracts.c -o "$ctest_probe_dir/probe"
for model in heisenberg_chain_real hubbard_chain_real heisenberg_chain_cmp \
  heisenberg_chain_fsz hubbard_chain_cmp hubbard_chain_fsz kondo_chain_real \
  kondo_chain_cmp kondo_chain_stot1_cmp general_rbm_cmp \
  hubbard_tetragonal_real hubbard_tetragonal_momentum_projection_real kondo_chain_fsz
do
  ctest_inputs="extern/Julia-mVMC/test/integration/reference/$model/inputs"
  ctest_sites=$(awk '$1=="Nsite" {print $2}' "$ctest_inputs/modpara.def")
  printf '%s AP ' "$model"
  "$ctest_probe_dir/probe" AP "$ctest_sites" "$ctest_inputs/orbitalidx.def"
  if test -f "$ctest_inputs/orbitalidxpara.def"; then
    printf '%s P ' "$model"
    "$ctest_probe_dir/probe" P "$ctest_sites" "$ctest_inputs/orbitalidxpara.def"
  fi
done
```

No C-result fixture was added. These observations belong to an explicitly
invoked developer audit, not an ordinary Rust test expectation. Full-model
coverage and outstanding evidence are tracked in
`../docs/reference/c-to-julia/verification/issue-180-model-coverage.md`.
