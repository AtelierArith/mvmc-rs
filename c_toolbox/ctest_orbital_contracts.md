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

## Independent canonical Julia prefix observer

`ctest_prefix_oracle.jl` runs canonical ctest inputs at their original seeds,
loading `initial.def` and `In*.def` in the production Julia runner order.
It overrides only the prefix length and averaging window. It writes to an
explicit external staging directory, never reads Rust results, and does not
write into the reference checkout. Rust's opt-in oracle gate reads the staged
data; it neither invokes nor reads toolbox programs.

Extraction boundaries are complete top-level Julia functions
`vmc_para_opt!` and `run_para_opt_from_namelist` from
`extern/Julia-mVMC/MVMCOptimizers.jl/src/`. The copies are renamed; read-only
hooks capture normalized pre-SR buffers and post-sync parameters/configurations.
Every original numerical operation and random draw remains in the extracted
body. The existing `scripts/reference_c_kernel_order.jl` supplies C Slater
coefficient retention and verifies its translated counter kernel against all
4,994 native C cases before running. This is explicitly a mixed reference,
not a complete C executable. No C compilation is performed by this observer.
FSZ requires the separately validated native C energy bridge, selected with
`MVMC_CTEST_NATIVE_FSZ_BRIDGE`; without it FSZ is rejected. Its provenance and
library SHA-256 are recorded alongside the observations. This adapter runs
the actual serial C local-energy kernels and independently verifies borrowed
projection counters, while Julia still owns initialization, sampling, and SR.
GeneralRBM's existing canonical gate supplies its counter-order evidence.

The output observer preserves Julia's original `zvo_var.dat` separately and
emits `zvo_c_slots_var.dat` from the same pre-SR data using `pack_parameters`.
This follows C `outputData()`'s formatted branch, `vmcmain.c:653-657`: its
loop emits every declared contiguous `Para[i]` exactly once, not mapped
orbital rows. Upstream SHA-256 is
`fdcd661c4eb028786fc58ae5e1f5232426c943b547f95a0d74e888e602256d63`.
The observer translates this five-line layout only; it does not call Rust,
change Julia's parameters, repair an algorithm, or claim full C execution.
The first twelve-model Rust comparison located this structural divergence
at prefix 1 after matching configurations/RNG, parameters, energy and SR.
Those failed runs do not establish complete prefix coverage.

Each staging directory records Julia/BLAS/thread metadata, the observer and
upstream function source SHA-256 hashes, and the Manifest-v1.13.toml hash.
Every successfully generated prefix records input-file SHA-256 hashes.
Unsuccessful prefixes receive `UNVERIFIED.txt`; they cannot pass Rust gates.
Full 624-word SFMT blocks and configurations/counters are captured separately
from decimal numerical parameters, energy, SR buffers, and formatted output.
The Rust gate checks exact discrete checkpoints first, then applies #190's
existing `1e-11` runner and `1e-12` SR budgets; a trajectory mismatch is fatal.

```sh
ctest_oracle_stage=$(mktemp -d /tmp/mvmc-ctest-oracles.XXXXXX)
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 julia +1.13.1 \
  --project=extern/Julia-mVMC c_toolbox/ctest_prefix_oracle.jl \
  "$ctest_oracle_stage" \
  hubbard_chain_cmp,kondo_chain_real,kondo_chain_cmp,kondo_chain_stot1_cmp,hubbard_tetragonal_real,hubbard_tetragonal_momentum_projection_real \
  1,2,3,20
MVMC_RS_CTEST_ORACLE_ROOT="$ctest_oracle_stage" \
MVMC_RS_CTEST_PREFIX_MODELS=hubbard_chain_cmp,kondo_chain_real,kondo_chain_cmp,kondo_chain_stot1_cmp,hubbard_tetragonal_real,hubbard_tetragonal_momentum_projection_real \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  cargo nextest run -p mvmc-core --locked --cargo-profile test-fast \
  --test ctest_model_prefixes --run-ignored only -E 'test(canonical_models_)' \
  --no-fail-fast --retries 0 --success-output immediate --failure-output immediate
```

Staged observations are not checked-in fixtures. Their presence or successful
generation does not count as Rust parity until the corresponding gate executes.

The optional bridge was built in `/tmp/mvmc-ctest-fsz-bridge.Xyzp6F` using
`uv run --no-project python scripts/check_native_fsz_runner_bridge.py --build-dir DIR`.
Its source verification and 72 complex plus 36 scalar energy gates passed.
The bridge's own `provenance.txt` records compiler options and extracted C
hashes; the observer copies that provenance. To generate the remaining models:

```sh
MVMC_CTEST_NATIVE_FSZ_BRIDGE=/tmp/mvmc-ctest-fsz-bridge.Xyzp6F \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 julia +1.13.1 \
  --project=extern/Julia-mVMC c_toolbox/ctest_prefix_oracle.jl \
  EXTERNAL_STAGE \
  heisenberg_chain_real,heisenberg_chain_cmp,hubbard_chain_real,heisenberg_chain_fsz,hubbard_chain_fsz,kondo_chain_fsz \
  1,2,3,20
```
