# Golden fixtures

This directory holds deterministic goldens consumed by the Rust port's
regression tests. **Fixtures are regenerated, not hand-edited.** The
upstream Julia dumpers that produce them live under
`extern/Julia-mVMC/tools/`:

* `dump_sfmt_c_reference.jl` -- SFMT19937 `gen_rand32` / `gen_rand64`
  / `genrand_real2` / `init_by_array` reference values. Output is
  hand-pasted into `crates/sfmt19937/tests/golden_vs_c.rs` because
  the constants are short.
* `dump_pfapack_reference.jl` -- LTL / Pfaffian / inverse text
  fixtures for `crates/pfapack/tests/golden_vs_julia.rs`, written
  under `pfapack/` below.
* `dump_calc_m_all_reference.jl` -- `calculate_m_all_{real,fcmp}!`
  `(slater_elm, ele_idx) -> (pf, inv_m)` fixtures for
  `crates/mvmc-core/tests/calc_m_all_vs_julia.rs`, written under
  `calc_m_all/` below.
* `dump_projection_reference.jl` -- `init_loc_spn` /
  `make_proj_cnt` / `update_proj_cnt` snapshots for
  `crates/mvmc-core/tests/projection_vs_julia.rs`, written under
  `projection/` below.
* `dump_rbm_reference.jl` -- synthetic RBM counter and log-weight
  snapshots for `crates/mvmc-core/tests/rbm_vs_julia.rs`, written
  under `rbm/` below. The upstream example namelists do not contain
  RBM blocks, so this fixture builds RBM terms by hand.
* `dump_candidate_reference.jl` -- normal + FSZ candidate-move outputs
  for `get_update_type`, `make_candidate_hopping`,
  `make_candidate_exchange`, `make_candidate_hopping_fsz`,
  local-spin-flip candidates, and `make_candidate_exchange_fsz`,
  written under `candidate/`. These fixtures pin SFMT draw order
  (`gen_rand32()%n`, `genrand_real2`) against Julia.
* `dump_pf_update_reference.jl` -- one-electron Pfaffian update helper
  fixtures for `calculate_new_pf_m2{,_fsz}{,_real}`, written under
  `pf_update/`.
* `dump_metropolis_reference.jl` -- Metropolis weight/draw/decision
  fixtures for `crates/mvmc-core/tests/metropolis_vs_julia.rs`, written
  under `metropolis/`.
* `dump_one_move_reference.jl` -- one normal hopping move, a precomputed
  deterministic mini-loop scaffold, and a generated-candidate mini-loop for
  `crates/mvmc-core/tests/one_move_vs_julia.rs`, written under
  `one_move/`.

## Layout

```
tests/fixtures/
├── pfapack/                              # produced by dump_pfapack_reference.jl
│   ├── {real,complex}_n{4,6,16}_seed*.orig.txt        # input skew matrix
│   ├── {real,complex}_n{4,6,16}_seed*.ltl.txt         # LTL form + 1-based pivots
│   ├── {real,complex}_n{4,6,16}_seed*.pfaffian.txt    # scalar Pfaffian
│   └── {real,complex}_n{4,6,16}_seed*.inverse.txt     # inverse from utu2inv
├── calc_m_all/                           # produced by dump_calc_m_all_reference.jl
│   ├── {real,complex}_ns*_ne*_nqp*_seed*.slater.txt    # column-major slater_elm
│   ├── {real,complex}_ns*_ne*_nqp*_seed*.ele_idx.txt   # 1D ele_idx (length 2*ne)
│   ├── {real,complex}_ns*_ne*_nqp*_seed*.pfaffian.txt  # Pfaffian per QP
│   └── {real,complex}_ns*_ne*_nqp*_seed*.inverse.txt   # inv_m per QP (post rmul!(-1))
├── projection/                          # produced by dump_projection_reference.jl
│   └── {case}.txt                       # one per upstream input dir;
│                                       #   loc_spn + initial proj_cnt + post-hop snapshots
├── rbm/                                 # produced by dump_rbm_reference.jl
│   └── synthetic.txt                    # hand-built RBM terms;
│                                       #   cnt_old/new + log_rbm_val/ratio
├── candidate/                           # produced by dump_candidate_reference.jl
│   ├── normal.txt                       # normal-mode get_update_type + hopping/exchange
│   └── fsz.txt                          # FSZ hopping/exchange + local-spin flip candidates
├── pf_update/                           # produced by dump_pf_update_reference.jl
│   └── pf_m2.txt                        # normal + FSZ, real + complex, one-electron Pf updates
├── metropolis/                          # produced by dump_metropolis_reference.jl
│   └── decision.txt                     # log deltas + SFMT draw + accept/reject
├── one_move/                            # produced by dump_one_move_reference.jl
│   └── hopping.txt                      # one-move + precomputed/generated mini-loop state snapshots
```

The 50-step `zvo_out.dat` regression goldens are stored at the repository
root under `reference/<model>/zvo_out_first50.dat`, outside this fixture
directory, so they are easy to inspect and replace independently.

Licensing note: `reference/*` derives from the GPL-3 C-mVMC reference
(via Julia-mVMC integration references) and is loaded only by `mvmc-core`
integration tests. `pfapack/*` and `sfmt19937/*` carry the same BSD-3 /
MPL-2.0 boundary as the producing crates.

Each `pfapack/*.txt` file is a tiny text format: a few `key value`
header lines, then a blank line, then one floating-point literal per
payload row (`%.17e`; complex rows print `re im` separated by a single
space). Re-run the dumper after touching either `PfaPack.jl/src/*.jl`
or `crates/pfapack/src/*.rs` to refresh the goldens.


## Input and parameter contracts

The historical Julia parameter checks use unmodified Julia 1.13.1 with
`extern/Julia-mVMC/Manifest-v1.13.toml`, canonical source HEAD `8bb1b9e8` and
numerical/parser source `c2ea4327`. These checks do not regenerate or relabel
the historical numerical fixtures described above.

C `extern/mVMC-1.3.0/` is now authoritative. The Julia descriptions below
record historical coverage; they do not establish C compatibility where
readers, flags or numerical behavior differ. C-derived expectations and
optional standalone generators are documented in [projection counts](projection_count/README.md),
[orbital contracts](orbital_general/README.md) and
[initial/fixed records](initial_records/README.md). Normal Rust tests use
checked-in fixtures and never compile, invoke or read `c_toolbox/`.

| Contract | Source checks | Rust coverage |
| --- | --- | --- |
| Component flags (#21) | `scripts/check_parser_contracts_parity.jl`; family parser scripts | `mvmc-expert-parsers/tests/optimization_flags.rs`; DH/RBM/OptTrans initialization fixtures |
| Optional In*.def overlays (#20) | `scripts/check_input_overlays_parity.jl`; family parser/loading scripts | `mvmc-expert-parsers/tests/input_overlays.rs`; DH/RBM/OptTrans parser and loaded-boundary fixtures |
| Atomic initial/fixed loading (#28) | `scripts/check_initial_params_parity.jl`; `scripts/check_opttrans_load_weights_parity.jl`; RBM production loading script | `mvmc-core/tests/initial_params.rs`, `opttrans.rs`, `runner_config.rs`; family full-record checks |
| C complete initial/fixed records (#28, partial) | `scripts/check_initial_records_c_parity.py` | `mvmc-core/tests/c_initial_records.rs`; C scalar expectations in `opttrans.rs` |

Global flags and parameter offsets include projection, DH2/DH4, all nine RBM
sections, declared Slater/AP/P slots and active OptTrans weights. Flags preserve
source distinctions between real/imaginary components, absent flags, unmapped
slots and shared indices. Fixed Slater slots skip initialization draws; Slater
normalization still rescales fixed components as Julia does. Fixed correlation
blocks disable gauge shifts, and repeated synchronization preserves them.
The parsed AP/P initialization contract explicitly checks every coefficient bit
and the complete following 624-word SFMT block hash against Rust's constants.

The runner preserves initialization → optional initial.def → ordered In*.def
updates → optimizer synchronization → QP initialization. Relative paths resolve
from the namelist; later overlapping overlays win and tied mappings share the
new coefficient. Gutzwiller/Jastrow/normal/AP/General/RBM files retain canonical
permissive numeric fallbacks. Parallel/DH2/DH4/OptTrans overlays validate their
whole indexed record before mutation. Missing optional files are skipped; an
invalid later strict overlay does not revert earlier successful overlays.

Initial and optimized loaders now share C complete-record conversion: six
diagnostics followed by parameter triples, with the final complete record
taking precedence. Empty files leave coefficients unchanged; numeric nonfinite
values and overflow/underflow are accepted as C converts them. This supersedes
the historical Julia finite/single-record rejection expectations. Declared
Slater storage includes unmapped slots; complete declared RBM storage remains
under #26. Malformed nonnumeric or incomplete files retain bounded atomic Rust
errors, independently of C's unchecked-scan behavior. Optional loading returns
false for missing/malformed files; strict fixed loading returns an error. The
runner's Auto/Path/None policy handles optional files separately.

Family details and reproduction commands are in [DH2](dh2/README.md),
[DH4](dh4/README.md), [RBM](rbm/README.md) and [OptTrans](opttrans/README.md).
Their deterministic production tests supplement the input-only contracts.
Full-record fixed loading is available independently of the remaining PhysCal
execution work (#29).

```sh
julia +1.13.1 --project=extern/Julia-mVMC scripts/check_parser_contracts_parity.jl
julia +1.13.1 --project=extern/Julia-mVMC scripts/check_input_overlays_parity.jl
julia +1.13.1 --project=extern/Julia-mVMC scripts/check_initial_params_parity.jl
cargo test -p mvmc-expert-parsers
cargo test -p mvmc-core --test initial_params --test opttrans --test runner_config
```
