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
