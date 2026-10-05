# StdFace 3D lattices (#356): C defects that are not reproduced

`Orthorhombic.c`, `FCOrtho.c` and `Pyrochlore.c` of `extern/mVMC-1.3.0` contain clear errors. The
Rust port (`crates/mvmc-stdface`) implements the correct behaviour instead of reproducing them.

Fixture layout for the cases in `c_toolbox/stdface/cases_3d.py`:

* `<case>/expected/`: the corrected behaviour. Produced by the same C sources with
  `c_toolbox/stdface/3d_defects.patch` applied to a copy (`build_reference_3d_fixed.sh`), so it is
  independent of the Rust code; the Rust output must match it byte for byte
  (`stdface_c_fixtures.rs`).
* `<case>/c_historical/` (only where the two differ, 28 cases): the unmodified C output. The test
  `stdface_3d_defects.rs` asserts that Rust differs from it and checks the corrected behaviour
  against properties derived without any C build.
* Cases without `c_historical/` are identical in both builds, i.e. byte-identical to the unmodified C.

The generator refuses to run if the set of differing cases is not `cases_3d.DEFECT_CASES`.
Provenance: `PROVENANCE_3d.md` (hashes of the patch and both binaries, commands).

## Defects

| # | Where | C behaviour | Corrected behaviour | Evidence case |
|---|-------|-------------|---------------------|---------------|
| 1 | `Pyrochlore.c`, Kondo local term | `StdFace_GeneralJ(J, 1, S2, isite + 3, jsite + isiteUC)`: all four localized spins of a cell couple to conduction site 3; conduction sites 0-2 have no Kondo coupling | `isite + isiteUC` | `pyrochlore_kondo_defect` (`exchange.def`: C pairs 19 with 0..3, corrected pairs 16+i with i) |
| 2 | `FCOrtho.c`, Kondo local term | the localized spin gets no `StdFace_MagField` (`h`, `Gamma`, `Gamma_y` ignored); `Orthorhombic.c`/`Pyrochlore.c`/`ChainLattice.c`/`Ladder.c` apply it | applied | `fcc_kondo_fields_defect` (8 localized spins x 4 terms more in `trans.def`) |
| 3 | all three, spin model | `V2`, `V0'`, `V1'`, `V2'`, `V''` are accepted silently (only `V`, `V0`, `V1`, `V'` checked) | `NotUsed` errors | `*_spin_V2_accepted_by_c`, `*_spin_Vpp_accepted_by_c`, `ortho_spin_Vp_component_accepted_by_c` |
| 4 | `Orthorhombic.c`, Hubbard/Kondo | rejects the nonexistent `J0''`, `J1''`, `J2''` but accepts the spin keys `J'` and `J''` | `J'`, `J''` rejected too | `ortho_hubbard_Jp_accepted_by_c`, `ortho_hubbard_Jpp_accepted_by_c`, `ortho_kondo_Jp_accepted_by_c` |
| 5 | `FCOrtho.c` | no third-neighbour bond exists: Hubbard/Kondo accept `t''`, `V''`, `J0''`..`J2''`; spin model reads `J0''`..`J2''` via `InputSpinNN` and drops them | rejected | `fcc_hubbard_tpp_accepted_by_c`, `fcc_hubbard_Vpp_accepted_by_c`, `fcc_hubbard_J0pp_accepted_by_c`, `fcc_spin_Jpp_accepted_by_c`, `fcc_spin_J0pp_accepted_by_c` |
| 6 | `Pyrochlore.c` | only `t`, `V`, `J` exist (primed values default from them) but `t'`, `t''`, `V'`, `V''`, `J'` (Hubbard/Kondo) and `J'`, `J''` (spin) are accepted silently | rejected | `pyrochlore_hubbard_tp_accepted_by_c`, `..._tpp_...`, `..._Vp_...`, `..._Vpp_...`, `..._Jp_...`, `pyrochlore_spin_Jp_accepted_by_c`, `pyrochlore_spin_Jpp_accepted_by_c` |
| 7 | all three, `ntransMax` | counts 4 local transfer terms per itinerant site; `StdFace_HubbardLocal` appends 6 raw terms when `Gamma` and `Gamma_y` are nonzero: heap overflow wherever hopping terms do not leave slack (pyrochlore: exit status -11) | no fixed bound needed; corrected build allocates 6 | `pyrochlore_hubbard_gc_fields_gamma_y`, `pyrochlore_kondo_gc_fields_defect` (`c_historical/exit_status` is `-11`) |

Item 7 is a generic C defect (it also affects other lattices whenever the sum of hopping and local
terms exceeds the allocation); on the orthorhombic and fcc lattices the unused `t'`/`t''` slack hides
it for the checked inputs.
