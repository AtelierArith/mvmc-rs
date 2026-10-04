# Complete C orbital sections and historical kernel models

These definitions provide complete AP/P/General sections for historical test
models. C acceptance is recorded in `../orbital_general/c_reader_contracts.txt`
and `../orbital_general/c_general_reader.txt`; broader acceptance of the other
projection/RBM/OptTrans files in those models is not asserted.

| Definition | Historical source | Change |
| --- | --- | --- |
| `ap_three.def` | Identical `dh2/orbital.def`, `dh4/orbital.def`, `rbm/orbital.def` | Complete nine spatial rows; retain four slots and all four flags |
| `p_three.def` | `dh4/parallel_orbital.def` | Retain three mapping rows and the first three parameter flag pairs |
| `ap_four.def` | `interall/orbital.def` | Complete sixteen spatial rows using fixed slot 2 for additional cells |
| `ap_general_three.def` | `orbital_general/ap.def` | Add all nine active parameter flag pairs |
| `p_general_three.def` | `orbital_general/parallel.def` | Add all three active parameter flag pairs |
| `general_three.def` | `orbital_general/general.def` | Convert combined coordinates to six columns; add fifteen active flags |
| `general_three_real.def` | `dh4/general_orbital.def` | Convert coordinates to six columns; retain all fifteen original flags |
| `general_heisenberg_six.def` | `orbital_general/heisenberg/general.def` | Convert all sixty-six upper-triangle rows; retain all twenty-two flags |
| `general_sparse_three.def` | `orbital_general/sparse.def` | Complete fifteen safe upper-triangle rows and four flags for a test constructor |

The original historical files and numerical fixtures remain unchanged. C's
actual header/AP/P readers reject all seven original inputs; those failures
are recorded alongside these accepted replacements. `namelist_ap_parallel.def`
uses the complete definitions directly for the spin-site matrix comparison.
`namelist_general.def` uses the six-column definition directly. The C General
reader accepts all four replacements for both boundaries, as recorded in
`../orbital_general/c_general_reader.txt`. `namelist_heisenberg_general.def`
retains the original non-orbital FSZ inputs for historical 50-step regression
checks, without claiming full C acceptance of those other files.
`namelist_interall_fsz.def` reuses that same sixty-six-row General section for
the historical InterAll initial-state, CG and direct-SR/RNG checks; all other
InterAll inputs remain the original files.

`tests/support/historical_orbital_model.rs` parses the complete replacements,
then constructs the original sparse spatial/spin-site tables programmatically for
historical coefficient/kernel regressions. Declared coefficient storage and
flags remain from the strict parser. These restored sparse models are not
presented as supported C input files or full C sampling evidence. Normal
production entry points use the strict parser and reject incomplete sections.
Neither the helper nor the Rust tests compile, invoke or read `c_toolbox/`.

Optional native C reproduction:

```sh
python3 scripts/check_orbital_contracts_c_parity.py
python3 scripts/check_orbital_contracts_c_parity.py --write
python3 scripts/check_general_orbital_c_parity.py
```

`ap_hubbard_six_complex.def` retains the complete thirty-six mappings, twelve
slots and active flags of the historical real Hubbard input, while explicitly
declaring complex orbitals. C's orbital flag writers use the orbital headers,
not the overall DH/G/J complex decision. The original mixed-DH Julia inputs
relied on global complex mode implicitly activating imaginary orbital flags;
C leaves those real-AP imaginary cells untouched. The five `namelist_*_cmp.def`
replacements use this explicit complex AP section for DH2/DH4/combined-DH/RBM/
OptTrans historical SR/RNG regressions. All other input paths and numerical
fixtures remain unchanged. The actual C AP reader accepts the section as the
additional case in `c_reader_contracts.txt`; this does not prove the other
families' complete C contracts or the original mixed-header C trajectory.

The nine files under `historical_binary_rbm/` copy the historical RBM mappings
and declarations, changing only the second flag from 2 to 1. Julia converted
both values to true; C initializes both but selects only 1 for SR. The five
RBM namelists and the combined OptTrans/RBM namelist use these explicit binary
inputs to retain the OLD Julia SR/RNG regression. Original inputs and goldens
are unchanged; production raw flag 2 remains fixed for SR, verified separately
by native C eligibility and both Rust solver tests. These RBM copies are legacy
numerical models, not evidence of complete C RBM input acceptance: the declared
width/mapping/count gaps remain #26. Reproduce all explicit input replacements
with `python3 scripts/prepare_historical_integer_flag_inputs.py --write` (omit
`--write` to verify).

`ap_hubbard_six_flag2.def` is a complete real AP section with all twelve raw
flags equal to 2, accepted separately by the actual C reader. The CLI test
compares one and three SR steps: initialized coefficients are nonzero and stay
unchanged, while the same workload with flag 1 changes them. This tests native
integer eligibility through the actual CLI; it does not assert full C sampling
or RNG equivalence. It uses the original Hamiltonian with G/J factors omitted.

For legacy three-site models, the test constructor also replaces the incomplete
DH2/DH4/RBM upper-only Jastrow input with `../jastrow/c_three.def`. It adds the
three reversed mappings using the original parameter indices, retaining all
three slots and original binary flags. Actual C accepts this replacement and
rejects the three original files, as recorded in `../jastrow/c_reader_contracts.txt`.
The directional production reader does not synthesize reverse rows. Original
inputs and numerical goldens remain unchanged, and the complete six-site
upstream Jastrow inputs retain their existing matrices and coefficients.
