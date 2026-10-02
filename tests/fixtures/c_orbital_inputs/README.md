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
