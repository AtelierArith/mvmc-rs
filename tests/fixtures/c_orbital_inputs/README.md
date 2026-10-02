# Complete C AP/P sections and historical kernel models

These five orbital definitions provide complete AP/P sections for historical
test models. C acceptance of each section is recorded in
`../orbital_general/c_reader_contracts.txt`; broader acceptance of the other
projection/RBM/OptTrans files in those models is not asserted.

| Definition | Historical source | Change |
| --- | --- | --- |
| `ap_three.def` | Identical `dh2/orbital.def`, `dh4/orbital.def`, `rbm/orbital.def` | Complete nine spatial rows; retain four slots and all four flags |
| `p_three.def` | `dh4/parallel_orbital.def` | Retain three mapping rows and the first three parameter flag pairs |
| `ap_four.def` | `interall/orbital.def` | Complete sixteen spatial rows using fixed slot 2 for additional cells |
| `ap_general_three.def` | `orbital_general/ap.def` | Add all nine active parameter flag pairs |
| `p_general_three.def` | `orbital_general/parallel.def` | Add all three active parameter flag pairs |

The original historical files and numerical fixtures remain unchanged. C's
actual header/AP/P readers reject all seven original inputs; those failures
are recorded alongside these accepted replacements. `namelist_ap_parallel.def`
uses the complete definitions directly for the spin-site matrix comparison.
General's six-column C definition reader remains separate #41 work.

`tests/support/historical_orbital_model.rs` parses the complete replacements,
then constructs the original sparse spatial tables programmatically for
historical coefficient/kernel regressions. Declared coefficient storage and
flags remain from the strict parser. These restored sparse models are not
presented as supported C input files or full C sampling evidence. Normal
production entry points use the strict parser and reject incomplete sections.
Neither the helper nor the Rust tests compile, invoke or read `c_toolbox/`.

Optional native C reproduction:

```sh
python3 scripts/check_orbital_contracts_c_parity.py
python3 scripts/check_orbital_contracts_c_parity.py --write
```
