# PairHop inputs

`hamiltonian.def`, `raw.def`, `replace.def`, `invalid.def`, `parser_cases.def` and the `namelist*.def` files exercise the PairHop reader and the runtime rejection of incomplete models.

The Julia 1.13.1 CPU outputs that used to live here (parser, layout, initialization, kernel, Green-function and SR expectations) were removed with the other Julia CPU references in #695; C 1.4.0 is the only CPU reference.
