# OptTrans inputs

`*.def`, `load_*.def` and `namelist_*.def` exercise the OptTrans and `InOptTrans` readers and loaders. `run_opt_*/` are complete OptTrans models; `run_opt_real/` is also read by the resident GPU tests.

The Julia 1.13.1 CPU outputs that used to live here (parser, layout, initialization, kernel, Green-function and SR expectations) were removed with the other Julia CPU references in #695; C 1.4.0 is the only CPU reference.
