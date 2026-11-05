# DH4 definition inputs

`*.def` and `namelist_*.def` exercise the strict DH4 reader (valid, malformed, boundary and replacement cases). `production_dh4_*/` and `production_dh24_*/` are complete DH4 and combined DH2/DH4 models read by the runtime tests and by the resident GPU tests.

The Julia 1.13.1 CPU outputs that used to live here (parser, layout, initialization, kernel, Green-function and SR expectations) were removed with the other Julia CPU references in #695; C 1.4.0 is the only CPU reference.
