# DH2 definition inputs

`*.def` and `namelist_*.def` exercise the strict DH2 reader (valid, malformed, boundary and replacement cases). `production_{real,cmp,fsz}/` are complete DH2 models read by the runtime tests and by the resident GPU tests.

The Julia 1.13.1 CPU outputs that used to live here (parser, layout, initialization, kernel, Green-function and SR expectations) were removed with the other Julia CPU references in #695; C 1.4.0 is the only CPU reference.
