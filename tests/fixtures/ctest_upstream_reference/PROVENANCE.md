# Upstream ctest references (issue #180)

`ref_mean.dat` / `ref_std.dat` per model are byte-identical copies of the C-shipped
files `extern/mVMC-1.3.0/test/python/data/<CModel>/ref/ref_{mean,std}.dat`
(mVMC v1.3.0, submodule commit `d73d06bd529d3b2573f38eb5817c4a5f52971006`; model
mapping in `models.tsv`). They are also the `ctest_ref/` files of
`extern/Julia-mVMC/test/integration/reference/<model>/` (submodule commit
`c0788c34a6a5753c611633a97cd1ea233203320c`); the copy script verified C and Julia
files are identical (`cmp`) before copying. SHA-256 of every reference file is in
`ref.sha256`.

`inputs.sha256` hashes the expanded input bundle in
`extern/Julia-mVMC/test/integration/reference/<model>/inputs/` (all files except the
historical `zqp_opt.dat`; `initial.def` is included because C's ctest passes it as the
second command-line argument). The gate reads inputs from the Julia reference checkout
(existing harness convention) and fails if any hash differs.

Rule (C `test/python/runtest.py`, Julia `test/integration/ctest_equivalent.jl`):
compare the first two window-averaged summary values; fail only when
`|diff| >= 3*ref_std` and `|diff| >= 1e-8`. Run length is the upstream
`NSROptItrStep`/`NSROptItrSmp` of each input. No expected value here was produced by Rust.
No toolbox program is used or read; normal tests need only these files.
