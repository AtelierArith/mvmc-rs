# Nodal-configuration Lanczos references (issue #481)

Standalone developer tools. Cargo builds and tests never read, compile or run anything here;
they consume only the checked-in files under `tests/fixtures/native_c_physcal_181/`
(`_sources/nodal_neel/`, `nodal_neel_lanczos1/`, `nodal_neel_lanczos2/`).

| File | Purpose |
| --- | --- |
| `make_sources.py` | Writes the hand-made inputs and fixed parameters: a Neel-type pairing state (orbital couples only up on even sites with down on odd sites, every other pair fixed at zero) on an 8-site periodic chain with 3 up and 3 down electrons. Every nearest-neighbour hop and every nearest-neighbour exchange leaves the non-zero-weight manifold, so the hopped Pfaffian ratio `checkGF1/checkGF2` is far below `1e-12` and C takes `calHCA2`/`calHCACA2`; next-nearest-neighbour transfers stay inside it and exercise the ordinary branch in the same runs. |
| `regenerate.sh` | Builds the unmodified `vmc.out` and the additive state-dump probe with `c_toolbox/physcal_native/build.sh` and runs the two `nodal_neel_*` rows of `scenarios.tsv` through `c_toolbox/physcal_native/generate.py` (one MPI rank, one thread; Linux x86_64 Dev Container). |

The inputs are not the output of an optimization: the parameters are fixed in `zqp_opt.dat` and
the 40 sampled configurations are all in the 16-configuration manifold. A single sample
(`NVMCSample 1`) makes C write uninitialised denormals for `zvo_ls_out` (singular alpha) and is
therefore not used. The complex variant (`ComplexType 1`) crashes native C
(`GreenFuncN` of `locgrn.c` passes `&n` and an uninitialised `lda` to `M_ZSKPFA`), so only the
real path has a native-C reference.
