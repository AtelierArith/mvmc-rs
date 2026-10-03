# Independent DH optimization-window outputs

Nine three-step/window-three independent Julia 1.13.1 histories: DH2, DH4 and
combined DH2/DH4 in real, complex and FSZ modes, at original input seeds.
Canonical overlays are loaded by the unchanged reference runner. Complex cases
use the explicit valid AP replacements under `c_orbital_inputs`, matching the
Rust runtime tests, not the malformed historical orbital definitions.
FSZ energy follows the separately verified native C bridge used by #180.

`c-window-input.txt` contains actual chronological post-SR/synchronization
`Etot, Etot2, declared Para` snapshots. Expected output is computed by the
verbatim authoritative `avevar.c` StoreOptData/CalcAveVar/OutputOptData bodies.
It is not Julia's final-parameter writer extension or Rust-generated output.
The original `c_zqp*` records used the generic non-FSZ auxiliary filename branch;
the runtime tests use the later `native_zqp*` records with actual input orbital
flags/counts, recorded separately in `orbital-aggregation-provenance.txt`.
The main streams are the same. FSZ AP/P auxiliary files are split at the actual
anti-parallel width, with twice the parallel index count, as C specifies.

Histories were generated externally in `/tmp/mvmc-dh-c-window.VOTXnv` by
`c_toolbox/ctest_dh_window_oracle.jl`; process 36421 completed successfully.
The native auxiliary-branch regeneration uses `ctest_dh_reaggregate.jl` and
`ctest_dh_opt_window.c`, without rerunning sampling or changing any history.
All source/input/binary hashes, settings, compiler flags, extraction provenance,
Julia/BLAS and C-FSZ bridge information are retained. `inputs.sha256` was added
after generation from the unchanged canonical namelist and referenced files;
verify from repository root. It includes initialization overlays where present.

`zvo_c_slots_var.dat` is the separate independent pre-SR complete declared-slot
stream captured from the original Julia runner's same data, following C
`vmcmain.c:653–657` contiguous Para layout. Every DH coefficient is present;
it is neither the post-SR window history nor Rust-observed coefficients.
The runtime tests compare this entire stream before separately checking the
historical Julia DH-omitting subsequence. Source hash/layout provenance is in
the shared/per-case generation records; archive file hashes pin this stream.

Normal Rust tests read only these standalone checked-in outputs and continue
using the existing 1e-12 absolute/relative bound with exact layouts/indices.
No C/Julia/toolbox program is compiled, read or invoked by Cargo tests.
