# Optional independent C declared-parameter acquisition

`generate_reviewed_parameter_c_audit.jl` orchestrates the C program
`reviewed_parameter_c_audit.c`; it imports only Julia standard libraries,
not Julia-mVMC or Rust. Ordinary Cargo builds/tests must never invoke/read
these programs. The acquired values are standalone text artifacts under
`tests/fixtures/reviewed_parameter_c_audit/`, not runtime oracle dependencies.
They are acquisition evidence, **not yet adopted reference expectations**.

## Authority, extraction and scope

Upstream mVMC 1.3.0 under `extern/mVMC-1.3.0/`:

- `src/mVMC/parameter.c`, SHA256
  `46ad04622f4475337028cee633bd76ce318a6d5058d03f202b55204500399fb0`:
  extract `void InitParameter()` through the end of `SetFlagShift`, excluding
  the final translation-unit `#endif`. Includes original `ReadInitParameter`,
  `SyncModifiedParameter`, `shiftGJ`, `shiftDH2`, `shiftDH4`, `SetFlagShift`.
- `src/mVMC/readdef.c`, SHA256
  `6c53cb832f93d6cbfd7cea955fbb693738af5536b913d36af32b98eed38c32d9`:
  `ReadInputParameters` through its following separator; `GetInfoOpt` through
  the start of `GetInfoGutzwiller`; **last** `GetInfoOptTrans` occurrence
  (definition, not prototype) through the start of `GetInfoInterAll`.
- Verbatim combined excerpt SHA256
  `70d50ac4dc34a28101fa5064625d38f989aee60b38ba35b326e5d4e1ce06d87e`.
  The original parameter.c copyright/GPL notice is retained. mVMC COPYING
  SHA256 `8ceb4b9ee5adedde47b31e975c1d90c73ad27b6b165a1dcd80c7c545eb65b903`;
  GPL-3.0-or-later. Native SFMT retains its upstream BSD license notices.

The driver includes original native `SFMT.c`. Its diagnostic wrapper counts
each original `genrand_real2` call (one original `gen_rand32` word); it does
not replace numerical functions, conversions, draws or arithmetic order.
Checkpoint copies native `sfmt` and `idx`, writes raw state, then saves/restores
them around the original next-624 calls. Peeking is not included in primitive
draw counts. Overlays/synchronization have no RNG calls.

Fifteen section headers are enumerated from the actual text inputs, independent
of any Julia/Rust pack. Projection widths use C's DH multipliers 6 and 10;
NPara is their sum plus all nine declared RBM widths, Slater and OptTrans.
AllComplexFlag uses C's Gutzwiller/Jastrow/DH/orbital header sum, **not RBM
headers**. Neuron totals use original signed modpara dimensions. Framing
validates complete geometry/flag counts, bounded mapped indices and ordered
flag labels; it rejects sparse missing flag records instead of expanding them.
Mapped markers are independently enumerated from original definition records;
they are not obtained from Julia `_foreach_parameter_location` or Rust state.
Geometry readers/site checks themselves are not executed by this scoped probe:
this is **not a full C reader/executable/MPI/sampling validation**.

The original `GetInfoOpt` computes the actual component flags. OptTrans uses
original `GetInfoOptTrans`, including its contiguous raw flag addressing at
NProj+NSlater, not a repaired interleaved tail. `flags-written.txt` observes
defined writes using INT_MIN markers before the C calls. Unwritten cells are
explicitly zeroed by the adapter afterward, **not claimed to be native C zero
values**. The fixture's source flags do not equal INT_MIN. Any comparison must
retain the mask and cannot claim C parity for those sentinel cells. Raw RNG
counts/state and numerical parameter values remain independently acquired.

Mode0 invokes original `SetFlagShift`; Mode1 leaves the shift flags zero, as
`readdef.c:1169–1177`. The stage order is actual `InitParameter`, optional
full-record `ReadInitParameter`, keyword `ReadInputParameters`, original
`SyncModifiedParameter`. Three observations are always present: initialized,
overlaid (after both overlay kinds), synchronized. No Julia-only overlay fix is
copied. Multiple explicitly requested groups each use a fresh original seeded
C initialization, corresponding to independent prefix acquisitions, not a
continuing optimization trajectory.

## Executed acquisition and reproduction

Linux x86_64 container `73c57e563c61`, GCC 13.3.0, Julia1.13.1 for orchestration,
BLAS none. Actual compiler options:
`-std=gnu11 -O0 -ffp-contract=off -DMEXP=19937`, source SFMT/readdef includes,
stage include directory and `-lm`. Actual command lines, source/input hashes,
executable hash, environment and timestamps are in each fixture's provenance.
GCC `-MM` confirmed compiler-local dependencies; their hashes are in
`compiler-inputs.sha256` (plus generated excerpt hash above). No numerical
acceptance budget was invented: provenance explicitly says UNREVIEWED.

Frozen adapter SHA256
`f9694cc7315b0e926aa1de8a551808c27e814857c022154d3f340cb8649b5da9`;
generator SHA256
`5f07817ee7678aa5fc2f89b5c5ab8b2a402e4dac2c1d153ed0c11a8d3f37dc3a`.

```sh
# Output paths must not exist and must be outside the checkout.
CC=gcc julia +1.13.1 c_toolbox/generate_reviewed_parameter_c_audit.jl extern/Julia-mVMC/test/integration/reference/general_rbm_cmp/inputs/namelist.def EXTERNAL_NEW_CANONICAL_STAGE 12395 extern/Julia-mVMC/test/integration/reference/general_rbm_cmp/inputs/initial.def 3
CC=gcc julia +1.13.1 c_toolbox/generate_reviewed_parameter_c_audit.jl tests/fixtures/physcal_181/hubbard_chain_dh_rbm_opttrans/inputs/namelist.def EXTERNAL_NEW_DH_STAGE 1 -
```

Actual terminal acquisitions:

| Acquisition | Handle / exit | NPara | Groups | Full phase records | Observed draws / idx |
| --- | --- | --- | --- | --- | --- |
| Canonical GeneralRBM | 74327 / 0 | 102 | 3 | 918 | 192 / 192 each group |
| DH2/DH4 + all nine RBM families + OptTrans (Mode1) | 46116 / 0 | 71 | 1 | 213 | 60 / 60 |

Frozen external stages are
`/home/vscode/.cache/mvmc/reviewed-c-parameter.AFmMKn/canonical-masked-groups3`
and `.../dh-masked`. All **44** persisted text artifacts were byte/hash compared
to those actual C-generated files. Fixture `archive.sha256` manifests contain
every persisted artifact hash. Earlier compiler failures 63710/37027 are not
execution coverage; preliminary unmasked acquisitions are not the frozen mask
evidence. No macOS run, Julia full-state/count equality, model trajectory,
new-CG reference adoption or numerical tolerance validation is claimed.

The suggested `rbm/run_rbm_general_cmp` and `opttrans/run_opt_dh24_rbm_cmp`
sparse runner fixtures contain RBM widths97 with only three flags. Original
C RBM readers require flag count == declared width. They are explicitly
excluded from this acquisition; no flags/geometry were expanded or rewritten.
The complete Mode1 DH acquisition above is a distinct input, not coverage of
those invalid sparse inputs or a Mode0 DH optimization trajectory.
