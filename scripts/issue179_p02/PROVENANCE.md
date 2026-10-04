# SOURCE and bounded synthetic verification

Publication baseline checked read-only: origin/main
`da979375739292e5e0c68a0413768a78094bee74` (remote and local ref agree).
The publication candidate is isolated from the shared historical checkout and
its unrelated dirty changes. Only this new directory and a Linux/default-only
CI step are in scope. At SOURCE review, no commit/push was performed. Cargo, production, fixtures
and existing tool implementations are unchanged. The CI step reuses the existing
uv setup and system Python selection and runs only the four synthetic modules.

The eleven Python files are byte copies of reviewed rank packet V3aJ5n, whose
14-member SOURCE inventory SHA is
`1ad067d2c5dbec61e3c1c65671c891e725bc7b04ee867f3bd8611d733b1ad9a8`.
The packet's three review documents are summarized here, not treated as runtime
expectations. Original h19YjL utility lineage is frozen separately. P02 changes
rename imports, bind NStore1 explicitly and add canonical source/typed seed-group
gates; `d:rank` is not emitted by either pinned writer and is rejected if injected.

Actual synthetic receipt (container73c):
`/tmp/mvmc-179-p02-synthetic-proof.0Wyffi`, session58953 terminal0.
`evidence/controls.stderr` records **29/29 PASS, 2.153s**, SHA256
`20aa2c833990a7f4a38e87919eda8c639b783e475674f08a4c8ea971238b2c12`.
`evidence/command.txt` records exactly the four README modules under uv0.12.21,
Python3.12.3, offline/no-project/no-downloads/-B. Source, exact membership,
tools, Python/providers, launch, authority-document POST checks and cleanup/
resource/outer aggregates were all0. Parent independently reviewed the full
command, 29 logs and recorded exit statuses. This is not a newly executed test
of the publication directory; source byte identity is the transfer association.

Actual model acquisition is **NOT_STARTED**: container SHM free32008KiB is
below the unchanged32768KiB startup guard. The approved one-pair launch remains
unused. No unknown shared-memory region was moved/deleted and no alternative
container/environment was substituted. No native/provider/discrete/numerical
model PASS is asserted here.

Frozen input/source interpretation: Rust recorder08e138... lines689–696 and
Julia recorderbc568... line328 emit seed/steps/window/NSRCG/NStore in that order.
P02 modpara4a1ce2... fixes [1,1,1,0,1], site6 and sample3. Rust state.rs614–617
allocates complete saved buffers as samples*n_size, samples*2*site (twice) and
samples*n_proj:18/36/36/6 here. These are complete chain buffers, not one sample.
Source filenames bind rank externally; width1 group[rank,0,1] and seed rank+1
are checked internally. No multi-rank GROUP equality follows from width1.

C-written-mask source SHA and the three original family-input SHA values are
literal in `c_written_mask.py`. Odd imaginary slots are unspecified by this
real-family reader contract, not expected zeros. The diagnostic preserves raw
flags and Rust's normalized-OO storage tail rather than fabricating matches.

`SOURCE.sha256` covers all proposed directory files except itself. Check its
relative checksums and exact membership from this directory; added files are
not authorized by checksum replay alone. Native/source hash pins remain
historical and do not automatically advance when main or Julia PR54 advances.
