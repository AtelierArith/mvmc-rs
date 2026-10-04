# QPTrans payload boundary (withdrawn guard; revised controls)

## Current correction — no production patch

The literal `NQPTrans` guard below is withdrawn. The previous rejection test was
not a valid C-domain negative: its second physical line `1 0.75 -0.5` defines
count zero under C `%s %d` scanning. The current main `read_definition` correctly
accepts this, and that is not a proven silent-default bug.

`readdef.c:508–510` calls `ReadBuffInt` rather than `ReadBuffIntCmpFlg`.
`ReadBuffInt:114–124` reads two physical lines, consumes an arbitrary label and
integer prefix, and permits zero. `ReadDefFileIdxPara:873–875` attempts the five
header skips; `GetInfoTransSym:2240` reads no body when the count is zero.
Canonical TransSym therefore permits arbitrary count labels and two-line
zero-count definitions. No literal label restriction is C-faithful.

The separate candidate now has **zero production diff**. Two SOURCE-only tests
preserve arbitrary positive-count labels with independent coefficient/map/sign
expectations, and two-line explicit/partial-scan zero-count inputs. They exercise
canonical TransSym and the existing indexed QPTrans alias through both public
loader modes. New test SHA-256:
`52f173b297b0fb2d87294a1679dffa47f0b5175353bb4a2cbf9b5cf232d1d59a`.
These revised controls subsequently passed the bounded acquisition below;
publication remains subject to parent review. There is still no production patch.

The generic alias domain cannot be inferred from payload bytes overlapping
defined C input. Any proposed generic-API rejection must be based on an explicit
entrypoint/keyword contract, not a payload heuristic. C excludes QPTrans as a
keyword, but Rust intentionally has an indexed alias; removing that alias is a
separate architecture decision, not justified by the former rejection fixture.
Julia's separate generic constructor/parser remains an API mapping gap; no
new Julia-only numerical support is presented as C parity.

## Revised controls acquisition (12c5)

Final owner session `63567` returned zero. Focused: 2PASS/0skip/0.007s.
Full parser UUID `9b86fcf2-2ba0-40be-83a1-f2a79df8ce6f`:
275PASS/0skip/0.425s. Targeted strict Clippy, rustfmt, full source/tool
before-after checks and aggregate pipeline all returned zero.

Receipt filesystem:
`happy_jackson:/tmp/issue184-qptrans-boundary-final-proof`.
Source checkout:
`happy_jackson:/tmp/issue184-qptrans-boundary-hydrated12c5`.
Dedicated target:
`/home/vscode/.cache/mvmc/issue184-qptrans-boundary-final-target`.
Exact commands/stdout/stderr/status and all selected ELF hashes are retained
in that receipt. The complete source manifest binds the hydrated static inputs.

Prerequisites: authoritative main12c5 gitlink Julia
`c0788c34a6a5753c611633a97cd1ea233203320c`, acquired with `git archive` of only
`examples/inputs/{heisenberg_chain_fsz,heisenberg_chain_cmp,heisenberg_chain_real,hubbard_chain_real}`.
No Julia process or fixture regeneration ran. The initial unhydrated acquisition
is retained at `/tmp/issue184-qptrans-boundary-controls-proof`: focused2PASS,
full270PASS/5missing-input FAIL, pipeline1. A subsequent reused-target attempt
is retained at `/tmp/issue184-qptrans-boundary-hydrated-proof`: old ELF absolute
paths still referenced the unhydrated checkout; source staging also overlapped
that attempt and failed its source check. Neither failed receipt is promoted.
The final acquisition began after staging completed and used its own target.

This final result adds only C-supported input controls. Generic Julia
`QPTransTerm`/parser semantics are not implemented by this Rust indexed API and
remain an explicit #184 unsupported API gap, not a C bug or full-audit success.
This README result update is post-acquisition documentation only; frozen
container source and previous S128/S129 sources are not edited.

## Historical, superseded proposal and acquisitions

The sections below preserve the original investigation and failed expectations.
Their interpretation as a public-boundary bug and proposed repair is superseded
by the C count-reader correction above. No historical execution is relabeled.

Related to #184 and #185. Baseline: main
`12c5bd83967c12b8d246fa81a102ada360bc19e1`; this is a separate candidate from
the frozen S128/S129 publication. No generic numerical implementation is included.

## Three distinct contracts

1. C namelist accepts `TransSym`, not `QPTrans`:
   `extern/mVMC-1.3.0/src/mVMC/include/readdef.h:33–60`, SHA-256
   `7e7a4a85b759bbbbe61657a9cd67e82112bdbe3f08405c20e2223c59669cca0d`.
   `GetFileName`, `readdef.c:1699–1750`, rejects an unknown keyword.
2. C `GetInfoTransSym`, `readdef.c:2232–2265`, reads indexed complex coefficients,
   site maps, signs and inverse maps. Complete `readdef.c` SHA-256:
   `6c53cb832f93d6cbfd7cea955fbb693738af5536b913d36af32b98eed38c32d9`.
   These are inspected source boundaries, not a new native acquisition.
3. Julia at `8bb1b9e8ae47b1512c00b321be05664ddcac0fd1` has a separate public
   `QPTransTerm(site, momentum::Vector{Float64}, phase::Float64)` constructor.
   `MVMCExpertModeParsers.jl/src/parsers/qptrans_parser.jl:69–100` consumes tokens
   two through last as momentum and also takes the last token as phase when
   present. Its generic loader pathway is not C indexed TransSym numerics.

Rust `canonical_namelist_keyword` explicitly retains `QPTrans` as a legacy
spelling alias of indexed `TransSym`. This is a Rust architecture extension, not
an assertion that C accepts that keyword or Julia's generic term semantics.
The candidate preserves this indexed alias and the canonical C keyword.

## Actual failure and proposed repair

The unmodified public loader accepted the two-row generic payload
`0 0.25 0.5` / `1 0.75 -0.5` as `Ok`, with empty `input_errors`, zero projection
count, and empty coefficients/maps. The low-level indexed parser defaulted to an
empty section when the indexed header was absent. Retained RED UUID
`b0bf49cd-709c-4efb-b5aa-1786f65d7024`: one rejection test failed and the indexed
positive control passed; exit 100. This is a public-boundary error, not RNG or
floating-point evidence.

The focused candidate requires the indexed second-line `NQPTrans` header before
calling the existing numerical parser. A different payload returns typed
`ParseError::InvalidInput` through the existing required-definition error path.
Both public parser modes are covered; both `QPTrans` and `TransSym` spellings have
an independent complete indexed positive control with coefficient `1+0.5i`,
map `[1,0]`, signs `[1,1]`. Generic rows are never reinterpreted as indexed data.

The first candidate acquisition on the previously reused checkout was also
terminal 100 (UUID `75fbb69c-7884-4995-a39a-4bb0836fb292`, five controls passed,
one rejection failed). It remains failed evidence; the candidate is NOT yet
validated GREEN. That checkout mutation affected a previous proof's live-source
association and was disclosed; the S128/S129 owner subsequently acquired a new
isolated successful proof. Future candidate verification must use the separate
checkout `happy_jackson:/tmp/issue184-qptrans-boundary-fresh12c5`, not either frozen
S128/S129 checkout. No failure is reclassified as a pass.

This header guard does not claim comprehensive validation of every malformed
indexed row, direct-parser behavior, generic Julia API coverage, native C/MPI
execution, sampling, or complete #184/#185 acceptance. No fixtures or tolerances
are changed. Full parser tests and targeted Clippy are required before publishing.
