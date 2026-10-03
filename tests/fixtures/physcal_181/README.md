# Issue #181 independent PhysCal trajectories

Current workload instruction: optimization long-run acceptance is20 steps,
not50, including serial/CLI/native-MPI/thread configurations. This PhysCal
fixture set records fixed-parameter measurement workloads, not optimization
prefix endpoints; its saved measurement counts and fixed-record input
NSROptItrSmp declarations must not be mechanically rewritten as20. Fresh
optimization20 endpoints belong to the corresponding prefix generators and
must be captured from reviewed PR54 `62b0f97`, with effective window metadata;
completed50 artifacts remain historical, never truncated into20 goldens.
CG kernel limits1–41 and refresh20/40 remain unchanged. Numerical bounds and
existing strict primitive RNG protections are not relaxed by this change.

These are newly generated Julia 1.13.1 references, not historical Julia 1.11
expectations and not Rust-generated snapshots. They were generated on Linux
x86_64 with OpenBLAS 0.3.30 ILP64, one Julia thread, one BLAS thread, seed 1,
and Julia-mVMC revision `8bb1b9e8ae47b1512c00b321be05664ddcac0fd1` using
`extern/Julia-mVMC/Manifest-v1.13.toml`. Each model records source/input hashes
and the actual environment in `provenance.txt`. The initial six-model
generator is recoverable at shared checkpoint `9e58590`, SHA-256
`61c99206319a0dbea9094f25fd3686bb3c7a6e7bb71b2c23170edb2717f9acd4`.

Normal Rust tests use the copied inputs, fixed records, and checked-in results
here. They do not read the vendor checkout, compile C, launch Julia, or invoke
the optional generator. The normal tests replay sampling only; the optional
full measurement gate additionally checks its saved configurations/counters
against these fixtures, before numerical output comparisons.

## Generation and observation boundaries

From the repository root:

```sh
JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  julia +1.13.1 --project=extern/Julia-mVMC scripts/generate_physcal_181.jl
# Regenerate a subset without repeating the six established workloads:
PHYSCAL181_MODELS=hubbard_chain_dh_overlays JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  julia +1.13.1 --project=extern/Julia-mVMC scripts/generate_physcal_181.jl
```

The optional generator copies input files to this directory and runs in a
temporary working directory. It instruments Julia source in memory only.
The original numerical/sampling functions, SFMT C calls, integer conversions,
and draw order are retained. It never edits vendor files. Each SFMT
`gen_rand32` or `genrand_real2` increments the observed primitive-word count;
unaccounted bulk/64-bit draw methods fail. `sfmt_dump_rand32` saves/restores
the global C SFMT state and therefore peeks without consuming random words.

`seeded/` is after seed 1, before loading/initialization draws.
`initialized/` is immediately after the single `InitParameter` call, before
fixed parameters are restored. `sample-0/` is after the complete unchanged
warmup and saved-sample workload, before measurement. Every stage contains
the cumulative primitive UInt32 draw count and the next 624 UInt32 outputs.
These are future outputs, not the internal 624-word state array.

| Model | Words after initialization | Words after complete sampling |
| --- | ---: | ---: |
| heisenberg_chain_real | 12 | 2680 |
| heisenberg_chain_cmp | 24 | 2690 |
| heisenberg_chain_fsz | 44 | 5339 |
| hubbard_chain_real | 12 | 3284 |
| hubbard_chain_dh_real | 12 | 3350 |
| kondo_chain_real | 64 | 42057 |
| hubbard_chain_dh_overlays | 12 | 3298 |
| hubbard_chain_dh_opttrans | 12 | 3365 |
| hubbard_chain_dh_rbm_opttrans | 60 | 3375 |

All five saved configuration buffers retain Julia/C's flat sample-major
ordering (`ele_idx`, `ele_cfg`, `ele_num`, `ele_proj_cnt`, `ele_spn`). Non-FSZ
spin buffers are empty. Counters retain Julia entries 1–9 followed by entry
11 (burn marker); reserved entry 10 is omitted to match Rust's public layout.
`fixed-parameters.txt` is the independent loaded/overlaid/synchronized block,
before initialization temporarily overwrites values.

The C input contract is explicit `zqp_opt.dat`, followed by named `In*`
overlays; a neighboring `initial.def` is not implicitly applied. Slater
rescaling remains active and correlation gauge shifts are disabled for
PhysCal. Sampling counts are never reduced or reseeded.

## Validation and first divergences

All six original models passed all 18 independent Julia-to-C Green output
comparisons using the existing per-output contracts. The largest absolute
difference was FSZ direct Green `7.276557828506114e-13`; complete per-model
results are recorded in provenance. These output checks do not establish an
independently traced C sampling trajectory.

The six focused Rust sampler tests passed exactly for seeded/initialized/final
next624 outputs, all saved configurations, and counters. No discrete
divergence was found at those observation boundaries. Tests report the first
different element if a checkpoint fails; they do not claim to localize every
individual proposed move or acceptance decision between checkpoints.

The full measurement gate exposed a formatting defect before numerical
comparison: Rust writes three rows of two fields for a three-term factored
Green, while C writes one row of six fields. The same defect affects Lanczos
factored Green. C authority is `vmcmain.c:671–675` and
`physcal_lanczos.c:137–140` (real), `259–262` (complex): pairs are emitted
in term order, with a newline after the complete loop. Rust's `io.rs` now
emits one ordered row for both outputs. The strict row check remains enabled.
The repaired explicit full gate passed all three scenarios, including both
Lanczos modes (94 seconds). This is not a floating tolerance
or parameter-layout discrepancy.

`hubbard_chain_dh_overlays` copies the fixed Hubbard DH PhysCal input and adds
the existing complete indexed `InDH2`/`InDH4` records from
`tests/fixtures/dh4/production_dh24_real/`. Their values replace the fixed
DH slices after loading. The reference is generated independently in Julia
without borrowing unoverlaid C outputs. Provenance labels the absence of a
C output/trajectory trace for this new parameter set.

## Reference scope and remaining evidence gaps

The original six workloads have no RBM or active OptTrans blocks. Existing
RBM/OptTrans optimization fixture trajectories are not PhysCal trajectories.
Julia's `vmc_phys_cal.jl` saves/restores only orbital, Gutzwiller, Jastrow,
DH2, and DH4 values around `init_parameter!`: active RBM values are replaced
by initialization; `init_parameter!` also resets OptTrans to definition
weights. Thus arbitrary fixed RBM/OptTrans zqp/overlay values are not safely
preserved by this reference runner. No such case is reported as a C-compatible
PhysCal pass; the generator's restoration guard rejects changed fixed blocks.
Dedicated C trajectory traces and a separately reviewed Julia reference
repair are still required to claim original-runner parity. The two new
combination fixtures instead use an explicitly labelled C phase-order
harness: initialize once, reload the independent fixed record, apply named
overlays, synchronize, then use unchanged Julia sampling/measurement kernels.
Both Rust sampling replays pass exactly. This is not a Julia runner repair or
full C executable parity claim.

`hubbard_chain_dh_opttrans` includes DH2/DH4 overlays, three nonidentity
OptTrans sectors and a complete `InOptTrans` record. The combined complex
case adds all nine RBM sections and complete InRBM records. Hidden sections
declare three slots (the unused fourth historical slot is removed explicitly),
so C declared widths and Julia inferred widths agree. Physical and
physical-hidden sections retain four slots. Complete fixed zqp records are
constructed from the independent DH input triples plus explicitly specified
RBM/OptTrans triples, never from Rust state. C OptTrans execution would require
its explicit `-o` activation; the Rust preparation API selects it explicitly.
Native C acceptance/trajectory of these composed inputs remains unverified.

`native-c-stages/` independently checks actual C InitParameter,
ReadInitParameter, ReadInputParameters and SyncModifiedParameter bodies for
three combinations and deliberately unnormalized, partially unmapped Slater
storage. Dimensions/definition flags are supplied explicitly; this is not a
full C definition-reader, MPI or sampling trace. Reproduction is
`julia +1.13.1 --project=extern/Julia-mVMC c_toolbox/generate_physcal_181_parameters.jl`.

`two-samples/` records both accumulated and averaged energy/ordered Green
arrays plus saved configurations, counters and next624 endpoints for all nine
models. Generate explicitly with `PHYSCAL181_TWO_SAMPLES=1`. Normal fixture-only
tests replay normalization for both frames of all nine cases. The optional
serial two-sample scenario compares all independent output files and the
genuine runner's final saved state and next624, not sampler-replay RNG alone.

`consumed-count.txt` is independently captured from Julia's fixed loader.
`optimization-flags.txt` and `optimization-flags-written.txt` independently
enumerate C's actual definition-reader writes, including the native contiguous
OptTrans offset and parallel-orbital duplication. Unwritten cells are masked
explicitly, not inferred from Julia Boolean flags or native allocation memory.
Regenerate this metadata without any sampling workloads using
`PHYSCAL181_METADATA_ONLY=1` with the generator command above.

`weighted-average/` is a standalone Julia energy/Green normalization fixture
with binary-rational synthetic inputs and a well-conditioned complex weight.
It checks retained Wc and ordered reciprocal multiplication for all three
Green families. It is not an independently captured full measurement trace.

## Acceptance inventory: assertions, not test names

| Behavior | Current evidence / exact gap |
| --- | --- |
| Nine saved trajectories | Exact five buffers, counters, seeded/initialized/final next624; consumed counts; original six plus DH overlay and two combinations |
| Fixed values / flags | Independent zqp expectations for original six; independent Julia fixed blocks for overlays/combinations; exact C-defined flag writes; optional full gate checks before/after preservation |
| Normal/FSZ factored output | Original six full output comparisons include FSZ; strict one-row ordered pairs; repaired full gate passes |
| No GEx order / duplicates | Synthetic writer test checks three ordered rows, distinct supplied values and repeated discrete keys; C `readdef.c:GetInfoOneBodyG` normal branch and `vmcmain.c:outputData` |
| Mode2 duplicate rejection | Actual C count/reader probe: two original rows reduce to count1, normal reader rejects 2 != 1 without GEx. Padded probe avoids C's extra out-of-range write; Rust early rejection is safe |
| Singular alpha | All-zero-moment writer test checks three NaNs and sixteen zero moments without abort; Julia fallback/C 0/0 singular arithmetic, not arbitrary native failure parity |
| Multiple NDataIdx / rerun | Independent two-frame output comparisons and final saved state/actual runner next624; exact same-environment rerun snapshot; signed -1→-01 unit contract |
| Out/var file lifecycle | Indexed per-sample truncate follows C `initfile.c:72–84`; explicit overwrite test; includes reserved DH/RBM/Slater/OptTrans storage |
| Weighted averages | Standalone kernel fixture plus independent accumulated/averaged frames for both samples of all nine cases; ordered arrays and retained Wc |
| Non-InterAll Lanczos | Optional modes1/2 compare exact file sets and per-output numbers for hopping+intra and exchange+spin-diagonal models; full counts retained; asserts actual terms, not model names alone |

The normal-test run passed 19 tests (including support helpers); the newly
added two-frame normalization test also passed separately. The explicit
three-scenario full gate passed once after the writer repair; the strengthened
serial second-sample independent endpoint check passed separately. No full
model/Lanczos workload was duplicated. Four output-block tests and six focused
IO tests pass; `cargo check -p mvmc-core --tests --locked` passes. Parent owns
the #180 full-prefix gate result and final milestone validation.

Main's #190/#193 policy was integrated at `d05cde3`. Native macOS
verification and final milestone validation remain separate requirements. No floating tolerance is loosened,
and no deterministic trajectory mismatch is treated as numerical noise.

## Final authority self-audit (not an issue-completion claim)

| Requirement | Observed evidence / remaining limit |
| --- | --- |
| Original six full PhysCal workloads | Explicit gate passes fixed preservation, exact flags/consumed widths, exact saved buffers/counters and actual runner final next624, strict rows/discrete indices and per-output numeric contracts |
| Non-InterAll Lanczos modes1/2 | Historical full C-reference gate passes three cases in both modes. New all-six-term nonzero combination passes independent Julia LS values, strict rows, actual saved state/RNG and C-reader-ordered PairHop metadata in both modes; composed native C measurement trace remains missing. See `all-terms-reader/README.md` |
| DH2/DH4, all nine RBM sections, OptTrans, In overlays | Independent Julia C-phase-order sampling fixtures pass; actual native C parameter-stage bodies pass. Full native C definition-reader acceptance and sampling/measurement traces for composed inputs are still missing |
| Nontrivial fixed normalization / reserved storage | Native C fixed→overlay→sync fixture uses Slater max16 and unmapped slots; exact stored phases, synchronized values, initial RNG/count and complete var storage pass |
| Julia reference retained-slot defect repair | PR54 commit `973184d49a16ea26b54a0ba609d10f86fa7c1857` repairs production retained-family initialization/load/overlay/SR/normalization; focused 171/171 passes, including independent C initialization and exact next624. Existing pinned fixtures were NOT regenerated with this commit; this is not a full trajectory or MPI claim |
| Two-sample trajectory / actual final RNG | Actual two-sample runners pass exact final saved buffers/counters/next624, fixed values/flags and ordered outputs for all nine fixtures. Hubbard mode2 now additionally requires and numerically compares all five independent LS files for BOTH frames; no file-presence-only fallback |
| Weighted measurements | Both captured accumulated/averaged frames pass normalization kernels. A no-op recording reducer observes actual raw energy sums and already-normalized ordered Green arrays for BOTH runner frames of all nine; stage comparisons pass. Pre-normalization Green sums are not exposed at that boundary |
| Indexed truncate / signed indices / zero Green | C-derived zero-Green file sets pass real runner modes0/1/2; mode2 empty bytes exactly newline. Indexed overwrite, complete var slots, signed -1→-01 and checked-overflow tests pass |
| Duplicate OneBodyG without GEx | Normal ordered duplicate output passes; mode2 actual C counter/reader rejects count2 vs reduced count1. No full LS claim from a forced bypass state |
| Duplicate OneBodyG WITH GEx | C probe returns canonical count1 and indirect reader success. Initial Rust count2 divergence is fixed; fixture-driven ordered canonical-layout and real runner mode0/mode2 checks pass. This is not a new native C sampling trace |
| Provenance / runtime independence | Julia1.13.1 Manifest-v1.13, Linux OpenBLAS and revision/input/source hashes recorded; native probes carry compiler/source/driver scope. Normal tests use fixtures only. Native macOS verification remains missing |
| First divergence localization | Checkpoints report first differing saved buffer/counter/word/output field. Individual proposed moves and acceptance decisions between checkpoints are not captured |

The new zero-Green and duplicate-reader tests passed separately after the
single full-gate pass. #180's 56-prefix gate is parent-owned; its captured
result is not inferred from process exit or the focused IO passes here.

After main's gate-reporting merge `6cd0374`, the latest exact out-byte patch
passed all six IO tests; 27 focused integration tests passed. The subsequent
all-nine two-frame endpoint/reduction-stage test passed separately (2 seconds)
without repeating the existing long Lanczos gate. These new normal tests read
only checked-in fixtures; their recording reducer changes no values or RNG.

Provenance clarification: older two-sample records describe `sample-0` in
their stages shorthand, but their `sample_quantity=2` and separately captured
`sample-1` files include the second frame. Future generator records explicitly
enumerate the frame range. Julia output files are numerical/layout references,
not exact C whitespace oracles; separate fixed-value native C formatting
fixtures establish bytes. Historical generator hashes/timestamps are retained
rather than relabeling old runs as regenerated by the current script.

## Latest all-term and two-frame LS checkpoint

New `hubbard_all_terms_lanczos1` and `hubbard_all_terms_lanczos2` independently
exercise all six non-InterAll term types together. See
`all-terms-reader/README.md` for the actual C ordered metadata, numeric
coefficients, explicit regeneration commands and scope limits.

Run `51fc938d-0902-44fe-92fe-ffe404e5c237` passed 30/30 focused tests
(three long optional scenarios not rerun). After regenerating only the
existing two-sample Hubbard reference to retain its previously omitted LS
outputs, run `d82baf71-1c5c-4f74-a8dd-b9c1f26dd9e4` passed both the
all-six-term modes and all-nine two-frame runner comparisons. Hubbard's
independent final draw count remains 6333; both frames now include numerical
LS moment, energy and corrected Green expectations.

The intervening run `001593eb-c2ba-41fe-98dc-5d38b052c4ad` failed an observer
call-count assertion before LS comparison: the shared runner now reduces
only the active SR branch, so real models have eight complex calls per frame,
not ten (two SR reductions use f64). The observer now checks that exact branch
and offsets; it still compares both raw energy and ordered Green stages.
No runner or numerical expectation was changed to repair this assertion.

Seven fixture chunk readers were changed to `as_chunks` with explicit
remainder assertions; targeted core-lib/#181 clippy passes. Native full
combo sampling/Lanczos traces, raw pre-normalization Green observer coverage,
move-by-move traces and macOS/final milestone verification remain distinct
evidence limits. The bounded C duplicate/pair readers are not promoted into
native sampling claims.

## Required contracts versus stronger optional native runs

The issue's acceptance text requires fixed parameters, consumed widths,
initial draws, saved configurations/RNG, weighted averages, file behavior
and supported non-InterAll Lanczos terms. It does not require a full native
C executable run for every combination. Current evidence is explicitly mixed:
independent Julia1.13.1 trajectories/numbers, historical C workload outputs,
and extracted actual C parameter/read/format/normalization kernels. Full
native composed trajectories would strengthen this evidence, but their
absence must not be described as a blanket missing acceptance cell.

`native-c-weighted-green/` now establishes C normalization of all nine
two-frame raw Julia arrays and records the actual reciprocal first divergence
for a non-real synthetic weight. The runner reducer checks actual normalized
arrays against both independent references. Actual runner pre-normalization
Green capture remains an explicitly requested strengthening in progress,
awaiting the runner owner's observer; raw fixture inputs are never derived
by multiplying Rust averages back by Wc. Proposed/accepted move-level traces
are likewise stronger localization evidence beyond the required saved
configuration/counter/next624 checkpoint comparisons; generic sampler
observer ownership remains with Wegener. Native macOS and final integrated
milestone verification remain parent-owned pending validation.

The optimization var writer's omitted DH families was a required C output
defect, not rounding: it now writes full projection/RBM/declared Slater/
OptTrans storage. Four actual C parameter-stage cases plus a binary-exact
C byte fixture verify all slots and lifecycle. Indexed PhysCal writing is
unchanged. Owned IO/output/#181 test directories now use exclusive create
with collision retry; they never remove pre-existing directories to start.
