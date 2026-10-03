# IO134 semantic handoff to the issue184 ledger owner

No TSV edits or blanket134-PASS claim. This maps every original assertion
range, including API/C differences and residual gaps, for Chandra's adoption.

Original Julia revision8bb1b9e8ae47b1512c00b321be05664ddcac0fd1:
factored_green99 SHA2566c69a832c474156c45711a2f2da13f72f4789aa6d06d8bda435cb55573eada12;
read_opt_para35 SHA256744bc2248c39395399a6c05341d5eed140471e692236cc2207458ceec178d86c.
Both are synthetic/unit contracts. The third Hubbard runner is separate.

Rust test source: crates/mvmc-core/tests/physcal_issue181.rs,
SHA2565f0960d3bdd368dc8564fcadc713b249d39dd3c6d7dbf00e3427e1c30eb5dc42.
Shared HEAD66e496fdbe5947a396a30be3b25ee07fc9a8c370 plus preserved drafts.
Latest owner session3613 terminal0, nextest9a33ef59-3de7-43d9-8f6c-ea792263854f:
15/15 pass,0.219s,27 excluded by explicit filter (not failed/unsupported skips).
This is a focused mutable-worktree result, not frozen full-workspace validation.

```sh
cargo nextest run --locked --cargo-profile test-fast -p mvmc-core \
  --test physcal_issue181 \
  -E 'test(original_io134) | test(third_hubbard_lanczos) | test(singular_lanczos_alpha) | test(no_factored_terms)' \
  --no-fail-fast --retries 0
```

Test aliases below all refer to that integration test file:

- Shapes: original_io134_physical_quantities_shapes_are_explicit
- Canonical: original_io134_canonical_pairs_keep_c_file_order_and_append_constituents
- NormalWriter: original_io134_normal_writer_keeps_literal_canonical_row_and_complex_pair
- QQQQ: original_io134_qqqq_cells_follow_c_flatten_and_conjugation
- Factor: original_io134_factored_accumulation_conjugates_and_adds
- Dispatch: original_io134_occupied_green_dispatch_and_fsz_state_publication
- Blocks: original_io134_fixed_loader_family_offsets_and_duplicates
- LoaderEdges: original_io134_loader_c_eof_nonfinite_and_atomic_tail_contracts
- PublicNormal: original_io134_public_runner_normal_routing_and_qcaq_are_repeatable
- Supported: original_io134_factored_supported_cases_reach_public_validation
- Index: original_io134_four_output_indices_use_c_signed_start_plus_sample
- LSWriter: original_io134_synthetic_lanczos_writes_energy_and_all_green_kinds
- Singular: singular_lanczos_alpha_writes_nan_without_aborting
- Duplicates: no_factored_terms_preserve_one_body_order_and_duplicate_rows

## Assertion-range mapping

| Original IDs | Exact condition, current evidence, distinction/gap |
| --- | --- |
| M0614–0615 | Initially empty Julia state index vectors: intentional storage ownership difference. Rust stores canonical terms/pairs on ExpertModeData, not PhysicalQuantities. Shapes does not pretend to assert absent Rust fields. |
| M0616–0622 | Shapes uses original constructor counts2/1/3; asserts2/1/3 Green sizes and16/8/4/12 Lanczos sizes. Exact scalar counts. |
| M0623–0624 | Canonical keeps original a=(0Up,1Up),b=(1Down,0Down) without GEx; with GEx dedups and appends c=(2Down,3Down) in order. Extra duplicate assertions follow authoritative C reader policy. |
| M0625 | Original Nsite2/site5 helper rejection is NOT tested by Canonical. Genuine remaining exact parser/bounds diagnostic gap; do not mark from model execution. Parser owner coordination needed. |
| M0626 | Julia invalid symbol :both cannot be constructed as Rust Spin enum. Intentional typed-API difference, not executable C-invalid-input proof. |
| M0627–0628 | Canonical asserts zero-based(0,1),(0,0), equivalent to Julia one-based(1,2),(1,1). API indexing distinction explicit. |
| M0629–0631 | Factor calls actual public production accumulator twice: fixed2+i,3−4i,weight.5 gives1+5.5i then2+11i. Exact binary-rational products; input locals unchanged. |
| M0632 | Dispatch calls actual normal public number-operator kernels for original occupations[1,0,0,1],idx[0,1],cfg[0,−1,−1,1],IP1: both local values1. |
| M0633–0634 | Private normal measurement routing is exercised by PublicNormal through public runner. Representative six-site/100-sample case, not original two-site helper w=.5. Sum of averaged number operators equals fixed particle count; exact original helper-boundary assertion remains distinct. |
| M0635–0639 | Dispatch calls actual public FSZ measurement using original two-site occupations/spins[0,1],weight.5. Locals[1,0],weighted[.5,0],direct counterparts[1,0]/[.5,0],factor.5. Rust publishes these into state. |
| M0640–0641 | Original Julia acc-overload leaves state locals/sums zero. Rust API publishes into state and has no equivalent detached overload. Intentional architecture difference, not a claim of these original unchanged-state assertions. |
| M0642–0646 | QQQQ calls actual accumulator at weight.5: complex H=2+3i,H2=5+7i checks cells.5,1−1.5i,37; real2/5 checks1,12.5. Zero-based cells0/2/15 follow C flattening. |
| M0647 | QQQQ checks exact length15 error and unchanged buffer. Bounded Rust API rejection; no C invalid-buffer execution claim. |
| M0648 | PublicNormal traverses actual private QCAQ path via public runner: all four number-operator blocks obey N*[1,E,E,H2]. Independent checked-in moment records plus particle-number algebra; repeated identical-input/seed/config output bytes. Representative case, NOT exact two-site helper arbitrary H1=2,H2=5,explicit half-weight [.5,1,1,2]. No invented test-only public API. |
| M0649–0654 | NormalWriter uses original Nsite4,one0Up→1Up,value1.25,factor2.5−i,index1. Exact file presence,one6columns,indices and factored single ordered2-column row. Rust additionally needs declared GEx term for C file-opening policy; Julia standalone helper supplies only state pairs. |
| M0655–0665 | LSWriter mode1,index7,original QQQQ values at2/3/10/11/15=(1,2,3,1,5). Asserts files through actual writer,3energy fields,E−3.5,ALPHA−.75 (not variance),all16 moments. Expectations independently substitute exact rational alpha into C formulas. |
| M0666–0677 | LSWriter mode2,index4 with original QPhysQ one[1,2,3,4],direct[5,6,7,8],factor[9,10,11,13]. Asserts ordered row/column counts,original index tuples,one−.8/direct−.4/factor.9 and imaginary0. Only scalar rounding budget2eps; not widened output policy. |
| M0678 | Julia warning assertion is separate; Rust singular fallback does not emit that Julia warning. Intentional diagnostic/API difference; no log-identity claim. |
| M0679–0682 | Singular executes writer with zero QQQQ,three NaNs,16 zero moments,no abort. Index7 vs original3: numerical/no-abort property tested, exact original filename not claimed. |
| M0683–0686 | Index executes actual writer for(start,sample)=(1,0),(1,3),(7,0),(7,2),verifies filenames1/4/7/9. |
| M0687–0692 | Intentional C authority difference: C indexed out/var files truncate independently; original Julia uses shared out/var append/truncate. Index asserts C-indexed paths and absence of shared paths. Existing io::tests::physcal_indexed_out_var_truncate_each_sample_and_keep_empty_green_contract supplies lifecycle assertions; its previous parent6/6 unit proof is separate from this15-test run. Do not label Julia shared-file line counts equivalent. |
| M0693–0697 | Canonical preserves original a,a,b without GEx. Duplicates executes writer with reverse-order equivalent duplicate-list property and literal ordered rows. Original exact a,a,b writer settings differ: classify property/representative evidence, not identical record replay. |
| M0698–0699 | Duplicates checks actual mode2 safe diagnostic and independently captured C-reader result2declared→1dedup count→normal-reader rejection without GEx. C supported/invalid-input classification, not a Julia-only feature guard; bounded reader only, no full C sampling claim. |
| M0700–0705 | Canonical asserts original single-term list/pair. Shapes supplies constructor layout; initialization wiring through private state_from_data is additionally traversed by public models. Exact original Nsite4 private initialization call is NOT separately exposed/replayed. |
| M0706–0709 | Canonical asserts original two shared swapped terms: list[a,c],pairs(0,1),(1,0),actual PhysicalQuantities lengths2/2/direct0. Canonical/constructor composition is tested; distinguish private initialization wiring as above. |
| M0710–0712 | Supported reaches actual public full validation with a valid parsed control,mode0: normal+GEx,general1+GEx,general1withoutGEx all accepted. Original isolated Julia hook/default-data settings differ; full validator case avoids unrelated guards masking the intended combinations. |
| M0713–0718 | crates/mvmc-core/tests/initial_params.rs::strict_loader_consumes_unique_parameters_verbatim_and_is_idempotent uses original five coefficient literals .1,.2,.3,.4−.1i,.5−.2i,consumed5. It also includes repeated orbital mapping. Loader-specific owner run16791fc2-a965-484e-96ab-00ad8a016fe4 terminal0,7/7; not part of latest15-test command. |
| M0719–0720 | Same initial_params test loads twice and optional delegation,asserts unchanged original coefficients. Loader has no RNG argument; do not infer whole-run draw identity from idempotence. |
| M0721–0727 | Original dynamic malformed loop includes missing,empty,short,trailing1,trailing3,garbage and three NaN/Inf/−Inf expansions. Existing initial_params/public/CLI tests cover bounded malformed diagnostics/atomicity (separate named runs). LoaderEdges tests complete-record last-wins plus malformed later record/tails atomically. C empty file returns0 unchanged; NaN/Inf/−Inf are accepted spellings, unlike Julia rejection. Explicit intentional C differences, not test omissions repaired by overrejecting. C unchecked malformed fscanf is not an atomicity oracle. |
| M0728–0731 | Blocks exact DH2 six literals1.1−.1i through1.6−.6i,consumed11,then Slater.4−.1i/.5−.2i. |
| M0732–0734 | Blocks active OptTrans tail.7−.8i,consumed6,Slater unchanged; actual declared tail enabled. |
| M0735–0743 | Blocks exact five RBM coefficients .61−.01i,.62−.02i,.71−.11i,.81−.21i,.82−.22i,consumed10,Slatercorrect. Phys-layer duplicateidx0 gets same value twice. Rust explicitly declares widths[0,0,2,0,0,1,0,0,2],not inferred storage from mappings. |
| M0744–0747 | Named initial_params test above asserts read_initial_def validtrue,originalvalues; missingfalse via missing_file_and_empty_parameter_model_have_distinct_contracts. Julia warning text is not Rust loader diagnostic identity. |

## Separate third Hubbard runner

third_hubbard_lanczos_consumed19_matches_independent_c_all_five_outputs
executes actual public runner,seed1,real,oneframe,mode2,consumed19,fixed params
unchanged. Strictly compares LS out3,QQQQ16,one/direct/factored row/index/value
contracts to third-hubbard/expected. It passed in the15-test run above.
See third-hubbard/README.md and original metadata for independent historical
C622166a native macOS acquisition/compiler/BLAS provenance and source hashes.
No new C build,Julia generation,MPI execution or bitwise computed-float claim.
This runner evidence does not retroactively label all134 synthetic assertions
as covered. No numerical tolerances are enlarged and no strict primitive RNG
protections removed. Reviewed CG62b lineage is unrelated to this fixed-parameter
historical C measurement oracle; no old-CG fixture is reused as a new oracle.

## Remaining precise gaps

1. M0625 exact invalid-site helper/reader diagnostic: parser-owner coordination.
2. Exact private two-site normal routing/QCAQ helper boundary: representative
   public-runner proof exists, but original arbitrary helper operands differ.
3. Original private initialize_phys_quantities! wiring: canonical/constructor
   and actual public runner each tested; identical original private call is not
   a Rust public API. Record architecture mapping rather than inventing one.

Typed symbols,detached accumulator routing,Julia warning text,shared append
files,and stricter nonfinite/empty/extra-record loader policies are explicitly
classified differences,not hidden PASS or a request to overreject C input.
