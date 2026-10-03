# Read-only SR semantic handoff to Chandra

This review supplies settings and proof gaps for the 149 previously syntax-only
rows: 22 parameter-sync, 63 stochastic-opt, 58 main-cal/SR, six weight-average.
It edits no inventory TSV, runs no new test, and assigns no PASS. Rust names
below are candidate assertion locations, not a claim that the original Julia
scenario has been replayed. Computed bounds remain #190-owned.

Current Julia source SHA256:

- `test_unit_parameter_sync.jl`: `49937cd1879e850aea8992d5ddafba96c5d1b1fcfe56261b94ff230bb8e0b171`
- `test_unit_stochastic_opt.jl`: `81249f43449c61fdd45803f6f8bc341341e4b83b98cce20cb9afd934667e4f97`
- `test_unit_vmc_main_cal_sr.jl`: `8600f1dc2493300a66da01fd6e97d47252ba409ee5c9eaa8a79276158ae26c7e`
- `test_unit_weight_average.jl`: `a5c5605d29c5fef3026589541a5146ab3e53584f3a55887b59f3eb78a7bdf9ff`

## Entry points S200/S205/S214

`test_unit_unsupported_inputs.jl:201-215` (S200): public para-opt with split3/CG1
must throw and mention `NSplitSize > 1 with SR-CG`; public PhysCal split3/general1
must throw and mention both NSplitSize and FSZ/general-orbital. Lines 253-259
(S205): public para-opt NSRCG2 must throw with `NSRCG >= 2`. Lines 346-360
(S214): public para-opt Lanczos1 must throw with NLanczosMode>0 and parameter
optimization; public PhysCal Lanczos2/general1 must throw with FSZ/general.

Rust `runtime_contract::rejects_modpara_solver_controls_instead_of_discarding_them`
actually calls `vmc_para_opt` for NSRCG2 and Lanczos1 and checks error substrings
plus unchanged parameters/flags/full624 RNG; it is a candidate for those two
public assertions, pending exact executed test evidence. Its split2-only input
is not S200's split3/CG1 case. `grouped_runtime_matrix_accepts_normal_physcal_and_rejects_unsupported_scopes`
and `rejects_unsupported_lanczos_physcal_combinations_before_sampling` call
validators only, not public PhysCal. They do not close the public entry-point
assertions. CLI grouped rejection tests are another API/context, not identical
library settings. Do not promote them merely because the validator is shared.

## Parameter sync: M0592-M0613 (22 rows)

| Rows / Julia lines | Exact scenario/expected condition | Candidate Rust coverage and remaining proof |
| --- | --- | --- |
| M0592-0595 / 14-44 | Gutz values 2+1.5i,-1+.25i; Jastrow5-.75i; empty flags: return0, real mean0, imag unchanged, real subtract mean | parser `sync_modified_parameter`; exact same-input return/imag/gauge assertions and executed ID needed |
| M0596-0597 / 47-71 | flags [true,false,...] for one Gutz real slot: all G/J unchanged | `optimization_flags::fixed_correlation_blocks_disable_gauge_shift_but_not_slater_normalization`; inspect exact fixture/settings rather than infer from name |
| M0598-0599 / 74-90 | Slater 2,-6+8i,.5-.5i; maxabs10; normalize to D_AMP_MAX | parser unit `sync_modified_parameter_rescales_slater_block`; current test/input equivalence and run ID pending |
| M0600-0603 / 93-116 | explicit shift_correlations=false: G/J/DH2 unchanged, Slater2/8i still rescaled | Rust fixed-block behavior is not proof of an explicit Julia keyword path; architecture/input-contract distinction needed |
| M0604-0606 / 119-161 | ParaQPTrans [1,-2+2i] unchanged; dedicated OptTrans[3+4i,1] -> [.6+.8i,.2] separately | `sr::opttrans_sync_matches_julia_with_declared_slater_normalization_from_c`; C active flag and declared widths must match, not a blanket no-normalization assertion |
| M0607-0608 / 164-187 | DH2 1+i..6+6i: subtract each three-bin mean, imag unchanged; Gutz10+.25i ->17+.25i | `dh4_projection::dh_gauge_matches_julia_declared_flags_compensation_and_shift_order` / DH2 gauge fixtures: require correct one-DH2 block and exact component flags |
| M0609-0610 / 190-211 | DH4 1..10 -> [-4,-4,-2,-2,0,0,2,2,4,4]; Gutz10->12 | same helper, DH4-only case not combined-only evidence |
| M0611-0613 / 214-227 | DH2 1..6 plus DH4 11..20; both centered; initial Gutz0->38 | same helper with combined=true; require original combined settings and executed ID |

The subsequent thirteen sync rows already semantically classified by another
batch are not counted again here. In particular empty flags and explicit
fixed-DH real flags are distinct, not interchangeable gauge eligibility.

## Stochastic opt: M0764-M0826 (63 rows)

| Rows / Julia lines | Exact scenario/expected condition | Candidate Rust coverage and remaining proof |
| --- | --- | --- |
| M0764-0765 / 22-32 | sequential xdot [1e16,100 ones,-1e16] dot ones ->0 | inspect CG scalar dot order; a solver residual or mathematical dot reference does not test cancellation order |
| M0766-0767 / 35-56 | store [[1,2],[3,4]], means [.25,-.5], diag[2,3], x[.5,-1], invW .25, shift .1; x unchanged, z[-2.18125,-4.8625] | `sr_cg::sampled_product_is_globally_summed_before_weight_mean_and_shift`; same matrix/storage/transposition and immutable x required |
| M0768-0776 / 99-131 | DH2 six slots at para3..8, delta-.125+.375i; following shared RBM at9 and Slater at18; unrelated blocks unchanged | `sr::rbm_parameter_updates_hit_all_shared_rows_and_leave_slater_at_final_offset` and DH index tests; exact two shared rows/last offset required |
| M0777-0785 / 134-166 | DH4 ten slots para3..12, delta.625-.25i; RBM13/Slater22; unaffected siblings | same candidate, not proof from DH2-only case |
| M0786-0790 / 169-244 | complex direct, nsite2/nelec1/sample1, cuts0/shift0/dt.25; only first DH2 real component active, OOdiag2/HO3: status0, delta-.75, remaining DH/RBM9families/Slater unchanged | fixed-update/direct observer tests alone do not establish this DH+all-nine-RBM enumeration scenario |
| M0791-0795 / 247-311 | Slater1 plus OptTrans2, only OptTrans[2] active, OOdiag2/HO3, dt.25; delta-.75, Slater unchanged, fixweight2 -> fullweights[2,2.5] | `sr::opttrans_direct_and_cg_updates_match_julia_component_flags_and_offsets`; actual C consecutive flags may intentionally differ from Julia pair-layout flags; label distinction |
| M0796-0801 / 313-322 | empty flags read component0/10 as0; flags[1,0,1] read0..3 as[1,0,1,0] | `sr::incomplete_flags_keep_missing_components_fixed`, `c_opttrans_flags_select_native_consecutive_writes`; generic accessor scenario must not be replaced by OptTrans exception |
| M0802-0804 / 325-371 | realCG, sample2/Wc4, size2, dt.5, mean0/diag2/HO1, raw stored O=2 each; status0, orbital10->9.5 | actual real-CG stored normalization/solver reference needed; direct108 audit is irrelevant and expanded179 discrete CG failures remain open |
| M0805-0820 / 374-414 | delta.25-.5i: G/J para1/2; nine RBM parameter blocks3..11 with duplicate mappings; Slater12/13; OptTrans14/15; unaffected siblings | `sr::rbm_indexed_updates_and_sparse_values_match_original_julia`, shared-row tests; declared-slot layout vs Julia row-values must be explicit |
| M0821-0826 / 417-465 | size2, complex OO16/HO8, mapping[0,1], shift.1/dt.05; asymmetric OO offdiagonal .3/.4 yields S[1.925,.4;.5,1.606], g[-.17,.046] | `sampled_direct_sr_matrix_gradient_factor_and_solution_match_julia` captures actual systems but does not automatically cover this asymmetric construction. U-factorization later must not erase lower-triangle constructor assertion |

## Main cal/SR: M1055-M1112 (58 rows)

| Rows / Julia lines | Exact scenario/expected condition | Candidate Rust coverage and remaining proof |
| --- | --- | --- |
| M1055-1058 / 18-29 | nProj3 counts[5,0,-2], complex sentinel99+99i: identity[1,0], count/zero pairs | `observables::set_projection_diff_writes_real_block`; compare full shape and overwrites |
| M1059-1066 / 33-77 | charge physical-only count2+3i; charge phys+hidden+coupling counts .1/.7, occupancy[1,1], xi1; derivative pairs value and i*value/tanh(.7) | `rbm_production::parsed_rbm_counter_incremental_ratios_and_all_derivatives_match_original_julia`; all-nine-family larger fixture is candidate, not evidence of these exact synthetic pair assertions |
| M1067-1070 / 81-114 | OptTrans2, fixweights[2,3], Pf[1+i,2-i,-1+.5i,.25-2i], ip5-2i: weighted QP group derivative/i pairs | `opttrans` derivative fixtures; exact grouping and ip needed |
| M1071-1076 / 117-206 | normal real nsite2/nelec1/QP1, specified inverse/Slater/Pf2, hop0->1 up; scratch/generic agreement; index/occupancy unchanged; allocated0 | `observables::real_transfer_fast_green_matches_generic_and_local_energy` is candidate numerical/restoration coverage; Rust allocation proof is separate and unverified |
| M1077-1082 / 209-351 | same real state, Transfer .7up/-.4down/.2both ->4 expanded entries; Gutz.31/Jastrow-.17 direct projection; repeat terms5times, enable inner workers; numerical equality and conditional scratch thread length=maxthreadid | `real_transfer_cache_eligibility_matches_julia_fast_path_requirements`, threaded182 transfer job; Julia workspace length is architecture-specific, neither fixture parity nor throughput proof |
| M1083-1089 / 362-449 | complex normal nsite2/nelec1/Slater4/QP1, specified complex inverse/Pf/ip; scratch/fallback8derivatives, scratch arrays length4, scratch allocation<baseline and<=128; identity OptTrans fast-vs-fallback equality | opttrans derivative and threaded182 scratch jobs candidates; allocation limits remain unverified, no numerical gate may stand in for them |
| M1090-1092 / 452-475 | nsite4 Gutz complex .11+.7i/-.03-.4i and real Jastrow, count deltas +/-; noalloc ratio==baseline; Julia<1.12 <=16 allocation ELSE ==0 | exact version branch: on required Julia1.13.1 only ==0 branch executes; <=16 row is historical-version branch, not fresh1.13 execution |
| M1093-1097 / 479-516 | complex size3/width6, w.25/e1.2-.5i, explicit O; first row wO/HO=e*wO, row1 untouched, rows2..5 wO*conj(O) | `threaded_issue182::independent_sr_and_copies`; analytic widths/layout/order and row1 preservation require individual matches |
| M1098-1099 / 521-539 | real size4 O[1,-2,.5,3], w.25/e-1.5, initialOO.1/HO-.2: += outerproduct/HO | same candidate, zero-initial test cannot establish += semantics |
| M1100-1101 / 542-574 | real size4/sample3, sin/cos store; sentinel -123.456; full-Gram active prefix numerical, trailing tail exact | threaded182 stored finalizers and `stored_direct_sr_gram_matches_sampled_julia_values`; require sentinel/tail assertions, not just output Gram |
| M1102-1105 / 577-657 | size3 total5/start2/count2, complex width6/real3; expected ONLY samples2/3; empty count0/start5 zeros full active Gram | grouped finalizer tests candidates; match offset and empty partition explicitly |
| M1106-1112 / 660-775 | size40, width80, w.375/e1.1-.25i; nonzero initialOO/HO; real/complex threaded==serial, store sample2 of5, full Gram5 | threaded182 jobs cover worker axes but Rust-to-Rust equality alone isn't independent oracle. Actual activation/tail/scratch evidence remains per-job, not blanket Julia source PASS |

## Weight averages: M1176-M1181 (six rows)

Julia state npara3: direct active OO length=size², Wc2, explicit MPI-size1
context skips reduction; CG active length=2*size, Wc4. Both require active values
divided, OO tail unchanged, all HO divided. Current Rust `average.rs:47` loops
the entire real OO allocation and takes no NSRCG argument. This is an identified
low-level active-range semantic mismatch, not a PASS. Production unused tails
do not satisfy the original tail assertion. C `average.c` uses direct
size*(size+1) or CG3*size over its contiguous OO+HO allocation; separate Rust
buffers need equivalent active extents. Parent/Chandra notified for owner
decision; no production change made here.

## Original PhysCal / third Hubbard Lanczos gaps

Original `phys_cal_equivalent.jl` six model counts are realHeis14/cmpHeis14/
FSZHeis24/realHubbard19/DHHubbard35/Kondo76. Each requires library status0,
consumed count exact, then one-body, direct two-body and factored Green comparison
`r.ok`; one sample filename per nested prefix, no extra match allowed. Existing
`physcal_issue181` trajectory cases check consumed counts against their own
new independent fixtures; they are not interchangeable with original input/
fixed-file/checkpoint settings. Status success in Rust is normally `Result::Ok`,
not a new integer field; map that architectural distinction explicitly.

Original `lanczos_equivalent.jl` third case is real `hubbard_chain_real`,
NPara19, has_two_body_ex=true: status0/count19, ls_out exactly3 at1e-8,
QQQQ exactly16 at1e-10, one/direct LS Green at1e-8 and factored LS Green at1e-8,
with both actual and expected files required. Current
`lanczos_transfer_physcal.rs:92` excludes all Green comparisons for that third
case, permits missing expected Green on other cases, uses QQQQ1e-8 rather than
original1e-10, and does not assert consumed NPara. Thus even an executed PASS
of that legacy gate cannot close these assertions. Two newer CLI Lanczos
families do not establish original third-case coverage without exact input and
per-file mapping. No tolerance repair or test edit was made in this review.
