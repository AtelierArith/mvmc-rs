# Source-derived scalar dimensions (not observed Rust expectations)

## NEW061 lineage staging (metadata only)

Current source base061a432a126cdcc772c93b14a55325e262fd0bb5. Approved current
types.rs SHA256 cc112e5108d723275c58dad281b1f6016ea6f1599fdab414781e73bb652750a3;
state.rs unchanged a912eab6001dc2ed336a2a15c87c0a869153a049e4702c323cb06d0fdb3b3e74.
Current emitter SHA2563eeabbafec3dba7f72e29f1536d0c265f71855021fd731c529ee7a0ba8adabaa.
types delta is only inverse-query helper, no layout changes. Source formulas below
retain their historical2a8 line anchors; current types anchors shift by79 lines
after the new helper. This is not a current parser/model execution claim.

case-schema-79.json is the exact447cf34d795268c270993e4a5ac99ace0bc2550e8393fc2318eb2e0d7d11000a
metadata-only producer76683 output. Its source_lineage explicitly separates current
source from historical input/inventory base. Four realFSZ closures remain missing.
Old2a8 JSON and94617 failure are preserved outside this NEW integration. No83 run.

## Preserved historical formula anchors

Frozen numerical/layout source: main2a8
`2a8e6cd33e7b2840a4935ca40bb7792621f9080a`.
state.rs SHA256 `a912eab6001dc2ed336a2a15c87c0a869153a049e4702c323cb06d0fdb3b3e74`;
parser types.rs `e34d3dd40535a92a3a0f77a2dce6d26d38488de15262935d53b73190d4ef439e`.

Let p=count_variational_parameters(), s=1+p, e=nelec (per spin),
q=allocated full QP planes, m=allocated sample count. Complex values are
flattened by the existing complex emitter to two scalar components each.

| Exact state record kind | Scalar dimension | Source anchor |
| --- | --- | --- |
| parameters | 2p | threaded parameters(): projection + RBM + declared Slater + OptTrans; parser types.rs:1052 active width |
| energy | 10 | emit_state literal five complex EnergyData fields |
| pf | 2q | state.rs:795 complex_pf allocation |
| inverse | 2q((2e)^2+1) | state.rs:179–186 InvMColMajor zeros and :249 whole backing as_slice |
| inverse-real | q((2e)^2+1), or 0 in complex mode | state.rs:796–809 inactive real allocation branch |
| pf-real | q, or 0 in complex mode | same real_pf branch |
| oo | 4s(2s+2) | state.rs:454 complex OO length 2s(2s+2), flattened complex doubles it |
| oo-real | s(s+2), or 0 in complex mode | state.rs:458–465 real branch |
| ho | 4s | state.rs:455 complex vector length 2s |
| ho-real | s, or 0 in complex mode | state.rs:464 |
| store | 4sm | state.rs:456 unconditional complex store allocation |
| store-real | sm, or 0 in complex mode | state.rs:465 |
| onebody | 2 * canonical GreenOne count | state.rs:1117 PhysicalQuantities allocation |
| twobody | 2 * GreenTwoEx count, NOT TwoBodyG/DC count | state.rs:1118 phys_cis_ajs_ckt_alt allocation |

The inverse `+1` is a pad entry in each QP plane, not an extra electron;
emit_state uses whole as_slice, not qp_matrix_slice. OO includes reserved/padded
trailing blocks because the emitter emits the full vector, not the active solver
submatrix. Store allocation does not branch on NStore: store0 remains nonempty
for positive p/q/m. An empty store0 is therefore not permitted by schema.
These are allocated test-observation shapes, not claims all pad slots are C
physical observables or all fields participate in a solver comparison.

## Static input progress

Original 8bb static input header resolution, not fixture result inference:

| Model | Declared projection | Slater | RBM | p before OptTrans | nelec |
| --- | --- | --- | --- | --- | --- |
| Heisenberg real/cmp | 1+1 | 12 | 0 | 14 | 3 |
| Heisenberg FSZ (and forced real-FSZ variant) | 1+1 | 12+2*5 | 0 | 24 | 3 |
| Hubbard real | 2+5 | 12 | 0 | 19 | 3 |
| GeneralRBM cmp | 1+5 | 10 | 2+4+80 | 102 | 5 |

AP/P interleaving authority: parser lib.rs:553 adds NArrayAP+2*NParallel.
Local-spin/conduction total determines nelec; headers alone are insufficient
without LocSpin closure. OptTrans is separately bound to parsing policy and
actual opt_trans.len(); these input namelists contain TransSym but no OptTrans.

45 optimization layouts set q/m=31/32/33, steps/window20, seed1 and explicit
store/CG. Fifteen synthetic PhysCal cases set m4 and q31/32/33, seed override1.
Canonical prefixes use input seed (General12395, others1), one step/window1,
and source-defined General store0/CG1 overrides. The input seed is not silently
substituted for the actual constructor/preparation seed.

Remaining input resolution: independent PhysCal includes TwoBodyGEx and its
OneBody canonicalization; do not reuse original no-GEx counts for those four
cases. Four independent real-FSZ cases require their external fixture closure;
missing closure remains MissingFixture/NotResolved, not a made-up model table.
Failure/callback record dimensions and exact completed-step counts remain
separate from state buffers. All 83 IDs remain inventoried, not executed.
