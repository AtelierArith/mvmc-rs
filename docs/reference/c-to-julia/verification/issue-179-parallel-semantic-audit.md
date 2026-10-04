# Parallel74 / QPsplit9 / MPI87: focused semantic audit

This is a source-to-contract audit, not a declaration that 74+9+87 individual
assertions passed in Rust. The #184 assertion ledger remains owned by its reviewer.
Pinned Julia sources were read directly, not replaced by generic MPI launch results.
Source hashes:

- `MVMCOptimizers.jl/test_unit/test_unit_parallel.jl`:
  `a2b582f61dc981c318f817fc0575cfadfdb8c93e34c5679aeaf7fdae48be5050`.
- `MVMCOptimizers.jl/test_unit/test_unit_vmc_sampling_qp_split.jl`:
  `0541d5924fba8baad29267f253b1150a91521e245b80f1a1dc962405a7ead54b`.
- `test/mpi/run_mpi_smoke.jl`:
  `dc5b078a3e0424d90d866e59aa0c868652d099481ef52196f9a9fa99b50ceeea`.

## Parallel unit contracts

| Julia assertion family and actual settings | Rust evidence boundary / remaining obligation |
|---|---|
| SplitLoop lengths/sizes (10,2), (10,3), (3,2), (5,4), (4,4), (2,4), (1,4), (0,2); every rank, half-open ranges | `parallel::c_splitloop_remainder_small_and_zero_work_contracts` and partition coverage tests follow C's last-rank remainder rule. Julia's 1-based `split_range` is an indexing adaptation, not Rust range equality. |
| QP range serial length4; fake world4/local2 ranks0/1; world4/local4 rank3 length1 empty | `mpi_issue179_mapping::issue179_group_width_endpoints` tests actual width1/2/3/world endpoint groups, QP lengths1/5, including empty rank ownership. Exact Julia fake-context cases still need individual assertion mapping. |
| Serial complex/real sum and integer maximum identity; broadcast/reduce/barrier/counter wrappers identity | Rust serial reducer no-op tests cover part of the contract. Maximum-scalar wrapper and Julia object-identity semantics are not automatically represented by Rust's in-place trait. |
| Serial context ranks0/sizes1/root=true; no communicator objects | Rust `SingleProcessReducer` represents this architecture without Julia's nullable communicator fields. Compare behavior, not field names. |
| OMPI/PMIx/PMI/Slurm detection, explicit JULIA_MVMC_MPI=0/1/invalid, guarded serial under launch | Rust `LaunchContext` detects rank/size; CLI feature/launch policy is separate. Julia's full environment truth table is NOT proved by rank detection or repeatability. In particular Slurm-only fail-fast and guarded-serial policies require focused CLI mapping. |
| Seed missing11272/zero0/positive123/time-negative/explicit777; group3 offset100→103 | Preserve primitive RNG tests and explicit global seed broadcast/group offset tests. Negative-clock tests must check shared payload, not equality to one clock timestamp. Repeatability inventory uses fixed seed1 and does not prove time-seed semantics. |
| SafeMPI chunk ranges n0/3/8/9 at chunk4 | No claim from ordinary small collective tests: chunk-boundary behavior needs explicit coverage or an architectural capacity declaration. |
| Pack/unpack first/last perturbation, short-buffer rejection, duplicate orbital repair, manual inconsistent RBM rejection, signed real/imaginary zero; serial sync equals legacy | `sync::broadcasts_and_applies_canonical_variational_parameters` is related but not by itself all these assertions. Duplicate/signed-zero families need individual test/setting evidence; raw root output repeatability does not substitute. |

## Public input validation versus reducer availability (S191/S192)

Julia accepts public settings `NSplitSize=2, NSRCG=0` in global and para-opt
parallel validators, including a serial context. Rust's public
`validate_supported_modpara` / `validate_grouped_runtime` accepts this ordinary
grouped direct-SR input; `validate_reducer_rank` separately rejects a serial
reducer without a group communicator. This is runtime availability, not invalid
input. Focused tests must retain before-mutation RNG/state/output snapshots rather
than infer the boundary from successful MPI execution.

Julia's grouped CG, grouped OptTrans and FSZ multi-QP rejection tests are capability
restrictions. Grouped CG cannot be declared C-invalid: the retained native C store
probe documents an unwritten-buffer dependency. The supported-contract decision
is separate from reference capability guards. Do not weaken those guards merely
to make a matrix pass, or copy a Julia restriction as a universal C input rule.

## QPsplit unit contract: useful exact primitive operands

Julia complex weights [2,3,5], Pfaffians [7,11,13] give full IP112, subrange33,
empty IP0, then log112. Real weights [2+9i,3], real Pfaffians [7,11] give47,
empty0, and log(abs(47)+1e-100). Real IP ignores the imaginary weight component.
Requested comm1 reduction without a context must error; serial context makes it
identity. Rust MPI mapping tests use owned-range-only buffers with stale NaNs
outside the range and empty local contributions, proving a related actual sampler
interface. They do not yet prove each of these nine literal assertions.

## MPI87 runner semantics reviewed

The runner launches distinct workers, not one generic helper:

- Real para-opt: serial versus world2 file-set equality, four output lines;
  world4 output existence/shape. Non-root result is minimal/NaN rather than root
  readback. Repeated Rust records prove repeatability, not those public result
  shapes or absence of duplicate writers.
- Weight average: local weight rank+1, weighted constants 10/100/2/3; global
  weight triangular sum, normalized constants; zero weight warning emitted once
  on root. Exact integer global reductions do not prove warning multiplicity.
- Hubbard and PhysCal: exact named file sets and root/non-root message counts;
  PhysCal includes Green files and start/end messages once. These require CLI /
  public runner assertions, not sampler state-only evidence.
- SR-CG operator: two ranks with different real and imaginary sample matrices;
  non-root x=[999,999] must become root x=[.5,-1]. Actual expected products are
  [-15.30625,-22.7375] real and [-14.4859375,-24.2375] complex. Serial operator
  tests and the independent world4 fixed-operand C diagnostic are related, not
  an executed equivalent of this two-rank literal case.
- Failures: NSRCG2/groupedCG/groupedOptTrans/FSZ multi-QP rejection before MPI
  initialization and no output; guarded serial under world2 exits nonzero;
  explicit MPI without launcher runs size1. Coordinated callback evidence covers
  a different post-initialization failure boundary and must not substitute.
- Self-consistency: direct/store world2-width1 versus world4-width2 on Hubbard
  real and Heisenberg FSZ, plus real/momentum/complex standard projection and
  real/complex PhysCal. Julia tolerances 1e-8/5e-8 are historical worker-level
  choices, NOT automatically adopted Rust numerical budgets. Input/world/store
  changes are not same-configuration repeatability comparisons.

Follow-up: expand individual ledger mappings and execute the missing literal
operator, launch-policy, public-result and warning/output-boundary assertions.
The 495 repeatability configurations and independent residual evidence remain
separately labelled; neither closes this semantic audit or #179.

The literal two-rank real/complex operator gap is now exercised by an actual
isolated Rust MPI regression, including poison non-root search broadcast and
all four product phases; see [its provenance and terminal proof](issue-179-literal-operator.md).
This closes that focused literal case, not the remaining launch/output/warning
families or all MPI87 assertions. The current average helper's silent tiny-weight
return is a concrete warning-contract gap, not an executed warning PASS.
