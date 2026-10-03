# Optimizer public failure boundaries (#184)

Owned test: `crates/mvmc-core/tests/issue184_public_rejection_boundaries.rs`.
New cases supplement, rather than replace, previously reviewed S200/S205/S214
and fixed-file preparation cases. No production code or verification TSV is
changed. These test existing Rust/Julia feature guards, **not** that authoritative
C rejects every combination. They do not reinstate a Julia-only grouped-CG guard.

Each public runtime case passes real parsed data and a pre-existing nonempty
caller-owned state/RNG to `vmc_para_opt` or `vmc_phys_cal_in_place`, with a fresh,
uncreated output subdirectory. It asserts the intended rejection family before
output creation, unchanged complete data/settings/parameters/flags, every state
plane and workspace/history, exact pre-existing RNG word count, and exact next
624 words versus a clone of that same caller-owned RNG. This is an error-boundary
atomicity check, not an independent C RNG trajectory oracle.

The seed is 11272 after one pre-existing draw. The exclusively-created owned
temporary parent is removed only by its owner. Inputs and fixed file now come
from checked-in `physcal_181/heisenberg_chain_real`, not `extern/`; optional
developer `cmp` verified every input and fixed record byte-identical to the
historical PhysCal reference. Ordinary tests invoke no C/Julia/toolbox runtime.

| Original family / IDs | Actual settings and assertion scope |
| --- | --- |
| S195/M0995–M0997; rejected portion S196/M0999–M1002 | Both public entries: split2, NSP1, NMP1, Lanczos0, general0, NQPOptTrans2, complex-valued storage `[1, .5]`, original synthetic optimized maps `[[0],[0]]`. Assert rejection family `NSplitSize > 1 with NQPOptTrans > 1 / OptTrans`. Maps are the original guard payload, not a complete six-site sampling input. This test does **not** prove S196's accepted singleton control M0998. For M1001, the actual caller is public PhysCal, but Rust's shared diagnostic does not repeat the literal `PhysCal` string: map the meaningful diagnostic plus caller context separately, not as byte-identical exception prose. |
| S197/M1003–M1009 | Public PhysCal split2/general1/Lanczos0 and split2/general0/Lanczos2. Assert distinct general/FSZ and grouped-Lanczos rejection families. Current general error contains `PhysCal`; grouped-Lanczos error does not repeat that word. M1009's literal prose is different, while its actual PhysCal rejection context is established by the public entrypoint. |
| S198/M1010–M1013 | Public ParaOpt split2/general1/CG0/single optimized sector, each original NSP/NMP pair `(2,1)`, `(1,2)`, `(1,-2)`. Assert FSZ standard-projection multi-QP rejection. M1012–M1013's exact settings are exercised in all three cases, but the shared Rust diagnostic identifies the family without repeating the NSP/NMP numeric values; no literal-value-message parity is claimed. S198 is not the zero-quadrature case. |
| S204/M1027–M1029 | Public ParaOpt separately `use_diag_scale=1`, `rescale_smat=1`; each other submode remains zero. Assert corresponding explicit unsupported option. |
| S203/M1024–M1026 | Existing S205 public CG2 test covers actual rejection and `NSRCG >= 2`, with the same atomicity checks. Rust says to use NSRCG0 or1 instead of Julia's literal `standard SR-CG solver`; map the meaningful supported-option guidance separately from exception-text identity. |

## Current proof

```sh
cargo nextest run --locked --cargo-profile test-fast -p mvmc-core \
  --test issue184_public_rejection_boundaries --no-fail-fast --retries 0
```

Terminal0, run `25fe8d06-de62-491d-bdab-feab21d9f654`: 10/10 passed,
0 skipped, 0.029s. Source SHA-256:
`f4fe7edda438d17dd20a48300211c45d0680cf19498573f5b37fb6f1f0cab212`.
Earlier run `04955115-4e38-4431-a95c-dde3857fcdec` passed ten cases before
the byte-identical offline fixture path substitution; it is not the latest hash.

The parent reviewed all new cases and independently ran the current offline
input version: `2b7a304c-20a5-4c21-8875-9f59a1e03126`, terminal0, 10/10 passed,
0.026s. Parent clarification: meaningful Rust diagnostics may be mapped
separately; copying Julia exception prose solely to satisfy literal substring
assertions is not required. The earlier context/value observations were sent to
the runtime owner and then clarified accordingly. No production diagnostic
change is requested solely for those literal differences.
Accepted S196/M0998 grouped singleton runtime has separate actual MPI evidence:
parent reports Wegener's corrected frozen-external run `98d22d`, terminal0,
worlds2/4 × workers1/2/4 × two repeats through public PhysCal. This is a bounded
frozen-checkpoint result, **not** current-main or whole #179 verification.
Its source/reducer/fixture metadata is owned by Wegener's #179 packaging report.
Wegener supplied the concrete proof: container `73c57e563c61`, artifact root
`/home/vscode/.cache/mvmc/issue179-S196-indexed.jIJsuw`, six terminal0 rows in
`results.tsv` and `w{2,4}-t{1,2,4}.log`. Actual `MpiGroupContext` group width2,
workers1/2/4 with threshold1, BLAS1, base seed1 plus group offset, three samples,
warmup1, one iteration repeated twice. The singleton sector is `[1+0i]` with a
complete six-site identity map: Julia's synthetic `[[0]]` guard payload is
adapted to the parsed six-site model, not mislabelled a complete input.
Retained test source SHA-256
`7dc64dab3d35dd35a7b70dc7eff6af5c61c43025f2c87c8454a48046c1217dd2`,
binary SHA-256
`27bb4706fd90ad217d6ebd2daa722de1172ce39506bea537d79469785a221551`.
Exact config/count/next624 repeatability accompanies `1e-12` absolute/relative
computed comparisons and density1. See `issue-179-S196-singleton.md` for the
full MPI proof; these are owner-reported metadata, not a new run by this owner.
These rejecting single-process cases still do not substitute for that accepted
MPI proof, and their own assertion scope remains unchanged.
