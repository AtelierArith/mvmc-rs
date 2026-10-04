# Default Reducer counter contracts: bounded pure-Rust evidence

Related to #179 and #185; neither issue is completed by this target.

`issue179_counter_scope` contains six ordinary, nonignored Rust tests. Its
literal mock adds100 and records integer reduction slices; this is independent
control logic, not a native MPI sum or Rust-generated numerical reference.
It checks exactly six statistical slots, root-only writeback, preservation of
the remaining logical/burn slots, short-buffer bounds, empty root/nonroot
participation and fail-closed default multirank integer/real broadcasts.
The default trait calls the integer reduction once even for an empty buffer;
MPI override empty-buffer behavior is outside this evidence.

## Actual bounded Linux verification

Historical compile SOURCE: main1b0e9d90de7ffb90c84d098c181cefa470042021 plus
this test only. Test SHA256:
`e676645bdd04329972a9a5213e768937cdff71cb8d5bbf7a54ecacd633a17e83`.
Publication base5b0874eb70b72e2a7993662af757d50b25dadc17 has identical
reducer.rs/counter.rs/core Cargo.toml/workspace Cargo.toml/Cargo.lock bytes;
the historical run is not relabeled as a fresh publication-base run.

Container73c receipt `/tmp/mvmc-179-counter-defaultguard.ACYB6Q`:
session93010 terminal0; UUID`c5966aef-7250-4e38-b29b-48fdf87d5d44`.
LIST selected exactly6, tests6PASS/0skipped in0.006s, targeted clippy8.12s and
fmt exit0. SOURCE/member/tools/providers/sysroot/ELF/fingerprint/packet POSTs
all0; cleanup0 and two empty-session observations each2. Parent independently
reviewed the native Rust test log and all statuses. This was default features,
locked/test-fast/jobs2, all BLAS/OpenMP/MKL/BLIS thread limits1, no MPI feature.
Actual tool, compiler-sysroot and provider hashes are retained in the sealed
packet; no C/Julia runtime or toolbox fixture is invoked by these tests.

`evidence/tests.stderr` SHA256:
`1f2efb5f0b8677363aa992abfe78f40396ba67857845d6b7f167fa5c10fa80af`.
Selected ELF SHA256:
`74c2a9fd33db1d57d269ce877cc3c5802855e8566e8fa0d65ccc7236d6488748`.
Packet binding SHA256:
`94e4c002a9e6bf62fd891d120abed9b595d10d3edc7b39f99e0b7fa5da5dee40`.

Earlier receipts remain failures, not retroactive passes:
session28087/TCsEeq compile101 (ambiguous empty expected array);
session68864/z7MylQ compile/LIST0 but harness feature guard1. The successor
typed both expected empty arrays and accepted only the actual named feature
`["default"]` while independently checking its default dependency list empty.
No build feature flags were changed to satisfy that guard.

## Reproduction and limitations

```sh
cargo nextest run -p mvmc-core --test issue179_counter_scope --locked --cargo-profile test-fast --no-fail-fast --retries 0
cargo clippy -p mvmc-core --test issue179_counter_scope --locked --profile test-fast -- -D warnings
cargo fmt --all --check
```

Fresh full Linux PR CI is still required before merge. This evidence does not
prove native MPI communicators, uneven/empty groups, collective order, model
sampling, cross-language numerical parity or any whole #179 matrix. Native P02
remains NOT_STARTED under its unchanged SHM admission floor. Historical provider
gaps and outstanding model/CG/store/width/long20 obligations are not waived.
