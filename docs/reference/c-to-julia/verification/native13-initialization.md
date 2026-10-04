# Native13 initializer regression candidate

Related to #180, #182, #185; none is completed by this initializer milestone.
Candidate initially base0130fdf0496742f8e49675722dea63d9cb3401d8; before any
candidate build advanced to83fd58ffa270cf211ca742ce842ebe3825fc8c4b retaining
owned uncommitted changes. This sparse dedicated
checkout contains no shared-tree changes. The83fd candidate was built and tested
as recorded below; no commit or push has occurred. After both live checks ended,
the candidate advanced to0f64bcb5f65fc7eb84af1487e978e9d3c74c3db8. The earlier
main4ce historical validation is a separate lineage, not relabelled0130/83fd.

## Contract and ordinary Rust independence

`public_initialization_matches_native13_defined_contracts` uses public Rust
parser/init/load/sync/QP initialization APIs and checked-in C-derived JSON plus
original13 model definitions. It never compiles, executes or reads C/Julia,
toolbox programs, native libraries or historical /tmp paths. Provenance paths
are historical identifiers, not test runtime dependencies. serde_json and sha2
are test-only dependencies; input names/full SHA set are independently checked.

All six seeded/workspace-query/random/loaded/synchronized/pre-sampling raw624,
cursor, drawcount and next624 contracts remain exact, with peek nonmutation.
C's workspace query has no invented Rust numerical counterpart. Parameters
and defined flag mask/layout are checked at the four stages containing initialized
parameters. Never read or compare C unwritten imaginary flag slots.13 parameter
segments, offsets, exact input settings/seed/20-step metadata are retained.
These20 settings do not mean this test runs a20-step sampler.

## Explicit numerical policy

Only GeneralRBM_cmp, random initialization, RBM parameter components receive
absolute1e-18 (relative0). All other initialized components/stages retain absolute0
against these fixed fixture values, as requested after their observed zero
difference. This is numeric equality, not computed-bit-pattern comparison.
No relative allowance, solver/noise waiver or other-model/stage propagation.
Nonfinite actual/reference values are rejected.

The empirical input/seed-specific threshold is explicitly user-authorized:
WP5PTo measured maximum8.673617379884035e-19 across this random RBM fixture;
first difference parameter6 imaginary1.0842021724855044e-19. The previous
unapproved7.77e-18 gamma/trig candidate is REMOVED, not presented as an accepted
accuracy bound. Neither a universal two-ULP library guarantee nor an arbitrary
forward solver tolerance is claimed.

First-operation trace on raw words324192925/3779980750:
real2,radius0.000754820473957806826,phase5.52980217845683786 andcos agree.
C cexp imaginary=-0.684110252064760282;
unchanged Rust Julia-style sine=-0.684110252064760394.
First trig difference1.1102230246251565e-16 at identical operands; radius times
this difference8.380190696464305e-20, plus product rounding, yields final imaginary
difference1.0842021724855044e-19. C final component matches original C fixture.
This localizes math-function error, not an RNG/order/input discrepancy. All13
pre-sampling parameter comparisons in the historical acquisition had zero difference;
the twelve historical full-C20 statistical failures cannot be blamed on this init.

## C provenance and independent acquisition

Static provenance.json retains original archives/manifests/bindings plus every
model definition SHA. First3/remaining10 init-only captures include6 checkpoints,
original seed/optional initialization, no sampler. C stop binarySHA
1cd80a8d3b168e447c2f7f3b33278c63e96f5c217c715396df382a923e827f4f;
originalvmcmain fdcd661c4eb028786fc58ae5e1f5232426c943b547f95a0d74e888e602256d63;
instrumented c033a58fc62be7ec01696a0b1b99f2a53a145c6a42c2825729a9e363388bbc4c;
readdef6c53cb832f93d6cbfd7cea955fbb693738af5536b913d36af32b98eed38c32d9;
original SFMT b61f0f0239193fd4c06f2879e6e8452225bec45b3bca426902c244fd31ad4e3a.

General seed12395 is bound by originalmodparaSHA
c22927f4ccf14ea548eac557383c4e21692d59611c8f362b234765caf4189e55.
Certification changes state word0 to12394 (XOR1), not a consumed draw.
No seed==post-certification-state shortcut remains.

First-operation C parameter.c SHA
46ad04622f4475337028cee633bd76ce318a6d5058d03f202b55204500399fb0, line53
`1e-2*genrand_real2()*cexp(2.0*I*M_PI*genrand_real2())`, upstream GPL-3.0-or-later.
Standalone C replay preserves this expression and original gcc13.3 flags
`-O3 -DNDEBUG -fopenmp -O3 -DNDEBUG`; no math fast flags. Actual direct libm SHA
f06f2ce1f1833df5f41cf13b6447ff07bea993ad9b27297d3428c2f70ab3f0e7,
libc511f825ee075610ac9c0f7f91e2c13de2000d0f7b859f6461137e809a0a009d0,
loader6222a16be7f2d458d6870efe6e715fc0c8d45766fb79cf7dcc3125538d703e28.
These are containerLinux providers, not native macOS evidence or a BLAS claim.
Original scalar Csource is unmodified; separate diagnostic subexpressions are
not asserted to be compiler-internal temporaries. Rust probe opt-level2 imports
immutable trig06d901487349c0a5f27cdc94077f9acec08ae35f8b96f2b2f31ef8f1858f6957
and exp09df0697f682135b8c821a1adef56fe940ed8b6446c0f56ac03b4ffb60f701e3.

## Actual receipts and candidate validation

WP5PTo standalone1PASS0.03s on main4ce bound libraries; allsource/input/library
posts0,52 numerical stage records,13 models. TestbinarySHA
aba5901d4a7a78991ed0f550d6037a18675f32699ac7c9c8c996854fecf28330.
Rust trace ytbPYz and container Ctrace BUEYwa aggregate/cleanup/allposts0.
TraceSHA Rustf1810f85943492c6652e262ffa93695f5c3ef48767f77f64d0156bdd8cde7579;
C722dd753c48c92fced52e54029d4392b5c3e6d84562f8deefb29b558c4575cc3.
Retain NgBN8U invalidseedcheck failure,5WWgNV missingmodule compile failure,
70zRAK capacity NOT_STARTED separately. No failure was reclassified as PASS.
ytbPYz binary recoverably relocated to HOSTSHM OYxYvE/original with old logical
path symlink; this membership change is recorded, not called an unchanged receipt.

On83fd, ONE ordinary Cargo test-fast acquisition completed17m03s compilation,
then run97c23c70-d319-4f6c-bdc4-55debe167c8c:1PASS/0skip in0.034s. Receipt
`/dev/shm/mvmc-native13-focused-proof.xDASU5` has nextest/outer/cleanup/source.post/
tools.post/aggregate all0. Full source manifests before/after shareSHA
1e366121e747d22f287ac8a5b794cd65e3faf8214731ba297a87037f8667f93b.
Selected binary `native13_initialization-87c3910d42c29500` SHA
9b541bacb8a137d1c88bbdc527ac75e4e6a680939532f48df06c060ef74e3698;
its depfile binds this candidate's CARGO_MANIFEST_DIR and its inventory contains
exactly the public test. Actual Linux toolchain rustc1.99.0(b940084d7)/LLVM23.1.1,
cargo1.99.0 and nextest0.9.146; buildjobs2 and BLAS/OMP/inner/Rayonthreads1.
This initialization binary's ldd lists libc/libgcc/loader, not a loaded BLAS
provider; C acquisition BLAS metadata is documented separately below.

Focused rustfmt check0. The sparse checkout's workspace-wide cargo fmt failed
because unrelated tests/support modules were omitted; not a workspacefmt PASS.
Focused clippy receiptG3sRti failed101 on one same-type i64 conversion at126;
source/binary/tools posts0. The later one-line removal changes no values,
operation order or tolerance. Fresh0f64 validation below is independent of
the retained83fd proof.

After that lint-only fix and documentation update,0f64 run
448c5191-3047-4c71-b8f8-e7820a79dfff completed1PASS/0skip in0.030s, warm
compilation3.86s. Receipt `/dev/shm/mvmc-native13-recheck-proof.nOjMHe` has
nextest/outer/cleanup/source.post/tools.post/aggregate all0. Frozen source
before/after manifestSHA
dedb07f8818ce19a5ae1e4cc1046d5b6df55c12e5bad226f0dcff466a5ed1e51;
implementationSHA
99cdd624e013b02aafeda8916fe0874be7c09bf6644c70cad742580cd117d0b7;
selected binarySHA
4926fce008feee26d6e1b30488ce9ee53b04b424c46250e8a8ec909587a7fd33.
The depfile binds this candidate's CARGO_MANIFEST_DIR, and binary inventory is
exactly one nonignored public test. Focused checks receipt
`/dev/shm/mvmc-native13-focused-checks.tBPpib`: clippy/fmt/outer/source.post/
binary.post/tools.post all0. Command:
`cargo clippy -p mvmc-core --test native13_initialization --profile test-fast --locked -- -D warnings`.
Vendored tenferro-runtime's deprecated fetch_update warning was nonfatal;
no allow or dependency modification was introduced. Formatting proof is
`rustfmt --edition 2021 --check crates/mvmc-core/tests/native13_initialization.rs`,
not a workspace-wide claim. This results-only document append was made AFTER
both checks ended; the implementation and all203 static fixtures are unchanged.

Durable C acquisition SOURCE and exact reproduction commands are in
`c_toolbox/native13_initialization/NATIVE-ACQUISITION.md` and
`reproduce-native-init.sh`; they include original source/gitlink hashes,
hook/extraction boundaries, compiler flags and the four-object replacement
link contract. The reference Linuxx86_64 environment used GCC/GFortran13.3.0,
MPICH4.2.0(ch4:ucx). Observed OpenBLAS0.3.26 pthread threads1 is a metadata
probe; statically linked BLISsv0.8.1+arm has raw getter-1 and actual threads
UNVERIFIED. The archived BLIS artifact differs from its inventoried gitlink.
Julia was not used to generate these C initializer fixtures. None of these
developer provenance files is a runtime dependency of normal Rust tests.

Candidate still needs parent regeneration-source review and publication
Linux/macOS checks. Sparse checkout intentionally
omits unrelatedlargefixtures; no full-workspace pass can be claimed here.
Full13model20-step sampling, proposals/acceptance/config,6PhysCal and complete
thread1/2/4+threshold/MPI coverage remain open independent acceptance tasks.
