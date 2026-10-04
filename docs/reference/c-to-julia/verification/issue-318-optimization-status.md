# A145 fallible optimization status reporting

Related to #184/#185; focused issue #318. This implementation has bounded local evidence below, not canonical ledger
promotion or full-workspace acceptance.

Original Julia8bb1b9e8ae47b1512c00b321be05664ddcac0fd1
`MVMCExpertModeParsers.jl/src/utils/opt_flag_utils.jl:161–206` reports G/J sites,
Slater first/last5 mapping rows and status totals. The Rust writer API keeps
G/J sites in explicitly labeled representative views, separates dense Slater
parameter totals from mapping rows, and queries actual shared parameter indices
rather than row ordinals. Output is deterministic ASCII with zero-based indices;
it is not byte-identical Japanese-format compatibility.

Current C authority d73d06bd529d3b2573f38eb5817c4a5f52971006
`src/mVMC/readdef.c:737–740,760–762,1013–1030` defines declared block counts and
Slater offsets. G/J parser vectors store one representative per coefficient;
missing/default representative metadata does not establish exhaustive spatial
mapping. Declared slots without metadata remain visible. DH/RBM display is not
added, but their complete declared widths participate in Slater offsets.

`write_optimization_status` accepts a caller-owned `io::Write` and returns
`io::Result<()>`. Only raw real flags exactly1 are eligible. Missing flags are
fixed. Report-local checked arithmetic precedes any write and does not call
unchecked layout/offset helpers. Extra G/J representatives, invalid Slater
indices/counts and arithmetic overflow return InvalidInput before output.
These are report-local checks, not new loader/pack/runner restrictions. Writer
failures preserve their error and may leave partial output; no data/RNG mutation.

Eight independent ordinary test controls cover: exact output with shared
indices/site representatives; empty/missing/imaginary flags; reserved/fallback
metadata; projection23+RBM45 literal136/140 offsets; independent dense/mapping
truncation; invalid/overflow/no-write; BrokenPipe partial output; Interrupted retry.
No test constructs an RNG or invokes C/Julia/model runtime. Existing numerical
algorithms, flags, initialization, fixtures and ledger remain unchanged.

Frozen d8fa local gate: exact8 LIST/RUN passed8/8, zero skipped,0.006s;
strict parser alltargets Clippy/fmt/postLIST and source/tool/sysroot/ELF/runtime
POSTs all0. Receipt `/tmp/mvmc-184-issue318-proof.20261004`; primary/posts/terminal0,
all five phase child/cleanup/postempty/writer statuses0. No inherited compiler
file-size ulimit; default features, locked test-fast, jobs2 and BLAS/OMP1.

Publication base29c6bc2cd6cbe1d09f54934724125a5bd7de5e0e preserves315metadata,
317directoffsettest and316reviewdoc. API/reexport/test bytes equal the frozen
executed d8fa source; this doc alone changed afterward. New-head full-workspace
Linux CI has NOT yet run. No C/Julia/MPI/model claim.
A145 has no dedicated original assertion identity; eventual evidence is bounded
reporting architecture only, not full #184/#185 acceptance. A180 warning/family
validator semantics remain unresolved and pending separately.
