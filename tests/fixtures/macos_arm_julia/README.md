# Independent macOS ARM reference fixtures

These are mixed Julia 1.13.1/C-kernel references using system LP64 OpenBLAS.
Rust tests read the checked-in short-run and fixed-input expectations without
running Julia or C. `vortexm4` and `neoversen1` select the matching BLAS core.

[Generation instructions and a single-artifact example](../../../docs/APPLE_SILICON_PARITY.md#independent-reference-generation)
cover dependencies, bridges, core/job selection, scratch output and packaging.
[The comparison policy](../../../docs/NUMERICAL_COMPARISONS.md) explains why
20-step long runs check repeatability instead of historical trajectories.

Each core's `PROVENANCE.txt` records the generator/backend environment;
`SHA256-all.tsv` and `storage.tsv` describe retained logical artifacts and
archive reuse. `omitted-auxiliary.tsv` preserves hashes of generated artifacts
that Rust does not consume and that are intentionally omitted from Git.
The root `SOURCE_SHA256.tsv` records reference/tool source hashes.

`.gz` is lossless storage with deterministic gzip timestamps. Artifact hashes
verify stored data, not newly computed floating-point output. Never replace
independent expected values with Rust-generated results.
