# Linux historical Julia FSZ references

These independent Julia 1.13.1 outputs supplement the preserved macOS
archives. They cover direct SR (with and without stored O) and CG prefixes
1, 2, 3 and 50 for FSZ, InterAll, PairHop, DH2, DH4, DH24, RBM FSZ and
OptTrans FSZ. `provenance.json` records the actual reference checkout,
manifest, generator hashes, BLAS providers and thread count.

`SHA256.json` records all 856 independently generated files, including
configuration and full 624-word RNG blocks. Files identical to their archived
counterparts are reused. The overlay stores changed files consumed by the
runner checks; unused solver/Gram snapshots are omitted. Test selection falls
back to the macOS archive for the byte-identical files. The Rust runner checks
configuration, spin, burn-in, counters and RNG before parameters and energy.
CG iteration counts, status, residual output and floating-point values remain
exact; no CG rounding tolerance is introduced.

Fixed FSZ setup, real/complex sampling and post-sync DH histories also use
independent Linux Julia outputs and exact floating-point bits. Their archived
macOS values remain unchanged. Full post-run RNG and configuration checkpoints
are checked before the history's numerical fields.

The first arithmetic divergence was the FSZ inverse's native C++ complex
division: GNU libgcc uses Smith's ratio while the previous Rust path used
norm-squared division. For the real pivot `0x3fe1b6db6db6db6e`, these produce
`0xbffce739ce739ce7` and `0xbffce739ce739ce6`, respectively. Passing the tested
GNU divider to the Rust FSZ inverse restores exact Linux Julia runner parity,
including all CG iterations. macOS preserves its original inverse path and
fixtures. This is a historical Julia runner comparison; standalone native C
Green and division fixtures remain separate strict checks, and neither proves
a complete native C sampling trajectory.

Generate these outputs independently in a Linux x86_64 Dev Container:

```sh
export JULIA_DEPOT_PATH=/home/vscode/.cache/mvmc/julia-depot
export JULIA_BINARY=/home/vscode/.cache/mvmc/tools/julia-1.13.1/bin/julia
scripts/generate-numerical-references.sh --suite julia-linux-all --output /tmp/julia-oracles
# A short independently checked subset:
scripts/generate-numerical-references.sh --suite julia-linux-cg-fsz --steps 1 --check
```

For each listed case, the relevant staged oracle commands are
`check_sr_direct_runner_parity.jl --case=CASE --store=0 --write`, the same
command with `--store=1`, and `check_sr_cg_runner_parity.jl --case=CASE --write`.
Run them with Julia 1.13.1, `--startup-file=no`, the pinned reference project
and one BLAS thread. Copy changed expected values only after reviewing the
staging manifest; ordinary Cargo tests invoke no oracle runtime or toolbox.


The full overlay suite is explicit and includes the setup/samplers/histories
and all Direct, stored-Direct and CG cases. `--list` exposes per-case suites;
`--steps` limits runner prefixes. Generation evaluates each existing generator
in an isolated module, qualifying only observer callbacks and recording its
output paths. It restores staged historical macOS fixtures, stores changed
consumed outputs beneath `linux_gnu_julia/`, and records every freshly generated
hash in the external staging manifest. `--check` compares those independent
values against the selected overlay or its identical archived fallback.
Review `regeneration.json`, `manifest.json` and `changes.json` before `--apply`.
Different BLAS providers may require justified error bounds for Julia parity;
the current recorded environment happened to satisfy the stronger exact gates.
