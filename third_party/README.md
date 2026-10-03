# Controlled tenferro security patches

These four crates are the only tenferro 0.7.1 packages that depend on lru. They are local Cargo patches for issue #192, independent of #190. Their package versions remain 0.7.1. Only the normalized Cargo.toml requirement changes:

```diff
 [dependencies.lru]
-version = "0.12"
+version = "0.18.5"
```

Registry lru 0.18.5 fixes both RUSTSEC-2026-0002 (fixed since 0.16.3) and RUSTSEC-2026-0253 (fixed since 0.18.2). No lru implementation is copied or relabeled.

## Provenance

Source: https://github.com/tensor4all/tenferro-rs, release v0.7.1. Each package retains its original Cargo.toml.orig, README and .cargo_vcs_info.json. The latter records the package's source revision; a final newline was added to those JSON files. All retained Rust sources, examples, benchmarks and tests are unchanged. Cargo.lock and .cargo-ok from the registry package were omitted: resolution belongs to the consuming workspace, and .cargo-ok is Cargo's cache marker.

The original crates.io archive SHA-256 values are:

| Package | SHA-256 |
| --- | --- |
| tenferro-ad | 22e31b8e20621d91ad9fc5bd5c2064cf578ce91a6f1fec5eff2ecdbda66da4a3 |
| tenferro-cpu | 4861b9617dcda7fc8a86f7e04a66f791cfaf088f86c812957eae51a80a1cd144 |
| tenferro-einsum | bf9a52646809d654ce638d5c3d3d741ab1d30f3744ddf2c95b7b9b3995944ae0 |
| tenferro-runtime | 81c16aa65fa58565d4ce5004f7dc73823e57f8b3a5a855e8055898080b3202d5 |

Archives can be obtained from `https://static.crates.io/crates/<package>/<package>-0.7.1.crate`; verify their hashes before extracting or replacing these files. Retained contents were compared against the corresponding registry packages with only the documented differences allowed.

The packages declare MIT OR Apache-2.0 but their published archives omit the license texts. LICENSE-MIT and LICENSE-APACHE in this directory reproduce the upstream texts from release commit `8a1839febeb3c868a502e26397ca10761bbc568d`, obtained through GitHub's contents API. Their copyright notices are retained.

## Scope and removal

The repository workspace excludes these packages as members and patches them as dependencies. The standalone benchmark workspace repeats all four patches. The ad crate is optional with default features but remains covered when autodiff is enabled. The packaged upstream test sources include references to files in the upstream monorepo; these packages are dependency snapshots, not a standalone copy of that monorepo's test environment.

Remove the path patches and these snapshots when a supported tenferro release requires an advisory-safe lru version. Update both consumer locks, rerun locked checks and audits, and confirm that no vulnerable lru remains. Do not dismiss the GitHub alert to account for the temporary patch.
