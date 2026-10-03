# Dependabot alert 5 dependency remediation

Local dependency remediation for [issue #192](https://github.com/AtelierArith/mvmc-rs/issues/192), investigated on 2026-10-03, independent of #190. Both workspace roots now patch the four lru-dependent tenferro 0.7.1 crates to controlled snapshots under third_party. Each snapshot changes only its lru requirement from 0.12 to 0.18.5. Both consumer lockfiles resolve registry lru 0.18.5, checksum `ef9ac18847474e638e3702b76c65d4eb93428471a74778ef0f1be711717f89b5`. Package source and licensing provenance are recorded in [third_party/README.md](../../third_party/README.md).

## Advisory coverage and release constraint

[Alert #5](https://github.com/AtelierArith/mvmc-rs/security/dependabot/5), read using `gh api repos/AtelierArith/mvmc-rs/dependabot/alerts/5`, concerns benchmark/pfapack_compare/Cargo.lock and lru 0.12.5. The root Cargo.lock had the same vulnerable dependency.

[RUSTSEC-2026-0002](https://rustsec.org/advisories/RUSTSEC-2026-0002.html), alias GHSA-rhfx-m35p-ff5j, describes the IterMut pointer defect; patched versions start at 0.16.3. [RUSTSEC-2026-0253](https://rustsec.org/advisories/RUSTSEC-2026-0253.html) describes panic safety in pop; patched versions start at 0.18.2. Selecting 0.18.5 covers both advisories rather than implementing only the original iterator backport.

Before patching, online and offline `cargo update -p lru --precise 0.16.3` failed for both workspaces with a requirement conflict: tenferro-cpu 0.7.1 requires lru ^0.12. The sparse registry index showed latest versions 0.7.1 for tenferro-ad, tenferro-cpu, tenferro-einsum and tenferro-runtime, all requiring ^0.12. Upstream HEAD `33e190fa367461e1e34f3797ddf396e05d7fefff` still declared lru 0.12. No compatible published fixed release was found, so the authorized local snapshots change the requirements without changing numerical code.

## API compatibility and verification

The original caches use LruCache constructors, get/get_mut, put/push, pop/pop_lru, iteration, clear and length operations. The selected lru source retains those APIs. Compilation checks validate the concrete calls and trait bounds. The default feature graph activates cpu, einsum and runtime; the optional ad crate is checked by enabling tenferro-einsum/autodiff.

The initial shared-checkout nextest run `cccaee13-a77f-4e54-be45-5370d5ca7e48` completed with exit 0: 9 passed, 390 skipped, 6.337 seconds of test execution. Its handle is closed. That result predates isolation from the unrelated runtime changes and is not the isolated branch's test result.

The focused security branch starts from origin/main `515a89b38dc49ead3f526c406643523e979c8e02` and restores only Cargo roots, locks, third_party and docs/security from checkpoint `9e58590`. The parent checkout's integration was left untouched. Verification uses separate target directories outside both checkouts: `/tmp/mvmc-security-192.rDsAn7/target` for the root and `/tmp/mvmc-security-192.rDsAn7/target-benchmark` for the standalone benchmark. Results below are for this isolated branch on native Linux x86_64, rustc 1.99.0 (`b940084d7`, 2026-09-28), Cargo 1.99.0 (`5f94df478`, 2026-08-27):

| Check | Status | Captured result |
| --- | --- | --- |
| Workspace locked check | Passed | Exit 0 |
| Workspace locked check with tenferro-einsum/autodiff | Passed | Exit 0; includes tenferro-ad |
| Standalone benchmark locked check | Passed | Exit 0 |
| Reverse lru trees for both roots | Passed | Exit 0; lru 0.18.5 through local patches |
| Retained file comparison | Passed | Exit 0; 381 files, only documented differences |
| Targeted einsum and gram nextest tests | Passed | Exit 0; 9 passed, 327 skipped, 11.867 seconds; run 4e977fef-bf03-46e2-8603-7f7cd8f8bbac |
| Workspace all-target Clippy with -D warnings | Passed | Exit 0; vendored warning remained a warning |
| Root Cargo audit | Failed overall | Exit 1; crossbeam-epoch finding, no lru finding |
| Benchmark Cargo audit | Failed overall | Exit 1; crossbeam-epoch finding, no lru finding |
| Git diff whitespace check | Passed | Exit 0 |
| Workspace formatting check | Passed | Exit 0 |
| GitHub alert closure | Pending | API state open; fixed_at and dismissed_at null |

Reproduction commands from the repository root (use isolated CARGO_TARGET_DIR values for builds in another checkout):

```sh
cargo check --workspace --locked
cargo check --workspace --locked --features tenferro-einsum/autodiff
cargo check --manifest-path benchmark/pfapack_compare/Cargo.toml --locked
cargo tree --locked -i lru
cargo tree --manifest-path benchmark/pfapack_compare/Cargo.toml --locked -i lru
cargo nextest run -p mvmc-core --cargo-profile test-fast --locked -E 'test(einsum) | test(gram)' --no-fail-fast --retries 0
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo audit --json
cargo audit --file benchmark/pfapack_compare/Cargo.lock --json
```

Both audits using advisory database commit `f8dee89e1b2f2f1eaf548312df7655fe5202a302` report no lru advisory. Full audit exit status remains 1 because both locks retain pre-existing crossbeam-epoch 0.9.18 / RUSTSEC-2026-0204. Additional warnings concern paste, and in the root lock custom_derive and anyhow. These findings were not ignored or changed as part of the lru remediation.

The original registry archives were SHA-256 verified. All 381 retained package files were compared against cached registry contents, allowing only the four lru requirement changes and final newlines on VCS metadata. No runner or project test files were edited. The benchmark lockfile's rayon entry was a pre-existing change in checkpoint 9e58590; origin/main already declares rayon in mvmc-core/Cargo.toml. The focused PR retains this necessary lock synchronization without introducing a new dependency declaration or threading change.

Upstream's existing deprecated fetch_update usage emits a warning when compiled as a local dependency; it is unchanged. The initial shared-checkout library/bin Clippy run passed, while all-target Clippy failed on an unrelated physcal_issue181 test lint. The isolated origin/main-based branch's all-target Clippy run passed with -D warnings; the upstream warning remained a dependency warning. Excluding vendor packages from workspace membership preserves the boundary between project lint targets and upstream dependency code; no global lint suppression or kernel change was added.

## Lifecycle

The initial implementation was recorded in checkpoint 9e58590. This validated focused milestone is recorded separately on security/192-lru-remediation for review; its PR must not be merged before parent review. Issue #192 and the GitHub security alert remain open; local resolution does not establish that GitHub has rescanned a merged change. No alert dismissal or write to the dependency's upstream repository was performed. Replace these snapshots with a supported fixed tenferro release when available, validate both locks again, and verify the GitHub alert's fixed state before claiming closure.
