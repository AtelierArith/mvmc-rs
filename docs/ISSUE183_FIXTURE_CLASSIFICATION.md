# Selected corrected-General fixture classification

Related to #183 and #185; this bounded change does not complete either issue.
Publication baseline: main14182afb0403931ff340ae93dd2ec3079114b0a8, whose tree the parent verified identical to PR241 head35ad2e1ab0470dfe8963d207864240cd0fe5f229.

Both explicitly ignored corrected-General numerical gates retain require_gate and the unchanged pinned archive/source/provenance/settings verifier. Preflight diagnoses absent metadata or archive members as MissingFixture before provenance unwraps. Malformed/duplicate/unsafe manifests, nonregular or unreadable files and verifier-detected corruption are terminal Failure. Manifest syntax is checked completely before checking member presence. Large numerical files are opened, not fully read by preflight. Success does not bypass the independent verifier or establish numerical PASS.

Root and descendant symlinks are rejected. System aliases above the chosen logical root are allowed. This static fixture check is not a race-proof hostile-filesystem sandbox. The fixed General consumer has no user-selected model string; Unsupported selection is only a unit-control contract, not falsely claimed as a production selector implementation.

## Actual local proof

HOST receipt /tmp/mvmc-183-classifier-controls.xuyawz, owner handle51185: standalone rustc --test --edition2021 -Dwarnings; exactly five named controls passed, zero ignored, primary/inner/outer terminals0. Rustfmt check passed. Actual resolved compiler/formatter, linker tools/startup objects, library providers, selected binary and sources were SHA256-bound before/after; selected runtime provider identity matched. The wrapper used timeout120s/kill-after5s and post-stage 8MiB/384MiB checks (not an enforced storage quota).

Helper SHA256: 0f020afa8eda32fba26e7437f4257fd16d1f787c0212dcf6b3c026771495e3c8.
Wrapper SHA256: 9a413d2222a0eb49e764d385f85c88043e845affb90435b6dcbdab0555cf4a7c.
Test stdout SHA256: b35301e1012b0dd45d8070468c558c17c9951a628ec8701315401eb65e40ff69.
Source manifest SHA256: 2a0ea0bc334c53118451bd812f8a74404f7a03930a697b1d6ad2546eab1eb7e0.

Controls cover absence/nonregular files; malformed/duplicate/traversal/invalidUTF8 manifests; missing-first plus malformed-later precedence; absent metadata/member preventing verifier invocation; independent fixed-byte valid/corrupt verifier behavior; final/child/root symlinks; and unknown/empty unit selections. PermissionDenied and post-observation read errors are deterministically tested through the actual pure error classifier, not chmod under root. This is not an actual OS PermissionDenied integration test.

## Unrun and remaining scope

The new ordinary corrected_general_classifier_accepts_checked_in_independent_bundle test runs preflight plus the unchanged verifier over the real checked-in archive, without sampling or solving. It is UNRUN locally. Cargo compilation, Clippy and complete workspace tests for this new source are pending exact-head CI; the five standalone controls do not substitute for them. No model numerical gates were executed by this milestone.

Other ctest/thread/MPI consumers and configurable selectors remain unmodified. MissingFixture, Failure, Unsupported, NotRun and independent numerical PASS must remain distinct in the broader #183/#185 matrix. Normal Cargo reads checked-in fixture inputs only: no toolbox/C/Julia oracle invocation, no changed expected values or tolerances, and no numerical InterAll coverage claim.
