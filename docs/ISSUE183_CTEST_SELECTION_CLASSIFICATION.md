# Configurable long-gate classification

Related to #183 and #185; neither issue is complete. Candidate baselinee41ffc38265310492546b33c7df008a378a99ee9. Minimal delta: ctest_equivalent.rs, merged fixture_status.rs helper and this doc; no production kernels, reference expectations or numerical bounds changed.

The actual MVMC_RS_CTEST_MODELS selection is validated before fixture discovery. Unknown, empty-segment, duplicate and nonUnicode explicit selections report Unsupported and cannot pass. Absent/empty/skip opt-in keeps existing require_gate fail-closed behavior. NonUnicode/empty explicit oracle root is Unsupported; only NotPresent uses the default root.

Missing regular fixture metadata or referenced input is MissingFixture. Symlinks/nonregular/inaccessible members and malformed manifests are Failure. Typed errors preserve those labels through the model catch, actual report_gate and aggregate terminal failure; independent verifier/numerical assertion panics remain Failure. The unchanged verify_inputs still rejects changed hashes or unhashed references. Flat-path checks are this pinned fixture bundle's existing contract, not new restrictions on the public C/Rust loader.

## Actual proof and unrun boundaries

HOST /tmp/mvmc-183-next-classifier-controls.chmRyh, owner92579: standalone helper6/6PASS0ignored0filtered0.01s; primary/inner/outer0, exact named inventory, formatting/rustc--test-Dwarnings and source/tool/provider/runtime/binary posts0. No Cargo or model executed. Wrapper source and executed helper SHA are in the receipt; source.before manifest SHA c3eb411c2faf2101df7fed1859072b11912f97367dfcd787a2420a74ec8d3e34.

The new classified_reporting_keeps_terminal_aggregate_failure and transitive_inputs_preserve_missing_and_changed_hash_classification controls are locally UNRUN. They call the actual report-and-aggregate function and existing namelist/hash verifier using fixed metadata literals; they are not numerical-model tests. Future targeted Cargo must also compile the helper in corrected-General and rerun its real-bundle offline compatibility test. Prior PR242 real-bundle CI success is historical evidence for the previous helper, not automatic evidence for this changed variant.

All13 default20 numerical references remain a separate MissingFixture/NotRun problem; this classifier cannot manufacture references, skip selected missing models or grant Pass from preflight. Full API/scenario/doc/MPI/thread coverage remains open. Normal Cargo remains independent of C/Julia/toolbox execution.

Candidate-only post-proof correction: remove redundant & on already-borrowed manifest bytes in from_utf8. This formatting/lint preparation is not retrospectively part of the chmRyh compiled helper. Targeted Cargo verification of the candidate remains required.
