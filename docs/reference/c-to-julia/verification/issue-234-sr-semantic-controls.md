# Issue234 SR protocol and saved-configuration developer controls

Related to #234, #178 and #185; this does not close any issue. Developer-only scripts under `scripts/issue234/` are not imported, read or executed by Cargo, build scripts or normal Rust tests. No C/Julia oracle runtime is required. Production numerical kernels and public APIs are unchanged.

## Standalone reproducible schema controls

From a complete checkout, choose a nonexistent output path whose parent already exists:

```sh
bash scripts/issue234/semantic_controls_wrapper.sh \
  /tmp/issue234-schema-controls-UNIQUE --schema-fixtures
```

The wrapper creates six deterministic analytic schema fixtures and executes the semantic checker on six positives and48 independently checked negative mutations. Fixture occupancy is one up electron at site0 and one down electron at site1: index `[0,1]`, map `[0,-1,-1,0]`, occupancy `[1,0,0,1]`. Raw624/next624 are literal zero lists with cursor/count0, solely a schema fixture, NOT an SFMT seeded stream or sampling oracle. Scratch arrays are empty grammar fixtures and their physical contents are not interpreted. No Rust-produced goldens, fixture generation by a sampler, MPI, model execution or numerical tolerance occurs.

This packaged synthetic mode was executed once and reviewed as bounded schema-only proof (details below). Its48 negatives are e999 overflow, missing final configuration brace, cursor625, raw2^32, raw-1, raw623 words, count2^128, and saved occupancy2. Each mutation changes exactly one intended field in the first CHECK; all other lines/fields must be byte-identical. Rejection requires exact status1 and the intended diagnostic, not arbitrary nonzero or syntax/I/O failure.

All configuration fields/array closures are required; saved-plane shape/range/occupancy/inverse identities are checked.4096cells per saved plane and bounded dimensions are diagnostic capture limits, not new supported-input restrictions. Scratch/burn fields are grammar-checked only. Historical Debug storage architecture is not reinterpreted as a C packed-memory contract.

Output cap16MiB, reserve384MiB, initial400MiB, external GNU timeout120s/kill10s. These are sampled guards, not filesystem quotas. Wrapper/runner/checker/mutation source, actual tool binaries/providers and input captures are bound pre/post. Existing output paths fail. Any first failure is retained, never automatically retried.

## Real or historical capture controls

A table consists of exactly six canonical `rank world absolute_capture_path` rows: worlds2/4 with ranks0..world-1; paths must not contain whitespace. Inputs must be complete normalized rank captures. An optional strict protocol validator may be supplied:

```sh
bash scripts/issue234/semantic_controls_wrapper.sh \
  /tmp/issue234-capture-controls-UNIQUE /absolute/six-capture-table.txt \
  /absolute/successful-stage-results /absolute/validate-sr.awk
```

Use `-` for the stage parameter when none applies. Protocol checker absence is explicitly reported as protocolChecked=false; synthetic mode never pretends to have a native protocol receipt. Historical18DONE negatives remain separate and are not repeated by these48 semantic controls.

## Historical acquired proof (not current-main execution)

### Packaged analytic schema execution

Owner66292 completed0 using candidate35ad2e1ab0470dfe8963d207864240cd0fe5f229 plus exactly six reviewed developer/provenance paths. CONTAINER receipt `/tmp/mvmc-234-packaged-schema.EOvU65`: outer/wrapper/inner all prior0/post0, six positives and48 negatives, protocolChecked=false/noModels=true. Parent independently verified all6source, runner, inner source/tools/inputs SHA manifests. Wrapper output2,285,132bytes, final free819,646,464bytes; owner wholepacket sample2,313,703bytes.16MiB cap and384MiB reserve maintained. External120s/k10 timeout; no Cargo/MPI/native/model execution.

The6-file executed manifest SHA is749ab2470f924fc0015ef7c5b7834457d41bc30677fd404141395b947cbd14af. Exact script SHA256 pins:

| Executed path | SHA256 |
|---|---|
| semantic_controls.sh |be8a4de0e75ac05a0c780f07241b48cf1f1aafc3e411bccb36aefde1564c1b96|
| semantic_controls_wrapper.sh |b1216c242f090ceb793d62c10d09f59ab44f219f39e6b162f2a03c9f7a9fa860|
| validate_current_native_semantics.sh |18f4f7313fdfb05da786a2b9649c63a79901567b2fecc0a59d11decb6740cab9|
| validate_sr_semantic.awk |4eef32525c4828d3bb57f71b6a5e1ec65379de784408fd8f75bf285749923391|
| verify_negative_mutation.awk |19433a974a9a72e2f1f78b5729882800b4b962a337bd752961c8853f321688cc|

The executed doc SHA was c6346af7246ae440d9496dfc6d1460a1b00795b7ace2c96fdbb396cb1f40b561. This proof paragraph is a later doc-only update; final doc SHA belongs to the publication manifest, not the execution manifest. Current-main hook remains UNRUN even though its source was bound among6paths. The new schema fixtures are analytic grammar controls, not C-compatible RNG/sampling expectations or complete234 acceptance.

### Earlier acquired real captures

ONE host-owner71185 execution completed0 in Linux x86_64 container happy_jackson. CONTAINER receipt `/tmp/mvmc-234-semantic48.PZ3Syb`: outer/inner prior0/post0; six positives and48 status1/intended-diagnostic negatives;48 intended-field mutation receipts; parent independently SHA-checked stage/runner/inputs/tools. Result subtree7,937,799bytes; whole packet owner sample7,970,670bytes; reserve384MiB maintained. PID/PGID was not captured before fast completion. These local receipt paths describe actual acquisition, not portable file dependencies or attached public artifacts.

Exact executed-source SHA256 pins:

| Input | SHA256 |
|---|---|
| runner |02765f8d065a47d8d2d684c835e7cf38c0a7ef13ee23b4ae7a768ab79c104cec|
| external wrapper |15cf6908d232cbeb2ecd1a77d56dd04b6458a8bfb793bc89bc4b4d5c3b507250|
| semantic checker |4eef32525c4828d3bb57f71b6a5e1ec65379de784408fd8f75bf285749923391|
| mutation verifier |19433a974a9a72e2f1f78b5729882800b4b962a337bd752961c8853f321688cc|
| unchanged strict protocol awk |1c90f4a998b53071f7a912cdc7b410b98d5082d77d1c4c0c1806aec4b109b6b8|
| six-capture table |28f7468781a900e56c5ce1b87b437c95ba9706cec235e63f9362e55f15081afd|

The six raw inputs were historical retained SR worlds2/4 captures under xW9KeD/HmL7y5, not regenerated by Rust for this proof. Fixed return/INFO/solve/callback protocol reflects actual native test assertions; the semantic checker adds lexical and saved-plane validity, not a numerical oracle. The separate packaged analytic-schema run above verifies its interface/reproducibility subset only; it does not repeat real capture acquisition or establish current-main native behavior.

## Current-main semantic hook

After a reviewed source-bound current-main build and successful fresh native SR launch:

```sh
bash scripts/issue234/validate_current_native_semantics.sh \
  FULL_MAIN_SHA /absolute/build-results /absolute/native-sr-root \
  /absolute/actual-native-pins.sha256 /absolute/validate-sr.awk \
  /tmp/issue234-current-audit-UNIQUE
```

The fresh launcher must record `association.txt` with exact main, selected binary path/SHA and `selected.before.sha256` BEFORE execution. The hook rejects historical captures with a newly invented main label as an inadmissible acquisition; it requires the recorded executable in successful build selection and launch pins, both worlds native0, six complete rank captures, source/input/tool closure. Association text alone is not proof: parent reviews launcher creation order and actual invocation receipts.4MiB audit output/384MiB reserve; no MPI/Cargo invocation by the hook. Use an owner external timeout120/k10 for audit execution as well.

Normal Rust authority remains the actual public runner tests `mpi_issue178_sr_failure` and `mpi_issue234_summary`, and existing independently referenced PhysCal/SFMT configuration/RNG tests. No artificial public Debug parser API is added merely to test developer text validation. A genuinely missing normal saved-domain assertion may be proposed as a separate Rust regression with analytic or existing independent fixture expectations.

Current-main build/native remain pending. Required final acceptance still includes current-source SR six-rank captures with widths1/2, healthy/root/last faults and exact repeat/checkpoints; summary MPI rootreadback/healthy cases; actual newly built CLI260 strict argv/diagnostic/output validation; final workspace/lint/docs and CI. Historical CLI260 over guarded binary c435a4e3ea3b5d175623d46e12b06e5055e4987d53647dd974c61b76c3defb58 is NOT current-main proof. Actual PR241 squash main14182afb0403931ff340ae93dd2ec3079114b0a8 and reviewed head35ad2e1ab0470dfe8963d207864240cd0fe5f229 both have tree d9c15dd2a8e768905df1608194e80b35d5a85493; complete tree diff0 verified before publication reconciliation. Publication candidate subsequently moved to actual PR242 main4cebcf361544f36fa65f8f813de76f38016bab9a; the14182-to4ce delta contains only three classifier/test/doc paths and no developer-source overlap. No historical whole-file overwrite occurred. EOvU65 remains historical35ad schema execution, not a newly executed current-main native proof; old d7cb stage proofs likewise keep their original lineage.
