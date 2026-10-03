# Public MPI CLI output and launch-policy checkpoint

Owned optional gate `scripts/verify_mpi_issue179_cli_policy.sh` runs the actual
public MPI-enabled CLI on worlds 2/4, not the state harness. One-step real direct
SR, width1/store0, fixed seed1, sample count3, final parameter window1.
Checks: successful bounded launch; each Rust CLI banner/model/completed-step/
output-path/energy/window summary appears exactly once; zvo_out/zvo_var have
one line and optimized parameters exist. These are output/message boundaries,
not independent numerical accuracy or proof of every Julia message string.

Build handle **42000** terminated **0**, frozen snapshot `issue179-62b.qAZUvg`,
named target `/home/vscode/.cache/mvmc/target/issue179-62b`:
`cargo build --locked --profile test-fast -p mvmc-cli --features mpi --message-format=json`.
CLI binary SHA-256
`8fe67d0a91d1e7da422e2532b971e528c54009ce26ff4c08af45563dd605769c`.
Actual build log reports 3.89 seconds. Build JSON/log hashes are respectively
`788e8a6867fcebcd24fe48826bac45c336cdaf9fdc8fe4d68376f080e9c5553a`
and `d80fd465cf0acd076b3699fa2a31934bef4becff208e174c3da0ac1c6bd23e75`.
Executed gate SHA-256
`cdef647af1d68f57dc0fd4664fcbaa97b118203cd4302966580c64fd1b276175`.

Actual command in that frozen snapshot:

```bash
bash scripts/verify_mpi_issue179_cli_policy.sh \
 /home/vscode/.cache/mvmc/target/issue179-62b/test-fast/mvmc \
 /home/vscode/.cache/mvmc/mvmc-issue179-evidence.2sRkYA/r2-real-s1-cg0-store0/inputs/namelist.def \
 /home/vscode/.cache/mvmc/issue179-cli-root-policy-first
```

Actual tool chunk **ef31dc**, terminal **0**; table rows `2 0 0`, `4 0 0`.
Evidence container `73c57e563c61`, repaired MPICH4.2 internal Hydra PMI, root
`/home/vscode/.cache/mvmc/issue179-cli-root-policy-first`.
Each launch uses `timeout --kill-after=5s 60s`; BLAS/OMP/MKL/BLIS threads1.
Retained files: `results.tsv`, `world2.log`, `world4.log`, `world2/`, `world4/`,
`binary-checker.sha256`, `binary-checker-check.txt`, `inputs.sha256`,
`input-check.txt`, `mpi-version.txt`, `binary-ldd.txt`, `terminal-summary.txt`.
All terminal binary/checker/input hash checks passed.
Parent independently reran the same frozen MPI binary and gate: **e960ec**
terminated **0**, world2/world4 table rows both launch0/boundary0, evidence root
`parent-cli185.EiTQL0/mpi` (parent-retained command uses the same documented
binary/input paths). This corroborates within-implementation public CLI boundaries
only, not independent numerical accuracy or Julia explicit-mode architecture.
Parent's separate fresh no-MPI child rerun subsequently completed; its definitive
result is recorded in the no-feature section below.
World2/world4 log hashes:
`fd187833e8e4420e26d76ed248a914b1f4dc092139fd3e89c5e9aa147e571e7a`,
`6c9f117f70c85069a0d5749f51e2d0cda917a1fe32a3b87bf623c72692ea8af6`.

## Actual no-feature launch rejection

Separate named target `issue179-policy-serial` built the same frozen source
without feature mpi. Build handle **41375**, terminal **0**.
Owned gate `scripts/verify_mpi_issue179_no_feature.sh`, SHA-256
`e08370602bd2f44a9e3fdcb905acfd5ef8941919d5efb4769730acc3d10fe35f`.
Actual no-feature CLI binary SHA-256
`c45a01c27167368ca479ba27392643b5e77bf76840be1f6549e41fb77a466c45`.
It uses the same namelist and explicit step/window overrides as the positive gate.
World2/4 launch exit is **1** in both cases, expected rejection appears once per
rank, and no requested output path is created. The gate rejects zero exits,
timeouts/signals, missing/wrong message multiplicity and output creation.
Launch tool chunk **df8f7c**, verifier terminal **0**, table rows `2 1 0`, `4 1 0`.
Evidence root `/home/vscode/.cache/mvmc/issue179-cli-no-feature-first` retains
results, raw logs, input and binary/checker hashes/postchecks, MPI version and
terminal summary. World2/world4 log hashes:
`905010adc123eebfbfa2450a22649ab96417a123dbe0fbe8af9c00732341950f`,
`f775112e4909507f3002d1baa8456401334616662e3f2b205fa5d8335cde1157`.
This is the Rust compile-feature launch policy, not Julia's explicit-mode policy.

Parent independent no-feature rerun **5b1143** definitively terminated **0**:
world2/world4 both launcher1/boundary0, same frozen `issue179-policy-serial`
binary. Evidence root
`/home/vscode/.cache/mvmc/parent-cli185.EiTQL0/no-mpi`.
Parent read the actual postchecks: binary and scripts all OK. This corroborates
only the documented no-feature launch rejection, with no new feature-policy or
numerical/Julia environment-policy claims.

Exact no-feature build and gate commands (same snapshot working directory):

```bash
CARGO_TARGET_DIR=/home/vscode/.cache/mvmc/target/issue179-policy-serial \
LIBCLANG_PATH=/usr/lib/llvm-18/lib OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
timeout --kill-after=5s 600s cargo build --locked --profile test-fast \
  -p mvmc-cli --message-format=json
bash scripts/verify_mpi_issue179_no_feature.sh \
 /home/vscode/.cache/mvmc/target/issue179-policy-serial/test-fast/mvmc \
 /home/vscode/.cache/mvmc/mvmc-issue179-evidence.2sRkYA/r2-real-s1-cg0-store0/inputs/namelist.def \
 /home/vscode/.cache/mvmc/issue179-cli-no-feature-first
```

Both executed CLI/checker manifests were rechecked successfully after parent
requested harvest of handle 42000. No build or MPI launcher was restarted.

Remaining: serial invocation boundaries;
public non-root result shape; full named PhysCal output/message cases; tiny-weight
root warning. No-warning current helper gap is assigned to Pauli for C-authority
review, not repaired here by printing a manufactured harness warning. Julia's
explicit MPI mode environment switches are not Rust CLI switches; architecture
differences require explicit policy mapping rather than assumed equivalence.

Pauli's C review confirms `average.c` WeightAverageWE41–75 has unconditional
division in both branches and no tiny-weight warning. Julia MPI87's warning
assertion is at `run_mpi_smoke.jl:179`; its worker explicitly injects zero weight
at `mpi_weight_average_smoke.jl:33`, not a small-positive sampling trajectory.
The Julia-only warning is an architectural diagnostic requirement, not a native
C numerical contract. No production guard/logging changes are made in this gate.
