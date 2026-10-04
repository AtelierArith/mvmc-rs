# Corrected GeneralRBM acquisition and separate post-run audit

Linux x86_64; container `73c57e563c61`; Julia executable
`/home/vscode/.cache/mvmc/tools/julia-1.13.1/bin/julia`, actual version 1.13.1.
Environment: `JULIA_DEPOT_PATH=/home/vscode/.cache/mvmc/julia-depot`,
`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JULIA_NUM_THREADS=1`,
`MVMC_CTEST_RBM_SOLVER=direct`; Julia flags `--startup-file=no
--compiled-modules=existing --project=/tmp/mvmc-general-pr54-3d0fd263.aEqff0/extern/Julia-mVMC`.
Actual BLAS: LBTConfig([ILP64] libopenblas64_.so), 1 thread. Default Julia
threads=1, interactive=0; serial, one worker.

GitHub API independently confirmed OPEN PR54 head
`3d0fd2638fd34de2a8f9609fcfaac2504caf02d2`. New isolated clone acquired
that exact head with its pinned submodules; historical62b archives unchanged.
Pinned Manifest SHA256
`09ebd06dab244510094b99fe7c6efa2fe7a3d22221d1336b951123a5a6e8befc`.
Full 63-file source manifest SHA256
`33730954e5ed3b831369723fbff577d0571fbad5067c4697c665c420a4b69605`.

Acquisition root: `/tmp/mvmc-general-pr54-3d0fd263.aEqff0` in the container.
The host `/proc/1038648/root` prefix in scripts is only access to that
container filesystem, not a reference identity; substitute its current PID
or copied artifact root for reproduction. Actual canonical input seed12395,
NSRCG0/NStore1, explicit steps/window1/2/3/20. Input hashes include all
namelist references and implicit initial.def.

Actual handles:20509 exit0 prefix1 actual100-local capture;94143 exit1
preflight no solves;78040 exit1 AFTER allfour original solves and numerical
captures, at missing snapshot stcopt.c metadata tail. The latter is NOT a
green producer. No rerun conceals that failure. Its partial provenance is
preserved. Separate `audit.cjs` post-run validation succeeds; `result.json`
retains exit1, hashes every case file, complete RAW624/index/drawcount,
source closure and explicit authoritative C root. On-disk original sources
remain unchanged; runtime injected observer body is separately hashed in
the acquisition root `direct-v2-injection-identity.txt`. The before-original
`_solve_direct_sr!` and after-original catch hooks do not change that helper.

Reproduce read-only audit (output must not exist):

```sh
node audit.cjs /proc/1038648/root/tmp/mvmc-general-pr54-3d0fd263.aEqff0 \
 /home/terasaki/work/atelierarith/mvmc-rs /tmp/new-posthoc-result.json
node compare.cjs /tmp/mvmc-ctest-prefix-2438630-1791024155673939246 \
 /proc/1038648/root/tmp/mvmc-general-pr54-3d0fd263.aEqff0-direct-v2/general_rbm_cmp \
 /home/terasaki/work/atelierarith/mvmc-rs/tests/fixtures/ctest_reviewed20_62b/general_rbm_cmp \
 /tmp/new-gen2-comparison.json
node compare-current.cjs \
 /proc/1038648/root/tmp/mvmc-general-pr54-3d0fd263.aEqff0-direct-v2/general_rbm_cmp \
 /tmp/new-current-comparison.json
```

Executed results: audit exit0; Gen2 comparison exit0; fresh Rust3/20
comparison exit0, all listed primitives/discrete/numeric quantities exact.
Rust3 actual nextest run7d8a34f4-7c13-40be-872b-a8988fd8ab42 exit100
old62b OO40 assertion; Rust20 e0697ffb-f80b-4ebf-8092-c99e3f22ce4c
exit100 old62b OO36 assertion. Artifacts retained before assertions;
new-reference comparisons separate, not historical gatePASS. Rust test
source SHA580c607510ca339b0fc820b7b494ea212ebeade270814f20573346469d4dcfa6;
focused clippy exit0. No tolerance changed, no C/Julia runtime in Cargo.

Optional C window authority is the actual vendored repository root, not
the incomplete acquisition C subset. Extraction/license boundaries are
documented in repository `c_toolbox/ctest_opt_window.md`: avevar.c lines
1–27 and34–260, SHA509a573944a5eabde93864346902674faaff6c23d0c88f89867263f8372dd51a.
No numerical source edits. Ubuntu GCC13.3.0-6ubuntu2~24.04.1:

```sh
cc -std=c11 -O0 -ffp-contract=off \
 /home/terasaki/work/atelierarith/mvmc-rs/c_toolbox/ctest_opt_window.c -lm \
 -o /tmp/mvmc-pr54-general-posthoc-audit.kBZ8du/ctest_opt_window
# For each step1/2/3/20 use its independent c-window-input.txt as argv1,
# and a NEW exclusive output-prefix as argv2.
```

Initial extra `-Wall -Wextra -Werror` compile failed on existing upstream
sprintf size warnings; documented compiler options subsequently succeeded.
Probe adapter validates output-head length with64 suffix bytes reserved.
Allfour optional Cprobe executions exit0. `compare-windows.cjs` exit0 compares
all generated main/family files including literal headers (not coerced to
NaN), finite means/deviations/reserved fields. Correct report is
`window-comparison-v2.json`; earlier `window-comparison.json` incorrectly
classified textual headers as nonfinite and is superseded, not acceptance
evidence. All actual numeric components are finite in this workload.

This is independent corrected Julia history + nativeC window aggregation,
NOT full C sampler/SR validation. Only GeneralRBM4 captures at3d0f: the
remaining historical model evidence retains62b provenance. No checked-in
fixture adoption or full52/new-version/milestone-completion claim.
