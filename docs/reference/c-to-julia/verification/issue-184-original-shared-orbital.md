# Original M0585: scoped LOCAL_ONLY shared-coefficient observation

Related184/185. Executed base9b6ed40754e88eb2dd8c3c3978309540df19b32d.
Draft publication base da979375739292e5e0c68a0413768a78094bee74 retains
PR295's two aggregation scripts and PR296 diagnostic; this main update is not relabelled
as the execution base. Validated test bytes remain identical.
Original8bb1b9e8ae47b1512c00b321be05664ddcac0fd1:
MVMCOptimizers.jl/test_unit/test_unit_parallel.jl:261–271,
SHA a2b582f61dc981c318f817fc0575cfadfdb8c93e34c5679aeaf7fdae48be5050.
Byte-original13 input files: tests/fixtures/original_heisenberg_parser_184,
inputs.sha256 and existing README bind original parser sample directory.
No Rust-generated expected fixture, coefficient initialization or RNG draws.

The test public-parses original Heisenberg, requires duplicate idx, then
unpacks original packed values + literal complex(0.1,-0.1). Each mapping
observes its canonical dense coefficient; independent before+delta expectations
cover every mapped index and every declared dense slot, including holes if any.
Holes are assigned the original full-vector delta, not preserved pre-unpack.
Mapping rows themselves remain unchanged. Signed coefficients/sign application
and M0586 independent Julia orbital-value repair are separate unverified
contracts, not implied by this observation. M0586 remains unchanged/pending.

Actual local receipt /tmp/mvmc-184-m0585-proof.20261004:
exact LIST1/RUN1 PASS,0 skip,0.006s; UUID43dd6e1b-bd40-4a08-9060-2b2aa4cee49d.
Strict target Clippy/fmt/postLIST, source/input membership+hash, tools/compiler
providers, executable/runtime SHA comparison and every owner cleanup0;
aggregate0. Transient proc/stat ENOENT diagnostics remain in original owner logs.
test SHA d6364dbc35aa73837ec43290d85d4159b74594bd7bccfb88ee6f4369e44095e0;
focused/command.stderr SHA320d1d265d73aa2e6742e63c523db2bb6f69a7a14098cb4785a5ae1d77312153.

Commands: cargo nextest list/run --locked --cargo-profile test-fast
-p mvmc-core --test issue184_original_shared_orbital; LIST --message-format json;
RUN --no-tests fail --no-fail-fast --retries 0.
cargo clippy --locked --profile test-fast -p mvmc-core
--test issue184_original_shared_orbital -- -D warnings; cargo fmt --all --check.
Default features/jobs2/BLAS OMP MKL1; exactc078 static gitlink hydrated before
freeze, not relabelled as original8bb. No C/Julia/MPI/model/fullworkspace run.
Draft test bytes match actual executed source. New-head CI pending.

M0585 is LOCAL_ONLY; a separate reviewed M0587 join binds merged PR296 evidence.
All2342 original rows and owners retained, M0586 and parent scenarios unchanged.
No full184/185 claim.
