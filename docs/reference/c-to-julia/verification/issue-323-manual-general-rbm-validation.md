# Manual General RBM PhysHidden diagnostics (#323)

Publication successor on main20dab9c5a12279e1d0337c77d13168e3846cc1cb. Related #184/#185, original API A180 only; no original assertion/scenario or canonical2342 status promotion. Six new controls plus one existing dense-validator regression passed the scoped Linux gate below. Full exact-head workspace/native CI remains pending.

`validate_general_rbm_phys_hidden_terms` is an explicit read-only term-slice utility returning `ValidationResult`. Both raw sites use caller-supplied `nsite`; errors are row/site1/site2 ordered. Term shadow magnitude above `1e10` gives a nonfatal row-ordered warning. Warnings alone do not make the result invalid. It neither checks spin/index/finiteness/dense consistency nor repairs values. No runner, parser, pack or automatic validator wiring, RNG or input acceptance changes.

Original Julia `8bb1b9e8ae47b1512c00b321be05664ddcac0fd1:MVMCExpertModeParsers.jl/src/utils/validation.jl:544–574` supplies the manual contract. C `d73d06bd529d3b2573f38eb5817c4a5f52971006:src/mVMC/readdef.c:1002–1007,2723–2749` uses physical `Nsite` and hidden `NneuronGeneral` plus spin/index/count input checks. These are separate runtime contracts: the manual utility must not reject C-valid hidden coordinates using the Julia architecture's diagnostic dimension.

Magnitude authority: Julia1.13.1 `base/complex.jl:277` calls `hypot(real,imag)`; `base/math.jl:746–801` checks infinity before NaN-bearing arithmetic, and rescales extreme finite operands. Locked Rust num-complex0.4.6 `src/lib.rs:217–219` calls `re.hypot(im)`. Rust [f64::hypot documentation](https://doc.rust-lang.org/std/primitive.f64.html#method.hypot) does not promise cross-platform exact rounding. The utility explicitly selects warning for any infinite component, no warning for remaining NaN-containing values, otherwise compares finite norm to the threshold. This preserves diagnostic special-value decisions without porting numerical runtime algorithms or asserting bitwise magnitude equality.

Independent controls: empty/boundary coordinates; four ordered site errors; exact1e10 versus1e10+1; literal3-4-5 complex magnitudes5e9/5e10; huge finite magnitudes and minimum subnormal; Inf+NaN in both operand orders, NaN-only, combined errors/warnings, unrelated spin/index preservation and complete raw-bit nonmutation. No C/Julia invocation or generated expectations in Cargo.

## Actual scoped Linux verification

ONE handle45497, receipt `/tmp/mvmc-184-issue323-proof.20261004`, executed source `/tmp/mvmc-184-a180-main20dab`, private target `/tmp/mvmc-184-issue323-target.20261004`. Default features, locked/test-fast/jobs2/BLAS-OMP1. Fixed selection:

```text
cargo nextest list --locked --cargo-profile test-fast -p mvmc-expert-parsers --test issue323_general_rbm_validation --test validation -E 'binary(issue323_general_rbm_validation) | test(rbm_validation_checks_declared_storage_and_c_mapping_coordinates)' --message-format json
cargo nextest run --locked --cargo-profile test-fast -p mvmc-expert-parsers --test issue323_general_rbm_validation --test validation -E 'binary(issue323_general_rbm_validation) | test(rbm_validation_checks_declared_storage_and_c_mapping_coordinates)' --no-tests fail --no-fail-fast --retries 0
cargo clippy --locked --profile test-fast -p mvmc-expert-parsers --all-targets -- -D warnings
cargo fmt --all --check
```

Actual seven selected tests PASS, eight unselected **NotRun**, 0.006s, UUID `89e60f3e-d580-4ecf-b63d-06ab4a1ee925`. Clippy6.48s/fmt/postLIST0; all45 status files plus primary/posts/terminal0. Every five-phase owner original0/cleanup0/reapNOT_STARTED; source/ELF independent replay0, PRE/POST LIST identity and runtime providers unchanged. No automatic retry, C/Julia/model/MPI execution or full-workspace claim.

Executed full-source26410-member manifest SHA `3a67bbe274a6f05c8f9cc80469b2d17ac3ae79f49644c9c3fdd5a0541a8b9556` remains immutable, as do its receipts. Three code files are byte-identical in publication: lib SHA `fb8923a5f3aa50ee303d670ad3c92baab3dbf38e5c2e5bef79763c8c3b9dee4d`, validation SHA `eb9968620761daf97f2587c22000ffff919949605e1b0a46b5326b0f997beb5d`, tests SHA `240387bc352afe0a2e3bada5079d7febaf31f070f980f6e377e33af807dd969f`. This documentation-only successor replaces the historical SOURCE/NotRun wording; original tested doc SHA `a9ccddf415d6b67560381feb8ea149a8ab493cd584ec0ad7814c612c6e5c30db` and original PRE are not rewritten/refrozen.

Static preparation hydrated c078 ordinary file closure; extra first blob-inspection mishandled nested gitlinks and is retained separately. Corrected read-only comparison verified869 blobs, with nested PfaPack/SFMT gitlinks explicitly not recursively hydrated. This does not certify Julia runtime dependencies. Existing C loader/dense validator/pack behavior and all numerical/fixture/RNG contracts are unchanged. Parent requires fresh full Linux CI before merge.
