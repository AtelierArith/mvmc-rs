# Original HeisenbergChain public parser input

Byte-original thirteen-file namelist closure from Julia-mVMC revision
`8bb1b9e8ae47b1512c00b321be05664ddcac0fd1`, directory
`MVMCExpertModeParsers.jl/test/samples/HeisenbergChain` (upstream
https://github.com/tmisawa/Julia-mVMC). `inputs.sha256` records each original
file identity. No `initial.def` exists in that source directory; no initial
overlay is named. Unreferenced StdFace/lattice/geometry files are not consumed
by this public parser and are not included.

This is original input, not generated Rust expectations. Tests parse it through
`parse_expert_mode_files`, without invoking Julia, C, or a numerical runner.
Original steps=300/window=30/seed=123456789 are retained as metadata only.
It covers original assertions M0193–198/M0200, not the absent-idx fallback
M0199, numerical sampling, initialization, or SR execution.

Source assertions: `test_parse_expert_mode_files.jl`, SHA256
`dcf0f3d8ed0a272e06696a32ec5e3e060b3acee153392144c6594b9cb662cc10`,
lines 118–184. C authority is `extern/mVMC-1.3.0/src/mVMC/readdef.c`, SHA256
`6c53cb832f93d6cbfd7cea955fbb693738af5536b913d36af32b98eed38c32d9`:
line 593 derives Ne=(NLocSpin+NCond)/2; GetInfoGutzwiller 2117–2142
requires 16 integer mappings plus one flag; GetInfoJastrow 2144–2175
requires 240 directional non-diagonal mappings plus one flag. Orbital has
256 site-pair rows plus 64 flags; TransSym has four weights plus 64 site
maps. This audit establishes supported input shapes, not a full native C run.

Reproduction: `cargo nextest run -p mvmc-expert-parsers --locked --test
issue184_original_heisenberg_public_loader --no-fail-fast --retries 0`.
The Rust fixture preflight rejects missing referenced files, omitted manifest
entries, changed bytes and unexpected implicit initial overlays. That stricter
fixture integrity policy is not falsely attributed to the public loader, whose
optional parameter-overlay handling is tested separately.
