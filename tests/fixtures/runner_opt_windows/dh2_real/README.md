# Independent DH2-real direct and CG windows

Same unchanged source snapshot, C extraction, compiler/runtime and aggregation
executable as `../dh24_real/direct-store0/README.md`. The observer commands use
`--case=dh2_real --steps=1,2,3,50` and respectively `direct` or `cg`.
Both original-script workloads passed twelve status assertions. External stages
are `/tmp/mvmc-runner-independent-dh2_real-{direct,cg}`; generation-loop handles
9527 and 72623 continued to other models after these completed workloads.

Each explicitly sets steps=window=N and NStore=0, seed=1, with the historical
nonzero `tests/fixtures/dh2/production_real/namelist.def` overlays, not a
substituted canonical model. Per-prefix provenance records all overlay input
hashes and actual modpara. NPara=25: thirteen projection slots (including all
six DH2 slots) plus twelve Slater slots. Window rows contain actual Julia
Etot/Etot2 and complete declared synchronized post-SR parameters.

Verbatim C StoreOptData/OutputOptData aggregation executed for all eight
method/prefix combinations. Every corresponding RNG and saved-configuration
checkpoint matched the unchanged archived `sr_{direct,cg}/dh2_real_runner`
record exactly by cmp. This is mixed independent Julia-history/C-aggregation
evidence, not a native C sampler/solver or MPI comparison.

Only checked-in files are read by Cargo. Full independent C filename/layout
manifests, pre-SR declared `zvo_var.dat` and computed-window values are asserted
with the existing abs=rel=1e-11 policy; missing DH output cannot be adapted away.
