# Issue #179 P02 schema controls (optional developer tooling)

This directory contains standard-library-only Python validators and independent
synthetic fixtures for one declared MPI scenario: ordinary real, direct SR,
NStore1, world2, width1, seed1, prefix1, samples3, worker1. It does not run MPI,
Rust binaries, Julia, C, an RNG, or a numerical solver. Cargo never imports or
invokes these tools. No reference runtime is needed to run the controls.

From this directory, run the complete four-module suite explicitly:

```sh
uv run --offline --no-project --no-python-downloads --python /usr/bin/python3.12 python -B -m unittest -v test_p02_binding test_p02_cross_discrete test_p02_typed_fullpack test_rank_binding
```

The absolute Python selection above reproduces the Linux verification receipt;
on other hosts select an already installed Python explicitly through uv. No
dependencies, project environment or lockfile are required or created here.

The controls exercise closed discrete inventories, seeded/initial/checkpoint/
final raw624 and cursor observations, nonconsuming future624, proposal/decision
word framing, saved-buffer shapes, counters, rank artifact ordering, typed SR
metadata, finite numerical arrays, matrix layout, output schema and C-written
flags. Fixtures are independently constructed synthetic values, not Rust/Julia
generated numerical expectations. Julia observed primitive consumption is not
a native counter; its native draw counter remains unavailable.

`p02_join.join` is an optional retained-artifact diagnostic function. It requires
original source bytes/hashes and C-reader/input-derived flags. It does not
authenticate a model run or providers. Its runtime/model/numerical acceptance
fields stay false. External caller authority, actual rank/provider association,
full source/input membership and PRE/POST checks remain required. Positive SR
records only are supported; failures/no-active/CG are not silently accepted.
No tolerance is applied to computed numerical differences. Native INFO and
Julia's checked-successful-return observation remain separately typed.

Source pins in `p02_binding.py` describe frozen historical Rust main34/Julia
c078 recorder/build inputs; they are not declarations about current main.
See [PROVENANCE.md](PROVENANCE.md) for source lineage and the bounded proof.

This suite is not the full #179 acceptance matrix. Complex/FSZ, world4, grouped
and uneven widths, requested-width>world, CG, other stores/prefixes/long20,
QP/OptTrans, empty ranges, actual workers/FUNNELED and coordinated failures
remain separate model-verification obligations. The approved P02 native pair
has not started because of its container SHM startup floor. Synthetic success
does not change that floor, repair historical provider gaps, or prove parity.
