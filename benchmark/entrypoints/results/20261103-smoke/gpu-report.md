# Hubbard L128 benchmark

Mode: gpu; steps=2; samples/walkers=16; warmup=1; measured repetitions=1; smoke=True.

CPU: C/Rust process wall (C includes mpirun, Docker launch excluded); Julia warm API including input/output, JIT and explicit GC excluded. GPU: both warm public APIs; setup/JIT/rep0/explicit GC/serialization excluded.

CPU uses c-order PF/SR, physical cores, BLAS=1. GPU Opt uses the resident ensemble; GPU PhysCal Rust uses a single SFMT chain while Julia uses XORWOW walkers (different trajectories/proposal counts). Common definitions and fixed parameters do not imply identical sampling work or kernel speedup.

## opt

| GPU | Median |
|---|---:|
| Julia GPU | 0.282 s |
| Rust GPU | 0.169 s |
| C | No GPU implementation |

## phys

| GPU | Median |
|---|---:|
| Julia GPU | 0.206 s |
| Rust GPU | 0.205 s |
| C | No GPU implementation |

This driver validates completed repetitions, finite timings and runtime output checks. It does not establish cross-implementation numerical/RNG parity. Performance acceptance is a separate recorded cohort (#629).
