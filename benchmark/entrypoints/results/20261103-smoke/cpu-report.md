# Hubbard L128 benchmark

Mode: cpu; steps=2; samples/walkers=16; warmup=1; measured repetitions=1; smoke=True.

CPU: C/Rust process wall (C includes mpirun, Docker launch excluded); Julia warm API including input/output, JIT and explicit GC excluded. GPU: both warm public APIs; setup/JIT/rep0/explicit GC/serialization excluded.

CPU uses c-order PF/SR, physical cores, BLAS=1. GPU Opt uses the resident ensemble; GPU PhysCal Rust uses a single SFMT chain while Julia uses XORWOW walkers (different trajectories/proposal counts). Common definitions and fixed parameters do not imply identical sampling work or kernel speedup.

## opt

| CPU threads | C | Julia | Rust |
|---:|---:|---:|---:|
| 1 | 0.699 s | 0.639 s | 0.558 s |
| 4 | 0.376 s | 0.598 s | 0.330 s |

## phys

| CPU threads | C | Julia | Rust |
|---:|---:|---:|---:|
| 1 | 0.419 s | 0.412 s | 0.315 s |
| 4 | 0.250 s | 0.506 s | 0.239 s |

This driver validates completed repetitions, finite timings and runtime output checks. It does not establish cross-implementation numerical/RNG parity. Performance acceptance is a separate recorded cohort (#629).
