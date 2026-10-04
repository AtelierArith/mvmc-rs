# Rust vs Julia Hubbard-chain benchmark

- generated: 2026-10-03T10:59:40Z
- platform: Darwin arm64
- rustc: rustc 1.98.1 (48a229cea 2026-09-01)
- julia: julia version 1.13.1
- steps/reps/warmups/threads: 300/3/1/1
- inputs: benchmark/hubbard_chain/inputs
- timing: internal `run_para_opt_from_namelist` wall clock; Julia JIT and
  process startup are excluded, and both sides are thread-pinned

| model | Rust median (s) | Julia median (s) | speedup (julia/rust) | |ΔE| |
|---|---:|---:|---:|---:|
| hubbard_chain_L16 | 4.210 | 3.102 | 0.737x | 0.00e0 |
| hubbard_chain_L24 | 9.040 | 6.539 | 0.723x | 0.00e0 |
| hubbard_chain_L32 | 16.190 | 11.966 | 0.739x | 0.00e0 |

`speedup = julia / rust`; values above `1.0x` mean Rust was faster.
