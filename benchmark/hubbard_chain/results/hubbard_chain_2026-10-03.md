# Rust vs Julia Hubbard-chain benchmark

- generated: 2026-10-03T11:16:20Z
- platform: Darwin arm64
- rustc: rustc 1.98.1 (48a229cea 2026-09-01)
- julia: julia version 1.13.1
- rust BLAS: /opt/homebrew/opt/openblas/lib/libopenblas.0.dylib
- julia BLAS: lbt (LBTConfig([ILP64] libopenblas64_.0.3.30.dylib))
- steps/reps/warmups/threads: 300/3/1/1
- inputs: benchmark/hubbard_chain/inputs
- scope: serial `R=1`; this is not the report's `R=4` MPI condition
- timing: internal `run_para_opt_from_namelist` wall clock; Julia JIT and
  process startup are excluded, and both sides are thread-pinned

| model | Rust median (s) | Julia median (s) | speedup (julia/rust) | |ΔE| |
|---|---:|---:|---:|---:|
| hubbard_chain_L16 | 4.183 | 3.069 | 0.734x | 0.00e0 |
| hubbard_chain_L24 | 8.881 | 6.488 | 0.731x | 0.00e0 |
| hubbard_chain_L32 | 16.254 | 12.228 | 0.752x | 0.00e0 |

`speedup = julia / rust`; values above `1.0x` mean Rust was faster.
