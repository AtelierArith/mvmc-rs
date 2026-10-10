# C / Julia / Rust rank and thread comparison

Periodic half-filled Hubbard chain, t=1, U=4. Opt300 / PhysCal100; total320 samples, BLAS1; warmup1, repetitions3; medians in seconds.

The 1×1 case is a baseline. The remaining five cases have ranks×threads=16. C uses rank-zero internal All; Julia/Rust use warmed maximum-rank production API time. Startup/JIT/setup are excluded; timing boundaries differ.

PhysCal uses a common C-generated optimized parameter file for the three implementations within each size/configuration. Different rank configurations generate their own parameter files and RNG trajectories, so this is an end-to-end workload comparison rather than a fixed-trajectory scaling experiment.

## 1 rank × 1 thread per rank

| 計算 | サイト数 | C | Julia | Rust |
|---|---:|---:|---:|---:|
| Opt | 32 | 15.529 | 11.523 | 11.964 |
| Opt | 64 | 69.062 | 50.137 | 56.532 |
| PhysCal | 32 | 7.465 | 4.515 | 4.658 |
| PhysCal | 64 | 39.351 | 19.474 | 22.703 |

## 1 rank × 16 thread per rank

| 計算 | サイト数 | C | Julia | Rust |
|---|---:|---:|---:|---:|
| Opt | 32 | 12.164 | 18.115 | 8.468 |
| Opt | 64 | 32.885 | 60.392 | 32.349 |
| PhysCal | 32 | 4.014 | 4.550 | 3.459 |
| PhysCal | 64 | 13.616 | 22.452 | 12.384 |

## 2 rank × 8 thread per rank

| 計算 | サイト数 | C | Julia | Rust |
|---|---:|---:|---:|---:|
| Opt | 32 | 5.860 | 6.012 | 4.739 |
| Opt | 64 | 17.635 | 19.715 | 18.286 |
| PhysCal | 32 | 2.220 | 2.340 | 2.095 |
| PhysCal | 64 | 8.238 | 8.250 | 7.558 |

## 4 rank × 4 thread per rank

| 計算 | サイト数 | C | Julia | Rust |
|---|---:|---:|---:|---:|
| Opt | 32 | 3.652 | 3.551 | 3.247 |
| Opt | 64 | 14.228 | 13.265 | 13.751 |
| PhysCal | 32 | 1.512 | 1.389 | 1.425 |
| PhysCal | 64 | 6.823 | 5.149 | 5.810 |

## 8 rank × 2 thread per rank

| 計算 | サイト数 | C | Julia | Rust |
|---|---:|---:|---:|---:|
| Opt | 32 | 3.023 | 3.007 | 2.950 |
| Opt | 64 | 13.515 | 12.411 | 12.995 |
| PhysCal | 32 | 1.289 | 1.282 | 1.096 |
| PhysCal | 64 | 6.386 | 5.136 | 4.853 |

## 16 rank × 1 thread per rank

| 計算 | サイト数 | C | Julia | Rust |
|---|---:|---:|---:|---:|
| Opt | 32 | 3.501 | 3.716 | 3.533 |
| Opt | 64 | 17.490 | 16.674 | 16.605 |
| PhysCal | 32 | 1.203 | 1.468 | 1.082 |
| PhysCal | 64 | 6.690 | 6.367 | 5.095 |
