# C / Julia / Rust rank and thread comparison

Periodic half-filled Hubbard chain, t=1, U=4. Opt300 / PhysCal100; total320 samples, BLAS1; warmup1, repetitions3; medians in seconds.

The 1×1 case is a baseline. The remaining five cases have ranks×threads=16. C uses rank-zero internal All; Julia/Rust use warmed maximum-rank production API time. Startup/JIT/setup are excluded; timing boundaries differ.

PhysCal uses a common C-generated optimized parameter file for the three implementations within each size/configuration. Different rank configurations generate their own parameter files and RNG trajectories, so this is an end-to-end workload comparison rather than a fixed-trajectory scaling experiment.

## 1 rank × 1 thread per rank

| 計算 | サイト数 | C | Julia | Rust |
|---|---:|---:|---:|---:|
| Opt | 32 | 15.451 | 11.563 | 12.006 |
| Opt | 64 | 68.306 | 49.805 | 55.072 |
| PhysCal | 32 | 7.424 | 4.460 | 4.612 |
| PhysCal | 64 | 39.266 | 19.455 | 22.601 |

## 1 rank × 16 thread per rank

| 計算 | サイト数 | C | Julia | Rust |
|---|---:|---:|---:|---:|
| Opt | 32 | 11.376 | 11.687 | 6.902 |
| Opt | 64 | 31.041 | 36.263 | 27.759 |
| PhysCal | 32 | 3.909 | 4.539 | 2.584 |
| PhysCal | 64 | 11.838 | 16.777 | 10.372 |

## 2 rank × 8 thread per rank

| 計算 | サイト数 | C | Julia | Rust |
|---|---:|---:|---:|---:|
| Opt | 32 | 5.554 | 5.607 | 4.545 |
| Opt | 64 | 17.449 | 19.298 | 17.968 |
| PhysCal | 32 | 2.167 | 2.325 | 2.023 |
| PhysCal | 64 | 8.006 | 8.221 | 7.604 |

## 4 rank × 4 thread per rank

| 計算 | サイト数 | C | Julia | Rust |
|---|---:|---:|---:|---:|
| Opt | 32 | 3.630 | 3.419 | 3.186 |
| Opt | 64 | 14.107 | 13.217 | 13.750 |
| PhysCal | 32 | 1.536 | 1.386 | 1.406 |
| PhysCal | 64 | 6.877 | 5.238 | 5.787 |

## 8 rank × 2 thread per rank

| 計算 | サイト数 | C | Julia | Rust |
|---|---:|---:|---:|---:|
| Opt | 32 | 3.105 | 2.966 | 2.919 |
| Opt | 64 | 13.410 | 12.084 | 13.094 |
| PhysCal | 32 | 1.216 | 1.267 | 1.065 |
| PhysCal | 64 | 6.433 | 5.135 | 4.849 |

## 16 rank × 1 thread per rank

| 計算 | サイト数 | C | Julia | Rust |
|---|---:|---:|---:|---:|
| Opt | 32 | 3.393 | 3.680 | 3.455 |
| Opt | 64 | 16.827 | 16.429 | 16.186 |
| PhysCal | 32 | 1.168 | 1.463 | 1.039 |
| PhysCal | 64 | 6.638 | 6.087 | 4.868 |
