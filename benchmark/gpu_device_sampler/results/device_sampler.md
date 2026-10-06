# Device-resident lock-step sampler vs CPU multichain (issue #434)

Environment: Intel Xeon E5-2699 v3 (36 hardware threads) shared with other jobs (load average
11 to 36 at the start and end of the runs, see the CSV headers; the numbers below carry that noise,
two complete campaigns agreed within about +-30 %, the L=64 `W=512` row of the 4-core table flipped
between 0.95x and 2.1x), 2x NVIDIA GeForce RTX 3060 (sm_86, 12 GB, device 0), driver 580.178.04,
CUDA driver API 13.0, container CUDA toolkit 12.9.2 (NVRTC 12.9), rustc 1.98.0, tenferro 0.7.1,
Linux x86_64. Docker image `tenferro-benchmark-cuda:full-verify-20260822`.

What is timed: one *call* is `vmc_make_sample_real` of every walker, the C `VMCMakeSample` of one
sample series: `(NVMCWarmUp + NVMCSample) * Nsite` hop attempts per walker (about 3000 per walker
and call; `NVMCSample` is chosen per size), the initial table construction (the CPU builds
its tables for the initial configuration in both variants; the device additionally builds its
resident tables from the configuration: a device `Begin` batch), and the periodic
recomputations. Hubbard chain, half filling, `Lsub = 4`, `U = 4`, `NSPGaussLeg = 8` (`NQP = 8`),
real normal mode, hopping updates, inputs `benchmark/hubbard_chain/inputs/hubbard_chain_L{16,32,64,128}`
(`L128` is new, generated with the Rust StdFace port like `L64`). All `W` walkers share one
wavefunction and have independent SFMT streams (`RndSeed + w`). Each method builds its own walkers,
runs one untimed warm-up call (burn-in), then three timed calls (median). Measurement
(`VMCMainCal`) is not part of the call. Thread creation per call (one thread per walker) and, for the
device, service construction excluded, Slater-table upload included.

* **CPU 1 thread**: the walkers one after the other on one thread (only for `W <= 64`).
* **CPU multichain**: one thread per walker with up to all host cores, the #425 execution model
  (independent walkers, single-threaded inside; BLAS pinned to one thread).
* **CUDA device-resident**: this design, pinned asynchronous transfers (#432), one host thread per
  walker, the Pfaffian stages on the GPU.

The 4-core tables run the same program in a container restricted to cores 0-3 (`docker
--cpuset-cpus=0-3`): a GPU attached to a small host, the CPU baselines and all walker threads of the
device run share exactly those cores. Columns "CUDA / CPU" are wall-time ratios (above 1 the device
is slower). The CSVs are `benchmark/gpu_device_sampler/results/device_sampler_cores{all,4}.csv`
(plus `_pageable.csv`); the metadata block (device report, host cores, load average) is in the
CSV/log header and in `results/device_sampler.md`.

**Wall time of one sampling call, 36 host cores (all of them; shared host)**

| L | W | CPU 1 thread | CPU multichain | CUDA device-resident | hops/s CPU multichain | hops/s CUDA | CUDA / CPU-multichain time | CUDA / CPU-1-thread time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 1 | 0.010 s | 0.010 s | 0.090 s | 298k | 33k | 8.96x | 9.24x |
| 16 | 8 | 0.065 s | 0.014 s | 0.142 s | 1.69M | 169k | 10.04x | 2.17x |
| 16 | 64 | 0.503 s | 0.082 s | 0.932 s | 2.34M | 206k | 11.40x | 1.85x |
| 16 | 512 | - | 0.644 s | 1.983 s | 2.38M | 773k | 3.08x | - |
| 32 | 1 | 0.020 s | 0.020 s | 0.106 s | 151k | 28k | 5.40x | 5.24x |
| 32 | 8 | 0.148 s | 0.024 s | 0.182 s | 1.01M | 131k | 7.72x | 1.23x |
| 32 | 64 | 1.079 s | 0.077 s | 0.852 s | 2.46M | 224k | 11.00x | 0.79x |
| 32 | 512 | - | 0.577 s | 2.142 s | 2.64M | 711k | 3.71x | - |
| 64 | 1 | 0.050 s | 0.052 s | 0.119 s | 56k | 25k | 2.28x | 2.36x |
| 64 | 8 | 0.438 s | 0.058 s | 0.206 s | 404k | 114k | 3.54x | 0.47x |
| 64 | 64 | 4.564 s | 0.249 s | 0.914 s | 757k | 206k | 3.67x | 0.20x |
| 64 | 512 | - | 1.725 s | 4.070 s | 874k | 370k | 2.36x | - |
| 128 | 1 | 0.127 s | 0.117 s | 0.135 s | 25k | 22k | 1.15x | 1.07x |
| 128 | 8 | 1.006 s | 0.160 s | 0.301 s | 147k | 78k | 1.88x | 0.30x |
| 128 | 64 | 8.092 s | 0.756 s | 2.750 s | 249k | 69k | 3.64x | 0.34x |
| 128 | 512 | - | 4.941 s | 10.119 s | 305k | 149k | 2.05x | - |

**Anatomy of the device run, 36 host cores (all of them; shared host)**

| L | W | wall | passes | ms/pass | service thread: wait for walkers / round / reply | in round: stage / upload / launch / device wait | slow batches |
|---|---:|---:|---:|---:|---|---|---:|
| 16 | 1 | 0.090 s | 2911 | 0.031 | 14.6 / 66.0 / 0.9 ms | 0.9 / 15.2 / 27.9 / 20.7 ms | 63 |
| 16 | 8 | 0.142 s | 3031 | 0.047 | 11.6 / 109.7 / 12.0 ms | 2.8 / 20.4 / 49.5 / 35.1 ms | 479 |
| 16 | 64 | 0.932 s | 2917 | 0.319 | 371.5 / 391.3 / 101.7 ms | 24.0 / 91.1 / 157.4 / 108.3 ms | 2210 |
| 16 | 512 | 1.983 s | 2921 | 0.679 | 866.4 / 487.5 / 289.1 ms | 59.6 / 91.1 / 130.2 / 188.9 ms | 2892 |
| 32 | 1 | 0.106 s | 2720 | 0.039 | 17.9 / 76.4 / 1.1 ms | 0.9 / 15.4 / 28.2 / 30.8 ms | 32 |
| 32 | 8 | 0.182 s | 2860 | 0.064 | 19.5 / 134.2 / 21.3 ms | 2.7 / 19.9 / 45.7 / 64.1 ms | 225 |
| 32 | 64 | 0.852 s | 2749 | 0.310 | 215.8 / 355.3 / 77.7 ms | 17.5 / 60.2 / 106.8 / 163.9 ms | 1343 |
| 32 | 512 | 2.142 s | 2723 | 0.787 | 640.9 / 780.9 / 235.1 ms | 50.1 / 80.9 / 118.8 / 514.7 ms | 2654 |
| 64 | 1 | 0.119 s | 2382 | 0.050 | 21.3 / 84.5 / 2.6 ms | 0.9 / 15.1 / 27.8 / 39.4 ms | 14 |
| 64 | 8 | 0.206 s | 2538 | 0.081 | 25.2 / 145.0 / 27.1 ms | 2.5 / 17.2 / 38.8 / 84.9 ms | 100 |
| 64 | 64 | 0.914 s | 2426 | 0.377 | 183.1 / 507.7 / 68.9 ms | 14.4 / 48.3 / 85.0 / 354.4 ms | 677 |
| 64 | 512 | 4.070 s | 2400 | 1.696 | 766.6 / 2313.8 / 264.1 ms | 50.1 / 73.7 / 114.2 / 2061.2 ms | 2074 |
| 128 | 1 | 0.135 s | 1797 | 0.075 | 26.2 / 82.4 / 1.8 ms | 0.7 / 10.8 / 19.1 / 51.0 ms | 5 |
| 128 | 8 | 0.301 s | 2015 | 0.150 | 42.9 / 207.2 / 23.9 ms | 2.0 / 14.6 / 32.8 / 156.5 ms | 36 |
| 128 | 64 | 2.750 s | 1824 | 1.508 | 1212.5 / 1025.9 / 114.0 ms | 17.6 / 69.8 / 110.7 / 820.6 ms | 262 |
| 128 | 512 | 10.119 s | 1810 | 5.591 | 1571.0 / 6430.9 / 200.2 ms | 47.2 / 69.6 / 103.2 / 6199.8 ms | 934 |

**Transfer path (pinned async vs pageable), 36 host cores (all of them; shared host)**

| L | W | pinned async | pageable | pageable / pinned |
|---|---:|---:|---:|---:|
| 16 | 8 | 0.142 s | 0.466 s | 3.29x |
| 16 | 64 | 0.932 s | 0.981 s | 1.05x |
| 64 | 8 | 0.206 s | 0.319 s | 1.55x |
| 64 | 64 | 0.914 s | 1.500 s | 1.64x |

**Wall time of one sampling call, 4 host cores (docker --cpuset-cpus=0-3)**

| L | W | CPU 1 thread | CPU multichain | CUDA device-resident | hops/s CPU multichain | hops/s CUDA | CUDA / CPU-multichain time | CUDA / CPU-1-thread time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 1 | 0.009 s | 0.009 s | 0.099 s | 317k | 30k | 10.49x | 10.60x |
| 16 | 8 | 0.072 s | 0.034 s | 0.304 s | 704k | 79k | 8.93x | 4.22x |
| 16 | 64 | 0.573 s | 0.200 s | 1.079 s | 957k | 177k | 5.39x | 1.88x |
| 16 | 512 | - | 1.360 s | 5.553 s | 1.13M | 276k | 4.08x | - |
| 32 | 1 | 0.021 s | 0.020 s | 0.164 s | 146k | 18k | 8.02x | 7.99x |
| 32 | 8 | 0.161 s | 0.073 s | 0.441 s | 324k | 54k | 6.00x | 2.74x |
| 32 | 64 | 1.264 s | 0.324 s | 1.056 s | 588k | 180k | 3.26x | 0.84x |
| 32 | 512 | - | 2.590 s | 5.060 s | 588k | 301k | 1.95x | - |
| 64 | 1 | 0.051 s | 0.051 s | 0.123 s | 58k | 24k | 2.42x | 2.45x |
| 64 | 8 | 0.396 s | 0.111 s | 0.286 s | 213k | 82k | 2.59x | 0.72x |
| 64 | 64 | 3.233 s | 0.873 s | 1.243 s | 216k | 152k | 1.42x | 0.38x |
| 64 | 512 | - | 6.879 s | 14.496 s | 219k | 104k | 2.11x | - |
| 128 | 1 | 0.207 s | 0.207 s | 0.261 s | 14k | 11k | 1.26x | 1.26x |
| 128 | 8 | 1.897 s | 0.729 s | 0.640 s | 32k | 37k | 0.88x | 0.34x |
| 128 | 64 | 14.905 s | 2.650 s | 2.057 s | 71k | 92k | 0.78x | 0.14x |
| 128 | 512 | - | 18.188 s | 13.194 s | 83k | 114k | 0.73x | - |

**Anatomy of the device run, 4 host cores (docker --cpuset-cpus=0-3)**

| L | W | wall | passes | ms/pass | service thread: wait for walkers / round / reply | in round: stage / upload / launch / device wait | slow batches |
|---|---:|---:|---:|---:|---|---|---:|
| 16 | 1 | 0.099 s | 2911 | 0.034 | 17.3 / 73.3 / 1.2 ms | 1.1 / 18.1 / 34.5 / 17.9 ms | 63 |
| 16 | 8 | 0.304 s | 2961 | 0.103 | 46.6 / 173.5 / 80.9 ms | 6.3 / 45.5 / 100.9 / 16.3 ms | 475 |
| 16 | 64 | 1.079 s | 2917 | 0.370 | 461.8 / 320.6 / 173.4 ms | 16.7 / 59.5 / 111.7 / 125.3 ms | 2210 |
| 16 | 512 | 5.553 s | 2921 | 1.901 | 5339.7 / 708.5 / 937.5 ms | 80.8 / 150.5 / 217.1 / 234.7 ms | 2892 |
| 32 | 1 | 0.164 s | 2720 | 0.060 | 33.8 / 111.3 / 4.5 ms | 2.2 / 31.9 / 60.3 / 13.9 ms | 32 |
| 32 | 8 | 0.441 s | 2778 | 0.159 | 116.5 / 184.0 / 120.7 ms | 7.1 / 49.9 / 101.9 / 20.4 ms | 224 |
| 32 | 64 | 1.056 s | 2729 | 0.387 | 372.1 / 332.8 / 150.8 ms | 15.3 / 50.2 / 90.0 / 171.3 ms | 1358 |
| 32 | 512 | 5.060 s | 2722 | 1.859 | 2747.7 / 831.0 / 799.9 ms | 54.6 / 86.4 / 126.4 / 543.3 ms | 2655 |
| 64 | 1 | 0.123 s | 2382 | 0.052 | 20.5 / 83.6 / 2.6 ms | 1.1 / 15.8 / 29.5 / 35.7 ms | 14 |
| 64 | 8 | 0.286 s | 2490 | 0.115 | 51.5 / 154.3 / 58.7 ms | 3.6 / 25.5 / 52.5 / 70.3 ms | 99 |
| 64 | 64 | 1.243 s | 2408 | 0.516 | 343.8 / 405.3 / 118.7 ms | 12.2 / 36.1 / 67.0 / 285.2 ms | 688 |
| 64 | 512 | 14.496 s | 2384 | 6.080 | 9038.5 / 2130.9 / 2315.8 ms | 88.4 / 235.1 / 281.9 / 1498.5 ms | 2078 |
| 128 | 1 | 0.261 s | 1797 | 0.145 | 113.9 / 112.6 / 12.7 ms | 1.7 / 26.2 / 43.7 / 38.8 ms | 5 |
| 128 | 8 | 0.640 s | 1879 | 0.340 | 209.7 / 238.8 / 120.6 ms | 5.1 / 39.3 / 70.9 / 120.2 ms | 36 |
| 128 | 64 | 2.057 s | 1839 | 1.119 | 499.4 / 982.7 / 60.6 ms | 8.9 / 26.0 / 46.1 / 898.6 ms | 265 |
| 128 | 512 | 13.194 s | 1807 | 7.302 | 2980.5 / 5471.6 / 665.8 ms | 34.0 / 48.7 / 74.6 / 5305.6 ms | 934 |

**Transfer path (pinned async vs pageable), 4 host cores (docker --cpuset-cpus=0-3)**

| L | W | pinned async | pageable | pageable / pinned |
|---|---:|---:|---:|---:|
| 16 | 8 | 0.304 s | 0.268 s | 0.88x |
| 16 | 64 | 1.079 s | 0.913 s | 0.85x |
| 64 | 8 | 0.286 s | 0.359 s | 1.26x |
| 64 | 64 | 1.243 s | 1.617 s | 1.30x |
