# CPU tenferro v0.8 migration: SR stage comparison

Exploratory measurements for #569, not an end-to-end Opt or PhysCal benchmark. `sr_backend_bench` generates deterministic input and reports medians of seven calls after a warmup, including tensor conversion and transfers. Outputs show C-order and tenferro milliseconds and maximum relative differences against C-order. The process-level CPU and wall times below also include startup, initialization, the C-order measurements and every configuration; they are too short for precise utilization analysis.

Linux x86_64, Xeon E5-2699 v3 (36 logical CPUs visible), Rust 1.99.0 (b940084d7), system `/lib/x86_64-linux-gnu/libopenblas.so.0`; `OPENBLAS_NUM_THREADS=1`, `RAYON_NUM_THREADS=1` or `4`. Before: main `5f5a490e3f4a4b77149b147f684ed2bbae252d77`, tenferro 0.7.1 with repository security patches. After: `e4240834b0b1ef0271e197931da0b77221662fb6`, tenferro-ad/tensor/einsum/linalg/runtime 0.8.0 and tenferro-cpu 0.8.1. Binaries were run sequentially with no other intentional GPU benchmark or test; no affinity or frequency control was applied. Each condition was executed once, and C-order timing changed noticeably across runs, so these numbers do not establish a statistically robust speedup. Worker participation, multi-walker and MPI behavior remain unmeasured.

Build in each checkout, then execute sequentially:

```sh
cargo build -p mvmc-core --profile test-fast --example sr_backend_bench
RAYON_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /usr/bin/time -v target/test-fast/examples/sr_backend_bench 64 128
RAYON_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 /usr/bin/time -v target/test-fast/examples/sr_backend_bench 64 128
```

## Raw output

### before, 1 thread(s)

```text
ms, median of 7, C-order / tenferro (cpu-faer)
| NPara | samples | Gram | S,g assembly | Cholesky solve | CG matvec | rel diff gram,asm,solve,cg |
|---:|---:|---|---|---|---|---|
| 64 | 300 | 0.12 / 0.35 | 0.02 / 0.40 | 0.03 / 0.32 | 0.010 / 0.188 | 7.8e-16 0.0e0 6.3e-16 1.0e-15 |
| 64 | 128 | 0.08 / 0.21 | 0.03 / 0.51 | 0.04 / 0.37 | 0.006 / 0.137 | 0.0e0 0.0e0 6.1e-16 8.5e-16 |
| 128 | 300 | 0.63 / 2.33 | 0.11 / 1.54 | 0.19 / 0.85 | 0.046 / 0.298 | 1.1e-15 0.0e0 1.0e-15 1.3e-15 |
| 128 | 256 | 0.61 / 2.14 | 0.13 / 1.70 | 0.20 / 0.89 | 0.038 / 0.287 | 0.0e0 0.0e0 9.5e-16 1.5e-15 |
	Command being timed: "/home/terasaki/work/atelierarith/mvmc-rs/target/test-fast/examples/sr_backend_bench 64 128"
	User time (seconds): 0.12
	System time (seconds): 0.05
	Percent of CPU this job got: 106%
	Elapsed (wall clock) time (h:mm:ss or m:ss): 0:00.17
	Average shared text size (kbytes): 0
	Average unshared data size (kbytes): 0
	Average stack size (kbytes): 0
	Average total size (kbytes): 0
	Maximum resident set size (kbytes): 21332
	Average resident set size (kbytes): 0
	Major (requiring I/O) page faults: 0
	Minor (reclaiming a frame) page faults: 7312
	Voluntary context switches: 591
	Involuntary context switches: 4
	Swaps: 0
	File system inputs: 0
	File system outputs: 8
	Socket messages sent: 0
	Socket messages received: 0
	Signals delivered: 0
	Page size (bytes): 4096
	Exit status: 0
```

### after, 1 thread(s)

```text
ms, median of 7, C-order / tenferro (cpu-faer)
| NPara | samples | Gram | S,g assembly | Cholesky solve | CG matvec | rel diff gram,asm,solve,cg |
|---:|---:|---|---|---|---|---|
| 64 | 300 | 0.12 / 0.45 | 0.02 / 0.27 | 0.03 / 0.21 | 0.011 / 0.124 | 7.8e-16 0.0e0 6.3e-16 1.0e-15 |
| 64 | 128 | 0.05 / 0.16 | 0.02 / 0.26 | 0.03 / 0.20 | 0.004 / 0.085 | 0.0e0 0.0e0 6.1e-16 8.5e-16 |
| 128 | 300 | 0.39 / 1.21 | 0.07 / 0.49 | 0.12 / 0.41 | 0.028 / 0.160 | 1.1e-15 0.0e0 1.0e-15 1.3e-15 |
| 128 | 256 | 0.35 / 0.93 | 0.08 / 0.47 | 0.12 / 0.40 | 0.020 / 0.140 | 0.0e0 0.0e0 9.5e-16 1.5e-15 |
	Command being timed: "/tmp/mvmc-570/target/test-fast/examples/sr_backend_bench 64 128"
	User time (seconds): 0.07
	System time (seconds): 0.02
	Percent of CPU this job got: 100%
	Elapsed (wall clock) time (h:mm:ss or m:ss): 0:00.09
	Average shared text size (kbytes): 0
	Average unshared data size (kbytes): 0
	Average stack size (kbytes): 0
	Average total size (kbytes): 0
	Maximum resident set size (kbytes): 24136
	Average resident set size (kbytes): 0
	Major (requiring I/O) page faults: 0
	Minor (reclaiming a frame) page faults: 3157
	Voluntary context switches: 1
	Involuntary context switches: 5
	Swaps: 0
	File system inputs: 0
	File system outputs: 8
	Socket messages sent: 0
	Socket messages received: 0
	Signals delivered: 0
	Page size (bytes): 4096
	Exit status: 0
```

### before, 4 thread(s)

```text
ms, median of 7, C-order / tenferro (cpu-faer)
| NPara | samples | Gram | S,g assembly | Cholesky solve | CG matvec | rel diff gram,asm,solve,cg |
|---:|---:|---|---|---|---|---|
| 64 | 300 | 0.11 / 0.42 | 0.02 / 0.45 | 0.03 / 0.30 | 0.010 / 0.157 | 7.8e-16 0.0e0 6.3e-16 1.0e-15 |
| 64 | 128 | 0.05 / 0.15 | 0.02 / 0.45 | 0.03 / 0.30 | 0.004 / 0.116 | 0.0e0 0.0e0 6.1e-16 8.5e-16 |
| 128 | 300 | 0.39 / 0.96 | 0.07 / 1.07 | 0.12 / 0.70 | 0.026 / 0.212 | 1.1e-15 0.0e0 1.0e-15 1.3e-15 |
| 128 | 256 | 0.33 / 0.86 | 0.07 / 1.12 | 0.11 / 0.68 | 0.016 / 0.212 | 0.0e0 0.0e0 9.5e-16 1.5e-15 |
	Command being timed: "/home/terasaki/work/atelierarith/mvmc-rs/target/test-fast/examples/sr_backend_bench 64 128"
	User time (seconds): 0.11
	System time (seconds): 0.08
	Percent of CPU this job got: 166%
	Elapsed (wall clock) time (h:mm:ss or m:ss): 0:00.11
	Average shared text size (kbytes): 0
	Average unshared data size (kbytes): 0
	Average stack size (kbytes): 0
	Average total size (kbytes): 0
	Maximum resident set size (kbytes): 22944
	Average resident set size (kbytes): 0
	Major (requiring I/O) page faults: 0
	Minor (reclaiming a frame) page faults: 8924
	Voluntary context switches: 1504
	Involuntary context switches: 6
	Swaps: 0
	File system inputs: 0
	File system outputs: 8
	Socket messages sent: 0
	Socket messages received: 0
	Signals delivered: 0
	Page size (bytes): 4096
	Exit status: 0
```

### after, 4 thread(s)

```text
ms, median of 7, C-order / tenferro (cpu-faer)
| NPara | samples | Gram | S,g assembly | Cholesky solve | CG matvec | rel diff gram,asm,solve,cg |
|---:|---:|---|---|---|---|---|
| 64 | 300 | 0.10 / 0.32 | 0.02 / 0.26 | 0.02 / 0.17 | 0.007 / 0.113 | 7.8e-16 0.0e0 6.3e-16 1.0e-15 |
| 64 | 128 | 0.04 / 0.12 | 0.01 / 0.24 | 0.02 / 0.18 | 0.004 / 0.073 | 0.0e0 0.0e0 6.1e-16 8.5e-16 |
| 128 | 300 | 0.31 / 0.43 | 0.06 / 0.39 | 0.09 / 0.31 | 0.019 / 0.134 | 1.1e-15 0.0e0 1.0e-15 1.3e-15 |
| 128 | 256 | 0.28 / 0.38 | 0.06 / 0.43 | 0.09 / 0.30 | 0.014 / 0.115 | 0.0e0 0.0e0 9.5e-16 1.5e-15 |
	Command being timed: "/tmp/mvmc-570/target/test-fast/examples/sr_backend_bench 64 128"
	User time (seconds): 0.15
	System time (seconds): 0.02
	Percent of CPU this job got: 250%
	Elapsed (wall clock) time (h:mm:ss or m:ss): 0:00.07
	Average shared text size (kbytes): 0
	Average unshared data size (kbytes): 0
	Average stack size (kbytes): 0
	Average total size (kbytes): 0
	Maximum resident set size (kbytes): 24108
	Average resident set size (kbytes): 0
	Major (requiring I/O) page faults: 0
	Minor (reclaiming a frame) page faults: 3249
	Voluntary context switches: 48
	Involuntary context switches: 4
	Swaps: 0
	File system inputs: 0
	File system outputs: 8
	Socket messages sent: 0
	Socket messages received: 0
	Signals delivered: 0
	Page size (bytes): 4096
	Exit status: 0
```
