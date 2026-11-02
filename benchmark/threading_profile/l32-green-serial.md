# L32 Green-only serial probe (#581)

The high-gate experiment in [L32 threshold](l32-threshold.md) changed many sites simultaneously. This follow-up changes only the two `ordinary_green_values` dispatch arguments to `false` using the accompanying [candidate patch](l32-green-serial.patch), leaving sample validation, other inner regions, tenferro settings and all production code unchanged. This is an experimental patch, not a proposed default. It is based on `ccb1fca4` (Rust sources equivalent to `f5324073b756b2491182fa605ea851ac25e30ee2` plus documentation); patch SHA-256 `3bb1013870034de36f29f6426df6dc5d5afb40ce5992882766e21c123d7af505`. Baseline binary SHA-256 `bc98b9ddcbafdcdd7f4e7d32b5f06979a07fcf63ba72570e47e69bda027d9a62`; candidate `eb2289c4debeaa2886482ac2dd830cff9fa2ae99da6732a09def9a0e284ae650`. Both were built with the Linux devcontainer release profile, MPI feature and locked dependencies, in separate target volumes.

Same L32 inputs, fixed PhysCal parameters, CPU placement, host, and environment as the threshold report; one process with 4 inner and 4 tenferro threads, single-thread OpenBLAS. Opt is an additional workload control; the changed routine is a PhysCal Green measurement path. Each mode/version had one warmup followed by four measured runs, alternating version order. The timer inside the container includes the CLI, initialization, computation and output, not Docker startup. [Raw logs and input hashes](results/l32-green-serial/) include wall, user and system CPU time.

| Workload | Baseline wall samples (seconds) | Candidate wall samples (seconds) | Median baseline / candidate (seconds) |
|---|---|---|---|
| Opt | 2.022, 2.248, 2.100, 2.045 | 2.165, 2.177, 2.119, 2.096 | 2.0725 / 2.1420 |
| PhysCal | 2.409, 2.411, 2.511, 3.026 | 2.236, 2.129, 2.260, 2.241 | 2.4610 / 2.2385 |

Within each workload, non-timing output files across eight measured runs were byte-identical under `diff -qr --exclude='*time*'`. This does not establish complete RNG state or acceptance-decision parity. These short shared-host samples have visible variation (including in the control); they identify a candidate for longer profiling, not proof of general improvement or grounds for a global default. Workloads, lattice sizes, threads and MPI/walker configurations remain to be tested. The candidate also does not solve ownership of tenferro's held team.

## Reproduce

Prepare the same inputs and a baseline release target volume as in [the threshold report](l32-threshold.md). Use a second clean worktree at the stated source revision and separate target volume; apply `git apply /path/to/report-checkout/benchmark/threading_profile/l32-green-serial.patch`, and build `cargo build -p mvmc-cli --features mpi --release --locked` in the devcontainer. Then run from a worktree containing this report, using fresh output directories:

```sh
benchmark/threading_profile/l32-green-serial.sh \
  DEVELOPMENT_IMAGE BEFORE_TARGET_VOLUME AFTER_TARGET_VOLUME \
  /tmp/placement-input /tmp/l32-green-results
```

The script labels the warmups `0` and measured runs `1`–`4`. Check `input-sha256.txt`, CPU topology, binary hashes and non-timing output files independently.

## Additional L64 PhysCal case

A separate sequence used the saved `r1-t1/L64/physcal-inputs.tar.gz` from the 2026-10-11 MPI comparison, including its fixed `zqp_opt.dat`, with `NDataQtySmp` changed from 100 to 20 and all other inputs left as recorded. The same binaries, four-CPU placement, pool/BLAS settings and timing boundary were used. One warmup preceded three measured runs per version, alternating order:

| L64 PhysCal | Measured wall samples (seconds) | Median (seconds) |
|---|---|---:|
| Baseline | 13.952, 14.063, 13.399 | 13.952 |
| Green-serial candidate | 9.749, 10.136, 10.034 | 10.034 |

The six measured non-timing output directories were byte-identical under `diff -qr --exclude='*time*'`; this is not a complete RNG trace. [Raw logs and exact input hashes](results/l64-green-serial/) include user/system CPU times. This is one L64 input on the same shared host and does not establish the effect for other Hamiltonians, thread counts or MPI ranks.

Reproduce with the same separately built binary volumes described above. From the repository root:

```sh
mkdir -p /tmp/green-l64-input
tar xzf benchmark/mpi_comparison/results/rank-thread-sweep-main-20261011/r1-t1/L64/physcal-inputs.tar.gz \
  --strip-components=1 -C /tmp/green-l64-input
sed -i -E 's/^NDataQtySmp[[:space:]].*/NDataQtySmp 20/' /tmp/green-l64-input/modpara.def
benchmark/threading_profile/l64-green-serial.sh \
  DEVELOPMENT_IMAGE BEFORE_TARGET_VOLUME AFTER_TARGET_VOLUME \
  /tmp/green-l64-input /tmp/green-l64-results
```
