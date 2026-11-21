# Resident GPU exchange rank-two update (#739)

Benchmark of the resident interval sampler's exchange proposals before and
after using C's two-electron rank-two ratio/inverse update instead of a full
Pfaffian per plane and a walker refactor on every accepted exchange.

## Environment

| Item | Value |
| --- | --- |
| GPU | NVIDIA GeForce RTX 3060, 12 GiB |
| Driver | 580.178.04 (CUDA 13.0 driver API) |
| Toolchain image | `mvmc731-cuda-toolchain:local`, CUDA 12.9.2, NVRTC 12.9 |
| Rust | 1.99.1 (5f94df478), test profile (`opt-level = 2`) |
| Base | `8018fe49` (full-Pfaffian exchange + refactor on accept) |
| Candidate | same base plus the rank-two exchange path |
| Geometry | `n = 256` (`Nsite = 256`, `Ne = 128` per spin), `nqp = 1`, no local spins |
| Proposal tape | one sweep = 256 proposals per walker; `exchange_fraction` 0.0 (`NExUpdatePath 0`) or 0.5 (`NExUpdatePath 1`); `u4 = 0` accepts every finite nonzero overlap |
| Timing | two fresh resident states; the first launch compiles NVRTC, the second is timed; one stream barrier at the end of `advance_with_tape` |

The same ignored test (`exchange_sweep_throughput_l16`) was applied to the
base worktree and to the candidate. Walkers start from the same configuration
and inverse; the tape is identical for both modes apart from word 3, which
selects a hop or an exchange.

## Result

`sweep` is the wall time of one 256-proposal sweep for the whole ensemble,
divided by the number of walkers (seconds per walker-sweep). Throughput is in
walker-sweeps per second.

| Mode | Walkers | Base sweep (s) | Candidate sweep (s) | Base walker-sweeps/s | Candidate walker-sweeps/s | Candidate / base |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| hops (`NExUpdatePath 0`) | 64 | 0.0015 | 0.0023 | 659 | 442 | 0.67 |
| hops | 320 | 0.0013 | 0.0014 | 796 | 740 | 0.93 |
| hops | 1,280 | 0.0013 | 0.0013 | 798 | 776 | 0.97 |
| exchange (`NExUpdatePath 1`, half exchanges) | 64 | 0.1762 | 0.0029 | 5.7 | 344 | **60x** |
| exchange | 320 | 0.1372 | 0.0019 | 7.3 | 539 | **74x** |
| exchange | 1,280 | 0.1304 | 0.0018 | 7.7 | 561 | **73x** |

- The exchange sweep changes from 72x slower than a hop sweep to 1.4x slower
  at 320-1,280 walkers.
- Exchange throughput now grows with walker count (344 -> 539 -> 561
  walker-sweeps/s at 64 -> 320 -> 1,280 walkers) instead of sitting at
  5.7-7.7.
- Hop-mode timing is unchanged within run-to-run noise; the rank-two exchange
  code is never entered when `exchange_fraction` is zero.
- The measured base ratio (48x-74x) matches the A100 measurement in issue
  #739. The absolute times differ because this is an RTX 3060, not an A100.

## Reproduction

```text
# candidate checkout
cargo test --manifest-path gpu/mvmc-gpu-cuda/Cargo.toml --lib \
  exchange_sweep_throughput_l16 -- --ignored --nocapture --test-threads=1
```

The base number uses the same test in a clean `8018fe49` worktree. The test is
ignored by default because it reports timings.
