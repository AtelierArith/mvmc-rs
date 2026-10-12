# NVIDIA XORWOW integer seed fixtures

`nvidia_seed_words.csv` contains 32 raw words for each of four unsigned 64-bit
seeds: 0, 1, 11272, and 18446744073709551615. Expected values were generated
independently of Rust by compiling extracted NVIDIA scalar seed/recurrence
statements, with subsequence=0 and offset=0. They are exact integer expectations.

Source: official NVIDIA `nvidia-curand-cu12==10.3.10.19` package, file
`include/curand_kernel.h`, SHA-256
`4a37c07a1d77c9b5c8c627a4720733cee6b4da4200a844e8a49291858bc26adf`.
Generation: Linux x86_64, GCC 13.3.0, `-std=c11 -O2 -Wall -Wextra`, 2026-10-12.
See `c_toolbox/xorwow/README.md` and `probe.py` for extraction boundaries and the
explicit optional reproduction command. Ordinary Rust tests require only this
CSV; no NVIDIA toolkit, C compilation, toolbox program, or GPU is involved.

These expectations cover device XORWOW words for an individual seeded state,
not cuRAND host-library generator interleaving or floating distribution APIs.
