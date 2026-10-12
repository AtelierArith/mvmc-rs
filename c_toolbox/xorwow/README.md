# Optional XORWOW header probe

This standalone scalar-word probe extracts seed initialization and the word recurrence
from NVIDIA's `curand_kernel.h` in the official PyPI distribution
`nvidia-curand-cu12==10.3.10.19`. It is not a full cuRAND library / sampling validation.
The input header SHA-256 is
`4a37c07a1d77c9b5c8c627a4720733cee6b4da4200a844e8a49291858bc26adf`.

Extraction boundaries: `_curand_init_inplace` statements from `unsigned int s0 =`
through the line before `_skipahead_sequence_inplace`, and the complete body of
`curand(curandStateXORWOW_t *state)`. Skipahead calls are omitted because both
subsequence and offset are zero. Normal-distribution cached fields are not used.
Unsigned 32-bit arithmetic is preserved. Source stays in the externally installed
NVIDIA package; the script creates a temporary standalone C harness.

Reproduce explicitly (ordinary Cargo builds/tests never invoke this):

```sh
uv run --no-project --with nvidia-curand-cu12==10.3.10.19 python -c 'import nvidia.curand; print(nvidia.curand.__path__[0])'
uv run --no-project python c_toolbox/xorwow/probe.py /path/to/nvidia/curand/include/curand_kernel.h tests/fixtures/xorwow/nvidia_seed_words.csv
```

Compiler: `cc -std=c11 -O2 -Wall -Wextra`; generated on Linux x86_64 with
GCC 13.3.0, Rust 1.99.0. The CSV covers 32 words per seed (0, 1, 11272, 2^64−1).
These integer RNG expectations are exact and platform independent. They do not
assert NVIDIA host generator ordering, which interleaves its own internal states.
