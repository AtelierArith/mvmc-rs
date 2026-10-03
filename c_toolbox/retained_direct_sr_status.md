# Retained direct-SR LAPACK status diagnostic

`retained_direct_sr_status.c` is an original standalone adapter, not extracted
mVMC source and not a Cargo dependency. It reads a retained finite column-major
matrix/RHS and calls original LAPACK DPOSV with upper triangle and one RHS.
It does not reconstruct the system, alter regularization or replace expectations.
Exit zero means diagnostic completion; positive INFO means factorization failed.

C authority: vendored mVMC 1.3.0,
`extern/mVMC-1.3.0/src/mVMC/stcopt_dposv.c` lines 44–45,
SHA-256 `2bd48d880dcbd95ea1b1b92931178c07fd08f981b8ebf04e7db1709e909e57ca`.
Original source copyright University of Tokyo; GPL-3.0-or-later with the upstream
BSD-derived history recorded in its header. No upstream source was copied or
modified: extraction boundaries are **none**; only the DPOSV calling contract
is followed. `readdef.c`/sampler/accumulator algorithms are not run by this probe.

Actual execution 2026-10-03 on Linux x86_64: GCC Ubuntu
13.3.0-6ubuntu2~24.04.1, `-std=c11 -Wall -Wextra -Werror -O0
-ffp-contract=off -llapack -lblas -lm`. System LAPACK/BLAS package versions
`3.12.0-3build1.1`; library SHA-256 values:

- `/lib/x86_64-linux-gnu/liblapack.so.3`: `1e82245607c58d13405580c71b9b0134b5a952686ac861f979fc7a886ebd4229`
- `/lib/x86_64-linux-gnu/libblas.so.3`: `8ba4a98f44d763c2e648755ef452025919e28633339f23ff605044de3483a25d`
- Adapter: `e60376b86872c67db282a46c4f67bf2bfe380c061f6b8c1038cef53a6c5aaf0a`
- Executable: `626a1bd68106c7892ca3f9092f49c3220396532dc4eeb550a76331aa4394f6ee`

Reproduction (optional developer command, never invoked by Rust tests):

```sh
gcc -std=c11 -Wall -Wextra -Werror -O0 -ffp-contract=off c_toolbox/retained_direct_sr_status.c -llapack -lblas -lm -o /tmp/mvmc-retained-direct-sr-status-2623231
/tmp/mvmc-retained-direct-sr-status-2623231 5 /tmp/mvmc-issue182-2623231-0/synthetic-hubbard_chain_real-qp32-samples32-store0-cg0/direct-sr-observed-10.matrix.txt /tmp/mvmc-issue182-2623231-0/synthetic-hubbard_chain_real-qp32-samples32-store0-cg0/direct-sr-observed-10.rhs.txt
```

Inputs are Banach's actual original before-factorization observation from failed
step 10 (zero-based), dimension 5, U triangle, Rust factor INFO=2, no substitution.
The original test run `39518` exited 100; it is not a 45-case pass. Matrix SHA
`bf89c6d1286e714b0fa3825bf000fc0e7b9f4a70bec8fc5d285f69b22d21a36b`,
RHS SHA `204814be0e9cf40e2c17961044386ed252c04467e3b0dbaf1beeac558ca9a90a`,
metadata SHA `6b3316dca7ea5641cd167ac54c0d829aa596398f5a7cff7ada6c9be8020635b2`.

Standalone native execution completed terminal zero and returned INFO=2, agreeing
with Rust on this **retained system**. Actual stdout was persisted at
`/tmp/mvmc-retained-direct-sr-status-2623231.stdout.txt`, SHA-256
`a9eb8b1f62e06188b1fbb358228629735bb939cac914d552bc1d295ef56e97ac`:

```text
leading_minor_2x2=-3.105441664702893643196e-34
dimension=5 UPLO=U NRHS=1 LAPACK_INFO=2
```

The negative leading principal minor demonstrates non-SPD input. The printed
minor uses extended precision, not a certified interval bound. This is neither
full-C sampling parity nor proof of how the upstream accumulator produced this
matrix. It does not justify a tolerance change, discarding the failed workload,
or asserting a solution/residual after failed factorization. Future threading
checks must separately verify repeatable failure state/configuration/RNG and
no parameter mutation on failure.
