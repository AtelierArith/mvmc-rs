# Fixed-value C Green formatting

Generated explicitly by `c_toolbox/physcal_181_format.c`, using the authoritative
`extern/mVMC-1.3.0/src/mVMC/vmcmain.c:666–687` fprintf templates and literal
binary-exact values, not Rust output or computed floating-point expectations.
The legacy indexed files have a terminal blank line; the factored file has
one ordered row and one terminal newline. Spaces are part of this contract.
Normal Rust tests read these checked-in bytes without running C or Julia.

Generated on Linux x86_64 with Ubuntu GCC 13.3.0-6ubuntu2~24.04.1,
`-std=c11 -O0`; no BLAS or RNG is used. Source SHA-256:

- `vmcmain.c`: `fdcd661c4eb028786fc58ae5e1f5232426c943b547f95a0d74e888e602256d63`
- `physcal_lanczos.c`: `d0f537e49c25af59140d8a00f1b2532df243f6cc5657f96d41b1f74d44aced15`
- probe (with full-storage var fixture): `60fca5b77c96d7e33119a39efbd604a9a4af7e9bf5f28ab5fd2f17d7cbb2eee2`

Reproduction from the repository root:

```sh
cc -std=c11 -O0 c_toolbox/physcal_181_format.c -o /tmp/physcal181-format
/tmp/physcal181-format tests/fixtures/physcal_181/formatting
```

`out.dat` additionally uses the `vmcmain.c:647` six-field template with
literal 1,1,2,-1,0.5,0.25 values. It verifies the extra space before Etot2
and the two unsigned-space Sztot fields, not computed variance bitwise parity.

Lanczos terminal formatting authority is `physcal_lanczos.c:78–86,110–141`
and `205–213,232–262`: ls_out has trailing spaces without a newline; indexed
Green files end in a blank line, including a newline for zero entries;
factored Green has one terminal newline even for zero entries. Real-mode
imaginary output is literal `0.0`, not exponential-format zero.

`var-full.dat` uses `vmcmain.c:655–657`'s complete `NPara` loop with
33 stored coefficients: Gutz2, Jast2, DH2 six, DH4 ten, all nine RBM
slots, Slater2 and OptTrans2. Zero-based slots1 and3 are zero; others
hold the literal binary-exact `(i+1)/8 - (i+1)/16*im`. Energy inputs
are -3+0im and 9+0im. The test verifies reserved-slot order, exactly
105 fields, literal zero third fields, spaces and one terminal newline.
It is a fixed-value writer probe, not a full input/sampling reference.
