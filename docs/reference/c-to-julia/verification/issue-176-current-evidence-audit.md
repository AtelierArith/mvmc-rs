# Issue #176 current evidence audit

Parent reported current `real_fsz_setup` binary: two tests passed, terminal exit
0, run `82c9decb-6861-448b-a167-96302f9d7b0b`. This is setup/initialization evidence, not full real-FSZ
sampling, SR or PhysCal verification. Existing numerical bounds are #190-owned
and unchanged by this audit.
The full UUID was subsequently supplied from the parent's original terminal
evidence; this is not a new fixture generation or rerun. Parent separately reported
16 memory-only shadow-copy probe cases; those establish alias/copy layout, not
numerical derivatives or sampling trajectory. No fresh real-FSZ full oracle
run is recorded yet; #176 remains open pending Banach's independent oracle and
its actual sampling/discrete, SR and PhysCal evidence.

The historical first computed difference is independently archived in
[real-FSZ provenance](../../../../tests/fixtures/real_fsz/README.md):
`ns=2, ne=1, polarized=1, complex_input=0`, inverse QP1 `(row=1,col=0)`,
macOS `bffce739ce739ce6`, Linux `bffce739ce739ce7`. The operation is complex
division during the tridiagonal inverse solve (norm-squared versus GNU Smith
scaling), not an RNG or control-flow difference. Current `pfaffian.rs` passes
`fsz_inverse_divide` to `utu2inv_complex_fsz`; Linux GNU selects `c_complex::divide`.
Current Linux fixture SHA256 is
`5d1c0bb40fd89796252c3b91e11b641d827cf66d25755a4249a3c8a5eefa47d1`;
current setup test SHA256 is
`7f30963fcd085f662f20e20c0f869d79968c4b4bd62f40104f682add3c96aa5e`.
[Linux provenance](../../../../tests/fixtures/linux_gnu_julia/provenance.json)
records Julia 1.13.1, actual checkout `8bb1b9e8`, manifest
`09ebd06dab244510094b99fe7c6efa2fe7a3d22221d1336b951123a5a6e8befc`,
Julia OpenBLAS 0.3.30 ILP64, Rust OpenBLAS 0.3.26 LP64, Ubuntu x86_64/GCC13.3.
The generator's older embedded `c2ea` label is distinguished there from the
actual checkout. No fresh Julia run is claimed by this source/hash audit.

## Defined C real-FSZ shadow lifecycle

Independent inspection resolved the apparent missing Pfaffian copy; this is
**not a C bug**. `setmemory.c:359-360` allocates complex `InvM` for
`NQPFull*(Nsize*Nsize+1)` elements and aliases `PfM` at offset
`NQPFull*Nsize*Nsize`. Lines 379-380 allocate/alias the real buffers identically.
Thus the trailing `NQPFull` elements of each allocation are Pfaffians.
`vmccal_fsz.c:82` copies precisely that full allocation element count from real
to complex after `CalculateMAll_fsz_real`; the copy includes both inverse planes
and Pfaffians, with its last index equal to capacity minus one. `slater_fsz.c:200`
then reads the now-defined `PfM[qpidx]`. The real sampler's real-only writes do
not undermine this per-sample calculation/copy before the derivative.

Upstream SHA256:

- `setmemory.c`: `573d1fb995de33386e7452f5e2fa34aeb05efe14b5bf4a904eb54d2ac7cf1b9e`
- `vmccal_fsz.c`: `9ac529ef2a18d80f61c5aca92aacce10391e2794ecc56aa8ca947691dbdd974e`
- `slater_fsz.c`: `2bb20af97f47c30cf2c81a856ee7235df5f45316546c2ae98b43785d99b90eec`
- `matrix.c`: `849488176375102fbc3bab7001e0dc27dfa7c8e049fd90ee1d6248ebfba10764`

Julia has separate vectors, so a faithful layout adaptation must copy BOTH
real inverse and real Pfaffian vectors into complex shadows before calling the
original derivative. Copying only the inverse is not the C contract. No vendored
source repair or undefined numeric probe was performed. Native real-energy ABI
36-case checks alone do not prove SR derivatives/OO/HO. Banach owns the new
independent real-FSZ oracle; remaining acceptance requires matching-input real
sampling/RNG, normalized OO/HO, original SR and PhysCal evidence, without
stripping imaginary parts from complex fixtures or widening tolerances.
