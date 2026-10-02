# C projection count contract

C comparison drivers and verbatim extracted source are stored in
`c_toolbox/`; generator commands below are optional developer checks. Rust
tests consume the checked-in fixtures and never compile, invoke or read the
toolbox. See [the toolbox documentation](../../../c_toolbox/README.md).

The authoritative reference is the local `extern/mVMC-1.3.0` C snapshot.
The fixture header records SHA-256 hashes of `readdef.c` and `qp.c`; this
directory is an extracted source tree, not a standalone Git checkout.

`readdef.c` reads NMPTrans unchanged, saves a negative sign in APFlag,
then uses the absolute count in NQPFix and NQPFull. Zero stays zero.
`qp.c` initializes spin quadrature independently but iterates over zero
translation sectors for a zero count. The English and Japanese Expert
manuals require 1 when translation projection is not applied. C's internal
default is zero despite the manual's documented default of one; execution
therefore requires a nonzero count rather than silently interpreting zero
as an identity sector.

Rust preserves this count behavior in QP initialization, state sizing and
Slater construction. Runtime entry points reject zero before parameter
initialization, sampling, RNG draws or output. Rust retains the signed
count as its boundary marker and uses its absolute value for dimensions;
its existing periodic/antiperiodic sign caches preserve the C APFlag
behavior.

`c_contracts.txt` contains ten cases: raw counts 0/1/-1/2/-2 with ordinary
and two-component OptTrans weights, NSPGaussLeg=1. It records converted
counts, APFlag, fixed/full widths, and exact real/imaginary bits of fixed
and full weights and spin cosine/sine. The generator compiles the actual
C count-conversion statements and unmodified InitQPWeight/UpdateQPWeight
kernels, with workspace allocation plumbing. GaussLeg and LegendrePoly
are linked directly from C; no numerical algorithm is rewritten in the
probe. It observes low-level zero behavior without executing an invalid
zero-projection Monte Carlo run.

Run `python3 scripts/check_projection_count_c_parity.py`; add `--write`
to regenerate or `--source PATH` to use a separate mVMC-1.3.0 source tree.
The C tree is an external prerequisite and is not added to this repository.
A C compiler and Python 3 are required. Rust's parser unit test verifies
all ten records, while core and CLI tests verify early rejection and a
full unchanged 624-word SFMT block. Existing signed multi-sector Slater,
derivative and QP-weight fixtures remain regression gates.

The earlier approved two-line Julia zero-to-one correction was withdrawn
after the user clarified that C is authoritative (Julia-mVMC PR #3).
Corrected-Julia experiment artifacts are kept outside this fixture tree
and are not used as C parity evidence. The Rust Julia submodule pin remains
unchanged.
