# C initial/fixed parameter records

`c_records.txt` records 35 standalone checks of the actual mVMC-1.3.0
`ReadInitParameter` function from `src/mVMC/parameter.c`, compiled with native
SFMT on Intel macOS using Apple clang 17, `-O0 -ffp-contract=off -DMEXP=19937`.
The fixture header records source SHA-256 hashes. The driver and verbatim
extracted function live in `c_toolbox/initial_records.c` and
`c_toolbox/initial_records_upstream.inc`; the authoritative C source is unchanged.

Each case has four lines: name and `NProj NRBM NSlater NOptTrans`, complete
input text, raw complex coefficient bit pairs in declared parameter order,
and the next 624 SFMT words after seed 1. Empty input and coefficient lines
are significant. The driver initializes each slot to `99+99i` so an empty
file's lack of writes can be checked explicitly.

Cases cover one/two/three complete records, final-record precedence, all
projection families and nine RBM sections, declared Slater slots without
mappings, OptTrans, empty and parameterless inputs, signed zeros, decimal/hex
conversion, subnormals, overflow/underflow, NaN and infinity. C accepts numeric
nonfinite diagnostics/gradients even though they are unused. Coefficients use
the C expression `tmp_real + tmp_comp*I`; its signed zeros and `0*infinity`
NaN affect the raw stored bits. Loading consumes no RNG draws and performs no
normalization. A nonfinite earlier record may be overwritten by a finite one.

The final 24 cases reuse historical OptTrans input payloads with explicitly
supplied C kernel dimensions. They supersede Julia-only numeric rejection
expectations in 48 optional/fixed loader checks. They do not establish C
acceptance of the historical mapping/flag files or their header widths.
The all-factor model uses every RBM index; sparse declared RBM storage remains
under #26, and C OptTrans activation and flag offsets remain under #27.

The Rust integration test consumes this fixture directly, compares every
coefficient bit through both loaders, preserves spatial mappings, and checks
all 624 RNG words. It neither compiles, invokes nor reads `c_toolbox/` or the
C reference. Malformed nonnumeric/incomplete records retain bounded, atomic
Rust errors; C's unchecked scans do not provide a useful equivalent failure
contract. No full C executable, MPI, keyword-overlay precedence or Monte Carlo
trajectory is exercised by this oracle. #28 remains open for those applicable
loader contracts and complete declared storage of other factors.

```sh
# Optional native C verification/regeneration:
python3 scripts/check_initial_records_c_parity.py
python3 scripts/check_initial_records_c_parity.py --write

# Normal Rust checks require only the checked-in fixture:
cargo test -p mvmc-core --locked --test c_initial_records --test initial_params --test opttrans
```
