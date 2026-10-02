# Native C directional Jastrow reader expectations

`c_reader_contracts.txt` holds 144 actual C cases: 69 accepted / 75 rejected.
Four-line records contain name/site count and native header/reader status,
the encoded definition (`|` replaces newline), flattened directional index
matrix and full real/imaginary flags. Rejected records have `-` array lines.

Source: authoritative `extern/mVMC-1.3.0/src/mVMC/readdef.c`, SHA-256
`6c53cb832f93d6cbfd7cea955fbb693738af5536b913d36af32b98eed38c32d9`.
The standalone driver and verbatim GPL excerpts live in `c_toolbox/`.
Verify with `python3 scripts/check_jastrow_contracts_c_parity.py`;
append `--write` to explicitly regenerate. Apple clang 17 uses `-O0`.
No BLAS, MPI, SFMT or Julia runtime is required for this reader check.

The actual `ReadBuffIntCmpFlg`, `GetInfoJastrow`, `GetInfoOpt` and site-check
bodies cover 2/3/6 sites, complex headers 0/1/2, raw flags -2/-1/0/1/2/3,
asymmetric directed indices, shared/unused slots, reordered and duplicate pairs,
ignored/duplicate/negative printed flag labels, folded/split C whitespace,
nonstandard header labels, diagonal/out-of-range sites, zero headers and
missing/excess mapping or flag fields. Jastrow is supplied at parameter offset
2 after the controlled real Gutzwiller prefix `[3,0,-2,0]`.

Untouched index cells are explicitly initialized to -1; imaginary flag cells
are initialized to zero. They are probe sentinels, not evidence of native malloc
contents. The driver does not validate unsafe parameter-index scans or malformed
integer input; bounded Rust diagnostics must be verified separately. This is a
reader-level oracle, not full C executable/MPI/sampling parity.

Rust section and namelist tests compare all accepted directional matrices,
declared coefficient storage and signed real/imaginary flags at the supplied
offset, and reject every native failure. Unsafe parameter dimensions, invalid
indices and malformed scans have separate bounded Rust error tests. The old
symmetric-matrix helper and parser signature are removed.

`c_three.def` copies the identical DH2/DH4/RBM three-site definitions, adding
only the reverse of each upper-triangle mapping with its original parameter
index. Width, complex header, coefficients and flags remain the same. The actual
C reader accepts the complete replacement and rejects all three original files;
five complete upstream six-site inputs are also accepted. The historical test
constructor substitutes the replacement without editing original files or
numerical goldens. Temporary fixed-SR tests likewise supply all six directed
triples. This preserves their coefficient/RNG checks, without presenting the
incomplete historical files as supported C inputs.

`c_projection_counts.txt` stores 1,795 actual `MakeProjCnt`/`UpdateProjCnt`
checks across 60 complete accepted tables. Three-line records contain the case,
site count, declared width and workload count; encoded definition; and `|`-joined
integer workload rows. Each workload row gives pattern/ri/rj/spin, the initial
2*Nsite occupancy, NProj initial counters and NProj updated counters. Spin -1 is
a no-hop workload. Other rows enumerate every legal single-spin directed hop
for eight empty/full/alternating occupancy patterns. Gutzwiller has two slots
with site-to-index i%2; DH2/DH4 are disabled. Rust compares initial counters,
incremental updates and complete post-hop rebuilds exactly. This proves C's
upper-triangle lookup with directional tables, including reverse hops, shared
indices and unused coefficient slots; it does not exercise full sampling/RNG.

Projection source: `src/mVMC/projection.c`, SHA-256
`2c01cc16ec75f6d5f3c9e7927bd68f96d1db329c124702b06b771ccd314522c1`.
Actual complete function bodies are extracted into
`c_toolbox/jastrow_projection_upstream.inc`. Kernels are excluded for duplicate
reader mappings that leave off-diagonal cells untouched. Such reader results
remain sentinel checks and never serve as native sampling evidence. All C
excerpts and both fixture files are verified by the optional command above.

Rust tests only consume checked-in text; they never compile, invoke or read the
C toolbox. Existing Julia numerical fixtures, tolerances and seeds are unchanged.
RBM complete widths/storage, OptTrans activation/layout, global signed complex
headers and full executable/MPI/sampling parity remain #26/#27/#44/#56 work.
