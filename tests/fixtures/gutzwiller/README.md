# Native C Gutzwiller reader expectations

`c_reader_contracts.txt` contains 144 actual C reader cases, with 72 accepted and
72 rejected. Each four-line record stores the case/site count and native header/
reader status, encoded definition (`|` represents a newline), site-index array,
and full real/imaginary flag array. Rejected records have `-` array lines.

Source: authoritative `extern/mVMC-1.3.0/src/mVMC/readdef.c`, SHA-256
`6c53cb832f93d6cbfd7cea955fbb693738af5536b913d36af32b98eed38c32d9`.
The optional driver and verbatim GPL excerpts are stored in `c_toolbox/`.
Reproduction: `python3 scripts/check_gutzwiller_contracts_c_parity.py`;
explicit regeneration: append `--write`. Apple clang 17 compiles with `-O0`;
no MPI, BLAS, SFMT or Julia runtime is used for this reader check.

Sites: 1/2/3/6. Complex headers: 0/1/2. Raw flags: -2/-1/0/1/2/3.
Mapping pairs are counted using Nsite, followed by exactly the declared flag
pairs, independent of physical line breaks. Printed flag indices and header
labels are ignored. Shared indices, trailing unused parameters, reordered and
duplicate sites, all C whitespace characters and incomplete/excess sections
are included. Native status, widths, assignments and component flags are tested
through both the Rust section parser and namelist orchestration.

Storage is explicitly zeroed for the probe. Omitted sites in duplicate-mapping
cases and unwritten real-mode imaginary components retain this sentinel value;
it is not a claim about C malloc initialization. Rust uses deterministic zeros.
Unsafe C site writes, malformed scans, invalid parameter indices and invalid
dimensions are excluded from native execution and tested separately as bounded
Rust errors. Coefficient placeholder values are not native reader output. This
fixture does not prove full C executable or Monte Carlo equivalence.

Rust tests consume this text directly and never read, compile or invoke
`c_toolbox/` or the native C source. Existing Julia numerical fixtures are
unchanged. Remaining Jastrow/RBM/OptTrans contracts and global complex-mode
selection are tracked separately by #21/#26/#27/#44.
