# Native comparison boundary probe (#492)

This optional developer command copies the authoritative `extern/mVMC-1.3.0`
source, then reuses the existing `sr_operand_dump/dump_sr_operands.patch` and
`physcal_native/state_dump.patch`. `add_probe.py` declares the existing state
writer and calls it immediately before the original SR solve. It changes no
numerical operation, input, sampling control, or RNG draw: the existing RNG
probe saves/restores the SFMT words, cursor and primitive draw count.

```sh
bash c_toolbox/native_comparison_492/build_probe.sh /home/vscode/.cache/mvmc/issue492-native-probe
```

Build inside the Linux x86_64 Dev Container. Options and compiler/linker flags
are those of `sr_operand_dump/build.sh`: CMake Release, GCC/GFortran, MPICH,
OpenBLAS plus upstream static BLIS, GEMMT enabled, blocked updates disabled.
Record source SHA-256 (`src/mVMC/{vmcmain.c,stcopt_dposv.c}` and
`src/sfmt/SFMT.c`), both patch hashes, generated build flags and binary hash
with each capture. Original copyright/license headers remain in copied files.
This is a full executable/sampling/SR observation, not a standalone kernel
oracle. Timing of the printing probe is not a performance measurement.

The Rust `native_c_diagnostics` example captures the actual pre-SR moments,
original unfactored S/g, active component map, solve results, saved chain,
primitive RNG count and next 624 words through existing read-only observers.
Neither Cargo builds nor Rust tests depend on compiling or running this C
probe. Generated numerical fixtures must be captured separately with provenance.
