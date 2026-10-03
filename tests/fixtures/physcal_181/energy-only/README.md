# Zero-Green file lifecycle

Supported Heisenberg PhysCal definitions with all optional OneBodyG, TwoBodyG,
and TwoBodyGEx entries omitted. Numeric fixed parameters are taken from the
independent sibling fixture, not generated from Rust.

`mode-*-files.txt` is produced by actual C `InitFilePhysCal` with zero counts
and Lanczos modes 0/1/2, not by a Rust file inventory. Regenerate explicitly:

```sh
julia +1.13.1 --project=extern/Julia-mVMC c_toolbox/generate_physcal_181_empty_files.jl
```

Authority: `initfile.c:93–106` opens normal Green files only for positive
counts; `108–137` opens LS moment files in mode1/2 and all three LS Green
files unconditionally in mode2 (without `_DEBUG`). The real/complex
`physcal_lanczos.c:112,132,141` / `234,253,262` unconditional post-loop
newlines mean each mode2 zero-entry Green file contains exactly one newline.
This standalone probe validates file creation, not full native C execution
or definition-reader acceptance; the fixture-only Rust test runs the real
PhysCal preparation/sampling/measurement path with supported definitions.
