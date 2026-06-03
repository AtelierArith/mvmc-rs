# Third-party license notices

The workspace as a whole is GPL-3.0-or-later. Two subcomponents retain
non-GPL licenses inherited from upstream:

## `crates/sfmt19937` — BSD-3-Clause

- Upstream: <https://github.com/tmisawa/SFMT.jl>
- Source: vendored in this repo at `extern/Julia-mVMC/SFMT.jl/`.
- Note: SFMT.jl in turn wraps the SFMT C library (BSD-3-Clause, Saito
  & Matsumoto, Hiroshima University). The Rust port translates the
  SFMT19937 algorithm directly; no FFI is required.

## `crates/pfapack` — BSD-3-Clause + MPL-2.0 (per-file)

- Upstream: <https://github.com/tmisawa/PfaPack.jl>
- Per-file license map (matches `extern/Julia-mVMC/PfaPack.jl/THIRD_PARTY_LICENSES.md`):

| Rust file | Upstream Julia file | License | Provenance |
|---|---|---|---|
| `src/pfaffian.rs` | `pfaffian.jl` | BSD-3-Clause | Julia translation of `pfaffian_LTL` from Wimmer PfaPack 2014-09 (`python/pfaffian.py:247-308`). |
| `src/ltl.rs`      | `ltl_decomposition.jl` | BSD-3-Clause | Julia translation of `zsktf2.f` / `dsktf2.f` (Wimmer PfaPack 2014-09). |
| `src/utu2.rs`     | `utu2.jl` | **MPL-2.0**  | Julia translation of `utu2pfa` / `utu2inv` / `sktdsmx` from xrq-phys/Pfaffine (`deps/{pfaffian,invert}.tcc`). Modifications inherit MPL-2.0 per §1.10. |

The Pure-Julia subset listed above is the only part of `PfaPack.jl`
that the Rust port translates. The FFI shims (`c_wrapper.jl`,
`fortran_wrapper.jl`) and their bundled C++ / Fortran sources
(`deps/ltl2inv.cc`, `deps/zsktf2.f`, `deps/dsktf2.f`) are intentionally
**not** ported — the upstream optimizer's hot path already uses the
pure-Julia routines, so the FFI surface is dead code there.

## `crates/mvmc-expert-parsers`, `crates/mvmc-core`, `crates/mvmc-cli`, `xtask` — GPL-3.0-or-later

These translate code from `MVMCOptimizers.jl` and
`MVMCExpertModeParsers.jl`, which themselves derive from C-mVMC
(GPL-3.0-or-later). See the top-level `LICENSE` file (to be added
alongside the first non-skeleton commit) for the full GPL-3.0 text.
