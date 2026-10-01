# Third-party license notices

The workspace as a whole is GPL-3.0-or-later. Subcomponents retain
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
**not** ported. The ordinary optimizer uses pure Julia kernels. The
upstream FSZ inverse calls a native wrapper; Rust reproduces its arithmetic
in the inverse pipeline and selects its BLAS/LAPACK provider without
calling or porting that wrapper.

## `crates/mvmc-expert-parsers`, `crates/mvmc-core`, `crates/mvmc-cli`, `xtask` — GPL-3.0-or-later

These translate code from `MVMCOptimizers.jl` and
`MVMCExpertModeParsers.jl`, which themselves derive from C-mVMC
(GPL-3.0-or-later). See the top-level `LICENSE` file (to be added
alongside the first non-skeleton commit) for the full GPL-3.0 text.

## Julia Base projection math — MIT + Sun Microsystems notice

`crates/mvmc-expert-parsers/src/utils/julia_trig.rs` ports the Float64
sine/cosine kernels and the small-angle Cody-Waite reduction from Julia
1.13.1 `base/special/trig.jl` and `base/special/rem_pio2.jl`. The supported
range is the projection angles [-pi, pi]; it does not port large-angle
Payne-Hanek reduction. Julia portions retain MIT; the kernel/reduction
portions retain the Sun Microsystems permission notice from FDLIBM.
The full notices are in `crates/mvmc-expert-parsers/LICENSE-julia-math`.

- Source: <https://github.com/JuliaLang/julia/tree/v1.13.1/base/special>
- Julia license: <https://github.com/JuliaLang/julia/blob/v1.13.1/LICENSE.md>

`crates/pfapack/src/julia_complex.rs` ports the robust `ComplexF64 /
ComplexF64` division and reciprocal from Julia 1.13.1 `base/complex.jl` under MIT. Its MIT
notice is also included in `crates/mvmc-expert-parsers/LICENSE-julia-math`.

`crates/mvmc-expert-parsers/src/utils/julia_hypot.rs` ports Julia 1.13.1
`base/math.jl` Float64 `hypot`, used for parameter normalization, under the
same Julia MIT notice.

`crates/mvmc-expert-parsers/src/utils/julia_exp.rs` ports Julia 1.13.1
`base/special/exp.jl` Float64 `exp` and its 256-entry reconstruction table
under the same Julia MIT notice. It preserves projection ratios and
Metropolis probabilities without platform-libm rounding differences.
