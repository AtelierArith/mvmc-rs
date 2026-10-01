# Pure General orbital parity

`general.def` uses Julia's combined site/spin coordinates directly: up sites
are 0–2 and down sites are 3–5. Its parameter indices and signs are equivalent
to `ap.def` plus `parallel.def`. `sparse.def` additionally covers reversed
down-down entries, cross-spin entries, an explicit diagonal, duplicate parameter
indices, unmapped cells, and an unused declared parameter slot.

`matrices.txt` records the authoritative parser index/sign matrices, Slater
tables, and Slater derivatives for both boundary conditions. Every case has two
QP planes, with a nontrivial site translation and antiperiodic translation
signs. The fixed inverse inputs are serialized in Julia's compact per-QP layout;
Rust copies those same matrices into its padded planes before computing the
derivatives. The `cached` case changes the supplied lookup matrix independently
of the original terms, proving that updates and derivatives use the cache.

Generate or verify with Julia 1.13.1:

```sh
julia +1.13.1 --project=extern/Julia-mVMC scripts/check_orbital_general_parity.jl --write
julia +1.13.1 --project=extern/Julia-mVMC scripts/check_orbital_general_parity.jl
```

The source version is Julia-mVMC v0.5.0 (`c2ea432`, numerical sources unchanged
in the pinned `8bb1b9e` reference), on Intel macOS with one BLAS thread.
Julia uses OpenBLAS 0.3.30 (ILP64, Haswell); Rust validation uses Homebrew
OpenBLAS 0.3.34 (LP64), with the FSZ inverse's macOS Accelerate kernels.
The fixture header records the Julia BLAS configuration. These checks follow
upstream `test_orbital_qptrans_utils.jl` and `test_unit_slater_update.jl`; they
compare all floating-point bits, including signed zeros.

`heisenberg/namelist.def` points to the original FSZ model's non-orbital inputs
and replaces its AP/P files with an explicit three/four-column General
definition. The parameter indices, signs, term order, and component flags are
preserved. Julia's parser contract is `site1 site2 idx [sign]` with combined
spin-site coordinates; the unused six-column General files bundled with the
upstream examples are not that parser's input contract.

The General run must match the existing AP/P fixtures, without regenerating
those fixtures:

```sh
julia +1.13.1 --project=extern/Julia-mVMC scripts/check_sr_cg_runner_parity.jl --case=fsz --general
julia +1.13.1 --project=extern/Julia-mVMC scripts/check_sr_direct_runner_parity.jl --case=fsz --general --store=0
julia +1.13.1 --project=extern/Julia-mVMC scripts/check_sr_direct_runner_parity.jl --case=fsz --general --store=1
```

Julia and Rust independently compare parameters, energy, all saved index,
configuration, occupancy, and projection arrays, and the next 624 RNG words at
prefixes 1, 2, 3, and 50. CG also compares the complete SR-info output.
General direct SR additionally verifies the existing matrix/factor/solution and
stored-Gram fixtures on the Julia side. The original AP/P tests remain enabled.
