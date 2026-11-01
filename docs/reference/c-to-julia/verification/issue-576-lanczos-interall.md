# Lanczos InterAll reference audit (issue #576)

The vendored C 1.3.0 fixed-Sz Lanczos path includes each `InterAll` row in
`calculateHW` and `calculateHW_real` (`src/mVMC/lslocgrn.c:164–173`,
`lslocgrn_real.c:421–430`). They call the same two-body `calHCACA` operator
used for PairHop and Exchange, in input order, after those two families. The
real path takes the real part of the coefficient; the complex path uses the
complex coefficient. The Hamiltonian kernels and the nested `calHCACA` paths
also include `InterAll`, so adding only the outer term is insufficient unless
the moved and nodal paths include it as well.

`GetInfoInterAll` (`readdef.c:2420–2456`) checks that each bilinear conserves
spin for a fixed `TwoSz`. The normal Lanczos kernels select the annihilation
spin from each pair. General orbitals are subject to C's Lanczos admission
check (`readdef.c:610–619`); that check has an `if`/`else if` involving the
spin projection count, so its exact control flow must be retained when
characterizing edge cases rather than treating an error string alone as proof
of all possible inputs. `GetTransferInfo` (`readdef.c:1973–1997`) accepts
spin-changing rows, but the normal Lanczos `calculateHK` and normal local
Hamiltonian select only `Transfer[idx][3]`, ignoring the creation-spin field
(`lslocgrn.c`, `lslocgrn_real.c`, `calham.c`, `calham_real.c`). Thus C's
normal-orbital result is for a different, spin-preserving operator, not the
specified spin-flip Hamiltonian. The FSZ Lanczos path is not implemented in C
(`readdef.c:573,615–619`). Rust keeps the explicit error instead of silently
changing that Hamiltonian.

The Rust normal local-energy path evaluates `InterAll`, including in moved
configurations and nodal expansions. `calculate_lanczos_h2_transfer` adds the
outer `InterAll` terms after Transfer, PairHop and Exchange, and reduces
coincident indices using the same two-body dispatch as the mode 2 Green
functions. End-to-end real, complex and real nodal C references for both modes
are in `tests/fixtures/native_c_lanczos_576/`. The real nodal scenario exercises
zero-overlap expansions but does not establish branch coverage for every
InterAll row; the upstream complex nodal reference has a known C crash, noted
in `c_toolbox/nodal_481/README.md`.
