# C raw proposal / RNG boundary probe (#557)

The authoritative vendored C revision is `d73d06` (mVMC 1.3.0).
`candidate_upstream.inc` is the verbatim function `makeCandidate_hopping` in
`src/mVMC/vmcmake.c`, from its declaration through the closing brace immediately
before the exchange-candidate comment (lines 515–546). No source substitutions
are made. Its `gen_rand32`/`genrand_real2` calls use a deterministic raw-word
stream in this probe; SFMT initialization is outside this kernel check.

The harness reproduces the hopping branch in `vmcmake_real.c:140–207`: rejected
candidates skip the acceptance draw; a valid candidate evaluates
`exp(2*(x+(logIpNew-logIpOld)))`, maps a nonfinite weight to -1, then compares
`weight > genrand_real2()`. The accepted-move rebuild boundary is `nAccept > Nsite`.
For the two-electron probe, the independent analytic overlap is
`S[up,down+Nsite] = 2^(up+down)` and `x=0`. No Pfaffian implementation or MPI
executable is borrowed. These are standalone candidate / decision expectations,
not a complete C Opt fixture.

The fixture covers occupied-site retries, the last permitted destination, a
final empty destination rejected at the cap, accepted and rejected Metropolis
moves, all four partial-proposal refill positions, and a rebuild crossing a
preview. Each `step` row is consumed words, source index, spin, old site, new
site, rejected-candidate flag, accepted flag, up site, down site, cumulative
accepts, accepted window after any rebuild. The first case completes before its
peers refill in the threaded engine regression.

Generated on Linux x86_64 with GCC 13.3.0. This probe uses libm only, no BLAS.
The chosen nonzero decision margins are at least 0.25, so ordinary log/exp
roundoff cannot change the expected branches. Rust tests read the checked-in
fixture without invoking C, Julia, or this toolbox.

From the repository root:

```sh
cc -std=c11 -O2 -Wall -Wextra -Werror -fno-fast-math -ffp-contract=off \
  c_toolbox/interval_c_rng_557/probe.c -lm -o /tmp/mvmc-557-c-probe
/tmp/mvmc-557-c-probe > tests/fixtures/interval_c_rng_557.txt
```

SHA-256 at generation:

```text
8431b58eaec53e5325c801b69aa8a56306ec6a154b54e099266f4e21e5dd42ed  extern/mVMC-1.3.0/src/mVMC/vmcmake.c
8126fe03d3a1cb92df728adb66e18e43ec659fe9d2f1e638edfedec672213f4e  extern/mVMC-1.3.0/src/mVMC/vmcmake_real.c
ed023e028cf80e65a2fb99e0862f71219b79fa5043353c9f8b082e4f3566ff7e  c_toolbox/interval_c_rng_557/candidate_upstream.inc
bf1eb86257212044668572d30fbce62855969d255b1091113414f7e917113250  c_toolbox/interval_c_rng_557/probe.c
78bd93f8cb4078415fe229b30d9b19d4815c8b7a0a08cb0e709eda5823e8c1a8  tests/fixtures/interval_c_rng_557.txt
```
