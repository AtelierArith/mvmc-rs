# Candidate retry-cap probe (#559)

Standalone C-authority check of the candidate retry loops in mVMC 1.4.0
(`extern/mVMC-1.4.0`, revision `e83aa6cb`). The four functions are byte-identical
to mVMC 1.3.0 (`d73d06bd`, lines 515–545, 594–664 and 756–788), and the probe
output is unchanged (#682). `generate.sh` verifies the SHA-256
of the upstream sources, then copies these functions verbatim (no substitutions):

| `.inc` | Source | Lines | Function |
| --- | --- | --- | --- |
| `hopping_upstream.inc` | `src/mVMC/vmcmake.c` | 620–650 | `makeCandidate_hopping` |
| `fsz_upstream.inc` | `src/mVMC/vmcmake_fsz.c` | 633–667 | `makeCandidate_hopping_fsz` |
| | | 669–703 | `makeCandidate_hopping_csz` |
| | | 866–898 | `makeCandidate_LocalSpinFlip_conduction` |

`probe.c` supplies only the globals those functions read (`Nsite`, `Ne`,
`Nsize`, `LocSpn`) and replaces SFMT with a fixed raw-word tape:
`genrand_real2()` is one word times 2^-32, as in SFMT. SFMT initialization,
the update-type draw, the FSZ hopping/flip coin and the Metropolis decision are
outside this kernel check. After each candidate the probe draws one marker word
only when the candidate is not rejected, mirroring `if(rejectFlag) continue;`
before the Metropolis `genrand_real2()` in `vmcmake*.c`.

C draws, then tests `icnt > Nsite*Nsite`, then tests occupancy, so the draw made
after the cap is rejected even when it names an empty site. The cases cover:

- ordinary hopping at Nsite=2 (the #559 minimal state) and Nsite=4: final empty
  destination rejected, last permitted destination, first draw empty, all draws
  occupied;
- ordinary source selection, which has no cap in C: eleven local-spin draws
  (more than `Nsite^2 + 1`) followed by an itinerant electron;
- the FSZ siblings (`hopping_fsz`, `hopping_csz`, `LocalSpinFlip_conduction`)
  at their final and last permitted draws. These already matched C; they are
  regression coverage from the #559 sibling audit.

Each fixture `result` row is `mi ri rj s t reject candidate_words
words_after_metropolis` (`t = s` for ordinary hopping). Rust reads
`tests/fixtures/candidate/hopping_retry_cap_559.txt` in
`crates/mvmc-core/tests/candidate_retry_cap_c.rs` without invoking C.

With no itinerant electron at all, C's ordinary source loop never terminates;
that input is unsupported and is not in the fixture. Rust keeps the Julia
guard's rejection after `Nsite^2 + 1` draws for that case only.

Integer-only probe: no libm, BLAS, MPI or SFMT library. Generated on Linux
x86_64 with GCC 13.3.0 (`cc -std=c11 -O0`).

From the repository root:

```sh
c_toolbox/hopping_retry_559/generate.sh          # verify the fixture against C
c_toolbox/hopping_retry_559/generate.sh --write  # regenerate it
```

SHA-256 at generation:

```text
a9518a622b4c170e0f0d6bca98270425a0042ccbe1c1deb1106c8bbec0a28fc9  extern/mVMC-1.4.0/src/mVMC/vmcmake.c
f6322ed48eb186c097074ea9287906fcf58129a44d91b932d0a6750ea9daba66  extern/mVMC-1.4.0/src/mVMC/vmcmake_fsz.c
30b1542c905dcd8bee68f1e3b601308bee04f3d020d214474bfe8128952a203a  c_toolbox/hopping_retry_559/hopping_upstream.inc
94b0431c870045895fbe24e05d817c3e4fdfafca8531d8f6aec7e01d1c365630  c_toolbox/hopping_retry_559/fsz_upstream.inc
8b175389b16ef651cf38c4782e4c118915505f7c4443b9741fc4f4637aa32446  c_toolbox/hopping_retry_559/probe.c
1b3354bafaf3567b4777994b28d117cd7e528d4f8f541a74e9f060dbd236330c  tests/fixtures/candidate/hopping_retry_cap_559.txt
```
