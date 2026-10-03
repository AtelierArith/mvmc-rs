# All six non-InterAll terms, Lanczos modes 1 and 2

`hubbard_all_terms_lanczos{1,2}` copy the independent six-site Hubbard
PhysCal input and fixed record, retaining its 100 samples, warmup10 and
interval1. Nonzero Transfer and CoulombIntra are retained; the added records
are CoulombInter(0,1)=0.375, Hund(0,1)=-0.125, Exchange(0,1)=0.25 and
PairHop(0,1)=-0.0625. No InterAll record is present.

`ordered-pairs.txt` comes from actual C `ReadPairDValue` and
`ReadPairHopValue`, not Rust parsing. The latter expands one declared pair
into (0,1) then (1,0), with the same coefficient; `readdef.c:697` doubles
the declared count. This standalone probe does not establish full C reader,
sampling or Lanczos numerical parity. Compiler, extraction boundaries and
source/driver SHA-256 are in `provenance.txt`.

Explicit optional regeneration (never invoked by Cargo):

```sh
julia +1.13.1 --project=extern/Julia-mVMC c_toolbox/generate_physcal_181_hamiltonian.jl
PHYSCAL181_MODELS=hubbard_all_terms_lanczos1,hubbard_all_terms_lanczos2 julia +1.13.1 --project=extern/Julia-mVMC scripts/generate_physcal_181.jl
PHYSCAL181_MODELS=hubbard_all_terms_lanczos1,hubbard_all_terms_lanczos2 PHYSCAL181_METADATA_ONLY=1 julia +1.13.1 --project=extern/Julia-mVMC scripts/generate_physcal_181.jl
```

New independent Julia1.13.1/Manifest-v1.13 references use unchanged sampling
and Lanczos kernels. Both modes consume 3284 UInt32 words, with 12 during
initialization. Each model records seed, checkpoints, counters, next624,
revision, input hashes and OpenBLAS0.3.30 ILP64/thread1 provenance.
Their numerical outputs are not exact C whitespace oracles and are not a
new native C measurement trace. The historical full C-reference Lanczos gate
remains separate; PairHop's composed measurement has no full native C trace.

Supplemental source hashes captured immediately after generation:

| Julia source | SHA-256 |
| --- | --- |
| `green_func_calc.jl` (Lanczos kernels) | `2baa297e891db88b3aa31ce7d1d44ed15a8aa7fde9bc85a68c415a58f3686513` |
| `vmc_main_cal.jl` (measurement) | `16def1b75c5a8b36462884983c18a4422882b62d3bb45543baf699756b8839d0` |
| `data_io.jl` (LS output) | `fc4c4de41bc1df052925f676803361f2ee1fbbc1b58bf57f683a202bd8cb6f5e` |

Focused Rust validation run `ddd71bbe-d85f-43a7-b015-e5982f72c5a9` passed:
ordered native C metadata, exact saved buffers/counters and actual final RNG,
fixed preservation, strict normal/LS rows and indices, independent numerical
values and the mode-dependent LS file set. No tolerance was changed.
