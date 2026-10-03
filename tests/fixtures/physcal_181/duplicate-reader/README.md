# Mode-2 duplicate OneBodyG without TwoBodyGEx

C does not accept this as a supported input. `readdef.c:577–585` runs
CountOneBodyGForLanczos in mode2 even with zero TwoBodyGEx records. Its
`285–296` first-occurrence map reduces two identical requested operators to
NCisAjs=1. The later reader call (`1046`) sets IndirectGFOn solely from
NCisAjsCktAlt>0, hence passes 0 without GEx. GetInfoOneBodyG reads both records
in the normal branch and rejects idx=2 versus NArray=1 (`2299–2300`).

The extracted actual counter and reader produce `result.txt`; the ReadGreen
adapter supplies two valid original rows via the same actual reader. Storage
for the second read is deliberately padded: upstream writes the extra row
before reporting the count error, so matching that out-of-range write is not
a supported Rust algorithm. The probe does not run full native parsing/MPI or
Lanczos. Calling LS directly with a forced two-record state would bypass the
rejecting input path and cannot prove that C supports this input.

Regenerate explicitly with Julia 1.13.1 / Manifest-v1.13:

```sh
julia +1.13.1 --project=extern/Julia-mVMC c_toolbox/generate_physcal_181_duplicate_reader.jl
```

No Cargo test invokes this tool. Rust rejects safely before RNG/files, with
a duplicate-specific diagnostic rather than C's generic file-count error.
Normal mode0/1 retains duplicate rows, as the separate ordered-output test
asserts. The current Julia rejection is therefore not the only authority for
the bounded mode2/no-GEx restriction.

The additional positive GEx case has constituent operators equal to the
duplicate OneBodyG key after C's second-pair reversal. Actual C deduplication
returns one canonical row and its indirect reader returns success. At the
initial checkpoint Rust retained both duplicate rows (first divergence:
count2 versus count1). The parser correction now deduplicates in first-occurrence
order; the parent reports all eight `green_two_ex` parser tests passing.
This does not loosen the no-GEx rejection or flatten output rows.

`namelist.def`, `onebody.def` and `factored.def` provide the positive-case
regression input using the independent Heisenberg fixture's supported
Hamiltonian/orbital definitions. The exact expected canonical list is
`[(site1=0,spin1=0,site2=1,spin2=0)]`; the sole GEx index pair is `(0,0)`.
These records reproduce the scalar probe's operators (all also valid for
Nsite=6). Parser owner must assert the list and pair before running PhysCal,
and preserve first-occurrence ordering for additional distinct operators.
This fixture is not a newly generated native C sampling reference.

`canonical-layout.txt` is emitted by the extracted actual C count and both
Green reader functions, not hand-transcribed from Rust. Its first line is
canonical one-body count and factored-pair count, followed by ordered
four-index rows and ordered two-index GEx references. The normal Rust
`duplicate_one_body_with_gex_matches_native_c_canonical_layout_fixture`
test reads these integers and the checked-in input files only. After the parser
correction it passes the canonical count, ordered row and pair assertions, and
the real runner mode0/mode2 checks. This is still a reader-only native C probe,
not a new native C sampling reference.
