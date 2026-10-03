# Issue183: native C zero-GEx Lanczos output contract

Explicit optional developer probe, never a Cargo dependency. Rust tests/builds
must not read, compile or invoke this directory. No generated fixture is added.
This checks original complete C writer functions, not full C executable, parser,
sampling, MPI, Julia or numerical trajectory parity. No BLAS or RNG is used.

## Authority and provenance

Origin: vendored mVMC1.3.0, `extern/mVMC-1.3.0/src/mVMC/`.
No extraction or source modification: the whole original `physcal_lanczos.c`
translation unit and its original header are compiled and linked. The probe is
a standalone caller. Upstream license is GNU GPLv3, as provided by
`extern/mVMC-1.3.0/COPYING` (SHA256
`8ceb4b9ee5adedde47b31e975c1d90c73ad27b6b165a1dcd80c7c545eb65b903`).

| Original input | SHA256 |
|---|---|
| physcal_lanczos.c | d0f537e49c25af59140d8a00f1b2532df243f6cc5657f96d41b1f74d44aced15 |
| include/physcal_lanczos.h | e3ef6136d196e2825373aa79c90e99192353b08eeef22c26f417d3c1bdaafcac |
| initfile.c | 8a7b20d54ab495cfac4df42646a30102aebaa228311ca3b17c1e29af5d0581f8 |
| readdef.c | 6c53cb832f93d6cbfd7cea955fbb693738af5536b913d36af32b98eed38c32d9 |
| vmcmain.c | fdcd661c4eb028786fc58ae5e1f5232426c943b547f95a0d74e888e602256d63 |
| probe.c | 10ce12bcb868cb612102081a1068216b609e18285203cdb2860e5c13158e5be8 |

`readdef.c:706` maps NCisAjsCktAlt to NTwoBodyGEx (default zero at1790).
`initfile.c:130-132` opens LS GEx file with `w` even at zero count when
NLanczosMode>1; `initfile.c:184` closes it. `vmcmain.c:696-709` passes that count
and stream to the real/complex functions. `physcal_lanczos.c:138-141` (real) and
`:259-262` (complex) execute zero pair-loop iterations then unconditionally
`fprintf(...,"\n")`. Therefore successful mode2 zero-GEx output is exactly
one LF, not zero bytes. Normal non-LS Green positive-count guards are different.
Historical Julia's missing expected file does not itself establish this rule.

## Reproduction

From the repository root, with an exclusively new temporary directory:

```sh
probe_dir=$(mktemp -d /tmp/mvmc-183-c-ls-empty.XXXXXX)
cc -std=c11 -O0 -Wall -Wextra \
  -I extern/mVMC-1.3.0/src/mVMC/include \
  c_toolbox/issue183_ls_empty/probe.c \
  extern/mVMC-1.3.0/src/mVMC/physcal_lanczos.c \
  -lm -o "$probe_dir/probe"
"$probe_dir/probe"
```

Actual initial audit: Linux x86_64, GCC `cc (Ubuntu
13.3.0-6ubuntu2~24.04.1) 13.3.0`, options above, terminal0. Original retained
source, binary and combined compiler diagnostics/stdout:
`/tmp/mvmc-183-c-ls-empty.9Ighp9/{probe.c,probe,actual-output.txt}`.
Executed binary SHA256:
`9e52c34d6046ff83b2d4d8bdbe2ebcfd7cbc897590b1853f873bf1099b3ceefd`.
Compiler emits existing unused-debug-stream-parameter warnings.

Actual output:

```text
mode=real bytes=1 first=10 next=-1
mode=cmp shared_bytes=2 first=10 second=10 end=-1 real_status=0 cmp_status=0
```

Input H1=0,H2=1,H22=1,H3=0,H4=4, nLSHam2, mode2 and all Green counts0.
Real writes one LF to a fresh tmpfile; complex appends one LF to the same
stream, so both independently contribute exactly one LF. The zero-count
malloc buffers are never dereferenced by the empty physical-count loops.
Analytic operands are only to reach the original successful writer boundary;
this is not a computed-energy reference or allowance.

The old Rust gate attempt79244 required zero bytes and failed correctly under
that erroneous assertion. Its artifacts are retained, not reassigned as PASS.
The corrected test contract requires both parsed GEx arrays empty, the specific
missing GEx expectation, and an actual regular file exactly `b"\n"`; it rejects
zero bytes, generic whitespace, CRLF, multiple LF, values, missing or symlink
outputs. EMPTY_CONTRACT is not an independent numerical comparison.
