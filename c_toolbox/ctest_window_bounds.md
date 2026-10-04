# C averaging-window bounds, without undefined numerical output

`ctest_window_bounds.c` executes the unchanged count assignments from
`readdef.c:683–684`, loop header from `vmcmain.c:339`, and complete collection
condition/index body from `vmcmain.c:511–513`. Its StoreOptData stub marks a
zero-initialized byte bitmap, rather than allocating or reading numerical data.
This is an assignment/index probe, **not** full C input-parser/executable parity.
No uninitialized C result is used as a numerical expectation.

Upstream source SHA-256:

```text
6c53cb832f93d6cbfd7cea955fbb693738af5536b913d36af32b98eed38c32d9  extern/mVMC-1.3.0/src/mVMC/readdef.c
fdcd661c4eb028786fc58ae5e1f5232426c943b547f95a0d74e888e602256d63  extern/mVMC-1.3.0/src/mVMC/vmcmain.c
573d1fb995de33386e7452f5e2fa34aeb05efe14b5bf4a904eb54d2ac7cf1b9e  extern/mVMC-1.3.0/src/mVMC/setmemory.c
509a573944a5eabde93864346902674faaff6c23d0c88f89867263f8372dd51a  extern/mVMC-1.3.0/src/mVMC/avevar.c
```

The reader's keyword branches `readdef.c:1906–1909` assign the requested values
to bufInt; the later assignments preserve them. `setmemory.c:424` allocates
`NSROptItrSmp*(2+NPara)` using malloc, not initialized padding. `avevar.c`
aggregates the requested window. No authoritative clamp was found.
An exhaustive `rg -n SROptData` over `src/mVMC` C/header files finds only the
allocation/free, global declaration, StoreOptData writes and aggregation reads;
there is no calloc, memset or other clearing path. The requested-but-unwritten
leading rows therefore contain indeterminate numerical storage, not defined
zero padding. The probe's byte bitmap is intentionally initialized and is not
a substitute for, or assertion about, those numerical contents.

```sh
stage=$(mktemp -d /tmp/mvmc-window-bounds.XXXXXX)
cc -O0 -Wall -Wextra -Werror c_toolbox/ctest_window_bounds.c -o "$stage/probe"
"$stage/probe" 1 100
"$stage/probe" 1 2
"$stage/probe" 3 5
"$stage/probe" 3 3
```

Native Linux x86_64, GCC 13.3.0, libc only, no BLAS or numerical solver.
Expected written index ranges are 99–99, 1–1, 2–4 and 0–2 respectively.
Thus an oversized requested window is unchanged but leaves leading rows
unwritten. Rejecting it is a safe supported-input boundary; silently returning
a smaller window invents a different output contract. Short-prefix validation
must explicitly override **both** step and window counts in Rust and the
independent reference. Existing #180 1/2/3/50 inputs already do this; their
successful trajectory evidence does not validate the oversized public API.

Executed all four bounds-only cases successfully with the compiler/options above.
For (steps,window)=(1,100),(1,2),(3,5),(3,3), the observed unwritten row counts
are 99,1,2,0; effective windows remain 100,2,5,3. Probe source SHA-256
`abc74c74352c14fce0b508ebe44ca238579dd52ec8c3e322d8ee611af3f9c1e2`;
executed binary SHA-256
`e0e5888afd2067752a78382978b806cf38a85bcbce2d5f450d75ab4de9c4b64a`.
