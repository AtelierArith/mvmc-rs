# Independent C parameter-stage acquisition (not yet adopted)

See [toolbox provenance](../../../c_toolbox/reviewed_parameter_c_audit.md)
for upstream source/license/extraction, compiler dependencies, actual terminal
executions, complete input hashes and regeneration commands. These expectations
come from original C parameter/overlay/synchronization calls, not Rust outputs
or Julia parameter snapshots. No normal Rust test invokes the toolbox or Julia.

`parameter-audit.tsv`: sequence, phase, zero-based declared index, mapped marker,
real, imaginary, raw real flag, raw imaginary flag; all slots at each of the
three phases. `flags-written.txt` has 2*NPara component markers. Undefined flag
cells have explicit adapter zero sentinels and must never be asserted as native
C zero parity. Numerical budget and Julia equality remain UNVERIFIED.

Each phase `*-state.txt` contains three lines: exactly 624 raw UInt32 SFMT words
in native memory order; the actual native index; observed primitive word count
since that group's seed. `*-next624.txt` contains exactly 624 non-consuming
future words. `*-draw-count.txt` repeats the observed count. Canonical groups
1–3 independently initialize with seed12395 and include original initial.def;
the DH Mode1 group uses seed1 and all actual keyword overlays.

`spec.txt` is the independent definition framing supplied to C; its acquisition
paths are recorded, not portable runtime dependencies. `provenance.txt` is
unchanged actual-generation provenance. `archive.sha256` checks all persisted
artifacts. `compiler-inputs.sha256` is rooted at the repository, generated after
GCC -MM confirmed the complete local compilation dependency set; generated
excerpt hashes are in provenance. No C binary or generated source is required
to consume the acquired text.

Executed: two C stage workloads, 1,131 complete slot records, all three phases,
44 artifact copies hash-identical to actual native output. Unrun: reviewed-fork
Julia full state/count comparison, authorized numerical-budget comparison,
fresh 56-prefix/13-long milestone gates, native macOS.
