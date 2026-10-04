# Available failure closure POSTs — bounded infrastructure milestone

Related to #183 and #185; no issue closure or model-completeness claim.

The driver attempts source, fixture and selected-binary POSTs independently in its finally path, preserving the primary failure and classification. Never-acquired closure is NOT_STARTED; missing/unreadable captured input is UNAVAILABLE; changed bytes are CHANGED; matching bytes are UNCHANGED, not model Pass. Successful execution with a bad POST fails. Source is captured before discovery, fixtures before provenance validation, and the binary immediately after selection.

Fresh metadata declares exact integer closure_post_schema=1. Pass requires terminal/file exact three-record join, UNCHANGED/error=null for every record, and matching captured before/after hashes. Missing schema remains historical/not-current-eligible; previously reviewed PR305 evidence is not rewritten. Literal positives precede 19 rehashed semantic negative variants across all four families (76 subcases). Real subprocess failures and real filesystem deletion/mutation test reporting without numerical model execution.

Executed source dependency: PR305 head `48b4d0f2ffd8337a869e468d5fd260c3af6c4e9c`, based on main5b087. Initial receipt `/tmp/issue183-failure-post-approved.WrzqIu` failed: first two focused identities passed, third had four file_list diagnostic-expectation failures; full suites NOT_RUN; prior1/post0, cleanup0 and all closure posts0. Strict read_json correctly rejected a JSON array with `expected JSON object`; the test expected a later join diagnostic. Only that test expectation received the approved one-line correction; production rejection was not weakened.

Approved successor `/tmp/issue183-failure-post-successor.3sqwyb`: focused1/1/1 passed (.428/.003/3.306s); full pipeline4, driver19, aggregate20, metadata2, provenance2 passed (47 identities). All eight command/group-empty statuses, prior/post/outer/cleanup and five source/tools/scripts/providers/75-prerequisite posts were zero. Offline uv used explicit Python3 and -B, 120s per command, overall300+k5, 16MiB and reserve384MiB. No model, Cargo build, native MPI, installer or C/Julia oracle ran.

Publication base `98064faeed46b10ff1d2e96a9cc47e50b47f4531`: the four scripts are byte-identical to the successful successor, and their baseline source is unchanged from PR305 head. This is source association, not a rerun on publication main. Existing regular Linux/default CI invokes pipeline4, aggregate20, metadata2 and provenance2 (28 prospective identities), not the driver19 suite or the complete local47. Exact publication-head CI remains pending.

SIGKILL/process destruction and unwritable receipt storage cannot fabricate completed POSTs; outer owner/aggregate failure or incomplete status still applies. This milestone reports available closures, not a guarantee that every missing acquisition can be completed. Broader #183/#185 acceptance remains open.
