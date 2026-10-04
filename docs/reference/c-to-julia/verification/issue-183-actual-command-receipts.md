# Actual command receipts: bounded infrastructure milestone

Related to #183 and #185; neither issue is completed by this milestone.

The driver now records actual command start/end times, return code, timeout or incomplete state, raw stream paths, and explicit selected-gate context. The aggregate validates acquisition and selected-call ordering, exact selection/profile/features/argv/environment joins, and completed zero-return execution. Version discovery is not selected-gate completion. Legacy receipts without the execution schema remain historical and are not current-eligible; this does not rewrite their original results.

The independent positive-first package builders and rehashed semantic negatives test infrastructure contracts, not numerical models. Actual small Python child processes exercise completed success, nonzero failure, timeout and incomplete launch without Cargo builds or reference oracles.

Executed source base: `1b0e9d90de7ffb90c84d098c181cefa470042021`. Receipt: `/tmp/issue183-command-approved.Aoze3z`, with commands and seven raw stderr logs retained. Focused identities passed 1 + 1; full suites passed 2 pipeline, 19 driver, 19 aggregate, 2 metadata and 2 provenance identities (44 total). Inner prior/post, outer, cleanup, and source/tools/scripts/providers/75-prerequisite closure posts were all zero. Protocol used offline uv, explicit `/usr/bin/python3`, `-B`, 120-second commands, 300-second overall deadline with 5-second grace, 16 MiB receipt limit and 384 MiB reserve.

Publication base: `5b0874eb70b72e2a7993662af757d50b25dadc17`. The four executed scripts and their baseline versions are unchanged between the executed and publication bases; publication transfer is byte-compared separately. The local receipt is not relabelled execution on publication main. Exact-head CI remains pending.

Previously failed or unexecuted acquisition receipts remain failures or unexecuted, not eligible success evidence. No model, native MPI, Julia/C oracle, fixture generation or numerical tolerance change is included. Universal failure-POST closure for every family remains the next separate #183 milestone; broader #183/#185 acceptance remains open.
