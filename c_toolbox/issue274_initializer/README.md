# Issue274 normal initializer caller-order oracle

Six optional C caller-control fixtures were acquired and independently reviewed.
Actual compiler, extraction/source hashes, commands and expected process statuses
are in `tests/fixtures/issue274_caller_control/PROVENANCE.txt`.
Cargo must not compile/invoke/read this directory.

Original C vmcmake.c SHA256:
`8431b58eaec53e5325c801b69aa8a56306ec6a154b54e099266f4e21e5dd42ed`.
Optional extraction preserves its complete shared initializer body and license.
The standalone control supplies explicit synthetic INFO and placement word-call
counters. It proves caller/control order ONLY, not actual SKTRF, MPI, SFMT
values, numerical sampling or a public model's first failed factorization.

Optional reproduction commands (fresh exclusive output paths):

```bash
mkdir exclusive-control-output
perl extract_shared_initializer.pl /pinned/original/vmcmake.c exclusive-control-output
cc -O2 -std=c11 -Iexclusive-control-output caller_control.c -o /owned/executable/caller-control
/owned/executable/caller-control success
/owned/executable/caller-control retry
/owned/executable/caller-control negative
/owned/executable/caller-control peer-retry
/owned/executable/caller-control call101-success
/owned/executable/caller-control exhaustion
```

Bind source/extracted-body SHA, actual compiler flags/version/native providers,
each command/status and outputs before fixture adoption. Last two cases must
exit42 after101 calls, including last-call-success: no real setup follows.
All other cases exit0 and print a distinct synthetic real setup whose status
does not trigger a retry. Native first-failure/retry kernel operands and actual
SFMT retry fixtures remain separate acceptance work under #274; these synthetic
fixtures do not supply them.

The pure-Rust tests consume checked-in C control output fixtures,
not these scripts/programs. Do not fill missing fixtures from Rust output.
