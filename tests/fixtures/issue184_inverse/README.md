# Offline inverse fixtures

See ../../../c_toolbox/issue184_inverse/README.md for source/license/compiler,
Julia1.13.1 pinned manifest, conditioning/residuals and reproduction.
Input: kind/n header, column-major complex pairs then n one-based pivots.
C/J output: full column-major A, M, vT pairs then n pivots.
Operator: independent U*T*transpose(U) with sequential permutation.
Small full/scalar paths only; no panel coverage claim. Real4 original literal;
complex6 NEW fixed dyadic input, not original unseeded12 Julia assertions.
Normal Cargo reads checked-in files only. Payload SHA256:

```
69ca02ba8532525fead34621c42e6a9ef811e5f03cec96c8aa64ee793705c8ad  tests/fixtures/issue184_inverse/complex6_pair_pivots.c.txt
72394f063a8c3d9d52a5b524ab797dbfc9a3118c90b41414ecca2fb10903553a  tests/fixtures/issue184_inverse/complex6_pair_pivots.input.txt
23c59e877a3391189b72b973878d2599e0e799b8456fd2f789bed3d9d5bb9412  tests/fixtures/issue184_inverse/complex6_pair_pivots.j.txt
6a2098e486df213dc711676e60cb8f627a9ff8a44830404c30efb51fc002c032  tests/fixtures/issue184_inverse/complex6_pair_pivots.operator.txt
84c8f2f28e34d492156c80e063c26ac404e97d5b6a4d3c012ad164d5f1a66f45  tests/fixtures/issue184_inverse/real4_identity.c.txt
4ff4aaa135d7653d1ba42bcd63c7b88aa92e883f61ef46b1a84e6a03070a126d  tests/fixtures/issue184_inverse/real4_identity.input.txt
84c8f2f28e34d492156c80e063c26ac404e97d5b6a4d3c012ad164d5f1a66f45  tests/fixtures/issue184_inverse/real4_identity.j.txt
89eb1b48cf0eb01ca3ec45d8b3e299a0a5385bf728346d40c1770801aa53c046  tests/fixtures/issue184_inverse/real4_identity.operator.txt
```
