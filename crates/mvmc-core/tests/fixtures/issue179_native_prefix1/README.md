# Native Heisenberg prefix1 diagnostic inputs

Related to #179/#185; this fixture does not complete either issue.

These 13 definitions originate from the independently generated Linux C input
collection `/tmp/mvmc-native13-collection.s7PmDg/HeisenbergChain`.
Twelve files retain their original bytes. Only `modpara.def` is adapted:
`NSROptItrStep 20 -> 1` and `NSROptItrSmp 20 -> 1`.
Original modpara SHA-256:
`8dfc9830a83c531e950e4847957297cc2179c2e0ea67e15693f8c5ea5853095d`;
adapted SHA-256:
`64804c8511ccc71c589837b3fa690b23a3cf71047511b948e4da0db32fb79706`.
`inputs.sha256` binds the exact 13 admitted diagnostic files.

External native C evidence (original Linux container 73c; not repository paths):

- Acquisition: `/dev/shm/issue179-native-C-offon-proof-G63DtS`.
- Capture: `on/observer.txt`, SHA-256
  `f2f53599dc4036c42c98e980ddb0677fd930756af26b3307adaeed4dc38683a4`.
- Separate read-only validation receipt:
  `/dev/shm/issue179-native-C-validation-readonly-G63DtS`.
- C source binding: `/tmp/issue179-native-C-reseals-KVvJwV/authority.sha256`;
  observer source `/tmp/issue179-native-C-reseal-packet-KVvJwV/overlay/observer.c`,
  SHA-256 `0428d32a2c6b08c8e2eeb8abc0b91fc3498e5fee77624f7057093bc4cf3b17e9`.
- Underlying SFMT source:
  `/tmp/mvmc-native13-build.sULMMH/source/src/sfmt/SFMT.c`, SHA-256
  `b61f0f0239193fd4c06f2879e6e8452225bec45b3bca426902c244fd31ad4e3a`.
- Data-only extracted fixture SHA-256:
  `a04f97ea935ece5f32c8c1462922b27d051b30aa80c804e7feb9d53d898e8ac1`.

The original C acquisition aggregate failure remains preserved; separate
capture/public-passivity validation does not rewrite that receipt. External
evidence is neither installed as a golden expectation nor read/invoked by Cargo.

Scope: seed1, world1, real/direct/store1, one step/window, 100 samples, derivative
width15, active SR dimension10. Rust OFF/ON compares its four scientific public
files byte-for-byte solely to check observer passivity. No C computed-bit gate,
numerical tolerance, full runtime authentication or cross-language numerical
PASS is claimed. The original 20-step/full13-model and broader MPI matrix
requirements remain open. Goodall is the sole combined integration publisher.
