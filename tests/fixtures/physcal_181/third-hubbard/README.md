# Third Hubbard Lanczos runner reference

Scope: third model in pinned Julia test/integration/lanczos_equivalent.jl,
HubbardChain / hubbard_chain_real / real / consumed19 / NLanczosMode2.
Copied acquisition records from Julia revision
8bb1b9e8ae47b1512c00b321be05664ddcac0fd1,
test/integration/reference/hubbard_chain_real/physcal_ref.
The complete original metadata.txt is preserved beside this file.

Expected LS results are historical independent native C execution at
issp-center-dev/mVMC commit 622166afe33c6be3402d7c926db7e9c0003a47c4,
single MPI rank, OMP_NUM_THREADS1; original metadata records macOS arm64,
Apple Clang15 / gfortran15.2 / Accelerate. Not a fresh Linux C execution,
not a fresh Julia1.13 whole-trajectory capture, and not synthetic IO134.
Optional reproduction uses the source/build environment and command recorded
in metadata.txt: vmc.out -e namelist.def ../zqp_opt.dat in staged inputs.

All five original numerical files are copied without numeric changes.
Four expected files retain exact source SHA256. The ls_out record had no
terminal newline; this copied text fixture adds one newline (source SHA
19e19c45cdad08a436ac31b7e1c717f64c10e469eb4cab50495d8d5aefca3da3).
The Rust numerical comparison ignores blank lines but strictly validates
ordered row/column count and discrete indices. This fixture is not a byte
format oracle. C writer byte formatting is separately tested elsewhere.

Independent original per-output bounds: LS out absolute1e-8;
QQQQ absolute1e-10; LS one/direct/factored absolute1e-8, relative0.
No bounds are enlarged. Normal Rust tests consume these offline records only,
never invoke C/Julia or inspect vendor checkout at runtime.

Copied records SHA256:

```text
2dee432769f18fdceede4942ac0d975e8ce5de8a39e482fed9aabd3caec698ae  tests/fixtures/physcal_181/third-hubbard/inputs/coulombintra.def
b30dd7042523b98a4d4627acbfc312efeb11bdcea30c7d1e363cc275e33f5c1a  tests/fixtures/physcal_181/third-hubbard/inputs/greenone.def
691a473126df3c13e933921c9576acbbac0ca72480aeefb2448eb50936b9518b  tests/fixtures/physcal_181/third-hubbard/inputs/greentwo.def
1b38b8a9f66906f6a8ee6019fffce5317153330779758120c5ca55d49c98f8c7  tests/fixtures/physcal_181/third-hubbard/inputs/greentwoex.def
ab689fc16e245bae343961a9a424398adaf5ddeef667911e4a4306f0c4834528  tests/fixtures/physcal_181/third-hubbard/inputs/gutzwilleridx.def
2feb386bcb243aa4b1cee3d3309c87b120757484756041b8ec5df9a637115956  tests/fixtures/physcal_181/third-hubbard/inputs/jastrowidx.def
d27306ba8703a1431ebc33846f98fecf5a812133b98adbeb0b6af61041fbaee2  tests/fixtures/physcal_181/third-hubbard/inputs/locspn.def
07f0f445f0e1a716c965ae244a87c9d7c3885bbdaa4de5e44e2f2328684c77f6  tests/fixtures/physcal_181/third-hubbard/inputs/modpara.def
17364bf19536df5ca8383186c67c5374dac019b9fae737c6cea2fb740fe256f5  tests/fixtures/physcal_181/third-hubbard/inputs/namelist.def
c2a1a275bfab65ff94a19e1ed6f9a630c02a0e1eb4ee80cd0787e166b5997ff6  tests/fixtures/physcal_181/third-hubbard/inputs/orbitalidx.def
c996af9119a1f2e5cec513dc08476050a5bdd60e02f42409f0ffd3d4a05d992d  tests/fixtures/physcal_181/third-hubbard/inputs/qptransidx.def
0f0ffb4737bdf45ecde7190df564f5c475c48538c0353804dbe534dbc9f7fec6  tests/fixtures/physcal_181/third-hubbard/inputs/trans.def
bc60804a5c6384785178d3db4ca8761a47a10165642f95dda4a5c02170d5aac9  tests/fixtures/physcal_181/third-hubbard/zqp_opt.dat
0cf333b8bdeac0fc5f9ee27737e760f978ccd97762462b936b9d38d8a98aad60  tests/fixtures/physcal_181/third-hubbard/expected/zvo_ls_cisajs_001.dat
ec96c65dcf24db1b81314224e5c394e865705f9bb633be9a27719ea8c58ccbe3  tests/fixtures/physcal_181/third-hubbard/expected/zvo_ls_cisajscktalt_001.dat
b5529686cd90c5e923412ccce36bab8f8bb29abf8d0aaf5f9b9e24c5b7674410  tests/fixtures/physcal_181/third-hubbard/expected/zvo_ls_cisajscktaltex_001.dat
8911a4d2253d84aa25dd7753d4dde75cf2248d749d78f69e3786deb82b600a2f  tests/fixtures/physcal_181/third-hubbard/expected/zvo_ls_out_001.dat
f1adf3d4a5f19a3ee4d0edeacb272b3242c9850b88d82e7207f8f59adfea5176  tests/fixtures/physcal_181/third-hubbard/expected/zvo_ls_qqqq_001.dat
```
