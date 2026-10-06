# 11. 互換性と相違点

[目次](README.md) · 前へ: [10. チュートリアル: 16サイトのHubbard鎖](10-tutorial.md) · 次へ: [12. アクセラレータ(GPU)バックエンド](12-accelerated-backends.md)

## 11.1 基準実装と関数の対応

Rustへの移植は、**Juliaの設計**(公開API、ランナーの構造、ライフサイクル、テスト)と**Cの数値規約**
(パラメータのレイアウト、乱数の初期化と抽出順序、演算順序、符号、ファイル形式)に従っています。CとJuliaが食い違う場合、RustはJuliaのアーキテクチャを保ちつつ、Cの
挙動を採用します([AGENTS.md](../../../AGENTS.md))。各実装は次のように対応します(Cの名前は `extern/mVMC-1.3.0/src/mVMC` から、Juliaの名前は
`extern/Julia-mVMC/docs/src/en/compatibility.md` から、Rustのパスは `crates/mvmc-core/src` 配下です)。

| C関数 | Julia | Rust |
|------------|-------|------|
| `VMCParaOpt` | `vmc_para_opt!` | `run::vmc_para_opt` |
| `VMCPhysCal` | `vmc_phys_cal!` | `run::vmc_phys_cal_in_place_timed` |
| `VMCMainCal`, `VMCMainCal_fsz` | `vmc_main_cal!`, `vmc_main_cal_fsz!` | `run::accumulate_observables_local` |
| `VMCMakeSample`, `_real`, `_fsz` | `vmc_make_sample!` | `sampling::driver::vmc_make_sample*` |
| `InitParameter`, `SyncModifiedParameter` | `init_parameter!`, `sync_modified_parameter!` | `parameter_init::init_parameter` / `sync_modified_parameter` |
| `ReadInitParameter` | `read_initial_def!` | `initial_params::read_initial_def` |
| `ReadInputParameters` | `read_input_parameters!` | `mvmc-expert-parsers` |
| `CalculateGreenFunc` | `calculate_green_func!` | `observables::ordinary_green_values` |
| `StochasticOpt`, `StochasticOptCG` | `stochastic_opt!`, CG | `sr::stochastic_opt_*`, `sr_cg::stochastic_opt_cg_with_reducer` |
| `CalculateMAll_*` | `calculate_m_all.jl` | `pfaffian::calc_m_all_*` |

第2章から第6章に、すべての式について行単位の対応表があります。

## 11.2 RustがJuliaから取り入れる点と、Julia固有の挙動をコピーしない点

| 項目 | Julia-mVMC | `mvmc-rs` |
|-------|-----------|-----------|
| オプションの内部スレッド化 | `JULIA_MVMC_INNER_THREADS=1`, `JULIA_NUM_THREADS` | `MVMC_RS_INNER_THREADS`(スレッド数)と `MVMC_RS_INNER_THRESHOLD`([8.5](08-running.md#85-環境変数)) |
| MPIの検出 | `JULIA_MVMC_MPI=1` とランチャーの変数 | ランチャーの変数のみ(`LaunchContext::from_env`)。`JULIA_MVMC_MPI` はテストのヘルパーにのみ現れます |
| デバッグダンプ | `MVMC_DEBUG_*` | 未実装 |
| タイマー出力 | `MVMC_C_TIMER=1` | 同じ変数、同じファイル `zvo_CalcTimer.dat`、加えて `MVMC_*_DIAG` 系列 |
| サポートされないオプション | BackFlow、スピンJastrow、CGでの `NSplitSize>1`(Cでも未定義: 保存した `O` が未書き込みメモリから読まれます)、`NSRCG>=2`、`useDiagScale`、`RescaleSmat`、FSZのLanczos法(Cも拒否) | 同じ一覧が `validation.rs` で拒否されます([7.5](07-input-files.md#75-サポートされる入力と拒否される入力)) |
| 高水準ランナー | `run_para_opt_from_namelist(...; nsteps, mode, nsmp, output_dir, seed, initial_def)` | `RunConfig` を伴う `run::run_para_opt_from_namelist`(同じ引数)。CLIはこれをラップします |
| `1e-14` のスレーター振幅カットオフ | 従来のJuliaのテーブル構築で適用 | 適用**しません**(Cは宣言された値を読み込みます、`slater_update.rs:46`) |
| 最適化中の出力名 | `zvo_out.dat`, `zvo_var.dat` | 同じ(Cとは異なります、[9.1](09-output-files.md#91-ステップごとのエネルギー-zvo_outdat-opt-と-zvo_out_nnndat-phys)) |
| Juliaの `exp`、`log`、`hypot`、`sin/cos`、複素数の除算 | Juliaのカーネルで使用 | Juliaの演算順序が保たれる箇所では移植されています(`julia_exp`、`julia_log`、`julia_hypot`、`julia_trig`、`julia_complex`)。Cの演算順序が必要な箇所ではC方式の変種(`c_complex`、`*_c_compat`) |

## 11.3 Cリファレンスとの既知の相違

厳密に再現されるもの: SFMT-19937生成器とそのシード設定(`RndSeed + group`)、変換関数(`genrand_real2`)、棄却された更新を含むすべての
制御経路における乱数の抽出回数と順序、[第9章](09-output-files.md)のファイル形式(書き出される形式についてはバイト単位)、パラメータのレイアウト、および[第2章から第6章](02-theory-vmc-hamiltonian.md)のアルゴリズムの
構造。

ユーザーが観測しうる相違:

| # | 領域 | C | Rust |
|---|------|---|------|
| 1 | ドライバーのオプション | `-b -h -m -o -F -e -s -v` | `-b -h -m -o -F -e -s -v` と位置引数 `initpara` は実装済み(`-m` は、C が `N` で割る `N <= 0` を拒否、[MultiDef モード](08-running.md#multidef-モード-m))、Rust 独自の長いオプションあり([8.1](08-running.md#81-mvmcコマンド)) |
| 2 | Standardモード / StdFace | 組み込み(`-s`) | [7.6](07-input-files.md#76-standard-モードstdface)の格子について `mvmc -s` / `--dry-run`。他は未移植。C の格子ルーチンの明らかな不具合は再現せず修正([7.6](07-input-files.md#76-standard-モードstdface)) |
| 3 | 最適化中の `zvo_out`/`zvo_var` | `zvo_out_NNN.dat`, `zvo_var_NNN.dat` | `zvo_out.dat`, `zvo_var.dat` |
| 4 | `zvo_SRinfo.dat` | 直接法とCGの両方のソルバーで書き出し | CGのみ |
| 5 | `zvo_time_NNN.dat`, `zvo_varbin_NNN.dat` | 書き出し(`-b` でバイナリ) | どちらも書き出し(`-b` は C のヘッダーで、C の切り詰められたブロックの代わりに完全な `2*NPara` ブロック。[8.1](08-running.md#バイナリ出力-b)) |
| 6 | タイマーファイルの接頭辞 | `CDataFileHead` | 常に `zvo` |
| 7 | `modpara.def` のパーサーの既定値 | `SetDefaultValuesModPara` | `ModParaParameters::default`([7.2](07-input-files.md#72-modparadef)) |
| 8 | `NMPTrans = 0` | リーダーでは拒否されない(既定値0で `NQPFix = 0` となる、`readdef.c:778`) | 拒否 |
| 9 | `NSROptItrSmp > NSROptItrStep` | 行が書き出されないまま残る | 拒否 |
| 10 | Lanczos法 | Cがサポートする任意の軌道/項の組(`InterAll` を含む) | FSZなし(Cも拒否)、`InterAll` なし、スピンを変える `Trans` なし。`NSplitSize>1` はCと同様にサポート。$\alpha$ が失敗すると `NaN` を書き出す |
| 11 | 最適化における `TwoBodyGEx` | 読み込まれ、オプティマイザでは無視される | 読み込まれ無視される(`validate_para_opt` は PhysCal と同様に受理) |
| 12 | BackFlow(`BF`, `NProjBF`) | サポート | 最適化では拒否。PhysCalでは**チェックされない**([11.5](#115-未解決の観測事項)) |
| 13 | `NSRCG >= 2`; `useDiagScale`, `RescaleSmat` | `NSRCG != 0` でCGを選択(`vmcmain.c:485`)。他の2つのキーはC 1.3.0には存在しない(Julia-mVMCのオプション) | `NSRCG >= 2` と、非ゼロの `useDiagScale`/`RescaleSmat` は拒否 |
| 14 | `exp`, `cosh`, `hypot`, `sin/cos` | libm / C99の複素数 | Julia互換の移植(最下位ビットの差が生じうる) |
| 15 | ローカルエネルギーと `NProj` の和におけるOpenMPのリダクション順序 | スレッド依存 | 固定された逐次順序 |
| 16 | $\lvert\langle H\rangle\rvert\le10^{-14}$ のときの分散の列 | $\pm\infty$/NaN | `0.0`(最適化の出力のみ) |
| 17 | 実数モードのLTL/逆行列の演算順序 | C PFAPACK(`DSKR2` の加算してから減算) | オプティマイザはJuliaの順序を用いる(`dsktf2`, `utu2inv_real`, `crates/mvmc-core/src/pfaffian.rs:693,706`)。C順序の変種 `dsktf2_c_compat` と `utu2inv_real_c_compat` は `crates/pfapack` に存在する(PR #334)が、ランナーでは選択されない |
| 18 | Hund / Exchange / CoulombInter の行 | 1行につき1項 | 同じ。`PairHop` の行は両方で両方の順序に展開 |
| 19 | MPIのグリーン関数のリダクション | ランク0への `MPI_Reduce` | all-reduce、ルートが書き出し |
| 20 | 負の `DSROptStepDt` | `SRFlag = 1`(「Diagonalization Mode」の注記、ヘッダー `sEigenMax sEigenMin`)を設定し、`dt` を `-dt` に置き換える(`readdef.c:746-752`) | 処理されない: 負の値はそのまま使われ、SRステップの向きが逆になる |
| 21 | `modpara.def` における $N_e$ のキー | `Nelectron`, `Ne` | `NElec`, `Nelec`(Cの綴りは無視される) |
| 22 | `MVMC_RS_SR_BACKEND=tenferro` または `cuda` での SR ステージ | 該当なし | オプトインの Rust 拡張([第 12 章](12-accelerated-backends.md)): RNG の消費と Metropolis のプロトコルは同じ。SR の Gram、求解、CG 積は C 順序パスと導出された境界内で一致し、バイト単位では一致しない。デフォルトの `c-order` パスは変更なし |

## 11.4 数値比較ポリシー

詳細なポリシーは [docs/NUMERICAL_COMPARISONS.md](../../NUMERICAL_COMPARISONS.md)(issue #186/#190)にあります。読者が知っておくべき規則は次のとおりです。

- **計算された浮動小数点の結果をビット単位で比較してはなりません**。RustとCの比較も同様です。`actual` と `expected` の比較は、
  $|{\rm actual}-{\rm expected}|\le{\rm abs}+{\rm rel}\cdot\max(|{\rm actual}|,|{\rm expected}|)$ のときに合格します。この上限は、演算、問題の規模、
  条件数、ソルバーの残差に基づいてテストが正当化するものです。複素数の値は成分ごとに比較します。NaNはNaNとのみ一致し、無限大は符号まで一致しなければなりません。
- **厳密**(許容誤差なし): シードの解決、乱数の初期化と状態、624ワードのブロック、固定された制御経路における乱数の抽出順序と抽出回数、添字、フラグ、
  次元、および離散的な入力の規約。
- 浮動小数点のわずかな差がメトロポリスの判定を反転させることがあります。その後の軌道、保存された配置、ソルバーの終了、最終的な乱数の状態は、
  実装間やプラットフォーム間で異なりうります。これは最初の数値的な乖離を特定した*後*に限って許容されます。異なる乱数アルゴリズムや、同一の制御経路における乱数の抽出漏れが許されることは決してありません。
  欠陥を隠すために、シードを変えたり実行を平均したりしてはなりません。
- 長い(20ステップの)実行は**再現性**(同じ実装、同じシード)で検査し、短い先頭部分は独立したリファレンスのチェックポイントに対して検査します。
- リファレンス環境: Linux x86_64(他のホストではDev Container)。macOSは移植性の確認です。JuliaのリファレンスにはJulia 1.13.1と
  `extern/Julia-mVMC/Manifest-v1.13.toml` を使用します。生成したフィクスチャとともにJulia/BLASのバージョンを記録してください。C由来の期待値は別途生成して
  `tests/fixtures/` に出所情報とともに登録します。通常のRustのテストはCやJuliaを呼び出さず、`c_toolbox/` も読み込みません。

## 11.5 未解決の観測事項

本マニュアルの執筆中に気づいた項目で、別々のissueで扱う予定のものです。「観測」とは `mvmc` のリリースビルドで再現したことを、「コード読解」とはソースで確認したが実行はしていないことを意味します。

1. **2電子更新、複素数の通常経路。** Cは4つのすべての2電子更新関数で `rsbOld = raOld + t*Nsite` を定義しています(`pfupdate_two_fcmp.c:227`、`pfupdate_two_real.c:227`、
   `pfupdate_two_fsz.c:232`、`pfupdate_two_fsz_real.c:218`)。すなわち、古い行列要素 $M^{\rm old}_{ab}$ の第2添字に*電子 a の古いサイト*を用います。Rustの*実数*版の更新は
   これを意図的に保っています(`crates/mvmc-core/src/sampling/updates.rs:588`)が、Rustの*複素数の通常*版の更新
   `update_m_all_two_complex_flat` は、Juliaと同様に電子 b の古いサイトを用います(`crates/mvmc-core/src/sampling/updates.rs:537`)。これが複素数の交換更新の数値に影響するかどうかは**検証していません**(コード読解)。
2. **PhysCalはネームリストのセクションをチェックしません。** `validate_phys_cal` には `validate_para_opt` のキーワードチェックがありません。`namelist.def` に `BF bf.def` の行があると、最適化は `unsupported namelist section BF` で停止しますが、
   `--physcal` は最後まで実行され、BackFlowは適用されません(観測)。未知のキーワード、`SpinJastrow`、未知の `In…` キーワードも、同様にPhysCalでは拒否されません。
3. **最適化における `TwoBodyGEx`** は C と同様に受理され無視されます(#349 までは古い「issue #30」メッセージで拒否されていました)。
4. **`--help` のテキスト**は、既定の出力ディレクトリが「namelist parent dir」であると述べていますが、コードは `<namelist parent>/output` を使います(観測)。
5. **`--mode`** は検証されるだけのラベルであり、一致しないラベル(たとえば実数モデルでの `--mode fsz`)も黙って受け付けられます(観測)。
6. **タイマーのファイル名**は `CDataFileHead` を無視します([9.5](09-output-files.md#95-タイマー))。
7. **直接法ソルバーの `zvo_SRinfo.dat`** は書き出されません([9.4](09-output-files.md#94-ソルバー情報-zvo_srinfodat))。
8. **実数モードでのOptTransの微分のレイアウト**(決定は #370 に記録)。C の `calculateOptTransDiff`(`vmccal.c:639`)は `double complex *` のポインタオフセット付きで呼ばれる(`vmccal.c:466`)ため、微分 *i* はパラメータスロット *i* ではなく複素スロット `offset + i` に書かれます。`NQPOptTrans = 3` の実数モード実行では、ネイティブCのステップ1オペランド(`tests/fixtures/c_order_sr_operands/opt_real-cg-store0.txt`、観測)で、微分1が欠落し、微分2が微分1のスロットに入り、最後のスロットは0になります。Rust の `opt_trans_diff`(`observables.rs:957`)は数学的に正しいレイアウトを保ち、`real_mode_opttrans_derivatives_use_their_own_slots`(`sum_i w_i O_i = 1`)で固定されています。この欠陥は tmisawa/Julia-mVMC#55 として報告済みで、各オペランド配列の影響を受ける末尾2要素はC比較から除外しています([NUMERICAL_COMPARISONS](../../NUMERICAL_COMPARISONS.md))。複素モードは調べていません。
9. **Lanczos法の失敗時の出力**が異なります([6.2](06-theory-observables-lanczos.md#62-1ステップ-lanczos-波動関数))。
10. **Rustの `modpara.def` の既定値**はCと異なります([7.2](07-input-files.md#72-modparadef))。したがって、キーを省略した実行の結果は、2つのプログラム間で互換ではありません。
11. **負の `DSROptStepDt`** は、Cでは注記つきで正のステップに変換されます(`readdef.c:746-752`)が、Rustではそのまま使われます(コード読解、`crates/` に `SRFlag` の使用なし)。
12. **`Nelectron`/`Ne` はRustの `modpara.def` パーサーに無視されます**(観測: `Nelectron 8` を持ち `Ncond` のないモデルは `Nelec=0` と報告し、「normal initialization precondition: ...」で失敗します)。Cマニュアル自身の例は `Nelectron` を使っています。
13. **Cは実数モードでRBMパラメータを無視します**(#379、tmisawa/Julia-mVMC#59 として報告済み)。実数モデル(軌道が `ComplexType 0`)がRBMセクションを宣言しても、Cは `FlagRBM=1` を黙って受理しますが、実数用サンプラー `vmcmake_real.c` にはRBMのコードがなく(`RBM` の出現は0回。複素用の `vmcmake.c` には27回)、RBMの重みは一度も適用されません。`tests/fixtures/c_orbital_inputs/namelist_rbm_real.def`(シード1、ステップ1)での観測:Cは NPara = 55(NProj 7、NRBM 36、NSlater 12)、ステップ1のエネルギー 5.984544891656925 を報告し、RBMの初期値を 0.125/-0.25 から全て0に変えても変化せず、RBM値を全て0にしたRustとビット単位で一致します。Rustは(エネルギー 6.228711216019723 のように)RBMの重みを適用します。RustはRBMの正しい数学を維持し、そのため `rbm_real` にはネイティブCのオペランド参照がありません。
14. **CはFSZ/一般軌道の実行でもRBMパラメータを無視します**(#397/#403、tmisawa/Julia-mVMC#59 に追記済み)。`vmcmake_fsz.c`、`calham_fsz.c`、`locgrn_fsz.c`、`calgrn_fsz.c`、`vmccal_fsz.c`、`slater_fsz.c` のいずれにもRBMのコードはなく(出現は0回。複素用の `vmcmake.c` には27回)、それでも `FlagRBM` は設定され、RBMパラメータは読み込まれ数えられます。そのためネイティブCはFSZ + RBM入力をRBMの重みなしでサンプリング・測定します(ネイティブCの `fsz_rbm_physcal` シナリオでの観測:`In*RBM*` オーバーレイを0にするとRustは 1.4e-14 で一致し、0でない値ではエネルギーがO(1)で異なります)。Rustは、FSZサンプラー(採択比、スピンを変える手の後に再構築するRBMカウンタ)、FSZハミルトニアン、FSZグリーン関数のすべてでRBM因子を一貫して適用します。#403以前は、RustのFSZサンプラーがCの欠陥をそのまま踏襲し、測定側は踏襲していませんでした。Rustは正しい数学を維持するため、0でないRBM値のFSZ + RBMにはネイティブCの数値参照がありません。値を0にしたオーバーレイのシナリオはネイティブCとの比較を保ち、`crates/mvmc-core/tests/issue403_fsz_rbm_sampler.rs` はサンプリングされた全ての手を、Pfaffian × 射影 × RBM の完全再計算と照合します。

## 11.6 本マニュアルで検証しなかったこと

MPIでの実行(ランチャーが利用できませんでした)、macOSでの挙動、`mvmc-cli` の `simd-backend` ビルド、`crates/mvmc-cli/examples` 配下のサンプルプログラム、OptTrans(`-o`)の実行、
FSZ/一般軌道の実行、RBMの実行、複素数モードの実行、および上記のいずれについてもCやJuliaとの数値的な整合性。[付録](appendix-checks.md)を参照してください。
