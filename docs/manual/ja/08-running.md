# 8. 実行

[目次](README.md) · 前へ: [7. 入力ファイル](07-input-files.md) · 次へ: [9. 出力ファイル](09-output-files.md)

## 8.1 `mvmc`コマンド

バイナリは `crates/mvmc-cli/src/main.rs` からビルドされます(`cargo build --release -p mvmc-cli` で
`target/release/mvmc` が得られます。`cargo run -p mvmc-cli -- ...` でも動作します)。書式:

```text
mvmc [options] <namelist.def> [initpara]
```

書式は C ドライバーの `vmc.out [option] NameListFile [OptParaFile]`(`vmcmain.c:714-723`)です。オプションは位置引数の後ろにも置けます。短いオプションはまとめて指定でき(`-bo`)、値は直後に続けても(`-F2`)次の語としても(`-F 2`)渡せます。C の `getopt` と同じです。

| オプション | 意味 | デフォルト |
|--------|---------|---------|
| `--nsteps <N>` | SR のステップ数。`NSROptItrStep` を上書きします | `modpara.def` の `NSROptItrStep`(最適化では $>0$ でなければなりません) |
| `--nsmp <N>` | 最終平均化ウィンドウ。`NSROptItrSmp` を上書きします(ステップ数に対して $\le$ を満たす必要があります) | `NSROptItrSmp` |
| `--out-dir <DIR>` | 出力ディレクトリ(存在しなければ作成されます) | `<directory of namelist.def>/output` |
| `--seed <N>` | RNG のシード。`RndSeed` を置き換えます(グループ/ランクのオフセットは引き続き加算されます。[4.6](04-theory-sampling.md#46-サンプラー内の並列化)) | `RndSeed` |
| `--mode real\|cmp\|fsz` | *検査されるラベル*: 数値モードを**選択しません**。数値モードは入力宣言から決まります([3.3](03-theory-wavefunction.md#33-実数モードと複素数モード))。推定されたモードと矛盾するラベルはエラー(終了ステータス 2、ファイル作成前)です。推定と重複しますがスクリプトのために残してあります | 推定(一般軌道なら `fsz`、複素の宣言があれば `cmp`、それ以外は `real`) |
| `--initial-def auto\|none\|PATH` | 初期パラメータファイル([7.4](07-input-files.md#74-初期パラメータ値)): `auto` は隣接する `initial.def` があれば読み込み、`none` は読み込まず、`PATH` は存在が必須です | `auto` |
| `-o`, `--opt-trans` | C の OptTrans モードを有効にします(C ドライバーの `-o`) | off |
| `-b` | バイナリのパラメータ出力: `zvo_var` テキストファイルの代わりに `zvo_varbin_NNN.dat` を書き出します([バイナリ出力](#バイナリ出力-b)) | off |
| `-F <N>`, `--flush-interval <N>` | `_time_`/`_SRinfo` ファイルを `N` ステップごとにフラッシュします。`N < 1` はエラーです([9.5](09-output-files.md)) | 1 |
| `-e`, `--expert` | Expert モード(既定)。受理されますが何もしません | on |
| `-v`, `--version` | `mvmc-rs version <crate version> (follows C mVMC 1.3.0)` を標準出力に出して終了ステータス `0` で終了します | |
| `-h`, `--help` | 使用法(C のオプション一覧と Rust 拡張)を C と同じく標準エラー出力に表示し、終了ステータス `0` で終了します | |
| `-s`, `--standard` | Standard モード: StdFace 入力から Expert ファイルを `--out-dir`(既定はカレントディレクトリ)に生成し、続けて `namelist.def` を実行します([7.6](07-input-files.md#76-standard-モードstdface)) | オフ |
| `--dry-run` | StdFace 入力から Expert ファイルを生成して停止します(C の `vmcdry.out`) | オフ |
| `-m <N>` | MultiDef モード: `mvmc -m N DirListFile NameListFile [OptParaFile]` は MPI グループとディレクトリごとに `N` 個の独立した計算を実行します([MultiDef モード](#multidef-モード-m)) | オフ |
| `--physcal <PATH>` | `NVMCCalMode=1` における位置引数 `initpara`(固定パラメータファイル。[8.2](#82-固定パラメータでの物理量計算))の別名です。`NVMCCalMode=0` ではエラーで、位置引数のファイルと併用してもエラーです | – |
| `--physcal-trace <NEW_DIR>` | 入力を消費しないシリアル PhysCal 診断(PhysCal 実行の段階ごとの記録)を*新しい*ディレクトリ `NEW_DIR` に書き出します。`NVMCCalMode=1` と明示的なパラメータファイル、単一プロセスでの起動が必要です | off |

終了ステータス: `0` は成功、`1` は入力/検証/実行時エラー(メッセージは `error:` で始まります)、`2` は使用法エラー
(不明なフラグ、値の欠落、引数個数の不一致)です **(観測)**。デフォルトの出力
ディレクトリは `<namelist parent>/output` で、`--help` もそのように表示します(以前は "namelist parent dir" と書かれていました)。C ドライバーは作業ディレクトリ相対の `output/` に書き出しますが、Rust のデフォルトは namelist 相対です。

C の `getopt` 文字列は `"bhm:oF:esv"`(`vmcmain.c:83`)です。すべてのオプションが実装されています。`-F` は C の `strtol` の規則に従います: 数字がない、または `int` の範囲外の値はエラー、数値の後ろの余分な文字は警告のみ、`N < 1` はエラーです。

### 位置引数 `initpara` ファイル

C(`vmcmain.c:177-182`, `:252-260`)と同様に、省略可能な 2 番目の位置引数は `zqp_opt.dat` 形式のパラメータファイルで、その役割は `NVMCCalMode` で決まります。

- `NVMCCalMode=0`: **初期**パラメータ。初期化の乱数の後、`In*` のオーバーレイの前に読み込まれます(C の `InitParameter` → `ReadInitParameter` → `ReadInputParameters`)。Rust では `--initial-def <path>` として読み込みます。両方を指定するとエラーで、ファイルが存在しない場合もエラーです(C はメッセージを出して続行します)。
- `NVMCCalMode=1`: **固定**パラメータ([8.2](#82-固定パラメータでの物理量計算))。`--physcal <PATH>` は別名です。

### MultiDef モード(`-m`)

`mvmc -m N DirListFile NameListFile [OptParaFile]` は C の `vmc.out -m N` オプション(`initMultiDefMode`, `vmcmain.c:727-800`)の移植です。1 回の MPI 起動の中で `N` 個の独立した計算を、それぞれ別のディレクトリと別の定義ファイルで実行します。

- **分割。** `size` 個のランクの world を `N` グループに分けます。`div = size / N`, `mod = size % N`, `threshold = (div+1)*mod` として、ランク `r` は `r < threshold` なら `r / (div+1)`、そうでなければ `mod + (r - threshold) / div` 番のグループに属します。先頭の `mod` グループは `div+1` ランク、残りは `div` ランクです。グループのコミュニケーターがその実行の world 全体(`comm0`)になります。グループ内のランク 0 が出力ルートでコンソールのバナーを表示し、`NSplitSize`(およびシードのオフセット `RndSeed + comm1 のグループ`、[4.6](04-theory-sampling.md#46-サンプラー内の並列化))はグループ内で働きます。グループ番号そのものはシードに入らないので、同じ入力を同じ幅で実行する 2 つのグループは同じチェインになります。
- **ディレクトリ。** ランク 0 が `DirListFile` の空白区切りの先頭 `N` 個の名前を読みます(C の `fscanf("%s")`。パスは起動ディレクトリ相対)。グループ `g` は `g` 番目のディレクトリに入ります。`NameListFile` と `OptParaFile` はグループのディレクトリ**内**で解決され、既定の出力ディレクトリと相対の `--out-dir` も同様です。C はランクごとに 1 プロセスなので `chdir` でプロセスの作業ディレクトリを変えます。`mvmc` も同じ(ランクごとに 1 プロセス)です。
- **`-e`/`-s`。** C と同様に `-e` と `-s` は複数定義フラグを解除するので、`-m 2 -e a b` は `a` を namelist として読み、`-e -m 2 dirs a` では `-m` が有効のままです。
- **メッセージと終了ステータス**(終了ステータス `1`。C の `exit(EXIT_FAILURE)`/`MPI_Abort` と同じ): `error: -m: N should be smaller than MPI size.`(world が `N` より小さい。シリアル起動は 1 ランクの world なので `-m 1` だけが動きます)、`warning: load imbalance. MPI_size=<size> nMultiDef=<N>`(ランク 0、`size % N != 0`)、`error: DirListFile does not exist.`、`error: <file> is incomplete.`(名前が `N` 個未満)、`error: chdir(): <dir>: <strerror>`。引数個数のエラーは Rust の使用法エラーのステータス `2` です(C は使用法を表示して `1` で終了します)。オプションの `mpi` フィーチャーでは、失敗時に全ランクが集団的に終了します。

C との相違(いずれも C 側の欠陥または未定義動作): `N <= 0` は `error: -m: N should be a positive integer.` で拒否されます(C は `N` で割るため、`0` では `SIGFPE`、`N < 0` では無効なコミュニケーターのカラーになります)。グループの全ランクがディレクトリを移動しますが、C はグループのランク 0(`group2 == 0`)だけが移動します(ファイルを読むのがそのランクだけなので C ではそれで足ります)。ランク 0 での失敗(リストファイル、ディレクトリ)は、以降の処理の前に全ランクを止めます。C では `MPI_Abort` が非同期なので、他のランクは初期化されていないディレクトリ名のまま続行します(`tests/fixtures/multidef_348/README.md`)。

参照: ネイティブ C の `vmc.out -m 2` を 2、3、4 ランクで実行したもの(入力の異なる 2 グループ。`tests/fixtures/multidef_348/`、`c_toolbox/multidef_348/generate_c_runs.sh` で再生成)。分割の算術は、すべての `1 <= N <= size <= 48` について C の式そのものと照合しています。明示的な MPI ゲート `multidef_groups_match_own_single_runs_and_c_fixture`(`scripts/run_explicit_mpi_gates.sh`)は、各グループの出力が、そのディレクトリをグループの幅で単独実行した結果(スケール差 `<= 1e-12`)および C のフィクスチャ(`<= 1e-9`、#349 の PhysCal の許容値)と一致することを確認します。

### バイナリ出力(`-b`)

`-b`(C の `FlagBinary`, `vmcmain.c:654-660`, `initfile.c:58-66`, `:82-90`)では `zvo_var` テキストファイルは**書き出されず**、他の出力ファイルは変わりません。パラメータは `zvo_varbin_NNN.dat`(`NNN` = `NDataIdxStart`。最適化では 1 ファイル、PhysCal ではサンプルごとに 1 ファイル)に書き出されます。

| バイト | 内容 |
|-------|---------|
| 0–3 | ネイティブエンディアンの `int32` `NPara` |
| 4–7 | ネイティブエンディアンの `int32` `NSROptItrStep`(PhysCal では `1`) |
| 以降、ステップごとに | ネイティブエンディアンの `f64` を `2*NPara` 個: 全パラメータを (re, im) の組で |

**C との意図的な相違(C の不具合)。** C は `double complex` 配列に対して `fwrite(Para, sizeof(double), NPara, ...)` を呼ぶため、ブロックには交互配置された記憶領域の先頭 `NPara` 個の double、すなわち先頭 `ceil(NPara/2)` 個のパラメータ(`NPara` が奇数なら最後の 1 つは実部のみ)しか入らず、残りのパラメータは書き出されません。`mvmc` は C のヘッダー(`NPara`、ステップ数)を保ったまま、完全な `2*NPara` 個の double のブロックを書き出します。したがってファイルサイズは `8 + steps*2*NPara*8` バイト(C は `8 + steps*NPara*8`)で、すべてのパラメータが含まれ、C のブロックは `mvmc` のブロックの先頭 `NPara` 個の double と一致します。`NPara` をブロック長として使う C ファイル用のリーダーは `mvmc` のファイルを誤って解釈するため、ステップごとに `2*NPara` 個を読んでください。未改変の C `vmc.out` で検証済みです(`tests/fixtures/issue347_varbin/PROVENANCE.md`、許容の方針は [`docs/NUMERICAL_COMPARISONS.md`](../../NUMERICAL_COMPARISONS.md)): ヘッダーはバイト単位で一致し、PhysCal のパラメータと最適化のステップ 0 のブロックは先頭 `NPara` 個の double で厳密に一致し、以降の最適化ブロックは約 `5e-10` で一致します(SR の求解における演算順序。テストの許容は `1e-8`)。

### 計算の選択

計算の選択は C ドライバーとまったく同じく、`modpara.def` の `NVMCCalMode` によって行われ
(`vmcmain.c`, `main`: `NVMCCalMode==0` → `VMCParaOpt`、`==1` → `VMCPhysCal`、それ以外はエラー)、コマンドラインオプションはこれと一致していなければなりません。
解析後(初期化やファイル作成の前に、MPI ランク全体で集団的に)、CLI は次の確認を行います。

| `NVMCCalMode` | `--physcal` | 結果 |
|---------------|-------------|--------|
| 0 | なし | パラメータ最適化(位置引数のファイル = 初期パラメータ) |
| 1 | あり/なし | PhysCal(位置引数のファイルまたは `--physcal` = 固定パラメータ。なし: C の `InitParameter` の乱数、続いて `In*` のオーバーレイと同期) |
| 0 | `--physcal` 指定 | エラー: "`--physcal` requires NVMCCalMode=1 in ModPara" |
| その他 | – | エラー: "unsupported NVMCCalMode=… the CLI supports 0 (optimization) and 1 (PhysCal)" |

この振り分け規則は PR #340(`select_calculation`, `crates/mvmc-cli/src/main.rs`)で導入されました。CLI は振り分けの*前に*入力を解析・
検証するため、`NVMCCalMode=1` の入力でのパラメータ最適化の実行は `select_calculation` のメッセージで拒否されます。検証メッセージ "cannot run parameter optimization; use fixed-parameter PhysCal"
(`validate_para_opt`)は、ライブラリレベルでの対応物です。

> **実装**
> - C: `main` — `extern/mVMC-1.3.0/src/mVMC/vmcmain.c:46`
> - C: `VMCParaOpt` — `extern/mVMC-1.3.0/src/mVMC/vmcmain.c:331`
> - C: `VMCPhysCal` — `extern/mVMC-1.3.0/src/mVMC/vmcmain.c:531`
> - Rust: `main` — `crates/mvmc-cli/src/main.rs:173`
> - Rust: `parse_c_int` — `crates/mvmc-cli/src/main.rs:147`
> - Rust: `select_calculation` — `crates/mvmc-cli/src/main.rs:901`
> - Rust: `run_with_selected_backend` — `crates/mvmc-cli/src/main.rs:1133`
> - Rust: `run_physcal_with_selected_backend` — `crates/mvmc-cli/src/main.rs:917`
> - Rust: `prepare_physcal` — `crates/mvmc-cli/src/main.rs:1086`
> - Rust: `output_data` — `crates/mvmc-core/src/io.rs:93`
> - Rust: `run_para_opt_from_namelist` — `crates/mvmc-core/src/run.rs:1444`
> - C: `initMultiDefMode` — `extern/mVMC-1.3.0/src/mVMC/vmcmain.c:727`
> - Rust: `init_multi_def` — `crates/mvmc-cli/src/main.rs:808`
> - Rust: `group_of_rank` — `crates/mvmc-core/src/multidef.rs:16`
> - Rust: `split_multi_def` — `crates/mvmc-core/src/mpi.rs:137`
> - 整合性: `main` の「定義ファイルの読み込み → メモリ設定 → パラメータ初期化(RNG は `RndSeed + group` でシード) → `InitFile` → 実行 → タイマーの書き出し」という順序は `run_para_opt_from_namelist` に踏襲されています。C ドライバーの `-m` オプションは [MultiDef モード](#multidef-モード-m) として移植されています(#348)。

### コンソール出力

出力ルート(ワールドランク 0)のみが出力します。パラメータ最適化の実行では、モデルのバナー、解決されたパス、
サマリーが出力されます。チュートリアルの実行での正確なテキストは[第10章](10-tutorial.md)に示します。サマリーのフィールド:

- `Completed N SR steps in T s` — 実行フェーズの実時間。
- `Final energy / site` — **最後のステップ**の $\langle H\rangle$ の実部を `Nsite` で割ったもの(ノイズを含みます。1 ステップのみ)。
- `Final-window means (n steps): [a, b]` — `zvo_out.dat` の最後の `n = NSROptItrSmp` 行にわたる $\langle H\rangle$ の実部と虚部の平均。

バナー行 `mode : NVMCCalMode=...` は `modpara.def` から読み込んだ値を表示します。

## 8.2 固定パラメータでの物理量計算

```bash
mvmc namelist.def zqp_opt.dat --out-dir phys    # --physcal zqp_opt.dat と同じ
```

`modpara.def` では `NVMCCalMode 1` とします。パラメータファイルは、最適化の実行で書き出された `zqp_opt.dat`(または
`initial.def` 形式のファイル。[7.4](07-input-files.md#74-初期パラメータ値))です。C と同様に省略可能で、`NVMCCalMode 1` で `mvmc namelist.def` とすると初期化の乱数によるパラメータで測定します(C で検証済み。`tests/fixtures/issue347_varbin`)。動作(`prepare_phys_cal_from_namelist`, `vmc_phys_cal_in_place_timed`):

1. Expert 入力が解析され、PhysCal 用に検証されます([7.5](07-input-files.md#75-サポートされる入力と拒否される入力))。指定されたパラメータファイルが存在しない場合はエラーです("fixed parameter file not found")。
2. 固定パラメータは、乱数を**消費する前に**読み込まれます(テスト `physcal_preparation_loads_fixed_parameters_before_rng_consumption`)。続いてオーバーレイと同期が行われ、その後 `UpdateSlaterElm` が実行されます。
3. `NDataQtySmp` 個のサンプルそれぞれについて、マルコフ連鎖のサンプリング、測定、ランク間の平均化が行われ、番号付きの 1 組のファイルとして
   `zvo_*_NNN.dat` が書き出されます(`NNN = NDataIdxStart + sample`、書式は `%03d`。開始値が負の場合は例えば `-01` と表示されます)。
4. パラメータは変更**されません**。各サンプルの `zvo_var_NNN.dat` には、読み込んだ値がそのまま繰り返し記録されます。

`--nsteps`/`--nsmp` は PhysCal には影響しません。コンソールのサマリーは `Completed K PhysCal samples in T s` です。

### 診断: `--physcal-trace`

`--physcal-trace <NEW_DIR>` は `NEW_DIR`(存在してはいけません)を作成して `schema.txt` と `request.txt` を置き、続いて
`fixed-loaded`, `overlaid`, `synchronized`, `seeded`, `initialized-clone`, `sample-0`, `sample-1`, ... の順に、段階ごとのサブディレクトリを 1 つずつ作成します。
各段階のディレクトリには、`parameters.txt`, `resolved.txt`, `settings.txt`、RNG が*次に生成するはずの* 624 ワード
(`next624.txt`。実行を乱さないようクローンから取得します)と `draw-count.txt`、そしてサンプリング段階では電子
配置の配列(`ele_idx.txt`, `ele_cfg.txt`, `ele_num.txt`, `ele_proj_cnt.txt`, `ele_spn.txt`, `counter.txt`)が入ります。実行の最後に
`stages.txt` と `terminal.txt` が書かれます。記録は Rust の境界に従っており、"not C chronological replay"(C の時系列の再現ではない)です
(`crates/mvmc-cli/src/physcal_trace.rs`)。整合性のデバッグ用であり、本番用の出力ではなく、マルチランクでの起動では拒否されます。

> **実装**
> - C: `VMCPhysCal` — `extern/mVMC-1.3.0/src/mVMC/vmcmain.c:531`
> - C: `InitFilePhysCal` — `extern/mVMC-1.3.0/src/mVMC/initfile.c:72`
> - Rust: `prepare_phys_cal_from_namelist` — `crates/mvmc-core/src/run.rs:517`
> - Rust: `vmc_phys_cal_in_place_timed` — `crates/mvmc-core/src/run.rs:840`
> - Rust: `read_opt_para_file` — `crates/mvmc-core/src/initial_params.rs:141`
> - 整合性: `vmc_phys_cal_in_place_timed` は、C の PhysCal 分岐が暗黙に行うとおり、作業用コピー上で `vmc_calc_mode = 1` を強制します(`run.rs:811`)。

## 8.3 スレッドとBLAS

`mvmc-rs` は OpenMP を使用しません。サンプラーと測定のループはデフォルトでは逐次実行されるため、マルコフ連鎖は
C コードとまったく同じように乱数を消費します。独立な作業項目に対するオプションの共有メモリ並列化(`calc_m_all_*` のパフィアン設定における射影セクター、
`OO`/`HO` の累積と保存されたグラム積の行、実数波動関数に対するローカルエネルギーの遷移項、対角項・PairHop・Exchange・InterAll のエネルギー項、`update_m_all_*`/`calculate_new_pf_m*` の QP ループ、Slater 要素の平面、doublon-holon カウンター、RBM の隠れユニット、SR 行列の構築と CG のベクトル更新、グリーン関数の要素)
は
`MVMC_RS_INNER_THREADS`([8.5](#85-環境変数))で有効になります。これはマルコフ連鎖や、各結果が形成される順序を変えません。
密な線形代数(`dgemv`, `dpotrf`, パフィアンカーネル)は OpenBLAS で実行され、そのスレッド数は通常の
`OPENBLAS_NUM_THREADS`/`OMP_NUM_THREADS` 変数で制御されます(`mvmc-rs` は読み取りません。プロジェクトのベンチマークでは 1 に固定しています)。

## 8.4 MPIとグループ実行

### ビルドと起動

```bash
cargo build --release -p mvmc-cli --features mpi
mpirun -np 4 target/release/mvmc namelist.def --out-dir out          # 4 independent chains
mpirun -np 8 target/release/mvmc namelist.def --out-dir out          # with NSplitSize 2 in modpara.def: 4 groups of 2 ranks
```

起動は、MPI が初期化される前に環境から検出されます: `MVMC_RS_MPI_RANK`/`MVMC_RS_MPI_SIZE`(明示的なテスト用変数)、
`OMPI_COMM_WORLD_RANK`/`_SIZE`(Open MPI)、`PMI_RANK`/`PMI_SIZE`(MPICH/PMI)、`PMIX_RANK`/`PMIX_SIZE`
(`LaunchContext::from_env`, `crates/mvmc-core/src/parallel.rs:20`)。`world_size > 1` で、かつバイナリが `mpi` フィーチャーなしでビルドされている場合、実行は
"MPI launcher detected; rebuild mvmc-cli with --features mpi to enable MPI execution" というメッセージで停止します。フィーチャーがある場合は、すべてのランクが
入力を解析・検証し、CLI はオプションと `ModPara` の値がランク間で一致していることを集団的に確認します
(`agree_controls`: "CLI run controls differ between MPI ranks")。したがって、すべてのランクが入力ファイルへの読み取りアクセスを必要とします。
MPI の実行は本マニュアルの執筆中には実行して**いません**(MPI ランチャーが利用できませんでした) **(未検証)**。ここでの記述は、コードと
`crates/mvmc-core/src/run_mpi_tests.rs` の `mpi` フィーチャーのテストに基づきます。

### コミュニケータと`NSplitSize`

ワールドコミュニケータは、C ドライバーと同様に分割されます(`vmcmain.c:239-256`)。

- `comm1`: ランク `0..NSplitSize-1`、`NSplitSize..2*NSplitSize-1`、... 各グループは**1 本のマルコフ連鎖**を扱います。`group = rank / NSplitSize` です。
  `NSplitSize` がワールドサイズを割り切らない場合、最後のグループは小さくなります(C は負荷不均衡の警告を出しますが、Rust は受け入れます)。
- `comm2`: グループ内で同じ位置にあるランク。6 つのサンプラー統計カウンタのリダクションにのみ使われます。
- [第2章](02-theory-vmc-hamiltonian.md)と[第5章](05-theory-sr.md)のすべての累積量は、**ワールド全体**で合計され、全
  重み $W$ で割られます。したがって、有効なサンプル数は $(\text{number of groups})\times\texttt{NVMCSample}$ です。

`NSplitSize = 1`(デフォルト)では、すべてのランクがそれぞれ 1 つのグループとなります。各ランクはシード
`RndSeed + rank` で、自身の `NVMCSample` 個の配置をサンプリングして測定します。`NSplitSize = n > 1` では、グループの `n` 個のランクが同じシードを使い、同一の乱数列を生成します。射影セクター
$[0,N_{\rm QP})$ はパフィアン更新のためにそれらの間で分割され(`SplitLoop`, `partition_range`)、グループの保存済みサンプルは測定のためにそれらの間で分割されます。
$\mathrm{IP}$ はグループ内でリダクションされます。したがって、グループ実行は `NQPFull` が大きい場合(スピン/運動量射影)に有用です。

### 制約

Rust が初期化の前に拒否するのは、C で未定義の唯一のグループ組合せだけです(`validate_grouped_runtime`, [7.5](07-input-files.md#75-サポートされる入力と拒否される入力))。

- CG ソルバー(`NSRCG != 0`)での `NSplitSize > 1`。`VMCMainCal` は `sqrt(w) O` をグローバルなサンプル位置に保存しますが(`vmccal.c:241,248`)、
  `calculateOO_Store` は基底ポインタから先頭の `sampleEnd - sampleStart` 列を読むため(`vmccal.c:314-318`)、最初以外のランクは未書き込みの `malloc` メモリを読みます(`setmemory.c:419-421`)。
  同じ欠陥により C では `NStore != 0` かつ `NSplitSize > 1` の直接SRも壊れます。Rust はこの場合を正しく計算するため、C と異なります。

サポート(ネイティブ C の 2/4 ランクと照合、`tests/fixtures/grouped_nsplit_349/`): 任意の `NSplitSize` での直接SR(`NSRCG = 0`)、`NSplitSize = 1` でのSR-CG(`NSRCG = 1`)、
sz 保存・FSZ/一般軌道・任意の `NQPFull` での PhysCal と最適化、`NQPOptTrans > 1` の OptTrans、`NLanczosMode = 1, 2` の PhysCal。

### 各ランクの書き出し内容

すべての出力ファイルは**ワールドランク 0 のみ**が書き出します(`Reducer::is_output_root`, `crates/mvmc-core/src/reducer.rs:134`。グループ化されたコミュニケータでは
グループ 0 のローカルランク 0 で、同じプロセスです)。他のランクは計算とリダクションには参加しますが、書き出しは行いません。明示的な `--out-dir` は
ルートだけがアクセスするため、ランクごとにローカルでもかまいません(`RunConfig` のドキュメント, `run.rs:1206`)。コンソール出力とタイマーファイルも同様です。
失敗は集団的に合意されるため、いずれかのランクが失敗すると、ハングせずにすべてのランクが停止します。

> **実装**
> - C: `main` (communicator split, `init_gen_rand(RndSeed+group1)`) — `extern/mVMC-1.3.0/src/mVMC/vmcmain.c:46`
> - C: `SplitLoop` — `extern/mVMC-1.3.0/src/mVMC/splitloop.c:31`
> - Rust: `MpiContext::split_groups` — `crates/mvmc-core/src/mpi.rs:154`
> - Rust: `MpiContext::initialize` — `crates/mvmc-core/src/mpi.rs:70`
> - Rust: `assign_group` — `crates/mvmc-core/src/parallel.rs:67`
> - Rust: `partition_range` — `crates/mvmc-core/src/parallel.rs:88`
> - Rust: `validate_grouped_runtime` — `crates/mvmc-core/src/validation.rs:23`
> - Rust: `run_para_opt_from_namelist_with_reducer` — `crates/mvmc-core/src/run.rs:1457`
> - Rust: `reduce_accumulators` — `crates/mvmc-core/src/run.rs:1750`
> - 整合性: コミュニケータの幅は `vmcmain.c:239-256` に従います(`NSplitSize` はコミュニケータの*幅*であり、連鎖の本数ではありません)。サンプルの範囲は `SplitLoop` に従います。C のグリーン関数のリダクションはランク 0 のみに集約されますが、Rust は累積量を all-reduce でリダクションしてルートが書き出すため、ファイルの内容は同じになります。

## 8.5 環境変数

| 変数 | 読み取る側 | 効果 |
|----------|---------|--------|
| `MVMC_NSTEPS` | CLI, examples | `--nsteps` と同じ(フラグが優先されます) |
| `MVMC_C_TIMER` | CLI/core (`TimerEnv`, `crates/mvmc-core/src/c_timer.rs:174`) | `0` 以外の任意の値で、C 互換のセクションタイマーが有効になります。`zvo_CalcTimer.dat` を書き出します([9.5](09-output-files.md#95-タイマー)) |
| `MVMC_TIMER` | 同上 | 非推奨のエイリアス。`MVMC_C_TIMER` なしで設定された場合、警告が `MVMC_C_TIMER=1` を推奨します |
| `MVMC_CALHAM1_DIAG`, `MVMC_SLATER_DIAG`, `MVMC_MAINCAL_DIAG`, `MVMC_WEIGHTAVG_DIAG` | 同上 | 対応する診断タイマー群を有効にします(ID は 966 まで)。いずれかを設定するとメインタイマーも有効になり、`zvo_CalcTimerDiag.dat` を書き出します |
| `MVMC_RS_INNER_THREADS` | `inner_thread_config` (`crates/mvmc-core/src/threading.rs:190`) | 独立な内部作業項目に対するワーカースレッド数。デフォルトは 1(逐次)。不正な値や 0 は 1 にフォールバックします。プロセスごとに一度だけ読み取られます。 |
| `MVMC_RS_INNER_THRESHOLD` | 同上 | ワーカープールを使用する最小の作業項目数。デフォルトは 32。不正な値や 0 は 32 にフォールバックします |
| `MVMC_RS_MPI_RANK`, `MVMC_RS_MPI_SIZE`, `OMPI_COMM_WORLD_*`, `PMI_*`, `PMIX_*` | `LaunchContext::from_env` | ランチャーの検出([8.4](#84-mpiとグループ実行)) |
| `JULIA_MVMC_ROOT`, `JULIA_MVMC_EXAMPLE_STEPS`, `MVMC_OUT_DIR` | `cargo run --example ...` プログラムのみ | `extern/Julia-mVMC` の入力の場所、ステップ数、出力ルート(デフォルトは `output/<model>/`) |
| `OPENBLAS_NUM_THREADS`, `OMP_NUM_THREADS`, ... | OpenBLAS(`mvmc-rs` は読み取りません) | BLAS のスレッド設定 |

元の issue で言及された Julia 固有のデバッグダンプ `MVMC_DEBUG_*` は Rust には実装されて**いません**。`MVMC_RS_PHASE`,
`MVMC_RS_CTEST_*`, `MVMC_RS_THREADED_*`, `MVMC_CG_DIAGNOSTICS`, `MVMC_GREEN_INDEX_CHILD` などの変数はテストコードにのみ現れ、ユーザー向けオプションではありません。

`MVMC_C_TIMER` と `*_DIAG` 変数の値は、リテラル文字列 `0` と比較されます。`MVMC_C_TIMER=0` で無効、それ以外の任意の文字列(空文字列を含む)で有効になります。

> **実装**
> - Rust: `TimerEnv::from_lookup` — `crates/mvmc-core/src/c_timer.rs:174`
> - Rust: `inner_thread_config` — `crates/mvmc-core/src/threading.rs:201`
> - Rust: `LaunchContext::from_env` — `crates/mvmc-core/src/parallel.rs:20`

## 8.6 再現性

同じバイナリを同じ入力、シード、プロセス配置で 2 回実行すると、バイト単位で同一の出力ファイルが得られます。チュートリアルの実行は
`--seed 5` で繰り返され、`cmp` は `zvo_out.dat` が同一であると報告しました **(観測)**。BLAS プロバイダー、プラットフォーム、スレッド配置が異なると、
浮動小数点の和の下位ビットが変わることがあり、メトロポリス判定を通じて後続の配置も変わりえます。これは想定された挙動であり、実装間の
比較で許容誤差を用いる理由でもあります([11.4](11-compatibility.md#114-数値比較ポリシー))。
