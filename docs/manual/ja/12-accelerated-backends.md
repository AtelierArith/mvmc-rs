# 12. アクセラレータ(GPU)バックエンド

[目次](README.md) · 前へ: [11. 互換性と相違点](11-compatibility.md) · 次へ: [付録: 引用のチェック方法](appendix-checks.md)

この章では、`mvmc-rs` のテンソル形状の処理に対するオプションの高速化実装(CPU 上の tenferro、および単体ワークスペース `gpu/mvmc-gpu-cuda` 経由の CUDA)について、ビルドと選択の方法、何がどの許容誤差で検証されているか、まだうまく動かないと分かっていることを説明します。設計記録は [docs/design/gpu-readiness.md](../../design/gpu-readiness.md)(issue #417、10〜15 節)、検証方針は [docs/NUMERICAL_COMPARISONS.md](../../NUMERICAL_COMPARISONS.md) の「Accelerated-backend validation (#424)」節、CUDA ゲートのディスパッチは [docs/OPTIONAL_GATES_DISPATCH.md](../../OPTIONAL_GATES_DISPATCH.md) にあります。ここに挙げる数値はそれらの文書から転記したもので、各文書が示すハードウェア(Linux x86_64、NVIDIA GeForce RTX 3060 を 2 枚、ドライバ 580.178.04、CUDA 12.9 ツールキット、tenferro 0.7.1、共有された 36 スレッドのホスト)で各文書の著者が測定したものです。このマニュアルのために**再測定はしていません**。

## 12.1 方針: 高速化パスとは何で、何ではないか

- **CPU の C 順序パスがデフォルトであり、基準(オラクル)です。** 高速化実装はすべて、ステージ単位のバックエンド trait の背後にある*追加の*実装です。CPU パスを置き換えたり、順序を変えたり、経路を変えたりすることはなく、デフォルトのビルド・テスト・フィクスチャは GPU にも tenferro の GPU クレートにも依存しません(設計 5.1、5.3)。
- **まず正しさ。** issue #417 に記録された方針決定は*選択肢 B* です。デバイス常駐サンプラ(PR #445)は検証済みの基盤として残しますが、現在は大規模な確率的再構成(GPU がすでに有利な領域)と CPU の最適化に注力しており、サンプラ側の GPU 作業は保留です。FP64 性能の高い GPU(A100/H100)で、issue #450 のベンチマークスイートのデータにより再評価します([12.9](#129-ベンチマークと検証スイート450))。本番のルーティングに触れるスループット作業(#452)は、正しさ優先の作業が許すまで保留です。
- **暗黙のフォールバックなし。** 未対応のステージ、dtype、デバイス、ビルドはエラー(`StageError::Unsupported`、`BackendError`)になり、CPU の結果で代用されることはありません。
- **RNG と Metropolis の判定はホスト側で厳密です。** SFMT の乱数、候補生成、採択判定、射影カウンタ、ファイル出力は、どのバックエンドでもホスト側に残ります。高速化バックエンドは重みや行列の*数値*を許容誤差内で変えることはありますが、乱数の消費順序と回数を変えてはなりません([12.5](#125-数値的保証と検証内容))。
- **明示的に選択します。** 自動選択はなく、デフォルトは `c-order` です。

## 12.2 何があるか

| 構成要素 | 場所 | 内容 | 通常の `mvmc` 実行に組み込まれているか |
|-----------|-------|--------------|---------------------------------|
| バックエンド選択とデバイスレポート | `crates/mvmc-core/src/backend.rs`(`BackendKind`、`DeviceReport`、`cuda_gate_decision`、feature `gpu-cuda`、#420) | CPU は常に利用可能。`Cuda(n)` は feature とプロバイダー登録がある場合のみ | いいえ(ライブラリ API) |
| 統一ステージバックエンド | `crates/mvmc-core/src/stage_backend.rs`、`sr_backend.rs`(#421、#437) | `SrStages`(Gram、S/g 組み立て、Cholesky 求解、CG 積、合成 `direct_step`/`cg_step`)と `PfaffianStages` を 1 つの `StageBackend` にまとめたもの | **SR ステージのみ**、`MVMC_RS_SR_BACKEND` 経由 |
| tenferro による SR | `sr_backend.rs` の `TenferroSr`(#421) | SR ステージを tenferro の `dot_general`/`cholesky`/`triangular_solve` で実行(CPU の `cpu-faer` または CUDA) | はい: `MVMC_RS_SR_BACKEND=tenferro`。`cuda[:N]` はプロバイダーを登録するプログラムから |
| バッチ Pfaffian と逆行列 | `crates/mvmc-gpu`(CPU 版、tenferro-native、`ExtensionOp`)と `gpu/mvmc-gpu-cuda`(CUDA カーネル)、#423 | `[n,n,NQP,B]` の歪対称平面の Pfaffian と逆行列(`f64` と `Complex64`、平面ごとの状態付き) | いいえ: ハーネスとゲートで使用。サンプラと測定は引き続き `calc_m_all_*` を呼ぶ |
| 検証ハーネス | `crates/mvmc-core/src/accel_validation.rs`(#424) | 教師強制リプレイ、判定反転検出器、再現性、ベンチマークメタデータ | テストとゲート用ツール |
| サンプルバッチ測定 | `crates/mvmc-core/src/measurement_batch.rs`(#422) | 独立なサンプルごとの作業の CPU 上での並べ替え。`MVMC_RS_MEASURE_BATCH` | はい、常時(CPU。出力はバッチサイズに依存しない) |
| マルチウォーカーランナー | `crates/mvmc-core/src/multichain.rs`(#425、#435) | 1 プロセス内の `W` 本の独立チェーン: 固定パラメータの PhysCal と、C 互換リダクション付きの最適化全体 | ライブラリ API(CLI フラグなし) |
| デバイス常駐サンプラ | `crates/mvmc-core/src/device_sampler.rs` と `gpu/mvmc-gpu-cuda/src/device_sampler.rs`(#434) | 状態を GPU 上に常駐させるロックステップのマルチウォーカー実数モードサンプラ | いいえ: 検証済みの基盤、保留中([12.8](#128-既知の制限)) |
| ピン留め/非同期転送ヘルパ | `gpu/mvmc-gpu-cuda/src/transfer.rs`(#432) | ピン留めメモリプール、非同期コピー、イベントによる順序付け | デバイスサンプラとデバイス常駐 SR が使用 |
| デバイス常駐 SR | `gpu/mvmc-gpu-cuda/src/sr_device.rs`、`stages.rs`(#447) | Gram、S/g、Cholesky、CG ループをデバイス上で実行(実数パラメータ) | **はい**(`MVMC_RS_SR_BACKEND=cuda[:N]`): 実数の直接 SR(`NStore != 0`)と実数・複素の CG。単一プロセス([12.4](#124-バックエンドの選択)) |

## 12.3 ビルドと実行

### デフォルトビルド(CPU のみ)

何も変わりません: `cargo build --release -p mvmc-cli`。GPU を必要としない高速化コード(`mvmc-gpu` クレート、CPU 上の tenferro、ホストのサンプラサービス、ハーネス)は通常のワークスペースの一部で、そのテストは通常の CI で実行されます。

```bash
cargo nextest run -p mvmc-gpu --cargo-profile test-fast          # バッチ Pfaffian の CPU 版とハーネス
cargo nextest run -p mvmc-core --cargo-profile test-fast \
  -E 'binary(multichain) | binary(multiwalker_sr_435) | binary(stage_backend_437) | binary(sr_backend_421) | binary(device_sampler)'
```

### `gpu-cuda` feature と単体ワークスペース

`mvmc-core` と `mvmc-cli` には feature `gpu-cuda` があります(`crates/mvmc-core/Cargo.toml`、`crates/mvmc-cli/Cargo.toml` で転送)。これは**依存を追加せず**、`Cargo.lock` も変更しません。`mvmc_core::backend` の `CudaProvider` 登録を有効にするだけです。したがって `cargo check -p mvmc-cli --features gpu-cuda` には CUDA ツールキットは不要で、**CUDA はリンクされません**。

CUDA の依存ツリー(`cuda` 付きの `tenferro-gpu`、cudarc、パッチ済みの `lru`)は単体ワークスペース `gpu/mvmc-gpu-cuda`(独自の `[workspace]` と `Cargo.lock` を持ち、ルートのワークスペースから除外)にあります(設計 10.1、10.2)。ルートのロックファイルに入れないのは意図的です。オプション依存として `tenferro-gpu` を追加すると、ロックファイルと、すべてのデフォルトビルドの監査対象が約 2000 行増えました。利用者への影響:

- GPU 関連は **`gpu/mvmc-gpu-cuda` からビルド**します(`cd gpu/mvmc-gpu-cuda && cargo ...`)。ルートで `cargo build --features gpu-cuda` としても CUDA にはリンクされません。
- プロバイダーは、CUDA を使いたいプログラムが `mvmc_gpu_cuda::install()`(`gpu/mvmc-gpu-cuda/src/lib.rs:69`)を呼んで登録します。標準の `mvmc` バイナリはこれを呼ばないため、そこで `MVMC_RS_SR_BACKEND=cuda` を指定すると使用法エラーで拒否されます([12.4](#124-バックエンドの選択))。**CUDA 対応のコマンドは `mvmc-cuda` です**: 単体ワークスペースの薄いバイナリ(`gpu/mvmc-gpu-cuda/src/bin/mvmc-cuda.rs`)で、`install()` を呼んだ後、変更のない `mvmc` ドライバ(`mvmc_cli::run_cli`)を実行します。したがってオプションは [8.1](08-running.md#81-mvmcコマンド) とまったく同じです(`mpi` feature は転送しません)。`gpu/mvmc-gpu-cuda` からビルドして実行します: `cargo build --release --bin mvmc-cuda`、続けて `MVMC_RS_SR_BACKEND=cuda target/release/mvmc-cuda namelist.def`。docker では `scripts/run_cuda_gate.sh` のイメージとマウントを使います。
- CUDA 実行の要件: CUDA ドライバ(`libcuda`)と、実行時に cuBLAS、cuSOLVER、NVRTC の CUDA ツールキットライブラリ(`dlopen` で読み込まれ、*ビルド*にはツールキット不要)。issue #417 の tenferro 調査では CUDA 12.6.2 が下限(完全な機能には 12.8)とされ、測定は 12.9 ツールキットで行われました。tenferro のリンクはホステッドランナーのディスクを使い切ることがあるため、単体ワークスペースは `debug = 0` にしています。

### CUDA ゲート(native または docker)

`scripts/run_cuda_gate.sh [native|docker] [追加の cargo-test 引数]` は、`gpu/mvmc-gpu-cuda/tests/` の ignore 指定されたゲートテスト(`cuda_gate`、`pfaffian_gate`、`transfer_gate`、`sampler_gate`、`sr_device_gate`)を `MVMC_RS_CUDA_GATE=1` でビルドして実行します。

```bash
scripts/run_cuda_gate.sh native                          # このホストにドライバと CUDA ツールキットライブラリが必要
MVMC_RS_CUDA_GATE=1 scripts/run_cuda_gate.sh docker      # NVIDIA CUDA ツールキットイメージ、--gpus all
MVMC_RS_CUDA_IMAGE=my/cuda:tag scripts/run_cuda_gate.sh docker
```

- `docker` コマンドがあれば `docker` がデフォルトです。デフォルトのイメージは `tenferro-benchmark-cuda:full-verify-20260822`(CUDA 12.9.2)で、`MVMC_RS_CUDA_IMAGE` で変更できます。ホストの `~/.rustup` と `~/.cargo` をマウントするため、イメージに Rust ツールチェーンは不要です。スクリプトはコンテナ内で(root として)OpenBLAS/LAPACK をインストールし、その後ユーザー ID に切り替えるので、`target/` のファイルの所有者はホストのままです。Cargo のターゲットは `gpu/mvmc-gpu-cuda/target` に置かれます。
- **フェイルクローズド。** `MVMC_RS_CUDA_GATE=1`(または `require`)でデバイスがなければハードな失敗です。変数が未設定でデバイスがなければ、テストは `cuda-gate: ExplicitSkip: skipped, no device (...)` を出力し、これは合格では*ありません*(`cuda_gate_decision`、`crates/mvmc-core/src/backend.rs:232`)。スクリプトは常にこの変数を `1` にします。
- **レポート。** 結果は `gpu/mvmc-gpu-cuda/results/` に書かれます(`cuda-gate.md`、`cuda-validation.md`、`cuda-roundtrip.md`、`cuda-transfer.md`。`MVMC_RS_CUDA_GATE_OUT`、`..._VALIDATION_OUT`、`..._ROUNDTRIP_OUT`、`..._TRANSFER_OUT` で変更可能。このディレクトリはバージョン管理の対象外)。各レポートには、デバイス、compute capability、メモリ、ドライバ(NVML)、CUDA ドライバ API、NVRTC、cuBLAS、cuSOLVER、tenferro のバージョンが記録されます(`DeviceReport`。取得できない項目は `unavailable` と表示)。
- **CI。** optional-gates ワークフローの入力 `cuda_gate` が、セルフホストランナー(`self-hosted, linux, x64, gpu`。ラベルは仮)でジョブ `cuda-gate` を起動します。これは bounded-family 台帳の外にあり、選択されなければ NotRun、デバイスがなければジョブは失敗します。ホステッドランナーには GPU がないので、デフォルトの CI がこれを実行することはありません([OPTIONAL_GATES_DISPATCH.md](../../OPTIONAL_GATES_DISPATCH.md) の「CUDA gate (#420)」)。

### 各構成要素のベンチマーク

各スクリプトは `native` か `docker` を取り、メタデータブロック(ホストのロードアベレージ、デバイスレポート)付きの CSV を書きます。

| スクリプト | 構成要素 | 出力 |
|--------|-----------|--------|
| `scripts/run_pfaffian_bench.sh` | バッチ Pfaffian/逆行列(#423) | `benchmark/gpu_pfaffian/results/pfaffian_batched.csv` |
| `scripts/run_device_sampler_bench.sh` | デバイス常駐サンプラと CPU マルチチェーンランナーの比較(#434)。`MVMC_RS_SAMPLER_CORES=0-3` でコンテナをホストの 4 コアに制限(`docker --cpuset-cpus`) | `benchmark/gpu_device_sampler/results/` |
| `scripts/run_sr_device_bench.sh` | デバイス常駐 SR ステップ(#447) | `benchmark/gpu_sr_device/results/` |

静かなホストで実行してください。CPU の行もデバイスの行のホスト側も、他の負荷に敏感であり、チェックイン済みの数値は共有ホストで取られています(設計 11.4、13.3、15.3)。

## 12.4 バックエンドの選択

`MVMC_RS_SR_BACKEND`([8.5](08-running.md#85-環境変数))は、本番が SR ステージに使うバックエンドを選びます。`selected_stage_backend`(`crates/mvmc-core/src/stage_backend.rs:411`)が読み、`parse_stage_backend`(`crates/mvmc-core/src/stage_backend.rs:370`)が解析します。

| 値 | バックエンド | 備考 |
|-------|---------|-------|
| 未設定、空、`c`、`c-order`、`corder`、`default` | C 順序 CPU(`COrderSr` + `COrderPfaffian`) | デフォルトであり整合性のオラクル。従来のインラインコードとバイト単位で同一 |
| `tenferro`、`tenferro-cpu` | CPU 上の tenferro eager 演算(`cpu-faer`)、`TenferroSr`。Pfaffian スロットは `Unsupported` | 実現性確認用の経路: 1 スレッドでは OpenBLAS/LAPACK より 1.1〜4 倍**遅い**(SYRK ではなく faer の GEMM、ステージごとのテンソルのアップロード/ダウンロード) |
| `cuda`、`cuda:N` | デバイス `N`(デフォルト 0)上の CUDA: デバイス常駐 SR ステップ(`ResidentCudaSr`。常駐ステップが扱わない部分は tenferro CUDA のステージ単位の経路)とバッチ CUDA Pfaffian カーネル | `mvmc-cuda` バイナリ(`gpu-cuda` feature と登録済みプロバイダー `mvmc_gpu_cuda::install()`)が必要 |

規則:

- **不正な値はエラーで、フォールバックはしません。** CLI は起動時に、ファイルを読み書きする前に一度だけセレクタを検証します(`stage_backend::validate_selected_stage_backend`)。不正な値(`MVMC_RS_SR_BACKEND is not one of c-order, tenferro, cuda[:N]`)や利用できないバックエンド(`gpu-cuda` feature なし、プロバイダー未登録、デバイス番号が範囲外、CUDA ランタイムライブラリがない)は、`error: MVMC_RS_SR_BACKEND: ...` を出力して、他のオプションエラーと同じ使用法エラーの終了ステータス `2` で終了します([8.1](08-running.md#81-mvmcコマンド))。出力ディレクトリは作られません。MPI では全ランクが合意に参加するので、全ランクが一緒に停止します。同じ事前検査が `MVMC_RS_MEASURE_PF_BACKEND`(測定の Pfaffian の生成元、[8.5](08-running.md#85-環境変数))も同じ契約で対象にします。この事前検査を経ずに本番コードに到達するライブラリ呼び出し(`selected_stage_backend`、`acquire`)は、同じメッセージで panic します。issue #464 より前は、CLI がここで panic して終了ステータス 101 でした。
- **ルーティングされる範囲。** SR ステージのみです。Gram 積(`observables.rs` の `finalize_oo_store_real` と複素版ファイナライザ)、S/g の組み立てと Cholesky 求解(`sr.rs`)、CG 積(`sr_cg.rs`)。サンプラと測定の Pfaffian(`calc_m_all_*`)は常に C 順序カーネルを使います。
- **デフォルト以外のバックエンドはプロセスごとに 1 回だけ構築され**共有されます(tenferro ランタイムや CUDA コンテキストは高価です)。定数の CG オペランドは、明示的なバージョンカウンタをキーにキャッシュされます。
- **`cuda` が実行するのはデバイス常駐 SR ステップです(issue #452)。** `ResidentCudaSr`(`gpu/mvmc-gpu-cuda/src/stages.rs`)が `SrStages` の常駐ステージ契約(`resident_direct`、`direct_begin`、`direct_assemble`、`direct_factor_solve`、`resident_cg`、`cg_step`)を実装し、プロバイダーがそれを本番へ渡します。
  - *実数の直接 SR*(`NSRCG = 0`、実数パラメータ、`NStore != 0`、単一プロセス): `run.rs` はホスト上の Gram `OO = O O^T` を作らなくなり、`sr.rs`(`stochastic_opt_real_resident`)は保存したサンプルストア `O`(`n x samples`、`n = 1 + NPara`)を**ステップごとに 1 回**アップロードします。デバイスが `G` を作り、ホストが受け取るのは `G` の対角と第 0 列だけです(`O(n)`。冗長成分のカットに十分)。`S`、`g`、Cholesky 因子はデバイスに残ります。ホストの `OO` に対して `weight_average_sr_opt_real` が行う `1 / wc` の重み正規化は、組み立てカーネルの中で Gram の各要素に適用されます(`DirectSolveInput::gram_scale`)。戻ってくるのは `x` だけです。SR オブザーバーを設置した場合に限り、キャプチャのために `S` と `g` をダウンロードします。
  - *CG の SR*(`NSRCG = 1`、実数または複素パラメータ、単一プロセス、CG オブザーバー未設置): `SampledSrOperator::solve_with_reducer`(`sr_cg.rs`)が `cg_step` を呼び、サンプル行列を 1 回アップロードして、ループ全体をデバイス上で実行します。
  - それ以外は**同じ選択バックエンド**のステージ単位の経路(tenferro CUDA のステージ)のままで、黙って CPU の結果になることはありません: 複素の直接 SR(複素 Gram とエルミート Cholesky は未実装)、複数ランクまたはグループ化サンプリング、`NStore = 0`(ホストがサンプルごとに `OO` を累積)、CG オブザーバー設置中の CG。
  - デフォルトの `c-order` 経路は変更されずバイト単位で同一です。常駐の分岐は、選択バックエンドが `resident_direct`/`resident_cg` を報告したときだけ取られます。
- **どのバイナリか。** `MVMC_RS_SR_BACKEND=tenferro mvmc ...` は標準のバイナリで動作します。`cuda[:N]` には `mvmc-cuda` が必要です([12.3](#gpu-cuda-feature-と単体ワークスペース))。**(観測)** デフォルト feature のリリースビルド(Linux x86_64)で `benchmark/hubbard_chain/inputs/hubbard_chain_L16`、`--nsteps 3 --nsmp 2` を実行: `tenferro` は `c-order` と同じファイル集合(`zqp_opt.dat`、`zqp_*_opt.dat`、`zvo_out.dat`、`zvo_var.dat`、`zvo_SRinfo.dat`、`zvo_time_001.dat`)を書き、`zvo_out.dat` の最初の 2 ステップは同一、3 ステップ目は末尾の桁が異なり、`zqp_opt.dat` の差は絶対値で最大 `5.7e-14` で、`tenferro` を 2 回実行するとバイト単位で一致しました。`mvmc-cuda` をゲートの docker イメージ内でビルドして RTX 3060(ドライバ 580.178.04、CUDA 12.9.2)で、同じ入力に `MVMC_RS_SR_BACKEND=cuda` を指定して実行すると、3 ステップとも実行され、`zvo_out.dat` は `c-order` と最大相対 `2.4e-15`、`zqp_opt.dat` は絶対値 `8.0e-14` で一致しました(同じコンテナでの `tenferro` は `8.6e-15` と `7.1e-15`)。[12.5](#125-数値的保証と検証内容) の SR の境界の十分内側です。GPU が 2 枚のこのホストで `cuda:9` は `CUDA device 9 requested but 2 device(s) found` で終了ステータス `2` になります。CUDA ランタイムライブラリのないホストでネイティブに実行すると、`mvmc-cuda` は `backend unavailable: Unable to dynamically load the "cudart" shared library` で終了ステータス `2` になります。`tenferro` と `cuda` の出力はデフォルトとバイト単位では**一致しません**。これがオプトインであり、境界で検証される理由です。これらの短い実行は例示であり、検証ではありません。
- **実際にサンプリングした最適化実行で検証済みです**(`gpu/mvmc-gpu-cuda/tests/sr_routing_gate.rs`、`MVMC_RS_CUDA_GATE=1`、ゲートの docker イメージ上の RTX 3060。デバイスなしでのルーティング確認は `crates/mvmc-core/tests/sr_resident_routing_452.rs`)。`hubbard_chain_dh_real` フィクスチャと `hubbard_chain_L16`/`L32`(`n = 36, 101, 197`)での直接 SR: SR ステップごとにちょうど `n x samples` 個の値のストアを 1 回アップロード。ステップ 1 の `S` は導出した上限 `6 (k + 3) eps (1 + DSROptStaDel) a_i a_j` の内側(観測した最悪値は上限の 3.1e-3、`g` はビット一致)、ステップ 1 の解は 1 次の摂動上限 `2 kappa (|B|_F / |S|_2 + |B_g| / |g| + 4 n eps)` の内側(観測した `|dx| / |x|` は最大 5.3e-14、上限は 2e-8〜5e-7、`kappa` は約 2e4〜3.5e4)。4 ステップの実行は、どの出力数値も C 順序との相対差が最大 4.8e-14 で、ビット単位で再現可能です(#358: 軌道は再現性と明示した上限で検査し、C に合わせ込むことはしません)。CG の実行(3 ステップ、実数フィクスチャと複素の `heisenberg_chain_cmp`): ステップごとにデバイス求解が 1 回、ファイル集合、更新前のエネルギー行、S のサイズとカットが C 順序と同一で、ビット単位で再現可能。CG の解そのものはステージ単位で検証されています(12.5)。複素の直接 SR はステージ単位の経路のままです(2 ステップの C 順序との差は 1.9e-11)。`mvmc-cuda` バイナリをエンドツーエンドで `benchmark/hubbard_chain/inputs/hubbard_chain_L16`(`--nsteps 3 --nsmp 2`、常駐の直接ステップ)に実行した結果: `zvo_out.dat` は `c-order` と最大相対 `2.1e-16`、`zqp_opt.dat` は `2.3e-14`、`zvo_var.dat` は `1.6e-15` で一致し、2 回の `cuda` 実行はバイト単位で同一でした。
- **状態はプロセス全体のものです。** バックエンドを指定する `mvmc` のコマンドラインオプションはありません。テストは `set_stage_backend_override` を使います。

### Rust からバックエンドを使う

```rust
use mvmc_core::stage_backend::{open_stage_backend, StageBackendKind};

mvmc_gpu_cuda::install();                       // CUDA プロバイダーを登録
let mut backend = open_stage_backend(StageBackendKind::Cuda(0))?;
println!("{} via {}", backend.label(), backend.provider());
```

`StageBackend::c_order()` がオラクルです。`with_pfaffian` は、別の Pfaffian 実装を指定した SR 実装と組み合わせます。ハーネスとゲートは、本番と同じ `open_stage_backend` の呼び出しでバックエンドを開きます(「検証されたものがデプロイされる」、設計 14.1)。

## 12.5 数値的保証と検証内容

### 契約

1. CPU の C 順序パスは、高速化コードが存在する前とビット単位で同一です。そのフィクスチャと 20 ステップの再現性テストは、このパスに対してのみ実行されます。
2. 高速化ステージは、演算・問題の規模・条件数から正当化された**明示的な絶対/相対許容誤差**で C 順序のオラクルと比較されます([11.4](11-compatibility.md#114-数値比較ポリシー))。計算された浮動小数点値をビット単位で比較することはなく、許容誤差は最初に測定された乖離から導出し、上方に調整することはしません。
3. バックエンドに**依存せず厳密**なもの: RNG の初期化と状態、乱数の消費順序と回数、更新種別と候補の乱数、添字、フラグ、離散的な判定プロトコル。
4. Metropolis の判定が変わりうるのは、*特定された*数値的反転を通してのみです。判定マージン `|w - u|` は、示された重みの誤差より小さくなければなりません。それより大きなマージンの反転や、同一の制御経路での乱数消費回数の違いは**欠陥**です。正当な反転の後は軌道が乖離してもよく、比較は同一実装での再現性に戻ります。
5. 同じバックエンド・入力・シード・デバイス・ライブラリバージョンでは、記載の許容誤差内で再現可能です(ビット単位の一致は検証された場合のみで、CUDA では仮定しません)。
6. 統計的チェックはこれらを補うものであり、代わりにはなりません。

### 検証ハーネス

`mvmc_core::accel_validation`(`crates/mvmc-core/src/accel_validation.rs`)がチェックを実装します。これは `StageBackend` を、線形代数ステージだけを実行する合成の対軌道系(検証用フィクスチャであり、物理モデルではありません)上で動かします。

- **教師強制リプレイ**(`replay`、`crates/mvmc-core/src/accel_validation.rs:382`)。オラクルが Metropolis の軌道を進め、各ステップで被検バックエンドが同じ候補を評価するので、偏差が累積しません。レポートは `pf`、`invM`、重み `(pf_new/pf_old)^2`、O ストア、`S`、`g` について、比較した要素数、絶対・相対の最大偏差、境界 `|a-b| <= abs + rel*max(|a|,|b|)` を外れた数を示します。デフォルト(`Tolerances`、`accel_validation.rs:76`): `abs = 1e-12`、`rel = 1e-10`。`n = 6`、要素が O(1)、Pfaffian の条件数が約 1e3 未満、加算項が高々数百、という前提から正当化され(誤差は O(10) eps で、約 1e4 の余裕を持ちつつ、レイアウト・符号・dtype の欠陥は検出できます)ます。
- **判定の記録と反転検出。** 各提案について、オラクルとバックエンドの重み、乱数、両方の判定、マージン、重みの誤差を記録します。各提案は電子、空きサイト、採択の 3 つの乱数を(棄却されても)常に引くため、乱数列と最終的な RNG 状態はバックエンドに依存しません。誤った重みを返すバックエンドが RNG の消費を変えないことは、テストで確認されています。
- **再現性**(`repeatability`、`accel_validation.rs:578`): 同じバックエンドを 20 ステップ 2 回実行します。離散的な状態は厳密、計算値は境界内。
- **ベンチマークメタデータ**(`BenchMetadata`、`accel_validation.rs:635`): リビジョン、CPU モデルとスレッド数、GPU とドライバ/CUDA/cuBLAS/cuSOLVER のバージョン、OS、rustc、tenferro のバージョン、プロバイダー、スレッド設定、dtype、バッチ、ウォームアップ、反復回数、アップロードとダウンロードを計時に含めるか。`bench_stages` は 5 回のウォームアップ後の 30 回の中央値を報告し、未対応のステージには 0 ではなく `None` を返します。比較の両側で同じスレッド数を使ってください。

バックエンドが実装していないステージは `Unsupported`/未比較と表示され、CPU の結果で代用されることはありません。tenferro CPU 版は通常の CI で実行され、CUDA 版はゲートの一部です。

### 許容誤差と観測された一致

| ステージ | 許容誤差(導出、調整なし) | 観測 | 場所 |
|-------|----------------------------|----------|-------|
| バッチ Pfaffian/逆行列、CUDA と `CpuPfapack` の比較 | 逆行列と Pfaffian の相対誤差のうち悪い方に対して `16 n eps cond_F(A)`(後退安定な逆行列の前進誤差のスケール)。独立なチェックとして `max abs(A A^-1 - I)`、`A^-1` の歪対称性の欠け、`Pf^2 = det A`(4 倍) | 実数 Pfaffian はすべてのサイズで pfapack とビット単位で同一。観測/許容は最大でも 1.4e-2(c64、`n = 2`)、`n >= 16` では 1e-3 未満。逆行列の相対誤差の最大は 7.7e-15(f64、`n = 128`)と 3.3e-13(c64、`n = 128`) | 設計 11.3、`pfaffian_gate.rs` |
| バッチ Pfaffian、CPU 版(通常の CI) | pfapack と同一、または `n = 16` の tensor-native と `ExtensionOp` で `1e-12`(Pfaffian)と `1e-11`(逆行列) | pfapack と `ExtensionOp` はビット単位で同一。tensor-native の逆行列の相対誤差は 7e-18〜1.4e-15(実数) | 設計 11.3、11.6 |
| ゼロピボットと非有限の平面 | 状態が厳密に一致、隣接する平面に影響しない | CPU と GPU の状態が厳密に一致 | 設計 11.3 |
| tenferro SR と C 順序の比較(Gram、CG 積) | 要素ごとに `2 gamma_k sum abs(terms)`、`gamma_k = k eps/(1-k eps)`(`k` はサンプル数、複素は `+3`、CG は `n + samples`) | 相対 1e-15〜1.5e-13。CUDA は最大 6.5e-15 | `NUMERICAL_COMPARISONS.md` の「tenferro SR backend vs C order」行、設計 12.3 |
| S と g の組み立て | 相対 `2 eps`(C と同じ IEEE 演算) | 完全に一致 | 同上 |
| Cholesky 求解 | `2 * 8 n eps kappa(S)` と、残差 `8 n eps norm(S) norm(x)` | 境界内 | 同上 |
| 直接 SR の短い実行 | `abs 1e-10 / rel 1e-8` | 境界内 | 同上 |
| デバイス常駐 Gram | 要素ごとに `2 k eps (abs(O) abs(O)^T)_ij` | 観測/境界の最悪値 6.2e-2 | 設計 15.2、`sr_device_gate.rs` |
| デバイス常駐の直接解 | 相対 `4 n eps kappa(S)`、残差 `1e-9` | 相対誤差は `n = 129` で 4.7e-14、`n = 400` で 2.4e-13(境界はホストでの kappa の推定から)、残差 5e-15〜9e-14 | 設計 15.2 |
| デバイス常駐 CG 積 | `4 (k + m) eps scale_i` | 観測/境界の最悪値 5.8e-4 | 設計 15.2 |
| デバイスサンプラと CPU サンプラの比較 | 重みは `1e-12 + 1e-10 w` 以内。判定列、配置、最終 RNG 状態は厳密 | ゲートされた 3 ケースすべてで反転 0、欠陥 0。相対重み差の最大は 5.9e-9(Hubbard L=32)、5.9e-12(L=16)、3.6e-13(Heisenberg)。最小マージン 1.9e-4 | 設計 13.2、`sampler_gate.rs` |
| ホストサービス上のデバイスサンプラ(CI) | ビット単位: 統計量、すべての `(weight, draw)`、SFMT の 624 ワード全部、保存された配置(`W = 1` と `W = 3`) | 完全に一致 | `crates/mvmc-core/tests/device_sampler.rs` |

CG はステップごとには比較**しません**。サンプルから作られる `S` は悪条件で(#358)、残差の 2 乗が 1e-16 までキャンセルし、1 ulp の変化が `max_iterations = n` での C 順序の解を 1e-4〜7e-4 動かします。デバイスの CG は、条件の良い問題ではホストと `2.9e-15` で一致し(反復回数も同じ)、悪条件の問題では 1〜8 反復で `1.8e-16〜2e-15`、119 反復後には `6.7e-4` 異なります。これは、計装した C に対して C 順序の Rust パスが示すのと同じ増幅であり、どちらの欠陥でもなく、方針は再現性です(デバイスの軌道はビット単位で再現可能で有限)。複素 Gram はデフォルトパスで C のサンプル順(逐次)を保ち、tenferro パスは純粋な再結合の境界で検証されます(設計 12.2 節「Complex Gram sample order」)。

この検証の限界(設計文書の記述): デバイスゲートの SR オペランドは合成(実数ストアの構造、制御した条件数)であり、最適化の実サンプル実行ではありません。GPU は RTX 3060 の 1 機種のみです。高速化パスを C に対して本番の最適化実行全体の中で検査したものはありません。

## 12.6 測定が示すこと

すべて設計文書からの引用です(RTX 3060、FP64 は FP32 の 1/64、共有ホスト。注意点は設計文書を参照)。これは実現性の記録であり、`mvmc` の高速化を主張するものではありません。

- **CPU で時間がかかる箇所**(Hubbard 鎖 L=16/32/64、1 スレッド、直接 SR): バッチ Pfaffian/逆行列が実時間の 58〜61 %、サンプラの rank-1 更新が 16〜30 %、局所エネルギーと Slater 微分が 6〜10 %、SR 行列は 0.2〜0.3 % のみ(O ストアと Gram を含めて 0.6〜1.5 %)。SR が支配的になるのは `NPara` が数千に達してからです。
- **バッチ Pfaffian**(データがデバイス上にある場合、`B >= 8`): pfapack 1 スレッドより 5.5〜14 倍(f64 `n = 16`)、9〜24 倍(`n = 32`)、11〜19 倍(`n = 64`)、約 9 倍(`n = 128`)高速。36 スレッド全部に対しては 1.5〜11 倍。呼び出しごとにアップロードとダウンロードを含めると、ほぼすべての場合に 36 スレッドより遅くなります。平面ごとにホストメモリ経由でオフロードしてはいけません。
- **デバイス上の SR**(設計 15.3、15.4): `NPara * samples >= 3e6` の CG は、行列がデバイス常駐のとき 36 コア全部より 8.5〜11 倍高速(エンドツーエンドで 5.7〜8.1 倍、1 スレッドに対しては 15〜22 倍。10^4 x 10^4、50 反復で 36 コアの 2.75 秒に対して 0.38 秒)。直接 SR は、`NPara >= 3000` でエンドツーエンドで 1 コアより 4.4〜5.5 倍速いものの、36 コアに対しては 1.1〜1.8 倍にとどまり、1000 x 10000 では 0.6 倍(遅い)です。`NPara * samples ~ 1e6` 未満では、デバイスは CPU より速くありません。
- **デバイスサンプラ**(設計 13.4): このマシンではマルチコアの CPU ランナーに勝てません。36 コア全部を使うと CPU ランナーが 1.15 倍(L=128、`W=1`)〜11.4 倍(L=16、`W=64`)速く、ホストが 4 コアの場合にデバイスが勝つのは L=128 のみ(`W >= 8` で 1.14〜1.37 倍)です。1 ラウンドは 31〜70 マイクロ秒で、CPU の 1 ホップは 3〜43 マイクロ秒、Pfaffian はサンプラの 32〜45 % にすぎないので、無限に速いデバイスでも約 1.5〜1.8 倍が上限です。
- **転送**(設計 10.7): ボトルネックは PCIe リンクでもピン留めメモリでもありません。tenferro 0.7.1 はアップロード 0.5〜1.7 GB/s、ダウンロード 1〜5 GB/s しか出ず、素の cudarc のコピーは 7〜13 GB/s です。リポジトリ内のヘルパ(`transfer.rs`)は、ホットな転送では tenferro の `upload_tensor`/`download_tensor` を迂回します。

## 12.7 マルチウォーカー実行

単一のマルコフ連鎖は GPU の恩恵を受けられません(1 回の試行ホップは 1 コアで 2〜13 マイクロ秒、デバイスの往復は数十マイクロ秒)。そのため、バッチ化の単位は独立な*ウォーカー*です。`crates/mvmc-core/src/multichain.rs` は 1 プロセスで `W` 本のウォーカーを実行します。**コマンドラインフラグはありません**。API は、固定パラメータのサンプリングと測定には `run_phys_cal_multichain(&MultiChainConfig)`、最適化全体には `run_para_opt_multichain(&ParaOptMultiChainConfig)` です。

- **シードは C のグループシードです。** ウォーカー `w` は `RndSeed + group_base + w`(`walker_seed`、`crates/mvmc-core/src/multichain.rs:88`)を使います。これは C のグループ `group1` のシード(`init_gen_rand(RndSeed + group1)`、[4.6](04-theory-sampling.md#46-サンプラー内の並列化))です。1 プロセス上の `W` 本のウォーカーは、グループ実行の C の `W` グループのチェーンと同じで、それぞれ独自のホスト SFMT ストリームを持ちます。時刻ベースのシード(`RndSeed < 0`)は全ウォーカーで 1 回だけ解決されます。各ウォーカーはシードオフセット `group_base + w` のシリアル実行とバイト単位で同一で、`W = 1` は通常のシリアル実行と等しくなります。
- **乱数の消費順序と回数は変わりません。** テストは各ウォーカーについて、SFMT の最終状態全体(624 ワードと位置)、消費ワード数、記録された `(weight, draw)` 列、サンプリング/観測量の状態全体をシリアル実行と比較します。結果はワーカープールのサイズに依存しません。
- **スレッド。** PhysCal: ウォーカーに対する rayon プール 1 つ(`threads`、デフォルト `min(W, cores)`)、各ウォーカーは単一スレッド、BLAS は 1 スレッドに固定。最適化: ウォーカー数と同じ OS スレッド。ウォーカー間リダクションに全ウォーカーの同時実行が必要なためです。
- **C 互換のリダクション付き最適化。** `run_para_opt_multichain` は、変更していない最適化器を、MPI コミュニケータの代わりをするプロセス内の `ThreadReducer`(`crates/mvmc-core/src/multichain.rs:433`)に対して実行します。`HO`、`OO`、エネルギー、重みはウォーカー間で合計され、全ウォーカーが同じ SR 系を解き、`SROptO` はウォーカーローカルのままで、`NSplitSize = 1` の分割なし MPI 実行の集団通信列と完全に同じです。和はランク順の左畳み込み `((v0 + v1) + v2) + ...` で、スレッドのスケジューリングに依存しません。`MPI_Allreduce` は順序を MPI ライブラリに任せるため、`W > 2` では Rust と C が最終ビットの丸めで異なりえます。C に対して使う境界は `|a - b| <= 1e-13 + 1e-12 |b|` です。
- **検証済み**(`crates/mvmc-core/tests/multiwalker_sr_435.rs`、MPI・C・Julia は不要): #179 マトリクスの、2 と 4 ランク・幅 1 の分割なし C の全セル(実数、複素、FSZ、OptTrans。直接法と CG)について、ウォーカーごとにカウンタ、保存配置、SFMT 状態が厳密に、リダクション後のオペランドが境界内で一致。2 ウォーカーは 4 ランク `NSplitSize = 2` セルの 2 グループと等しい。`W = 1` はシリアル最適化とバイト単位で同一。4 ステップ・4 ウォーカーの実行はビット単位で再現可能(1 ステップ目以降は悪条件の求解が丸めを増幅する(#358)ため、軌道は C に合わせず再現性でチェック)。tenferro SR バックエンドでも同様です。
- **出力ファイル**は、同等な MPI 実行の出力ルートと同じく、ウォーカー 0 のみが書き出します。
- **対象外。** `NSplitSize > 1`(1 本のチェーンの QP/サンプルをランク間で分割)はウォーカーではありません。MPI(プロセス間でスケールさせるには、`mpi` feature 付きの `mvmc` が引き続きその手段です、[8.4](08-running.md#84-mpiとグループ実行))。
- **スケーリングの記録。** Heisenberg 鎖のフィクスチャでの CPU ウォーカーランナーの効率 `T(1)/T(W)` は、他のジョブで負荷のかかったホスト(36 スレッドでロードアベレージ 58〜81)で `W = 2` が 0.94、`W = 8` が 0.56 でした。これはランナーのコストの上限であり、ランナーの性質ではありません(設計 10.6)。
- **判定マージン**は観測的に記録されます(`DecisionSummary`: 提案数、採択数、最小マージン `|w - u|`、反転すれすれの数)。Heisenberg フィクスチャでの最小マージンは 1.5e-7〜3.2e-7 で、`1e-10` の重みの摂動より 3 桁大きいため、ハーネスの境界内のバックエンドは、この入力では判定を反転できません。

## 12.8 既知の制限

- **RTX 3060 の FP64。** 測定したカードの FP64 は FP32 の 1/64 です。ここでのデバイス対 CPU の比はすべてこのカードによるもので、GEMM 律速のステージ(直接 SR の Gram と Cholesky、`n >= 1024` の FP64 GEMM)は 36 コアのホストと同程度です。データセンター向け GPU(FP64 が FP32 の 1/2)では比が変わると期待されますが、これは**測定されていません**。#450 のスイートによる A100 での評価は、そのためにあります([12.9](#129-ベンチマークと検証スイート450))。
- **tenferro の転送経路。** tenferro 0.7.1 はホストとデバイス間のコピーが 0.5〜1.7 GB/s(アップロード)、1〜5 GB/s(ダウンロード)しか出ません。`upload_tensor`/`download_tensor` がデバイス同期、CubeCL のステージング、冗長なホストコピー(256 MB のアップロードの 87 %)を加えるためです。上流の issue は **tensor4all/tenferro-rs#2009** で、要望文は `docs/design/tenferro-transfer-request-draft.md` に下書きがあります。リポジトリ内のピン留め/非同期ヘルパが、デバイスサンプラとデバイス常駐 SR 経路の回避策であり、大きな平面のアップロードは、平面をデバイス上で構築することで避けるべきです。設計が記録した他の上流のギャップ(Pfaffian や歪対称分解がない、決定性の契約がない、argmax がない、`Tensor` が `Clone` でない、`Module` が `Send` でない)は、設計 4.7 と 11.5 にあります。
- **保留中のサンプラ側 GPU 作業。** デバイス常駐サンプラ(#434)は実装・検証済みで保持されていますが、方針 B により、以降の作業(射影因子と `log_proj_ratio` のデバイスへの移行、共有メモリ版 LTL^T によるカーネル調整、永続カーネルや CUDA グラフ、バッチ局所エネルギー #426)は延期されています。対象は、hopping と exchange の実数モードの通常サンプリングのみで、複素・FSZ モード、測定ステージ、MPI の QP 分割は含みません。デバイス実行後は、ホストの `inv_m_real` は古くなっています(正しい値はデバイスにあります)。
- **本番へのルーティング(#452)。** デバイス常駐 SR ステップは、`MVMC_RS_SR_BACKEND=cuda[:N]` を指定した通常の `mvmc-cuda` 実行で動きます(どのステップかは[12.4](#124-バックエンドの選択)参照)。バッチ Pfaffian は依然として通常の実行からは呼ばれません。サンプラと測定は C 順序の `calc_m_all_*` を使います。標準の `mvmc` バイナリには CUDA プロバイダーがありません。
- **デバイス常駐 SR** は、直接 Cholesky ステップでは実数パラメータ、CG では実数または複素パラメータを扱います。複素の直接 SR には複素 Gram とエルミート Cholesky が必要ですが未実装で、ステージ単位の経路で動きます。複数ランクにまたがる CG には、常駐ループが行わないランク間リダクションが必要なので、常駐ステップは単一プロセスの実行用です。メモリは、直接 SR で `8 (n^2 + n_active^2 + n samples)` バイト(10^4 x 10^4 で 2.4 GB)、CG で `8 n samples` バイトです。
- **バッチ Pfaffian エンジンの制限。** `n <= 1024`、`n` は偶数、`f64` または `Complex64`。平面は、デバイスバッファが最大 3 GiB のチャンクで処理されます。上記の CPU ウォーカーのスケーリングとベンチマーク表は、共有ホストでの 1 回限りの数値です(キャンペーン間で約 ±30 %)。
- **再現性**は、上流で決定的なリダクション順序が使えるようになるまで、GPU では「同じデバイス、同じライブラリバージョン、許容誤差内」です。
- **CI に GPU はありません。** すべてのチェックの CUDA 版は、セルフホストランナー上のオプションのゲートであり、この章の数値は 1 台のマシンでのローカル実行によるものです。

## 12.9 ベンチマークと検証スイート(#450)

issue #450 は実装済みです([PR #462](https://github.com/AtelierArith/mvmc-rs/pull/462))。これは移植可能な関数レベルのスイートで、GPU・tenferro・CPU の各バリアントの**数値**をまず C 順序のオラクルと照合し、その後で初めて時間を記録します。保守者が別サーバーで実行する FP64 性能の高い GPU(A100)で、保留中のサンプラ側の作業を再評価するためです。正式なリファレンスは `benchmark/function_suite/README.md`、ハーネスは `gpu/mvmc-gpu-cuda` の `function_suite` example、RTX 3060 での参考レポートは `benchmark/function_suite/results/rtx3060-reference.md` です。

### 実行方法

```bash
git submodule update --init --recursive
scripts/bench/run_all.sh --native --quick --out bench-out          # 約 10 分の動作確認
scripts/bench/run_all.sh --native --full --gpu 0 --out bench-out   # 1 GPU で約 2 時間
scripts/bench/run_all.sh --docker --full --gpu 0 --out bench-out   # ホストのツールキットの代わりに nvidia/cuda イメージ
scripts/bench/run_all.sh --native --preflight-only                 # チェックのみ
```

- **要件。** NVIDIA ドライバ(`nvidia-smi` が動くこと)、Rust 1.96 以上、python3、および(native)NVRTC・cuBLAS・cuSOLVER を含む CUDA ツールキット 12.6 以上と OpenBLAS の開発ファイル(`libopenblas.so`)、または(docker)NVIDIA Container Toolkit 付きの docker。イメージは `MVMC_RS_CUDA_IMAGE` を設定しない限り `nvidia/cuda:12.9.1-devel-ubuntu24.04` で、OpenBLAS はコンテナ内にインストールされ、ホストの `~/.rustup` と `~/.cargo` がマウントされます。
- **プリフライト。** ドライバ、`--gpu` で指定した GPU、rustc、ツールキットのバージョン、CUDA ライブラリ、OpenBLAS のいずれかがない場合、ビルド前にメッセージを出して停止します。`--native`/`--docker` を省略すると、ツールキットがあれば native、なければ docker を選びます。
- **手順。** リリースビルド、Rust の StdFace 移植(`mvmc --dry-run`)による Hubbard 鎖の入力生成(`--full` では L = 16〜256)、続いて `pfaffian`、`sr`(別途 1 コアの CPU パス付き)、`sr_resident`、`sampler`、`transfers` の各ファミリー。あるファミリーが失敗しても他は実行されます。`--full` のサンプラ計時グリッドは `W * L^2 <= 2.1e6` に制限され(`MVMC_BENCH_WORK_CAP` で引き上げ可)、省略された点は明示的な `SKIPPED` 行になります。
- **出力。** `DIR/results-<host>-<date>.tar.gz`。ファミリーごとの CSV(`csv/`)、`report.md`、生ログ(`logs/`)、`metadata.txt`(GPU 型番、compute capability、スクリプトの表にあれば FP64 ピーク、ドライバ、CUDA・tenferro・cudarc のバージョン、CPU、OS、rustc、git リビジョン、スレッド環境)を含みます。CSV のスキーマは `family,function,variant,dtype,params,reps,median_s,min_s,max_s,dev_metric,dev_value,dev_bound,dev_ratio,verdict,note` です。
- **デバイス常駐 SR のフック(#447)。** `run_all.sh` は `gpu/mvmc-gpu-cuda/examples/bench_sr_resident.rs` があればそれを実行し、なければ `NotAvailable` 行を 1 つ記録します。数値をでっち上げることはありません。既存の `bench_sr_device` example は別のプログラムで、このスキーマをまだ書かないため、現時点ではこのファミリーは `NotAvailable` です。

### 判定の意味

**数値判定が主要な結果**であり、`FAIL` か `ERROR` が 1 つでもあればプロセスは非 0 で終了します。時間(ウォームアップ後の中央値・最小・最大)は副次的な参考列であり、レポートの「GPU 化すべきか」の推奨は、チェックに失敗した関数に対して `YES` になることはありません。

| 判定 | 意味 |
|---------|---------|
| `PASS` | オラクルからの偏差が上限以内(`dev_ratio` = 観測値/上限 <= 1)、または厳密チェック(RNG 状態、乱数消費数、配置、ステータスコード、ビット一致のコピー)が成立 |
| `FAIL` | 上限を超えた、または厳密チェックが失敗 |
| `ERROR` | 予期しない実行時エラー。失敗として数える |
| `ORACLE` | 基準の行そのもの(pfapack、OpenBLAS を使う `COrderSr`、CPU サンプラ) |
| `INFO` | 計時のみ |
| `NotAvailable` | この環境に関数またはフックが存在しない。数値なし |
| `SKIPPED` | 実行しなかった(メモリや計算量の上限)。note に理由 |
| `KNOWN-ISSUE` | CSV の値ではなくレポート上のラベル。追跡中の不具合に該当する `FAIL`([#465](https://github.com/AtelierArith/mvmc-rs/issues/465) サンプラのウォーカーの常駐逆行列、[#466](https://github.com/AtelierArith/mvmc-rs/issues/466) tenferro-native の c64 Pfaffian n = 128)。`FAIL` として数えることに変わりはなく、追跡されていない失敗は `NEW` と表示される |

ファミリーごとの上限(導出は `gpu/mvmc-gpu-cuda/examples/function_suite/*.rs` のソースヘッダにあります。デバイスに合わせて調整したものではなく、実行を通すために緩めてはなりません)。

- **Pfaffian と逆行列:** 平面ごとに `max(逆行列の相対誤差, Pfaffian の相対誤差) <= 16 n eps cond(A)`(`cond = ||A||_F ||A^-1||_F` は実際の平面のもの)。加えて独立な不変量 `A inv = I`、`inv` の歪対称性、`Pf^2 = det`、および平面ごとのステータスコード(ゼロピボット、NaN)がオラクルと一致すること。
- **SR ステージ:** `COrderSr` に対する相対最大ノルムで、上限は `4 k eps`(Gram)、`8 eps`(S/g)、`8 n eps kappa`(Cholesky 求解、kappa は Gershgorin)、`4 (k + n) eps`(CG 積)、`2 K kappa 4 (k + n) eps`(K 反復の CG 求解)。
- **サンプラ:** 厳密なものが必須条件です。RNG 状態と乱数消費数、電子配置、自由実行の判定列が CPU とビット単位で一致し、#424 の teacher による乱数不一致と説明のつかない判定反転(「defects」)が 0 であること。自由実行の発散は、teacher が判定反転を特定した場合にのみ、数値方針が許す範囲として合格になります。teacher 強制実行の重み偏差は、さらにウォーカーごとに `2 * 16 * (n + s) * eps * kappa / sqrt(w_min)` と照合されます。
- **転送:** ビット単位で一致する往復(コピーに演算はありません)。

サンプラの重みの上限は、証明ではなく明示した仮定に基づきます。定数 16 は Pfaffian の上限と同じ値です。再計算の間の更新数 `s` は、再計算の区間内の提案がすべて採択される最悪の場合としています。`kappa` は平面の条件数の最大値で、各ウォーカーの実行の開始時と終了時にのみ採取します(軌跡に沿ってではありません)。`w_min`(参照の非ゼロ重みの最小値)は比の内積の桁落ち因子の代わりであり、射影因子も含みます。このため上限は緩く、RTX 3060 で観測した偏差の 600〜20000 倍です(L = 16〜256 で観測値/許容値は 5e-5〜1.7e-3)。厳しくするには、teacher が提案ごとの偏差を公開する必要があります。

### 結果の送り返しとマシン間の比較

アーカイブ `bench-out/results-<host>-<date>.tar.gz`(数百キロバイト)1 つを送ってください。終了コードが非 0 なのは `FAIL` か `ERROR` が記録されたことを意味しますが、アーカイブはどちらの場合も書かれ、どちらの場合も送るべきです。アーカイブを統合するには(標準ライブラリのみ)次のようにします。

```bash
uv run --no-project scripts/bench/analyze.py results-a100.tar.gz results-rtx3060.tar.gz --out comparison.md
```

各引数が 1 台のマシン(アーカイブまたは展開済みディレクトリ)です。レポートは次の順です。(1) 数値検証: マシンごとの判定数、KNOWN-ISSUE または NEW ラベル付きの失敗一覧、関数・バリアントごとのマシン横断の最悪偏差/上限。(2) 参考としての時間: 速度向上の表と損益分岐点(バッチサイズ、NPara、ウォーカー数)。(3) 数値でゲートされた関数ごとの「GPU 化すべきか」の推奨。(4) `NotAvailable` と `SKIPPED` の行。続いて各マシンのメタデータ。

サンプラの時間は GPU と同じくらいホストのコア数に依存する(そのホスト側はウォーカーごとに 1 スレッドで動く)ので、同じマシンの CPU multichain の行と比較してください。CPU の行は他の負荷に敏感なので、空いているホストで実行してください。

## 12.10 数値の再現

```bash
cargo nextest run -p mvmc-gpu --cargo-profile test-fast                         # CPU 版、ハーネス
MVMC_RS_CUDA_GATE=1 scripts/run_cuda_gate.sh docker                             # すべての CUDA ゲート
scripts/run_pfaffian_bench.sh docker                                            # 設計 11.7
scripts/run_device_sampler_bench.sh docker                                      # 設計 13.5
scripts/run_sr_device_bench.sh docker                                           # 設計 15.5
cargo run --release -p mvmc-core --example sr_backend_bench                     # SR ステージ、C 順序と tenferro CPU
```

すべての結果にメタデータブロック(デバイスモデル、ドライバ、CUDA/cuBLAS/cuSOLVER のバージョン、tenferro のバージョン、ホスト CPU とロード、スレッド設定、リビジョン)を記録してください。それがない GPU の結果は比較できず、数値比較の基準環境は引き続き Linux x86_64 です([11.4](11-compatibility.md#114-数値比較ポリシー))。
