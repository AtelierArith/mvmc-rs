# 12. アクセラレータ(GPU)バックエンド

[目次](README.md) · 前へ: [11. 互換性と相違点](11-compatibility.md) · 次へ: [付録: 引用のチェック方法](appendix-checks.md)

この章では、`mvmc-rs` のオプションの高速化実装、すなわち通常の `mvmc` バイナリで選択できる tenferro の CPU ステージと、単体ワークスペース `gpu/mvmc-gpu-cuda` のデバイス常駐 GPU Opt/PhysCal ランナーについて、ビルドと選択の方法、何がどの許容誤差で検証されているか、既知の制限を説明します。設計記録は [docs/design/gpu-readiness.md](../../design/gpu-readiness.md)(tenferro ステージ、issue #417)、[docs/design/gpu-resident-opt.md](../../design/gpu-resident-opt.md) と [docs/design/gpu-resident-physcal.md](../../design/gpu-resident-physcal.md)(常駐 GPU ランナー)、検証方針は [docs/NUMERICAL_COMPARISONS.md](../../NUMERICAL_COMPARISONS.md) の「Accelerated-backend validation (#424)」節にあります。

従来の C 順序/SFMT の GPU 経路(`MVMC_RS_SR_BACKEND`/`MVMC_RS_MEASURE_PF_BACKEND`/`MVMC_RS_MEASURE_BACKEND=cuda[:N]`、`MVMC_RS_SAMPLER_BACKEND`、`gpu-cuda` feature、CUDA プロバイダー、CPU ドライバのラッパーとしての `mvmc-cuda`)は #643 で削除しました。その測定結果は、日付付きの `benchmark/*/results/` ディレクトリに履歴として残っています。

## 12.1 方針

- **CPU の C 順序パスがデフォルトであり、基準(オラクル)です。** 高速化実装はすべて*追加の*実装です。CPU パスを置き換えたり、順序を変えたり、経路を変えたりすることはなく、デフォルトのビルド・テスト・フィクスチャは GPU にも tenferro の GPU クレートにも依存しません。
- **暗黙のフォールバックなし。** 未対応のステージ、dtype、入力はエラー(`StageError::Unsupported`)になり、CPU の結果で代用されることはありません。
- **明示的に選択します。** 自動選択はなく、デフォルトは `c-order` です。
- **2 つの数値契約。** tenferro の CPU ステージは C 順序の乱数の消費順序と回数を保ち、浮動小数点の加算順序だけが明示的な許容誤差の範囲で異なります([12.5](#125-数値的保証と検証内容))。常駐 GPU ランナーは Julia の GPU 契約(`AGENTS.md` で認められた cuRAND XORWOW のウォーカー集団)に従い、C/SFMT の軌道ではなく Julia GPU のフィクスチャに対して検証されます。

## 12.2 何があるか

| 構成要素 | 場所 | 内容 | 通常の `mvmc` 実行で使われるか |
|-----------|-------|--------------|---------------------------------|
| 統一ステージバックエンド | `crates/mvmc-core/src/stage_backend.rs`、`sr_backend.rs`(#421、#437) | `SrStages`(Gram、S/g 組み立て、Cholesky 求解、CG 積)、`PfaffianStages`、`MeasurementStages` を 1 つの `StageBackend` にまとめたもの | はい: `MVMC_RS_SR_BACKEND`、`MVMC_RS_MEASURE_PF_BACKEND`、`MVMC_RS_MEASURE_BACKEND` 経由 |
| tenferro による SR | `sr_backend.rs` の `TenferroSr`(#421) | SR ステージを CPU(`cpu-faer`)上の tenferro `dot_general`/`cholesky`/`triangular_solve` で実行 | はい: `MVMC_RS_SR_BACKEND=tenferro` |
| tenferro による測定ステージ B | `measurement_backend.rs` の `TenferroMeasurement`(#518) | 局所エネルギーの比テーブルと Slater 微分をテンソル縮約で計算(実数、非 FSZ) | はい: `MVMC_RS_MEASURE_BACKEND=tenferro` |
| バッチ Pfaffian と逆行列 | `crates/mvmc-gpu`(#423) | `[n,n,NQP,B]` 平面の Pfaffian/逆行列の CPU 版、tenferro-native、`ExtensionOp` | いいえ: ライブラリとハーネス |
| 検証ハーネス | `crates/mvmc-core/src/accel_validation.rs`(#424) | 教師強制リプレイ、判定反転検出器、再現性、ベンチマークメタデータ | テスト用ツール |
| サンプルバッチ測定 | `crates/mvmc-core/src/measurement_batch.rs`(#422) | 独立なサンプルごとの作業の CPU 上での並べ替え。`MVMC_RS_MEASURE_BATCH` | はい、常時(出力はバッチサイズに依存しない) |
| マルチウォーカーランナー | `crates/mvmc-core/src/multichain.rs`(#425、#435) | 1 プロセス内の `W` 本の独立チェーン: 固定パラメータの PhysCal と、C 互換リダクション付きの最適化全体 | ライブラリ API(CLI フラグなし) |
| 常駐 GPU Opt | `gpu/mvmc-gpu-cuda/src/opt.rs`、`opt/model.rs`(#560、#574) | `run_gpu_para_opt_from_namelist`: ウォーカー集団、サンプリング、観測量、SR をデバイス上に常駐。ファミリーごとの対応表: [`docs/design/gpu-resident-model-support.md`](../../design/gpu-resident-model-support.md)(#572) | いいえ: 単体ワークスペース |
| 常駐 GPU PhysCal | `gpu/mvmc-gpu-cuda/src/opt/physcal.rs`(#636) | `run_gpu_phys_cal_from_namelist`。`mvmc-cuda resident-physcal` | いいえ: 単体ワークスペース |

## 12.3 ビルドと実行

### デフォルトビルド(CPU のみ)

何も変わりません: `cargo build --release -p mvmc-cli`。GPU を必要としない高速化コード(`mvmc-gpu` クレート、CPU 上の tenferro ステージ、ハーネス)は通常のワークスペースの一部で、そのテストは通常の CI で実行されます。

```bash
cargo nextest run -p mvmc-gpu --cargo-profile test-fast          # バッチ Pfaffian の CPU 版とハーネス
cargo nextest run -p mvmc-core --cargo-profile test-fast \
  -E 'binary(multichain) | binary(multiwalker_sr_435) | binary(stage_backend_437) | binary(sr_backend_421)'
```

実験設定 `MVMC_RS_TENFERRO_WORKER_BUDGET=N` は、同一プロセス内で mvmc-rs が生成する CPU Engine の追加ワーカー予約数を制限します。参加数 `T` の Engine は `T - 1` 個を予約し、空きがなければ待機せず逐次 Engine を使用します。予約は共有所有者がすべて解放されて Engine が破棄された後に戻ります。スコープ付きチームの交換にも適用します。`0` は逐次実行のみを許可します。最初の CPU Engine 生成前に設定し、その後の変更・設定・解除はエラーです。未設定時は従来の割当方式を使用します。walker の呼び出しスレッド、内部 Rayon、BLAS、外部から渡した Engine、別 MPI ランクはこの上限に含まれません。それらの CPU 使用量を別途考慮してください。予約の取得順によって並列・逐次の選択が変わるため、検証用の設定として扱い、一般的な高速化はまだ保証しません。

### 実験的な Linux MPI CPU 割当

[所有権の評価と設定方針](../../../benchmark/tenferro_thread_budget/ownership-evaluation-20261102/README.md)に、予算の数え方と測定結果を記録しています。CLI の単一ランクでは、driver を含む内部参加数と Engine の追加ワーカー予約数を数えます。ライブラリの複数 walker では独立した呼び出しスレッドも必要で、内部1スレッド・最適化 walker 数 `W`・追加ワーカー上限 `E` なら計算参加数は `W + E` です。collective のため全 optimizer 呼び出しスレッドを実行可能に保ってください。BLAS は1スレッドとし、ランタイム補助スレッド・外部プール・物理コアの割当・quota は別途考慮します。既定の Rayon と作業量の閾値を維持し、shared team は実測で確認したワークロードと割当で選ぶ実験設定として扱います。

`MVMC_RS_MPI_CPUS_PER_RANK=N` は、各ランクの初期 affinity の範囲内で、同一ノードのランク間で重複しない `N` 個の論理 CPU を割り当てます。Linux の MPI ビルドを複数ランクで起動し、`MVMC_RS_TENFERRO_WORKER_BUDGET` を明示してください。内部並列の参加数と Engine の追加ワーカー上限の合計が `N` 以下である必要があります。

```bash
cargo build --release -p mvmc-cli --features mpi
MVMC_RS_MPI_CPUS_PER_RANK=2 MVMC_RS_INNER_THREADS=1 \
MVMC_RS_TENFERRO_WORKER_BUDGET=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
mpirun -n 4 target/release/mvmc namelist.def
```

割当設定は全ランクで揃えてください。割当不能や設定の不一致は入出力前に全ランクでエラーになります。MPI 呼び出しスレッドと既存の mvmc 内部 Rayon ワーカーの affinity を変更し、その後の生成スレッドに継承させます。既存の MPI・BLAS スレッドには適用しません。cgroup の CPU quota、SMT の物理コア共有、外部生成スレッドの上限は扱わないため、BLAS のスレッド数も別途設定してください。ライブラリの guard は破棄時に呼び出しスレッドと内部ワーカーを元の affinity に戻し、明示的な `restore()` では復元エラーを確認できます。この設定だけで高速化を保証するものではありません。

### 単体 GPU ワークスペース

CUDA の依存ツリー(`cuda` 付きの `tenferro-gpu`、cudarc、取り込んだ raw 実行ヘルパ `gpu/tenferro-ext`)は単体ワークスペース `gpu/mvmc-gpu-cuda`(独自の `[workspace]` と `Cargo.lock` を持ち、ルートのワークスペースから除外)にあります。ルートのロックファイルに入れないのは意図的です。オプション依存として `tenferro-gpu` を追加すると、ロックファイルと、すべてのデフォルトビルドの監査対象が約 2000 行増えました。このワークスペースが `mvmc-core` を使うのは、入力の解析、検証、パラメータの入出力だけです。

- **`gpu/mvmc-gpu-cuda` から**ビルドします(`cargo build --release --manifest-path gpu/mvmc-gpu-cuda/Cargo.toml ...`)。
- **`mvmc-cuda`**(`gpu/mvmc-gpu-cuda/src/bin/mvmc-cuda.rs`)は常駐サブコマンド `mvmc-cuda resident-physcal NAMELIST PARAMS NEW_OUTPUT_DIR` だけを実行し、`energy.dat`、`green_one.dat`、`green_two.dat`、`green_two_ex.dat`、`sampling.txt` を書き出します。それ以外の呼び出しはエラーです。CPU ランナーは `mvmc` です。
- 常駐最適化はライブラリ呼び出し `mvmc_gpu_cuda::opt::run_gpu_para_opt_from_namelist` です。ベンチマークでは `examples/resident_opt_worker.rs` と `examples/resident_physcal_worker.rs` が 2 つのランナーを駆動します([gpu-resident-opt.md](../../design/gpu-resident-opt.md)、[gpu-resident-physcal.md](../../design/gpu-resident-physcal.md))。
- CUDA 実行の要件: CUDA ドライバ(`libcuda`)と、実行時の NVRTC、cuBLAS、cuSOLVER、cuRAND(`dlopen` で読み込まれ、*ビルド*にはツールキット不要)。tenferro のリンクはホステッドランナーのディスクを使い切ることがあるため、単体ワークスペースは `debug = 0` にしています。

### CUDA ゲート(native または docker)

`scripts/run_cuda_gate.sh [native|docker] [追加の cargo-test 引数]` は単体ワークスペースをビルドし、その CUDA テストを `--include-ignored` で実行します。対象は常駐 Opt/PhysCal のライブラリテスト(Julia フィクスチャ、20 ステップの再現性、転送と同期の契約)、常駐インターバルサンプラのテスト(`resident_interval`)、インターバルカーネルのゲート(`interval_gate`、`interval_parity`)です。2 デバイスのコンテキストテストとスループットのベンチマークはゲートに含みません。

```bash
scripts/run_cuda_gate.sh native                          # ドライバと CUDA ライブラリが LD_LIBRARY_PATH にあること
scripts/run_cuda_gate.sh docker                          # NVIDIA CUDA ツールキットイメージ、--gpus all
MVMC_RS_CUDA_IMAGE=my/cuda:tag scripts/run_cuda_gate.sh docker
```

- `docker` が存在すれば `docker` がデフォルトです。デフォルトのイメージは `tenferro-benchmark-cuda:full-verify-20260822` で、`MVMC_RS_CUDA_IMAGE` で変更できます。ホストの `~/.rustup` と `~/.cargo` をマウントするのでイメージに Rust ツールチェインは不要です。スクリプトはコンテナ内で(root として)OpenBLAS/LAPACK をインストールし、その後ユーザー ID に切り替えるので、`target/` のファイルはホストの所有権のままです。Cargo のターゲットは `gpu/mvmc-gpu-cuda/target` に置かれます。
- **フェイルクローズド。** テストの前に `nvidia-smi -L` がデバイスを列挙できなければなりません。デバイスがなければハードな失敗で、合格にはなりません。
- **ログ。** テストログは `gpu/mvmc-gpu-cuda/results/cuda-gate.log` に書かれます(`MVMC_RS_CUDA_GATE_OUT` で変更可能。このディレクトリは git の無視対象です)。
- **CI。** optional-gates ワークフローの入力 `cuda_gate` が、セルフホストランナー(`self-hosted, linux, x64, gpu`。ラベルは仮置き)上でジョブ `cuda-gate` を起動し、`scripts/run_cuda_gate.sh native` を呼びます。ホステッドランナーには GPU がないため、デフォルトの CI では実行されません([OPTIONAL_GATES_DISPATCH.md](../../OPTIONAL_GATES_DISPATCH.md))。

### エンドツーエンドの GPU ベンチマーク

`scripts/benchmark_gpu.sh`(`scripts/benchmark/run.py gpu`)は、共通の L128 入力について、1 枚の NVIDIA GPU 上で常駐 Rust Opt/PhysCal ワーカーと Julia GPU の時間を測ります。`--smoke` は短い構成で実行し、`--dry-run` は計画を表示します。`scripts/benchmark/README.md` を参照してください。

## 12.4 バックエンドの選択

`MVMC_RS_SR_BACKEND`([8.5](08-running.md#85-環境変数))は本番が SR ステージに使うバックエンドを、`MVMC_RS_MEASURE_PF_BACKEND` と `MVMC_RS_MEASURE_BACKEND` は測定ステージ A の Pfaffian の生成元とステージ B の消費側を選びます。セレクタは `parse_stage_backend`、`parse_measurement_pfaffian`、`parse_measurement_consumer`(`crates/mvmc-core/src/stage_backend.rs`、`crates/mvmc-core/src/measurement_batch.rs`)が解析します。

| 値 | バックエンド | 備考 |
|-------|---------|-------|
| 未設定、空、`c`、`c-order`、`corder`、`default` | C 順序 CPU(`COrderSr` + `COrderPfaffian` + `COrderMeasurement`) | デフォルトであり整合性のオラクル。従来のインラインコードとバイト単位で同一 |
| `tenferro`、`tenferro-cpu` | CPU 上の tenferro eager 演算(`cpu-faer`)。Pfaffian スロットは PfaPack 拡張演算の準備済み tenferro プログラム(`TenferroPfaffian`、#613) | 実現性確認用の経路: 1 スレッドでは OpenBLAS/LAPACK より 1.1〜4 倍**遅い**(SYRK の代わりに faer の GEMM、ステージごとのテンソル生成と読み出し) |

`MVMC_RS_MEASURE_PF_BACKEND` はさらに `calc-m-all`(デフォルト、サンプルごとの C 順序カーネル)を、`MVMC_RS_MEASURE_BACKEND` はさらに `direct`/`inline`(インラインの C 順序呼び出し)を受け付けます。

規則:

- **不正な値はエラーで、フォールバックはしません。** CLI は起動時に、ファイルを読み書きする前に一度だけ全セレクタを検証します。削除された `cuda`/`cuda:N` を含む不正な値(`MVMC_RS_SR_BACKEND="cuda" is not one of c-order, tenferro`)や利用できないバックエンドは `error: <変数名>: ...` を表示し、他のオプションエラーと同じく使用法エラーの終了ステータス `2` で終了します([8.1](08-running.md#81-mvmcコマンド))。出力ディレクトリは作られません。MPI では全ランクが合意に参加するので、全ランクがそろって停止します。この事前チェックを経ずに本番コードに到達したライブラリ呼び出し(`selected_stage_backend`、`acquire`)は同じメッセージで panic します。
- **ルーティングされる範囲。** `MVMC_RS_SR_BACKEND` がルーティングするのは SR ステージのみです。Gram 積(`observables.rs` の `finalize_oo_store_real` と複素版ファイナライザ)、S/g の組み立てと Cholesky 求解(`sr.rs`)、CG 積(`sr_cg.rs`)。サンプラは常に C 順序のカーネルを使います。
- **デフォルト以外のバックエンドはプロセスごとに 1 回だけ構築され**共有されます(tenferro ランタイムは高価です)。定数の CG オペランドは明示的なバージョンカウンタのもとでキャッシュされます。
- **(観測)** デフォルト feature のリリースビルド(Linux x86_64)で `benchmark/hubbard_chain/inputs/hubbard_chain_L16`、`--nsteps 3 --nsmp 2` を実行すると、`tenferro` は `c-order` と同じファイル群(`zqp_opt.dat`、`zqp_*_opt.dat`、`zvo_out.dat`、`zvo_var.dat`、`zvo_SRinfo.dat`、`zvo_time_001.dat`)を書きます。`zvo_out.dat` の最初の 2 ステップは同一で、3 ステップ目は最後の桁が異なり、`zqp_opt.dat` の差は絶対値で最大 `5.7e-14` です。`tenferro` の 2 回の実行はバイト単位で同一です。出力はデフォルトとバイト単位では**一致しない**ため、オプトインで、許容誤差に対して確認します。この短い実行は例示であり、検証ではありません。
- **状態はプロセス全体のものです。** バックエンドを指定する `mvmc` のコマンドラインオプションはありません。テストは `set_stage_backend_override` を使います。

### Rust からバックエンドを使う

```rust
use mvmc_core::stage_backend::{open_stage_backend, StageBackendKind};

let mut backend = open_stage_backend(StageBackendKind::TenferroCpu)?;
println!("{} via {}", backend.label(), backend.provider());
```

`StageBackend::c_order()` がオラクルです。`with_pfaffian` は別の Pfaffian 実装を所与の SR 実装と組み合わせます。ハーネスは本番と同じ `open_stage_backend` 呼び出しでバックエンドを開きます(「検証されたものがそのまま本番で使われる」、設計 14.1)。

## 12.5 数値的保証と検証内容

### 契約(tenferro の CPU ステージ)

1. CPU の C 順序パスは、高速化コードが存在する前とビット単位で同一です。そのフィクスチャと 20 ステップの再現性テストは、このパスに対してのみ実行されます。
2. 高速化ステージは、演算・問題の規模・条件数から正当化された**明示的な絶対/相対許容誤差**で C 順序のオラクルと比較されます([11.4](11-compatibility.md#114-数値比較ポリシー))。計算された浮動小数点値をビット単位で比較することはなく、許容誤差は最初に測定された乖離から導出し、上方に調整することはしません。
3. バックエンドに**依存せず厳密**なもの: RNG の初期化と状態、乱数の消費順序と回数、更新種別と候補の乱数、添字、フラグ、離散的な判定プロトコル。
4. Metropolis の判定が変わりうるのは、*特定された*数値的反転を通してのみです。判定マージン `|w - u|` は、示された重みの誤差より小さくなければなりません。それより大きなマージンの反転や、同一の制御経路での乱数消費回数の違いは**欠陥**です。正当な反転の後は軌道が乖離してもよく、比較は同一実装での再現性に戻ります。
5. 同じバックエンド・入力・シードでは、記載の許容誤差内で再現可能です(ビット単位の一致は検証された場合のみ)。
6. 統計的チェックはこれらを補うものであり、代わりにはなりません。

常駐 GPU ランナーはこれに代えて Julia の GPU 契約に従います。Julia GPU 実装の初期化・提案・判定・更新の順序と cuRAND XORWOW ストリームを使い、明示的な許容誤差のもとで独立な Julia フィクスチャ(`gpu/mvmc-gpu-cuda/tests/fixtures/opt_julia.md`、`physcal_julia.md`)に対して検証し、さらに同一実装での 20 ステップの再現性と、転送・同期のカウンタを確認します。

### 検証ハーネス

`mvmc_core::accel_validation`(`crates/mvmc-core/src/accel_validation.rs`)がチェックを実装します。これは `StageBackend` を、線形代数ステージだけを実行する合成の対軌道系(検証用フィクスチャであり、物理モデルではありません)上で動かします。

- **教師強制リプレイ**(`replay`)。オラクルが Metropolis の軌道を進め、各ステップで被検バックエンドが同じ候補を評価するので、偏差が累積しません。レポートは `pf`、`invM`、重み `(pf_new/pf_old)^2`、O ストア、`S`、`g` について、比較した要素数、絶対・相対の最大偏差、境界 `|a-b| <= abs + rel*max(|a|,|b|)` を外れた数を示します。デフォルト(`Tolerances`): `abs = 1e-12`、`rel = 1e-10`。`n = 6`、要素が O(1)、Pfaffian の条件数が約 1e3 未満、加算項が高々数百、という前提から正当化され(誤差は O(10) eps で、約 1e4 の余裕を持ちつつ、レイアウト・符号・dtype の欠陥は検出できます)ます。
- **判定の記録と反転検出。** 各提案について、オラクルとバックエンドの重み、乱数、両方の判定、マージン、重みの誤差を記録します。各提案は電子、空きサイト、採択の 3 つの乱数を(棄却されても)常に引くため、乱数列と最終的な RNG 状態はバックエンドに依存しません。誤った重みを返すバックエンドが RNG の消費を変えないことは、テストで確認されています。
- **再現性**(`repeatability`): 同じバックエンドを 20 ステップ 2 回実行します。離散的な状態は厳密、計算値は境界内。
- **ベンチマークメタデータ**(`BenchMetadata`): リビジョン、CPU モデルとスレッド数、OS、rustc、tenferro のバージョン、プロバイダー、スレッド設定、dtype、バッチ、ウォームアップ、反復回数、テンソルの生成と読み出しを計時に含めるか。`bench_stages` は 5 回のウォームアップ後の 30 回の中央値を報告し、未対応のステージには 0 ではなく `None` を返します。比較の両側で同じスレッド数を使ってください。

バックエンドが実装していないステージは `Unsupported`/未比較と表示され、CPU の結果で代用されることはありません。tenferro CPU 版は通常の CI で実行されます。

### 許容誤差と観測された一致

| ステージ | 許容誤差(導出、調整なし) | 観測 | 場所 |
|-------|----------------------------|----------|-------|
| バッチ Pfaffian、CPU 版(通常の CI) | pfapack と同一、または `n = 16` の tensor-native と `ExtensionOp` で `1e-12`(Pfaffian)と `1e-11`(逆行列) | pfapack と `ExtensionOp` はビット単位で同一。tensor-native の逆行列の相対誤差は 7e-18〜1.4e-15(実数) | 設計 11.3、11.6 |
| tenferro SR と C 順序(Gram、CG 積) | 要素ごとに `2 gamma_k sum abs(terms)`、`gamma_k = k eps/(1-k eps)`(`k` はサンプル数、複素では `+3`、CG では `n + samples`) | 相対 1e-15〜1.5e-13 | `NUMERICAL_COMPARISONS.md` の行「tenferro SR backend vs C order」、設計 12.3 |
| S と g の組み立て | 相対 `2 eps`(C と同じ IEEE 演算) | 完全に一致 | 同上 |
| Cholesky 求解 | `2 * 8 n eps kappa(S)` と残差 `8 n eps norm(S) norm(x)` | 境界内 | 同上 |
| 直接法 SR の短い実行 | `abs 1e-10 / rel 1e-8` | 境界内 | 同上 |
| tenferro 測定ステージ B と `COrderMeasurement` | サンプルごとに `abs 1e-13 / rel 1e-13`、エンドツーエンドで `abs 1e-11 / rel 1e-8` | 相対 6.8e-16(局所エネルギー)、絶対 8.9e-16(Slater 微分) | 設計 14.5、`measurement_tenferro_518.rs` |

CG はステップごとには**比較しません**。サンプリングされた `S` は悪条件で(#358)、残差の二乗は 1e-16 まで桁落ちし、`max_iterations = n` では 1 ulp の変化で C 順序の解が 1e-4〜7e-4 動くため、方針は再現性です。複素 Gram はデフォルトの経路では C の逐次的なサンプル順を保ち、tenferro の経路は純粋な結合順変更の境界で検証されます(設計 12.2 節「Complex Gram sample order」)。

## 12.6 測定が示すこと

- **CPU での時間の内訳**(Hubbard 鎖 L=16/32/64、1 スレッド、直接法 SR): バッチ Pfaffian/逆行列が実時間の 58〜61 %、サンプラのランク 1 更新が 16〜30 %、局所エネルギーと Slater 微分が 6〜10 %、SR 行列はわずか 0.2〜0.3 %(O ストアと Gram を含めて 0.6〜1.5 %)。SR が支配的になるのは `NPara` が数千に達した場合だけです(設計 3)。
- **tenferro の CPU ステージ**は実現性確認用の経路で、1 スレッドでは OpenBLAS/LAPACK より 1.1〜4 倍遅くなります(設計 12.3)。
- **常駐 GPU ランナー**: RTX 3060 上での Julia GPU に対する L128 の最適化と PhysCal の時間は、[gpu-resident-opt.md](../../design/gpu-resident-opt.md)、[gpu-resident-physcal.md](../../design/gpu-resident-physcal.md)、`scripts/benchmark_gpu.sh` のレポートに記録しています。

## 12.7 マルチウォーカー実行

単一のマルコフ連鎖は GPU の恩恵を受けられません(1 回の試行ホップは 1 コアで 2〜13 マイクロ秒、デバイスの往復は数十マイクロ秒)。そのため、バッチ化の単位は独立な*ウォーカー*です。`crates/mvmc-core/src/multichain.rs` は 1 プロセスで `W` 本のウォーカーを実行します。**コマンドラインフラグはありません**。API は、固定パラメータのサンプリングと測定には `run_phys_cal_multichain(&MultiChainConfig)`、最適化全体には `run_para_opt_multichain(&ParaOptMultiChainConfig)` です。

- **シードは C のグループシードです。** ウォーカー `w` は `RndSeed + group_base + w`(`walker_seed`、`crates/mvmc-core/src/multichain.rs:88`)を使います。これは C のグループ `group1` のシード(`init_gen_rand(RndSeed + group1)`、[4.6](04-theory-sampling.md#46-サンプラー内の並列化))です。1 プロセス上の `W` 本のウォーカーは、グループ実行の C の `W` グループのチェーンと同じで、それぞれ独自のホスト SFMT ストリームを持ちます。時刻ベースのシード(`RndSeed < 0`)は全ウォーカーで 1 回だけ解決されます。各ウォーカーはシードオフセット `group_base + w` のシリアル実行とバイト単位で同一で、`W = 1` は通常のシリアル実行と等しくなります。
- **乱数の消費順序と回数は変わりません。** テストは各ウォーカーについて、SFMT の最終状態全体(624 ワードと位置)、消費ワード数、記録された `(weight, draw)` 列、サンプリング/観測量の状態全体をシリアル実行と比較します。結果はワーカープールのサイズに依存しません。
- **スレッド。** PhysCal: ウォーカーに対する rayon プール 1 つ(`threads`、デフォルト `min(W, cores)`)、各ウォーカーは単一スレッド、BLAS は 1 スレッドに固定。最適化: ウォーカー数と同じ OS スレッド。ウォーカー間リダクションに全ウォーカーの同時実行が必要なためです。ウォーカースレッド数が OpenBLAS の `MAX_THREADS` を超えると、OpenBLAS は `precompiled NUM_THREADS exceeded, adding auxiliary array for thread metadata` を表示しますが、これは警告でありエラーではありません。Hubbard L128 の 320 ウォーカー(OpenBLAS 0.3.26 `MAX_THREADS=64`、36 コア)は 3 回とも完走し、ウォーカーごとの SFMT 状態・配置・出力は一致しました。ピークメモリは約 24 GB でした(#642) **(観測)**。
- **C 互換のリダクション付き最適化。** `run_para_opt_multichain` は、変更していない最適化器を、MPI コミュニケータの代わりをするプロセス内の `ThreadReducer`(`crates/mvmc-core/src/multichain.rs:433`)に対して実行します。`HO`、`OO`、エネルギー、重みはウォーカー間で合計され、全ウォーカーが同じ SR 系を解き、`SROptO` はウォーカーローカルのままで、`NSplitSize = 1` の分割なし MPI 実行の集団通信列と完全に同じです。和はランク順の左畳み込み `((v0 + v1) + v2) + ...` で、スレッドのスケジューリングに依存しません。`MPI_Allreduce` は順序を MPI ライブラリに任せるため、`W > 2` では Rust と C が最終ビットの丸めで異なりえます。C に対して使う境界は `|a - b| <= 1e-13 + 1e-12 |b|` です。
- **検証済み**(`crates/mvmc-core/tests/multiwalker_sr_435.rs`、MPI・C・Julia は不要): #179 マトリクスの、2 と 4 ランク・幅 1 の分割なし C の全セル(実数、複素、FSZ、OptTrans。直接法と CG)について、ウォーカーごとにカウンタ、保存配置、SFMT 状態が厳密に、リダクション後のオペランドが境界内で一致。2 ウォーカーは 4 ランク `NSplitSize = 2` セルの 2 グループと等しい。`W = 1` はシリアル最適化とバイト単位で同一。4 ステップ・4 ウォーカーの実行はビット単位で再現可能(1 ステップ目以降は悪条件の求解が丸めを増幅する(#358)ため、軌道は C に合わせず再現性でチェック)。tenferro SR バックエンドでも同様です。
- **出力ファイル**は、同等な MPI 実行の出力ルートと同じく、ウォーカー 0 のみが書き出します。
- **対象外。** `NSplitSize > 1`(1 本のチェーンの QP/サンプルをランク間で分割)はウォーカーではありません。MPI(プロセス間でスケールさせるには、`mpi` feature 付きの `mvmc` が引き続きその手段です、[8.4](08-running.md#84-mpiとグループ実行))。
- **スケーリングの記録。** Heisenberg 鎖のフィクスチャでの CPU ウォーカーランナーの効率 `T(1)/T(W)` は、他のジョブで負荷のかかったホスト(36 スレッドでロードアベレージ 58〜81)で `W = 2` が 0.94、`W = 8` が 0.56 でした。これはランナーのコストの上限であり、ランナーの性質ではありません(設計 10.6)。
- **判定マージン**は観測的に記録されます(`DecisionSummary`: 提案数、採択数、最小マージン `|w - u|`、反転すれすれの数)。Heisenberg フィクスチャでの最小マージンは 1.5e-7〜3.2e-7 で、`1e-10` の重みの摂動より 3 桁大きいため、ハーネスの境界内のバックエンドは、この入力では判定を反転できません。

## 12.8 既知の制限

- **tenferro ステージ**は実数・複素の SR と、実数・非 FSZ の測定ステージ B(RBM と二体 Green 項を除く)を扱います。上流の tenferro には歪対称分解がないため、tenferro の測定 Pfaffian 生成元は純 Rust の PfaPack カーネルを tenferro 拡張演算として登録したものです(`mvmc_core::tenferro_pfaffian`)。テンソルプログラム、その準備と実行は tenferro、分解は C 順序の PfaPack 計算列です(#613)。設計が記録しているその他の上流の欠落(決定性の契約なし、argmax なし、`Tensor` が `Clone` でない、`Module` が `Send` でない)は設計 4.7 と 11.5 を参照してください。
- **常駐 GPU ランナー**が受け付けるのは、有限な実数パラメータの実数通常軌道、Gutzwiller/Jastrow、DH2/DH4 と実数 RBM 因子、`NExUpdatePath` 0-2 の局所スピン、直接 SR または CG SR(`NSRCG`。PhysCal では無視)、OptTrans なし、逐次実行、`2 Ne <= 1024`(PhysCal はさらに `NLanczosMode = 0`)です。それ以外はデバイス上の作業の前に拒否されます(`gpu/mvmc-gpu-cuda/src/opt/model.rs`)。設計上、C/SFMT の軌道は再現しません。
- **RTX 3060 の FP64。** 記録された GPU の測定値は、FP64 が FP32 の 1/64 のカードによるものです。データセンター向け GPU では比率が変わると見込まれます。
- **CI に GPU なし。** CUDA ゲートはセルフホストランナー上のオプションです。設計文書の GPU の数値はローカル実行によるものです。

## 12.9 数値の再現

```bash
cargo nextest run -p mvmc-gpu --cargo-profile test-fast         # バッチ Pfaffian の CPU 版、ハーネス
cargo run --release -p mvmc-core --example sr_backend_bench     # SR ステージ、C 順序と tenferro CPU
scripts/run_cuda_gate.sh docker                                 # 常駐 GPU Opt/PhysCal とインターバルのテスト
bash scripts/benchmark_gpu.sh --smoke                           # 常駐 Rust と Julia GPU、短い実行
```

結果には必ずメタデータ(デバイスモデル、ドライバ、CUDA ライブラリのバージョン、tenferro のバージョン、ホスト CPU と負荷、スレッド設定、git リビジョン)を記録してください。これのない GPU の結果は比較できません。数値比較の基準環境は引き続き Linux x86_64 です([11.4](11-compatibility.md#114-数値比較ポリシー))。
