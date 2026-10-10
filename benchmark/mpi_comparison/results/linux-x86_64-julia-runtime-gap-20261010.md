# Julia / Rust 残存性能差の調査（2026-10-10、Linux x86_64）

Julia側の型不安定性、一時ベクトル確保、SR更新の繰り返し走査、実数行列カーネルの境界検査を改善した。9構成で実行時間を12.9–32.8%短縮したが、Rustとの差は1.108–1.390倍残る。次の主対象はサンプルごとの行列生成・Pfaffian・逆行列計算。

原因は、Slater更新中の`Any`由来のボックス化、proposalごとの投影係数ベクトル生成、accepted hopの作業ベクトル確保、SR各成分の全パラメータ再走査、実数行列カーネルの境界検査。型をループ前に確定し、係数を直接参照し、作業領域を再利用し、SRの走査をまとめた。演算順と乱数処理は維持した。

L32・1 process × 1 threadの診断では、総割当量が5.529 GBから0.539 GBへ減少。これは累積割当量で、常駐メモリ量ではない。変更前のサンプルごとのPfaffian・逆行列計算は全体の50.3%を占めた。GCだけが原因ではなく、残る速度差を詰めるには行列生成・分解・逆行列の各段階を同じ入力で測る必要がある。Rustを参考にした4列同時solveはJuliaの固定入力プローブで遅くなったため不採用。

検証: 1 thread 22,158件、4 threads 22,159件の全Optimizerテストが成功。20ステップ・9構成で42組のSFMT状態ブロックと6,552,597回の乱数primitive消費順・回数が一致した。計算値88,794項目はabs=rel=1e-11以内（実測差0）。process数変更によるseed・trajectory差があるため、収束や有効サンプル数の同等性は主張しない。

上流 issue: https://github.com/tmisawa/Julia-mVMC/issues/73 。[上流PR #74（julia-patch）](https://github.com/tmisawa/Julia-mVMC/pull/74)を作成した。実装・検証・測定資料のcommitは`0d86b1f3c14dbb7219d038d9b9780c437b52ad24`。測定時の28個のOptimizerソースSHA-256は、このcommitのファイルと照合済み。元のreference submoduleは上流`7b1ffd5`のまま保持している。

CI追加commit `e67cf84e982c5878679ebc6808f3938a09ae777e`で、Linux/macOSのJulia 1.11・1.12・1.13を検証する6構成とした。Linux 1.13でもMPI smoke・失敗時処理・default threadingのC parity検証を実行する。性能測定後の変更はworkflowのみで、Optimizerソースは変わっていない。

CIの最終workflowは`d9037d61859549301f89b317d2cf417b97f424a8`。`JULIA_NUM_THREADS=1` / `2,0`でinteractive poolを0に固定した。Julia 1.13では`2`だけの指定がdefault 2 / interactive 1になることを実機で確認したため、作業領域のthread IDと測定環境に合わせている。通常テストの`1`指定は、Julia 1.11のPkg.testが生成するCLI引数との互換性も維持する。旧workflowの`1,0`では1.11のテスト起動が失敗したため修正した。

Baseline: upstream `7b1ffd519e24f1794807ad326e09ee49a6a8cd3f`, which includes PR #72.
Candidate: the source hashes and diff in `runtime_gap_20261010_environment.json` identify the exact measured code.
Linux x86_64 Dev Container, AMD Ryzen 9 PRO 8945HS (8 cores / 16 logical CPUs), Julia 1.13.1,
Rust 1.99.0, MPICH 4.2.0 `/opt/mpich` (`ch4:ucx`, Hydra / PMI1).
Rust and Julia's native helper use OpenBLAS 0.3.26 LP64; Julia LinearAlgebra uses OpenBLAS 0.3.30 ILP64.
All BLAS pools are verified at one thread per process; native helper and Julia BLAS are checked independently.
GC threads: one. Julia default/interactive pools: `4,0`, `2,0`, `1,0`.

300 SR steps, 300 total samples per step (300 / 150 / 75 per process), one full warmup, three repetitions.
Median of the slowest process's warmed production API time, including production parsing/output but excluding launch,
JIT and warmup. Baseline and candidate phases ran sequentially with no concurrent heavy builds/tests.
Rust was remeasured in both phases; no historical baseline is substituted. Three repetitions are descriptive, not a confidence interval.
Julia executed MVMC frames on 4 / 2 / 1 threads, including the driver. Rust's default gates selected serial inner kernels
on these small inputs (observed pooled workers 0); a configured pool size is not evidence of kernel parallel execution.
Processes use different chain seeds and trajectories. These data compare throughput, not effective samples or convergence.

| モデル | process数 × threads/process | Julia変更前（秒） | Julia変更後（秒） | 短縮率 | Rust前 / 後（秒） | Julia後 / Rust後 |
|---|---:|---:|---:|---:|---:|---:|
| L16 | 1 × 4 | 4.991915 | 4.286273 | 14.1% | 3.112387 / 3.103700 | 1.381 |
| L16 | 2 × 2 | 2.911010 | 2.335425 | 19.8% | 1.720491 / 1.680704 | 1.390 |
| L16 | 4 × 1 | 1.712879 | 1.151904 | 32.8% | 0.976469 / 0.969523 | 1.188 |
| L24 | 1 × 4 | 9.311548 | 8.107485 | 12.9% | 6.422753 / 6.436101 | 1.260 |
| L24 | 2 × 2 | 5.514556 | 4.386669 | 20.5% | 3.447813 / 3.449969 | 1.272 |
| L24 | 4 × 1 | 3.506979 | 2.388856 | 31.9% | 2.036145 / 2.044399 | 1.168 |
| L32 | 1 × 4 | 16.676494 | 14.036661 | 15.8% | 11.953876 / 11.956682 | 1.174 |
| L32 | 2 × 2 | 9.611197 | 7.735210 | 19.5% | 6.461263 / 6.443195 | 1.201 |
| L32 | 4 × 1 | 5.999136 | 4.284880 | 28.6% | 3.877829 / 3.867876 | 1.108 |

Julia elapsed time fell by 12.9–32.8% across these nine cases.
The remaining Julia/Rust ratio is 1.108–1.390; this patch does not establish equal speed.

## Where the remaining time goes

The following is a separate warmed, instrumented L32 one-process/one-thread run, **not** the headline medians.
Timers are nested and must not be added; profiling and timer probes have overhead. Cross-language phase comparisons
are diagnostic because floating-point acceptance differences can change trajectories.

| Instrumented L32, 1 process × 1 thread | Julia before (s) | Julia after (s) | Rust (s) |
|---|---:|---:|---:|
| Total | 16.39686 | 13.72662 | 12.77368 |
| Sampling | 4.70703 | 3.91240 | 4.22677 |
| Main calculation | 10.28693 | 9.33564 | 8.16783 |
| Per-sample Pfaffian / inverse | 8.24760 | 7.40312 | 6.14992 |
| Slater update | 0.68751 | 0.16489 | 0.04928 |
| SR total | 0.54086 | 0.14498 | 0.05967 |
| SR postprocess | 0.41094 | 0.01430 | 0.00023 |
| DPOSV | 0.04194 | 0.04034 | 0.04397 |

Baseline per-sample Pfaffian/inverse work was 8.24760 / 16.39686 s (50.3% of total),
and it remains the largest kernel target. CPU samples identified factorization/inverse scalar loops and array access.
GC alone does not explain the gap: baseline GC was 0.376 s.
The same final timer probe allocated 5,529,366,920 bytes before and
539,275,144 bytes after (90.2% less).
These are cumulative allocations in a production call, including parsing/output, not resident memory.

## Implemented changes and evidence

* Resolve `qp_weights::Any` once to `QuantumProjectionWeights` before the Slater site-pair loops.
  A 0.1% allocation sample on the **intermediate** candidate identified boxed `ComplexF64` arithmetic here;
  the allocation-site file is diagnostic sampled bytes, not an extrapolated exact total.
* Read live projection coefficients in flattened C layout without building a new parameter vector for every proposal;
  preserve missing header slots, coefficient updates and the scalar accumulation order.
* Reuse two rank-local serial hop-update vectors in `SamplingWorkspace`; poisoned-buffer tests check overwrite and tails.
* Batch direct-SR parameter changes into one layout/duplicate scan; preserve sequential real/imaginary additions,
  last-wins duplicate lookup, untouched fixed duplicates, retained slots and QPOptTrans weight refresh.
* Check dimensions/indexing once before bounds-check-free ordinary real factor/inverse loops; SIMD only across
  independent rank-2 update elements. No `@fastmath`, fused `muladd`, solver replacement or RNG changes.

A four-column blocked tridiagonal solve inspired by Rust was tested on fixed input and rejected:
at n=32 the existing Julia solve took about 0.900 µs, the blocked trial 1.240 µs.
The committed solver keeps its original loop and arithmetic order. Adjacent child-kernel records compare fixed n=16/24/32
skew operators with preallocated workspaces (three batches of 10,000 calls); they are not whole-workload timings.

Next priorities are the per-sample matrix assembly/factorization/inverse pipeline: isolate each stage on identical matrices,
measure strided `SubArray` access and redundant scratch copies/initialization, and consider contiguous specialized kernels
only with independent residuals and unchanged arithmetic/RNG contracts. Whole-pipeline replacement, reciprocal division,
BLAS provider changes and additional thread spawning are not validated shortcuts in this report.

## Correctness and reproducibility

Full MVMCOptimizers suite on the final source: 22,158 assertions passed with one thread, 22,159 with four threads.
Focused coverage includes projection slots/live values/zero allocations, poisoned reusable buffers, SR duplicate and
repeated-component semantics, invalid dimensions, strided views, independent `T \ B` results and residuals (256 eps).
The exchange-counter provenance guard now hashes the two independently extracted upstream sampler bodies rather than
unrelated helpers in the whole file; both sampler bodies and all acceptance expectations remain unchanged.

All nine 20-step model/layout prefixes: 42 initial/final SFMT 624-word blocks matched exactly;
88,794 finite computed fields compared with abs=rel=1e-11 (maximum observed absolute difference 0).
This preserves the previous numerical tolerance; computed results are not compared bitwise.
An additional isolated audit forwards the original C RNG primitives and hashes integer words plus primitive kind and seed:
21 rank/model streams, 6,550,581 `gen_rand32` calls and
2,016 `genrand_real2` calls matched exactly in count and order. This runtime instrumentation is excluded from timings.
The real2 conversion is checked against C `to_real2` for its consumed word. No SFMT library/source changes.
Local verification is Linux x86_64 Julia 1.13.1; CI coverage is reported in the pull request, not assumed here.

Raw Julia CSVs, complete phase environment/hashes/diffs, executed drivers, timer/allocation records and parity summary
are adjacent to this report. The Rust checkout was dirty at measurement: its HEAD alone is not the executed source identifier.
Tracked input/lock provenance starts at Rust repository commit `72ca4e99e6b5c0e24b653a1b6176dea7c9540d48`;
the isolated resolved Manifest, preferences and prepared-input hashes are retained in metadata.
The original Julia submodule pin remains at upstream baseline until this new PR is reviewed and merged.
