# mvmc-rs function-level GPU/CPU suite report

## 1. Numerical validation

Primary result. Every GPU/tenferro/CPU variant is compared with the C-order CPU oracle (pfapack / `COrderSr` / CPU sampler) using an explicit bound per function (see `benchmark/function_suite/README.md`). `ratio` = observed deviation / bound; PASS needs ratio <= 1. RNG state, draw counts and configurations are compared exactly.

| machine | PASS | FAIL | ERROR | of which known issues | NotAvailable | SKIPPED | overall |
|---|---|---|---|---|---|---|---|
| NVIDIA GeForce RTX 3060 (witch) | 116 | 2 | 0 | 2 | 1 | 1 | **FAIL** |


### Failures

| machine | family | function | variant | dtype | params | verdict / label | ratio | note |
|---|---|---|---|---|---|---|---|---|
| NVIDIA GeForce RTX 3060 (witch) | sampler | sampler_resident_inverse | cuda-pinned | f64 | L=16;Wc=2;NVMCSample=20;setup=shared-wavefunction | FAIL KNOWN-ISSUE [#465](https://github.com/AtelierArith/mvmc-rs/issues/465) | 5.34e+07 | settled download (0.5 s later): inv_rel=6.92e-1 pf_rel=5.34e-1; w0:inv=6.9e-1;pf=5.3e-1 [8 of 8 QPs differ; qp0: cpu_pf=2.245e5 dev_pf=2.123e5] w1:inv=2.5e-16;p |
| NVIDIA GeForce RTX 3060 (witch) | sampler | sampler_resident_inverse | cuda-pageable | f64 | L=16;Wc=2;NVMCSample=20;setup=shared-wavefunction | FAIL KNOWN-ISSUE [#465](https://github.com/AtelierArith/mvmc-rs/issues/465) | 5.34e+07 | settled download (0.5 s later): inv_rel=6.92e-1 pf_rel=5.34e-1; w0:inv=6.9e-1;pf=5.3e-1 [8 of 8 QPs differ; qp0: cpu_pf=2.245e5 dev_pf=2.123e5] w1:inv=2.5e-16;p |


KNOWN-ISSUE rows are tracked defects (still FAIL in the counts above); NEW rows are untracked.


### Worst deviation/bound per function and variant, by machine

Values are the worst ratio over all sizes (its parameters in parentheses); a cell is `FAIL` when any size fails.


| family | function | variant | dtype | NVIDIA GeForce RTX 3060 (witch) |
|---|---|---|---|---|
| pfaffian | pfaffian_inverse | cpu-pfapack-rayon | c64 | 0.00e+00 (n=16;NQP=8;B=1;planes=8) |
| pfaffian | pfaffian_inverse | cpu-pfapack-rayon | f64 | 0.00e+00 (n=16;NQP=8;B=1;planes=8) |
| pfaffian | pfaffian_inverse | cuda-total-with-transfers | c64 | 8.83e-04 (n=16;NQP=8;B=16;planes=128) |
| pfaffian | pfaffian_inverse | cuda-total-with-transfers | f64 | 3.50e-04 (n=16;NQP=8;B=16;planes=128) |
| pfaffian | pfaffian_inverse | tenferro-extop-cpu | c64 | 0.00e+00 (n=16;NQP=8;B=1;planes=8) |
| pfaffian | pfaffian_inverse | tenferro-extop-cpu | f64 | 0.00e+00 (n=16;NQP=8;B=1;planes=8) |
| pfaffian | pfaffian_inverse | tenferro-native-cpu | c64 | 1.63e-01 (n=64;NQP=8;B=16;planes=128) |
| pfaffian | pfaffian_inverse | tenferro-native-cpu | f64 | 4.93e-03 (n=64;NQP=8;B=16;planes=128) |
| pfaffian | pfaffian_inverse_invariants | cuda | c64 | 4.71e-04 (n=16;NQP=8;B=1;planes=2) |
| pfaffian | pfaffian_inverse_invariants | cuda | f64 | 8.48e-04 (n=16;NQP=8;B=1;planes=2) |
| sampler | sampler_resident_inverse | cuda-pageable | f64 | FAIL 5.34e+07 (L=16;Wc=2;NVMCSample=20;setup=shared-wavefunction) |
| sampler | sampler_resident_inverse | cuda-pinned | f64 | FAIL 5.34e+07 (L=16;Wc=2;NVMCSample=20;setup=shared-wavefunction) |
| sampler | sampler_teacher_forced | cuda-pageable | f64 | 2.77e-04 (L=32;Wc=2;NVMCSample=20;setup=seed7) |
| sampler | sampler_teacher_forced | cuda-pinned | f64 | 2.77e-04 (L=32;Wc=2;NVMCSample=20;setup=seed7) |
| sr | assemble_s_g | tenferro-cpu-faer | f64 | 0.00e+00 (NPara=200;samples=1000;cg_iters=25) |
| sr | assemble_s_g | tenferro-cuda | f64 | 0.00e+00 (NPara=200;samples=1000;cg_iters=25) |
| sr | cg_matvec | tenferro-cpu-faer | f64 | 6.67e-04 (NPara=200;samples=1000;cg_iters=25) |
| sr | cg_matvec | tenferro-cuda | f64 | 7.78e-04 (NPara=200;samples=1000;cg_iters=25) |
| sr | cg_solve | tenferro-cpu-faer | f64 | 3.25e-06 (NPara=200;samples=1000;cg_iters=25) |
| sr | cg_solve | tenferro-cuda | f64 | 3.25e-06 (NPara=200;samples=1000;cg_iters=25) |
| sr | cholesky | tenferro-cuda | c64 | 7.84e-06 (n=512) |
| sr | cholesky | tenferro-cuda | f64 | 3.14e-06 (n=512) |
| sr | cholesky_solve | tenferro-cpu-faer | f64 | 8.97e-04 (NPara=200;samples=1000;cg_iters=25) |
| sr | cholesky_solve | tenferro-cuda | f64 | 8.97e-04 (NPara=200;samples=1000;cg_iters=25) |
| sr | dot_general | tenferro-cuda | c64 | 2.63e-04 (n=512) |
| sr | dot_general | tenferro-cuda | f64 | 0.00e+00 (n=128) |
| sr | gram | tenferro-cpu-faer | f64 | 1.44e-03 (NPara=200;samples=1000;cg_iters=25) |
| sr | gram | tenferro-cuda | f64 | 1.26e-03 (NPara=1000;samples=2000;cg_iters=25) |
| transfers | roundtrip_integrity | cudarc-pageable | f64 | 0.00e+00 (bytes=512) |
| transfers | roundtrip_integrity | cudarc-pinned | f64 | 0.00e+00 (bytes=512) |
| transfers | roundtrip_integrity | tenferro | f64 | 0.00e+00 (bytes=512) |


### Exact-match checks (no tolerance)

| machine | function | variant | dtype | params | note |
|---|---|---|---|---|---|
| NVIDIA GeForce RTX 3060 (witch) | pfaffian_inverse_degenerate | cuda | f64 | n=8;planes=4 | status codes identical to the oracle |
| NVIDIA GeForce RTX 3060 (witch) | pfaffian_inverse_degenerate | cuda | c64 | n=8;planes=4 | status codes identical to the oracle |
| NVIDIA GeForce RTX 3060 (witch) | sampler_free_run_vs_cpu | cuda-pinned | f64 | L=16;Wc=2;NVMCSample=20;setup=seed7 | bit-identical trajectory; proposals=960 accepts=377 recomputes=21 passes=531 |
| NVIDIA GeForce RTX 3060 (witch) | sampler_free_run_vs_cpu | cuda-pageable | f64 | L=16;Wc=2;NVMCSample=20;setup=seed7 | bit-identical trajectory; proposals=960 accepts=377 recomputes=21 passes=493 |
| NVIDIA GeForce RTX 3060 (witch) | sampler_free_run_vs_cpu | cuda-pinned | f64 | L=16;Wc=2;NVMCSample=20;setup=shared-wavefunction | bit-identical trajectory; proposals=960 accepts=374 recomputes=22 passes=525 |
| NVIDIA GeForce RTX 3060 (witch) | sampler_free_run_vs_cpu | cuda-pageable | f64 | L=16;Wc=2;NVMCSample=20;setup=shared-wavefunction | bit-identical trajectory; proposals=960 accepts=374 recomputes=22 passes=492 |
| NVIDIA GeForce RTX 3060 (witch) | sampler_free_run_vs_cpu | cuda-pinned | f64 | L=32;Wc=2;NVMCSample=20;setup=seed7 | bit-identical trajectory; proposals=1920 accepts=762 recomputes=22 passes=1099 |
| NVIDIA GeForce RTX 3060 (witch) | sampler_free_run_vs_cpu | cuda-pageable | f64 | L=32;Wc=2;NVMCSample=20;setup=seed7 | bit-identical trajectory; proposals=1920 accepts=762 recomputes=22 passes=972 |
| NVIDIA GeForce RTX 3060 (witch) | sampler_free_run_vs_cpu | cuda-pinned | f64 | L=32;Wc=2;NVMCSample=20;setup=shared-wavefunction | bit-identical trajectory; proposals=1920 accepts=711 recomputes=21 passes=1083 |
| NVIDIA GeForce RTX 3060 (witch) | sampler_free_run_vs_cpu | cuda-pageable | f64 | L=32;Wc=2;NVMCSample=20;setup=shared-wavefunction | bit-identical trajectory; proposals=1920 accepts=711 recomputes=21 passes=972 |


## 2. Timing (reference)

Secondary result: medians after warm-up. CUDA times include host-device transfers unless the row says kernel-only. Speedup > 1 means the GPU is faster. Timings are only meaningful for variants whose numerical verdict in section 1 is PASS.


### NVIDIA GeForce RTX 3060 (witch)


#### Batched Pfaffian + inverse (time per plane, NQP = 8 planes per batch element)

| dtype | n | B | 1 thread | all cores (rayon) | CUDA total | CUDA kernel | x vs 1 thread | x vs all cores | kernel x vs all cores |
|---|---|---|---|---|---|---|---|---|---|
| c64 | 16 | 1 | 12.3 us | 19.3 us | 50.9 us | 21.5 us | 0.2x | 0.4x | 0.9x |
| c64 | 16 | 16 | 9.5 us | 4.7 us | 7.7 us | 2.4 us | 1.2x | 0.6x | 1.9x |
| c64 | 64 | 1 | 828.1 us | 255.1 us | 442.0 us | 293.0 us | 1.9x | 0.6x | 0.9x |
| c64 | 64 | 16 | 795.1 us | 75.0 us | 288.0 us | 70.5 us | 2.8x | 0.3x | 1.1x |
| c64 | 256 | 1 | 31.52 ms | 11.64 ms | 13.58 ms | 10.53 ms | 2.3x | 0.9x | 1.1x |
| f64 | 16 | 1 | 5.0 us | 19.9 us | 38.6 us | 13.7 us | 0.1x | 0.5x | 1.4x |
| f64 | 16 | 16 | 5.7 us | 3.2 us | 4.1 us | 1.1 us | 1.4x | 0.8x | 3.0x |
| f64 | 64 | 1 | 282.4 us | 107.0 us | 209.6 us | 124.7 us | 1.3x | 0.5x | 0.9x |
| f64 | 64 | 16 | 271.0 us | 56.6 us | 199.5 us | 18.7 us | 1.4x | 0.3x | 3.0x |
| f64 | 256 | 1 | 9.70 ms | 2.71 ms | 3.56 ms | 2.94 ms | 2.7x | 0.8x | 0.9x |
| f64 | 256 | 16 | 7.43 ms | 1.42 ms | 3.68 ms | 1.29 ms | 2.0x | 0.4x | 1.1x |


Break-even (smallest B where CUDA incl. transfers beats the CPU): c64 n=16: all cores B>=never, 1 thread B>=16; c64 n=64: all cores B>=never, 1 thread B>=1; c64 n=256: all cores B>=never, 1 thread B>=1; f64 n=16: all cores B>=never, 1 thread B>=16; f64 n=64: all cores B>=never, 1 thread B>=1; f64 n=256: all cores B>=never, 1 thread B>=1


#### SR stages

| stage | NPara | samples | C-order 1 core | C-order all cores | tenferro CPU | tenferro CUDA | x vs all cores | x vs 1 core |
|---|---|---|---|---|---|---|---|---|
| assemble_s_g | 200 | 1000 | 130.9 us | 334.7 us | 1.47 ms | 2.69 ms | 0.1x | 0.0x |
| assemble_s_g | 1000 | 2000 | 2.91 ms | 2.90 ms | 47.63 ms | 40.35 ms | 0.1x | 0.1x |
| cg_matvec | 200 | 1000 | 124.5 us | 208.7 us | 466.4 us | 503.4 us | 0.4x | 0.2x |
| cg_matvec | 1000 | 2000 | 1.06 ms | 1.09 ms | 757.1 us | 353.4 us | 3.1x | 3.0x |
| cg_operand_upload | 200 | 1000 | - | - | - | 8.00 ms | - | - |
| cg_operand_upload | 1000 | 2000 | - | - | - | 47.31 ms | - | - |
| cg_solve | 200 | 1000 | 3.17 ms | 5.30 ms | 11.75 ms | 13.29 ms | 0.4x | 0.2x |
| cg_solve | 1000 | 2000 | 27.34 ms | 28.87 ms | 26.88 ms | 55.66 ms | 0.5x | 0.5x |
| cholesky_solve | 200 | 1000 | 342.7 us | 669.2 us | 3.14 ms | 2.72 ms | 0.2x | 0.1x |
| cholesky_solve | 1000 | 2000 | 15.11 ms | 14.42 ms | 39.30 ms | 26.36 ms | 0.5x | 0.6x |
| gram | 200 | 1000 | 2.90 ms | 5.67 ms | 3.58 ms | 2.00 ms | 2.8x | 1.5x |
| gram | 1000 | 2000 | 91.06 ms | 95.65 ms | 56.68 ms | 55.76 ms | 1.7x | 1.6x |


#### Sampler end to end (one `VMCMakeSample` call per walker)

| L | W | CPU 1 thread | CPU multichain | CUDA pinned | CUDA pageable | x vs multichain |
|---|---|---|---|---|---|---|
| 16 | 1 | 5.34 ms | 6.25 ms | 47.28 ms | 40.68 ms | 0.13x |
| 16 | 8 | 41.50 ms | 6.90 ms | 123.48 ms | 269.62 ms | 0.06x |
| 16 | 64 | 326.22 ms | 41.32 ms | 387.30 ms | 429.42 ms | 0.11x |
| 32 | 1 | 11.48 ms | 11.57 ms | 48.42 ms | 43.79 ms | 0.24x |
| 32 | 8 | 84.12 ms | 12.11 ms | 83.51 ms | 251.44 ms | 0.15x |
| 32 | 64 | 490.19 ms | 35.06 ms | 344.89 ms | 523.38 ms | 0.10x |


#### Sampler stages (device-event time of one profiled call)

| stage | L | W | time |
|---|---|---|---|
| sampler_propose_ratios | 16 | 1 | 8.96 ms |
| sampler_accept_updates | 16 | 1 | 7.06 ms |
| sampler_recompute_slow_lane | 16 | 1 | 2.65 ms |
| sampler_host_staging | 16 | 1 | 507.7 us |
| sampler_upload | 16 | 1 | 8.66 ms |
| sampler_launch | 16 | 1 | 22.46 ms |
| sampler_wait_fast_lane | 16 | 1 | 8.48 ms |
| sampler_propose_ratios | 16 | 8 | 28.45 ms |
| sampler_accept_updates | 16 | 8 | 33.71 ms |
| sampler_recompute_slow_lane | 16 | 8 | 22.99 ms |
| sampler_host_staging | 16 | 8 | 2.59 ms |
| sampler_upload | 16 | 8 | 20.16 ms |
| sampler_launch | 16 | 8 | 48.23 ms |
| sampler_wait_fast_lane | 16 | 8 | 188.45 ms |
| sampler_propose_ratios | 16 | 64 | 91.43 ms |
| sampler_accept_updates | 16 | 64 | 30.09 ms |
| sampler_recompute_slow_lane | 16 | 64 | 97.68 ms |
| sampler_host_staging | 16 | 64 | 9.61 ms |
| sampler_upload | 16 | 64 | 37.92 ms |
| sampler_launch | 16 | 64 | 81.52 ms |
| sampler_wait_fast_lane | 16 | 64 | 49.49 ms |
| sampler_propose_ratios | 32 | 1 | 8.23 ms |
| sampler_accept_updates | 32 | 1 | 9.44 ms |
| sampler_recompute_slow_lane | 32 | 1 | 4.31 ms |
| sampler_host_staging | 32 | 1 | 434.2 us |
| sampler_upload | 32 | 1 | 7.46 ms |
| sampler_launch | 32 | 1 | 19.34 ms |
| sampler_wait_fast_lane | 32 | 1 | 10.87 ms |
| sampler_propose_ratios | 32 | 8 | 11.64 ms |
| sampler_accept_updates | 32 | 8 | 23.54 ms |
| sampler_recompute_slow_lane | 32 | 8 | 31.06 ms |
| sampler_host_staging | 32 | 8 | 1.20 ms |
| sampler_upload | 32 | 8 | 9.63 ms |
| sampler_launch | 32 | 8 | 28.49 ms |
| sampler_wait_fast_lane | 32 | 8 | 17.88 ms |
| sampler_propose_ratios | 32 | 64 | 66.26 ms |
| sampler_accept_updates | 32 | 64 | 58.02 ms |
| sampler_recompute_slow_lane | 32 | 64 | 186.78 ms |
| sampler_host_staging | 32 | 64 | 7.55 ms |
| sampler_upload | 32 | 64 | 28.08 ms |
| sampler_launch | 32 | 64 | 58.67 ms |
| sampler_wait_fast_lane | 32 | 64 | 75.89 ms |


#### Transfers

| bytes | path | time | bandwidth |
|---|---|---|---|
| 32768 | pageable_up | 17.8 us | 1.8 GB/s |
| 32768 | pageable_down | 14.1 us | 2.3 GB/s |
| 32768 | pinned_wc_up | 10.3 us | 3.2 GB/s |
| 32768 | pinned_cached_up | 8.9 us | 3.7 GB/s |
| 32768 | pinned_down | 9.2 us | 3.6 GB/s |
| 32768 | pinned_up_enqueue | 3.1 us | 10.5 GB/s |
| 32768 | tenferro_up | 39.6 us | 0.8 GB/s |
| 32768 | tenferro_down | 29.8 us | 1.1 GB/s |
| 1048576 | pageable_up | 220.1 us | 4.8 GB/s |
| 1048576 | pageable_down | 210.4 us | 5.0 GB/s |
| 1048576 | pinned_wc_up | 93.8 us | 11.2 GB/s |
| 1048576 | pinned_cached_up | 90.9 us | 11.5 GB/s |
| 1048576 | pinned_down | 86.9 us | 12.1 GB/s |
| 1048576 | pinned_up_enqueue | 3.2 us | 324.1 GB/s |
| 1048576 | tenferro_up | 574.5 us | 1.8 GB/s |
| 1048576 | tenferro_down | 208.4 us | 5.0 GB/s |
| 16777216 | pageable_up | 2.21 ms | 7.6 GB/s |
| 16777216 | pageable_down | 2.07 ms | 8.1 GB/s |
| 16777216 | pinned_wc_up | 1.36 ms | 12.3 GB/s |
| 16777216 | pinned_cached_up | 1.35 ms | 12.4 GB/s |
| 16777216 | pinned_down | 1.28 ms | 13.1 GB/s |
| 16777216 | pinned_up_enqueue | 4.2 us | 3970.9 GB/s |
| 16777216 | tenferro_up | 25.23 ms | 0.7 GB/s |
| 16777216 | tenferro_down | 17.97 ms | 0.9 GB/s |


## 3. GPU-ize? recommendation

Rule: a function is only recommended when its GPU variant passes every numerical check on every machine. Speed classes (vs the best CPU baseline, all cores, transfers included): >= 2x at a realistic size = YES; 1x-2x = MARGINAL (depends on host core count and transfer overlap); < 1x = NO. Break-even sizes are per machine.

| machine | function | numerics | best speedup vs CPU all cores | break-even (B / NPara / W) | GPU-ize? | reasoning |
|---|---|---|---|---|---|---|
| NVIDIA GeForce RTX 3060 (witch) | batched Pfaffian+inverse c64 n=16 | PASS | 0.6x (B=16) | never | NO | transfers are 3.2x the kernel time; FP64 peak 0.2 (FP32/64 = 64:1) TFLOPS |
| NVIDIA GeForce RTX 3060 (witch) | batched Pfaffian+inverse c64 n=64 | PASS | 0.6x (B=1) | never | NO | transfers are 1.5x the kernel time; FP64 peak 0.2 (FP32/64 = 64:1) TFLOPS |
| NVIDIA GeForce RTX 3060 (witch) | batched Pfaffian+inverse c64 n=256 | PASS | 0.9x (B=1) | never | NO | transfers are 1.3x the kernel time; FP64 peak 0.2 (FP32/64 = 64:1) TFLOPS |
| NVIDIA GeForce RTX 3060 (witch) | batched Pfaffian+inverse f64 n=16 | PASS | 0.8x (B=16) | never | NO | transfers are 3.8x the kernel time; FP64 peak 0.2 (FP32/64 = 64:1) TFLOPS |
| NVIDIA GeForce RTX 3060 (witch) | batched Pfaffian+inverse f64 n=64 | PASS | 0.5x (B=1) | never | NO | transfers are 1.7x the kernel time; FP64 peak 0.2 (FP32/64 = 64:1) TFLOPS |
| NVIDIA GeForce RTX 3060 (witch) | batched Pfaffian+inverse f64 n=256 | PASS | 0.8x (B=1) | never | NO | transfers are 1.2x the kernel time; FP64 peak 0.2 (FP32/64 = 64:1) TFLOPS |
| NVIDIA GeForce RTX 3060 (witch) | SR gram | PASS | 2.8x (NPara=200, samples=1000) | 200 | YES | per-call upload/download of the stage operands included |
| NVIDIA GeForce RTX 3060 (witch) | SR assemble_s_g | PASS | 0.1x (NPara=200, samples=1000) | never | NO | per-call upload/download of the stage operands included |
| NVIDIA GeForce RTX 3060 (witch) | SR cholesky_solve | PASS | 0.5x (NPara=1000, samples=2000) | never | NO | per-call upload/download of the stage operands included |
| NVIDIA GeForce RTX 3060 (witch) | SR cg_matvec | PASS | 3.1x (NPara=1000, samples=2000) | 1000 | YES | constant operand uploaded once per solve |
| NVIDIA GeForce RTX 3060 (witch) | SR cg_solve | PASS | 0.5x (NPara=1000, samples=2000) | never | NO | per-call upload/download of the stage operands included |
| NVIDIA GeForce RTX 3060 (witch) | sampler lock-step L=16 | FAIL | 0.13x (W=1) | never | NO (fix numerics first) | vs CPU multichain on 36 cores |
| NVIDIA GeForce RTX 3060 (witch) | sampler lock-step L=32 | FAIL | 0.24x (W=1) | never | NO (fix numerics first) | vs CPU multichain on 36 cores |


## 4. Not available and skipped

| machine | family | function | variant | params | status | reason |
|---|---|---|---|---|---|---|
| NVIDIA GeForce RTX 3060 (witch) | pfaffian | pfaffian_inverse | all | n=256;NQP=8;B=16;planes=128 | SKIPPED | exceeds --max-bytes |
| NVIDIA GeForce RTX 3060 (witch) | sr_resident | all | - |  | NotAvailable | device-resident SR (#447) is not in this checkout (examples/bench_sr_resident.rs absent) |


## Metadata

| key | NVIDIA GeForce RTX 3060 (witch) |
|---|---|
| host | witch |
| profile | quick |
| git_rev | 8e75fa7120ba72fcc98ce018f1508d4f72b9976a |
| mode | native |
| gpu_model | NVIDIA GeForce RTX 3060 |
| gpu_compute_capability | 8.6 |
| gpu_memory_total | 12288 MiB |
| gpu_driver | 580.178.04 |
| gpu_fp64_peak_tflops | 0.2 (FP32/64 = 64:1) |
| cuda_toolkit | 12.9 |
| cpu_model | Intel(R) Xeon(R) CPU E5-2699 v3 @ 2.30GHz |
| cpu_logical_cores | 36 |
| cpu_available_cores | 36 |
| os | Ubuntu 24.04.5 LTS / kernel 6.8.0-146-generic |
| rustc | rustc 1.99.0 (b940084d7 2026-09-28) |
| crate_tenferro-gpu | 0.7.1 |
| crate_cudarc | 0.19.10 |
| env_OMP_NUM_THREADS | <unset> |
| env_OPENBLAS_NUM_THREADS | <unset> |
| env_RAYON_NUM_THREADS | <unset> |

