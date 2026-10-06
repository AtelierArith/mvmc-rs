# mvmc-rs function-level GPU/CPU suite report

## 1. Numerical validation

Primary result. Every GPU/tenferro/CPU variant is compared with the C-order CPU oracle (pfapack / `COrderSr` / CPU sampler) using an explicit bound per function (see `benchmark/function_suite/README.md`). `ratio` = observed deviation / bound; PASS needs ratio <= 1. RNG state, draw counts and configurations are compared exactly.

| machine | PASS | FAIL | ERROR | NotAvailable | SKIPPED | overall |
|---|---|---|---|---|---|---|
| NVIDIA GeForce RTX 3060 (witch) | 116 | 2 | 0 | 1 | 1 | **FAIL** |


### Failures

| machine | family | function | variant | dtype | params | verdict | ratio | note |
|---|---|---|---|---|---|---|---|---|
| NVIDIA GeForce RTX 3060 (witch) | sampler | sampler_resident_inverse | cuda-pinned | f64 | L=16;Wc=2;NVMCSample=20;setup=shared-wavefunction | FAIL | 5.34e+07 | settled download (0.5 s later): inv_rel=6.92e-1 pf_rel=5.34e-1; w0:inv=6.9e-1;pf=5.3e-1 [8 of 8 QPs differ; qp0: cpu_pf=2.245e5 dev_pf=2.123e5] w1:inv=2.5e-16;p |
| NVIDIA GeForce RTX 3060 (witch) | sampler | sampler_resident_inverse | cuda-pageable | f64 | L=16;Wc=2;NVMCSample=20;setup=shared-wavefunction | FAIL | 5.34e+07 | settled download (0.5 s later): inv_rel=6.92e-1 pf_rel=5.34e-1; w0:inv=6.9e-1;pf=5.3e-1 [8 of 8 QPs differ; qp0: cpu_pf=2.245e5 dev_pf=2.123e5] w1:inv=2.5e-16;p |


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
| sampler | sampler_teacher_forced | cuda-pageable | f64 | 5.86e-01 (L=32;Wc=2;NVMCSample=20;setup=seed7) |
| sampler | sampler_teacher_forced | cuda-pinned | f64 | 5.86e-01 (L=32;Wc=2;NVMCSample=20;setup=seed7) |
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
| NVIDIA GeForce RTX 3060 (witch) | sampler_free_run_vs_cpu | cuda-pinned | f64 | L=16;Wc=2;NVMCSample=20;setup=seed7 | bit-identical trajectory; proposals=960 accepts=377 recomputes=21 passes=527 |
| NVIDIA GeForce RTX 3060 (witch) | sampler_free_run_vs_cpu | cuda-pageable | f64 | L=16;Wc=2;NVMCSample=20;setup=seed7 | bit-identical trajectory; proposals=960 accepts=377 recomputes=21 passes=493 |
| NVIDIA GeForce RTX 3060 (witch) | sampler_free_run_vs_cpu | cuda-pinned | f64 | L=16;Wc=2;NVMCSample=20;setup=shared-wavefunction | bit-identical trajectory; proposals=960 accepts=374 recomputes=22 passes=519 |
| NVIDIA GeForce RTX 3060 (witch) | sampler_free_run_vs_cpu | cuda-pageable | f64 | L=16;Wc=2;NVMCSample=20;setup=shared-wavefunction | bit-identical trajectory; proposals=960 accepts=374 recomputes=22 passes=492 |
| NVIDIA GeForce RTX 3060 (witch) | sampler_free_run_vs_cpu | cuda-pinned | f64 | L=32;Wc=2;NVMCSample=20;setup=seed7 | bit-identical trajectory; proposals=1920 accepts=762 recomputes=22 passes=1099 |
| NVIDIA GeForce RTX 3060 (witch) | sampler_free_run_vs_cpu | cuda-pageable | f64 | L=32;Wc=2;NVMCSample=20;setup=seed7 | bit-identical trajectory; proposals=1920 accepts=762 recomputes=22 passes=972 |
| NVIDIA GeForce RTX 3060 (witch) | sampler_free_run_vs_cpu | cuda-pinned | f64 | L=32;Wc=2;NVMCSample=20;setup=shared-wavefunction | bit-identical trajectory; proposals=1920 accepts=711 recomputes=21 passes=1080 |
| NVIDIA GeForce RTX 3060 (witch) | sampler_free_run_vs_cpu | cuda-pageable | f64 | L=32;Wc=2;NVMCSample=20;setup=shared-wavefunction | bit-identical trajectory; proposals=1920 accepts=711 recomputes=21 passes=972 |


## 2. Timing (reference)

Secondary result: medians after warm-up. CUDA times include host-device transfers unless the row says kernel-only. Speedup > 1 means the GPU is faster. Timings are only meaningful for variants whose numerical verdict in section 1 is PASS.


### NVIDIA GeForce RTX 3060 (witch)


#### Batched Pfaffian + inverse (time per plane, NQP = 8 planes per batch element)

| dtype | n | B | 1 thread | all cores (rayon) | CUDA total | CUDA kernel | x vs 1 thread | x vs all cores | kernel x vs all cores |
|---|---|---|---|---|---|---|---|---|---|
| c64 | 16 | 1 | 11.4 us | 19.8 us | 53.0 us | 21.7 us | 0.2x | 0.4x | 0.9x |
| c64 | 16 | 16 | 9.9 us | 4.1 us | 7.0 us | 2.4 us | 1.4x | 0.6x | 1.7x |
| c64 | 64 | 1 | 712.4 us | 225.5 us | 434.2 us | 292.8 us | 1.6x | 0.5x | 0.8x |
| c64 | 64 | 16 | 2.08 ms | 127.0 us | 330.6 us | 70.5 us | 6.3x | 0.4x | 1.8x |
| c64 | 256 | 1 | 29.02 ms | 9.38 ms | 14.05 ms | 10.58 ms | 2.1x | 0.7x | 0.9x |
| f64 | 16 | 1 | 5.0 us | 20.1 us | 38.8 us | 13.7 us | 0.1x | 0.5x | 1.5x |
| f64 | 16 | 16 | 5.8 us | 3.4 us | 3.8 us | 1.0 us | 1.5x | 0.9x | 3.3x |
| f64 | 64 | 1 | 203.3 us | 85.4 us | 227.0 us | 126.3 us | 0.9x | 0.4x | 0.7x |
| f64 | 64 | 16 | 3.05 ms | 173.3 us | 162.6 us | 18.7 us | 18.8x | 1.1x | 9.3x |
| f64 | 256 | 1 | 7.16 ms | 2.32 ms | 3.69 ms | 2.94 ms | 1.9x | 0.6x | 0.8x |
| f64 | 256 | 16 | 10.50 ms | 1.59 ms | 4.22 ms | 1.29 ms | 2.5x | 0.4x | 1.2x |


Break-even (smallest B where CUDA incl. transfers beats the CPU): c64 n=16: all cores B>=never, 1 thread B>=16; c64 n=64: all cores B>=never, 1 thread B>=1; c64 n=256: all cores B>=never, 1 thread B>=1; f64 n=16: all cores B>=never, 1 thread B>=16; f64 n=64: all cores B>=16, 1 thread B>=16; f64 n=256: all cores B>=never, 1 thread B>=1


#### SR stages

| stage | NPara | samples | C-order 1 core | C-order all cores | tenferro CPU | tenferro CUDA | x vs all cores | x vs 1 core |
|---|---|---|---|---|---|---|---|---|
| assemble_s_g | 200 | 1000 | 126.0 us | 121.1 us | 1.68 ms | 3.03 ms | 0.0x | 0.0x |
| assemble_s_g | 1000 | 2000 | 9.10 ms | 10.37 ms | 58.80 ms | 48.79 ms | 0.2x | 0.2x |
| cg_matvec | 200 | 1000 | 120.4 us | 120.1 us | 732.7 us | 529.8 us | 0.2x | 0.2x |
| cg_matvec | 1000 | 2000 | 1.37 ms | 1.32 ms | 1.08 ms | 370.7 us | 3.6x | 3.7x |
| cg_operand_upload | 200 | 1000 | - | - | - | 8.59 ms | - | - |
| cg_operand_upload | 1000 | 2000 | - | - | - | 59.69 ms | - | - |
| cg_solve | 200 | 1000 | 3.07 ms | 3.05 ms | 14.90 ms | 14.03 ms | 0.2x | 0.2x |
| cg_solve | 1000 | 2000 | 30.81 ms | 34.69 ms | 27.89 ms | 68.58 ms | 0.5x | 0.4x |
| cholesky_solve | 200 | 1000 | 339.2 us | 333.0 us | 15.66 ms | 2.90 ms | 0.1x | 0.1x |
| cholesky_solve | 1000 | 2000 | 17.30 ms | 19.77 ms | 86.36 ms | 32.48 ms | 0.6x | 0.5x |
| gram | 200 | 1000 | 2.00 ms | 1.79 ms | 6.84 ms | 2.34 ms | 0.8x | 0.9x |
| gram | 1000 | 2000 | 77.10 ms | 89.92 ms | 61.73 ms | 62.53 ms | 1.4x | 1.2x |


#### Sampler end to end (one `VMCMakeSample` call per walker)

| L | W | CPU 1 thread | CPU multichain | CUDA pinned | CUDA pageable | x vs multichain |
|---|---|---|---|---|---|---|
| 16 | 1 | 5.68 ms | 5.66 ms | 47.43 ms | 40.84 ms | 0.12x |
| 16 | 8 | 42.35 ms | 7.20 ms | 75.84 ms | 81.61 ms | 0.09x |
| 16 | 64 | 310.87 ms | 40.32 ms | 376.83 ms | 1.08 s | 0.11x |
| 32 | 1 | 20.95 ms | 18.44 ms | 126.53 ms | 56.87 ms | 0.15x |
| 32 | 8 | 82.51 ms | 19.35 ms | 110.02 ms | 128.93 ms | 0.18x |
| 32 | 64 | 601.11 ms | 59.20 ms | 385.37 ms | 524.06 ms | 0.15x |


#### Sampler stages (device-event time of one profiled call)

| stage | L | W | time |
|---|---|---|---|
| sampler_propose_ratios | 16 | 1 | 8.91 ms |
| sampler_accept_updates | 16 | 1 | 6.86 ms |
| sampler_recompute_slow_lane | 16 | 1 | 2.69 ms |
| sampler_host_staging | 16 | 1 | 516.4 us |
| sampler_upload | 16 | 1 | 8.57 ms |
| sampler_launch | 16 | 1 | 22.47 ms |
| sampler_wait_fast_lane | 16 | 1 | 8.40 ms |
| sampler_propose_ratios | 16 | 8 | 13.71 ms |
| sampler_accept_updates | 16 | 8 | 17.43 ms |
| sampler_recompute_slow_lane | 16 | 8 | 19.56 ms |
| sampler_host_staging | 16 | 8 | 1.55 ms |
| sampler_upload | 16 | 8 | 11.48 ms |
| sampler_launch | 16 | 8 | 34.86 ms |
| sampler_wait_fast_lane | 16 | 8 | 8.30 ms |
| sampler_propose_ratios | 16 | 64 | 90.22 ms |
| sampler_accept_updates | 16 | 64 | 29.14 ms |
| sampler_recompute_slow_lane | 16 | 64 | 97.24 ms |
| sampler_host_staging | 16 | 64 | 9.29 ms |
| sampler_upload | 16 | 64 | 35.27 ms |
| sampler_launch | 16 | 64 | 76.85 ms |
| sampler_wait_fast_lane | 16 | 64 | 49.32 ms |
| sampler_propose_ratios | 32 | 1 | 11.79 ms |
| sampler_accept_updates | 32 | 1 | 11.49 ms |
| sampler_recompute_slow_lane | 32 | 1 | 4.50 ms |
| sampler_host_staging | 32 | 1 | 1.06 ms |
| sampler_upload | 32 | 1 | 15.36 ms |
| sampler_launch | 32 | 1 | 36.38 ms |
| sampler_wait_fast_lane | 32 | 1 | 5.74 ms |
| sampler_propose_ratios | 32 | 8 | 12.62 ms |
| sampler_accept_updates | 32 | 8 | 26.93 ms |
| sampler_recompute_slow_lane | 32 | 8 | 33.53 ms |
| sampler_host_staging | 32 | 8 | 1.92 ms |
| sampler_upload | 32 | 8 | 13.99 ms |
| sampler_launch | 32 | 8 | 37.95 ms |
| sampler_wait_fast_lane | 32 | 8 | 14.41 ms |
| sampler_propose_ratios | 32 | 64 | 76.33 ms |
| sampler_accept_updates | 32 | 64 | 55.89 ms |
| sampler_recompute_slow_lane | 32 | 64 | 189.53 ms |
| sampler_host_staging | 32 | 64 | 8.35 ms |
| sampler_upload | 32 | 64 | 28.64 ms |
| sampler_launch | 32 | 64 | 69.76 ms |
| sampler_wait_fast_lane | 32 | 64 | 72.08 ms |


#### Transfers

| bytes | path | time | bandwidth |
|---|---|---|---|
| 32768 | pageable_up | 19.1 us | 1.7 GB/s |
| 32768 | pageable_down | 15.4 us | 2.1 GB/s |
| 32768 | pinned_wc_up | 11.2 us | 2.9 GB/s |
| 32768 | pinned_cached_up | 9.5 us | 3.5 GB/s |
| 32768 | pinned_down | 9.8 us | 3.3 GB/s |
| 32768 | pinned_up_enqueue | 4.0 us | 8.1 GB/s |
| 32768 | tenferro_up | 43.1 us | 0.8 GB/s |
| 32768 | tenferro_down | 33.5 us | 1.0 GB/s |
| 1048576 | pageable_up | 239.9 us | 4.4 GB/s |
| 1048576 | pageable_down | 225.1 us | 4.7 GB/s |
| 1048576 | pinned_wc_up | 95.8 us | 11.0 GB/s |
| 1048576 | pinned_cached_up | 92.0 us | 11.4 GB/s |
| 1048576 | pinned_down | 88.3 us | 11.9 GB/s |
| 1048576 | pinned_up_enqueue | 4.6 us | 228.9 GB/s |
| 1048576 | tenferro_up | 709.2 us | 1.5 GB/s |
| 1048576 | tenferro_down | 228.1 us | 4.6 GB/s |
| 16777216 | pageable_up | 2.54 ms | 6.6 GB/s |
| 16777216 | pageable_down | 2.03 ms | 8.3 GB/s |
| 16777216 | pinned_wc_up | 1.36 ms | 12.3 GB/s |
| 16777216 | pinned_cached_up | 1.36 ms | 12.4 GB/s |
| 16777216 | pinned_down | 1.28 ms | 13.1 GB/s |
| 16777216 | pinned_up_enqueue | 11.0 us | 1528.3 GB/s |
| 16777216 | tenferro_up | 32.63 ms | 0.5 GB/s |
| 16777216 | tenferro_down | 21.56 ms | 0.8 GB/s |


## 3. GPU-ize? recommendation

Rule: a function is only recommended when its GPU variant passes every numerical check on every machine. Speed classes (vs the best CPU baseline, all cores, transfers included): >= 2x at a realistic size = YES; 1x-2x = MARGINAL (depends on host core count and transfer overlap); < 1x = NO. Break-even sizes are per machine.

| machine | function | numerics | best speedup vs CPU all cores | break-even (B / NPara / W) | GPU-ize? | reasoning |
|---|---|---|---|---|---|---|
| NVIDIA GeForce RTX 3060 (witch) | batched Pfaffian+inverse c64 n=16 | PASS | 0.6x (B=16) | never | NO | transfers are 2.9x the kernel time; FP64 peak 0.2 (FP32/64 = 64:1) TFLOPS |
| NVIDIA GeForce RTX 3060 (witch) | batched Pfaffian+inverse c64 n=64 | PASS | 0.5x (B=1) | never | NO | transfers are 1.5x the kernel time; FP64 peak 0.2 (FP32/64 = 64:1) TFLOPS |
| NVIDIA GeForce RTX 3060 (witch) | batched Pfaffian+inverse c64 n=256 | PASS | 0.7x (B=1) | never | NO | transfers are 1.3x the kernel time; FP64 peak 0.2 (FP32/64 = 64:1) TFLOPS |
| NVIDIA GeForce RTX 3060 (witch) | batched Pfaffian+inverse f64 n=16 | PASS | 0.9x (B=16) | never | NO | transfers are 3.7x the kernel time; FP64 peak 0.2 (FP32/64 = 64:1) TFLOPS |
| NVIDIA GeForce RTX 3060 (witch) | batched Pfaffian+inverse f64 n=64 | PASS | 1.1x (B=16) | 16 | MARGINAL | transfers are 8.7x the kernel time; FP64 peak 0.2 (FP32/64 = 64:1) TFLOPS |
| NVIDIA GeForce RTX 3060 (witch) | batched Pfaffian+inverse f64 n=256 | PASS | 0.6x (B=1) | never | NO | transfers are 1.3x the kernel time; FP64 peak 0.2 (FP32/64 = 64:1) TFLOPS |
| NVIDIA GeForce RTX 3060 (witch) | SR gram | PASS | 1.4x (NPara=1000, samples=2000) | 1000 | MARGINAL | per-call upload/download of the stage operands included |
| NVIDIA GeForce RTX 3060 (witch) | SR assemble_s_g | PASS | 0.2x (NPara=1000, samples=2000) | never | NO | per-call upload/download of the stage operands included |
| NVIDIA GeForce RTX 3060 (witch) | SR cholesky_solve | PASS | 0.6x (NPara=1000, samples=2000) | never | NO | per-call upload/download of the stage operands included |
| NVIDIA GeForce RTX 3060 (witch) | SR cg_matvec | PASS | 3.6x (NPara=1000, samples=2000) | 1000 | YES | constant operand uploaded once per solve |
| NVIDIA GeForce RTX 3060 (witch) | SR cg_solve | PASS | 0.5x (NPara=1000, samples=2000) | never | NO | per-call upload/download of the stage operands included |
| NVIDIA GeForce RTX 3060 (witch) | sampler lock-step L=16 | FAIL | 0.12x (W=1) | never | NO (fix numerics first) | vs CPU multichain on 36 cores |
| NVIDIA GeForce RTX 3060 (witch) | sampler lock-step L=32 | FAIL | 0.18x (W=8) | never | NO (fix numerics first) | vs CPU multichain on 36 cores |


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
| git_rev | d620833a10d217cbb934522ec6e09ba893adab88 |
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

