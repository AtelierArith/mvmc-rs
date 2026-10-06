# Draft upstream request for tensor4all/tenferro-rs (NOT filed)

Status: draft prepared for issue #432. Filing requires the user's approval; do not file without
it. Measurements are from `docs/design/gpu-readiness.md` section 10.7 (reproduce with
`scripts/run_cuda_gate.sh docker` from the mvmc-rs repository, test
`transfer_microbenchmark_report`).

## Title

`tenferro-gpu`: host-to-device / device-to-host transfers are 2 to 15x slower than plain
cudarc, and there is no pinned or asynchronous transfer API

## Environment

NVIDIA GeForce RTX 3060 (sm_86, PCIe 4.0 x16, 2 copy engines), driver 580.178.04 (CUDA driver
API 13.0), CUDA toolkit 12.9, tenferro 0.7.1 (crates.io, `tenferro-gpu` feature `cuda`), cudarc
0.19, Linux x86_64.

## Observation

`upload_tensor` / `download_tensor` (f64 vector, synchronized, medians) against cudarc 0.19
`memcpy_htod` / `memcpy_dtoh` in the same process:

| bytes | tenferro up GB/s | cudarc pageable up GB/s | cudarc pinned up GB/s | tenferro down GB/s | cudarc pageable down GB/s | cudarc pinned down GB/s |
|---|---|---|---|---|---|---|
| 1 MB | 1.59 | 4.48 | 11.07 | 4.95 | 4.73 | 11.85 |
| 4 MB | 0.72 | 6.13 | 12.03 | 1.00 | 7.22 | 12.56 |
| 16 MB | 0.91 | 6.96 | 12.31 | 3.33 | 7.81 | 12.74 |
| 64 MB | 0.52 | 7.72 | 12.41 | 1.36 | 10.09 | 13.17 |
| 256 MB | 0.53 | 7.79 | 12.44 | 1.43 | 12.07 | 13.19 |

Per-call fixed cost (8 bytes): tenferro 19 to 26 us, cudarc pageable 5 to 7 us, cudarc pinned
enqueue 4 us. The link is not the limit (pinned reaches 12.4 to 13.2 GB/s).

Reading the 0.7.1 sources (`tenferro-gpu/src/cubecl/memory.rs`): `upload_tensor` goes through CubeCL
`create_from_slice`; `download_tensor` calls `rt.synchronize()`, CubeCL `read_one`, then copies
the returned bytes once more (`T::from_bytes(..).to_vec()`). We have not profiled inside CubeCL;
the extra synchronization, staging and host copies are the candidates.

## Why it matters

Downstream (mVMC) keeps the Metropolis decision and the SFMT stream on the host for C parity, so
every step moves small buffers host to device and back (walker configurations in; Pfaffians,
energies and O vectors out, 4 KB to 1 MB). A device-resident design removes the large planes,
but the per-step round trip stays, and with a 2-stream ping-pong of walker groups the measured
end-to-end gain is only 1.3 to 1.6x (the ideal for two groups is 2x), so the transfer path must
not add cost. With two copy engines, host-to-device and device-to-host copies run concurrently
with each other and with a kernel (overlap 1.00 measured), but tenferro exposes no way to use
that.

## Requests

1. Make the existing synchronous path match cudarc: direct `cuMemcpyHtoD` / `cuMemcpyDtoH`
   (or the CubeCL equivalent) without the extra device synchronization and host copies; target
   at least the pageable cudarc rates above and a fixed cost near 5 to 10 us.
2. `download_into(rt, device_tensor, &mut [T])` (caller-owned destination) to avoid the
   allocation and the second host copy.
3. Pinned host buffers: an allocator (`cudaHostAlloc`, cached and write-combined variants) and
   transfer entry points that accept them (`upload_from_pinned`, `download_to_pinned`).
   Note write-combined memory is slow for small copies (4 KB: 51 us against 7 us cached).
4. Asynchronous transfers on a caller-selectable stream returning an event-like token
   (`TransferEvent` with `is_complete` / `wait`), and a way to make a compute stream wait on
   it, so groups can be pipelined without a device-wide synchronize.
5. Document the transfer path and its expected cost in `docs/guides/devices-and-gpu.md`.

## Workaround in use

mvmc-rs `gpu/mvmc-gpu-cuda/src/transfer.rs` implements pinned buffers and stream-ordered async
copies directly on cudarc for the hot transfers; tenferro stays in use for compute.
