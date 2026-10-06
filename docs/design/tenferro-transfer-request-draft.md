# tenferro-gpu: host-device transfers up to 26x slower than plain cudarc, caused by redundant host copies (upstream issue body, NOT filed)

Status: ready-to-file draft for tensor4all/tenferro-rs. The user approved filing after review;
do not file before that review. Numbers: RTX 3060, `docs/design/gpu-readiness.md` section 10.7
and the MWE below. Source references are to the crates.io 0.7.1 sources (`tenferro-gpu 0.7.1`,
`t4a-cubecl-runtime 0.10.1`, `t4a-cubecl-cuda 0.10.1`).

---

## Summary

`tenferro_gpu::cuda::upload_tensor` / `download_tensor` reach only 0.5 to 1.8 GB/s (up) and
1.3 to 6.6 GB/s (down) on a PCIe 4.0 x16 GPU where plain cudarc `memcpy_htod` / `memcpy_dtoh`
reaches 5 to 13 GB/s. Instrumenting the sources shows that the time is not spent on the PCIe
link or in the driver: **68 to 87 % of an upload and 61 to 82 % of a download is spent in
redundant host-side copies** (two extra `Vec<u8>` clones on upload, one extra `to_vec()` on
download), each into freshly allocated, page-faulting memory. The per-call fixed cost is also
about 3x cudarc's. There is no way to supply a pinned host buffer, to avoid the host copy, or
to run a transfer asynchronously.

## Environment

* GPU NVIDIA GeForce RTX 3060 (sm_86, PCIe 4.0 x16, 2 copy engines), driver 580.178.04
  (CUDA driver API 13.0)
* CUDA toolkit 12.9.2 (container `nvidia/cuda`-based; `nvcc` 12.9.86)
* rustc 1.99.0 (b940084d7 2026-09-28), Linux x86_64 (Ubuntu 24.04, Xeon E5-2699 v3)
* tenferro-gpu 0.7.1 (crates.io, feature `cuda`), t4a-cubecl-runtime / -cuda 0.10.1, cudarc 0.19.10
* Note: `tenferro-gpu` 0.7.1 with `cuda` pulls `lru ^0.12` (RUSTSEC-2026-0002 / -0253, #1958). It
  builds and runs unchanged; the MWE needs no patch.

## Minimal example

`Cargo.toml`:

```toml
[package]
name = "tenferro-transfer-mwe"
version = "0.0.0"
edition = "2021"

# crates.io only. tenferro-gpu 0.7.1 requires `lru ^0.12` when `cuda` is enabled; that range has
# RUSTSEC-2026-0002/-0253 advisories, which matters for `cargo audit` but not for this benchmark
# (tenferro #1958 tracks the bump). No patch is needed to build and run.
[dependencies]
tenferro-gpu = { version = "0.7.1", features = ["cuda"] }
tenferro-tensor = { version = "0.7.1", default-features = false }
cudarc = { version = "0.19", default-features = false, features = ["driver", "dynamic-loading", "cuda-12080"] }

[profile.release]
debug = 0
```

`src/main.rs`:

```rust
//! Host<->device transfer cost: tenferro-gpu `upload_tensor`/`download_tensor` vs cudarc memcpy.
//! Every call is followed by a stream/runtime synchronize; medians; time per call and GB/s.
use std::time::Instant;

use cudarc::driver::CudaContext;
use tenferro_gpu::cuda::{cuda_devices, download_tensor, upload_tensor, CudaBackend};
use tenferro_tensor::Tensor;

const SIZES: [usize; 6] = [8, 4 << 10, 64 << 10, 1 << 20, 16 << 20, 256 << 20];

fn median_us(reps: usize, mut f: impl FnMut()) -> f64 {
    f(); // warm-up
    let mut t: Vec<f64> = (0..reps)
        .map(|_| {
            let s = Instant::now();
            f();
            s.elapsed().as_secs_f64() * 1e6
        })
        .collect();
    t.sort_by(|a, b| a.total_cmp(b));
    t[t.len() / 2]
}

fn main() {
    // tenferro-gpu (CubeCL CUDA) runtime
    let dev = cuda_devices().unwrap().into_iter().next().expect("no CUDA device");
    let backend = CudaBackend::new(dev.id()).unwrap();
    // cudarc, same device
    let ctx = CudaContext::new(0).unwrap();
    let stream = ctx.new_stream().unwrap();
    println!("device: {} ({} bytes)", dev.name(), dev.total_memory_bytes());
    println!(
        "{:>10} | {:>24} | {:>24} | {:>24} | {:>24} | {:>24}",
        "bytes", "tenferro up", "tenferro down", "cudarc pageable up", "cudarc pinned up", "cudarc pinned down"
    );
    println!("{:>10} | {}", "", vec!["      us/call   GB/s"; 5].join(" | "));
    for &bytes in &SIZES {
        let n = bytes / 8;
        let reps = if bytes >= 64 << 20 { 8 } else { 30 };
        let cell = |us: f64| format!("{:>12.1} {:>10.2}", us, bytes as f64 / (us * 1e-6) / 1e9);

        // tenferro: upload_tensor (+ runtime synchronize), download_tensor (synchronizes itself)
        let host = Tensor::from_vec_col_major(vec![n], vec![1.0_f64; n]).unwrap();
        let mut last = None;
        let t_up = median_us(reps, || {
            let t = upload_tensor(backend.runtime(), &host).unwrap();
            backend.runtime().synchronize().unwrap();
            last = Some(t);
        });
        let resident = last.take().unwrap();
        let t_down = median_us(reps, || {
            std::hint::black_box(download_tensor(backend.runtime(), &resident).unwrap());
        });
        drop(resident);

        // cudarc pageable
        let mut dev_buf = unsafe { stream.alloc::<f64>(n) }.unwrap();
        let pageable = vec![1.0_f64; n];
        let c_up = median_us(reps, || {
            stream.memcpy_htod(&pageable, &mut dev_buf).unwrap();
            stream.synchronize().unwrap();
        });
        // cudarc pinned (ordinary page-locked memory, not write-combined)
        let mut pinned = unsafe { ctx.alloc_pinned_with_flags::<f64>(n, 0) }.unwrap();
        pinned.as_mut_slice().unwrap().fill(1.0);
        let p_up = median_us(reps, || {
            stream.memcpy_htod(&pinned, &mut dev_buf).unwrap();
            stream.synchronize().unwrap();
        });
        let p_down = median_us(reps, || {
            stream.memcpy_dtoh(&dev_buf, &mut pinned).unwrap();
            stream.synchronize().unwrap();
        });
        println!(
            "{:>10} | {} | {} | {} | {} | {}",
            bytes,
            cell(t_up),
            cell(t_down),
            cell(c_up),
            cell(p_up),
            cell(p_down)
        );
    }
}
```

Run on a machine with the NVIDIA container runtime and a Rust toolchain on the host (the CUDA
toolkit image needs no Rust): mount `~/.rustup` and `~/.cargo`, `cargo run --release`. Output
from the machine above (medians; 30 reps, 8 for 256 MB; every call includes its synchronize;
"pinned" is ordinary cudarc page-locked memory):

```
device: NVIDIA GeForce RTX 3060 (12488343552 bytes)
     bytes |              tenferro up |            tenferro down |       cudarc pageable up |         cudarc pinned up |       cudarc pinned down
         8 |         15.6       0.00 |         17.0       0.00 |          4.8       0.00 |          6.9       0.00 |          6.7       0.00
      4096 |         18.8       0.22 |         24.5       0.17 |          6.1       0.67 |          7.8       0.53 |          6.9       0.59
     65536 |         46.6       1.40 |         31.3       2.09 |         21.1       3.10 |         12.6       5.21 |         11.6       5.67
   1048576 |        587.4       1.79 |        207.2       5.06 |        192.7       5.44 |         92.7      11.31 |         87.6      11.98
  16777216 |       9977.4       1.68 |       2547.9       6.58 |       1476.7      11.36 |       1418.6      11.83 |       1293.1      12.97
 268435456 |     568242.1       0.47 |     206634.4       1.30 |      37997.1       7.06 |      21639.0      12.41 |      20530.7      13.07
```

Ratios at 256 MB: upload 15x slower than cudarc pageable and 26x slower than pinned; download
10x slower than pinned (cudarc pageable download reaches 12 GB/s at this size). At 4 KB to 64 KB
a call costs 2 to 3x cudarc's.

## Root cause (source-level, with a measured breakdown)

Upload, `upload_tensor` -> `upload_typed`:

1. `tenferro-gpu/src/cubecl/memory.rs:137` `client.create_from_slice(T::as_bytes(host_data))`.
2. `t4a-cubecl-runtime/src/client.rs:296` `vec![slice.to_vec()]`: **host copy #1** of the whole
   buffer, into a freshly allocated `Vec<u8>`.
3. `client.rs:232` `Bytes::from_bytes_vec(data.to_vec())` in `do_create_from_slices`:
   **host copy #2**, cloning the `Vec` that was just created (it is passed by value; the clone
   is not needed).
4. `client.rs:238` `self.device.submit(..)` hands the buffer to the server thread, which runs
   `initialize_memory` (device reserve, `t4a-cubecl-cuda/src/compute/server.rs`) and
   `write_to_gpu` (`command.rs:361`), which issues `memcpy_htod_async` from the **pageable**
   `Vec` (`command.rs:542`; the `Bytes` is not pinned, only `AllocationProperty::File` data is
   restaged, `command.rs:377`) and keeps the buffer in the drop queue (`command.rs:401`).

Download, `download_typed`:

1. `memory.rs:217` `rt.synchronize()` (stream sync before the read).
2. `memory.rs:220` `client.read_one(handle)` -> server `read` -> `command.rs:281`
   `copy_to_bytes` -> `reserve_cpu` (`command.rs:155`): a pinned staging buffer from the CubeCL
   host pool for sizes up to 100 MB, but **`vec![0; size]` (zero-filled pageable memory) above
   100 MB** (`command.rs:162`); then `memcpy_dtoh_async` and a fence wait (`command.rs:200-215`).
3. `memory.rs:222` `T::from_bytes(&bytes).to_vec()`: **host copy** of the whole result into a
   new `Vec<T>`.

Measured breakdown (the same sources with `Instant` timers at these points, `eprintln!`; RTX 3060,
third iteration, i.e. after warm-up):

| step | 16 MB | 256 MB |
|---|---|---|
| upload: `slice.to_vec()` (`client.rs:296`) | 2.3 ms | 181 ms |
| upload: `data.to_vec()` (`client.rs:232`) | 4.7 ms | 381 ms |
| upload: caller side in total | 7.0 ms | 586 ms |
| upload: `memcpy_htod_async` issue + `runtime.synchronize` (the real transfer) | 3.1 + 3.3 ms | 38 + 58 ms |
| upload share spent in the two redundant host copies | 68 % | 87 % |
| download: `read_one` (`memcpy_dtoh_async` + fence) | 1.3 ms | 20.5 ms |
| download: `T::from_bytes(..).to_vec()` (`memory.rs:222`) | 2.2 ms | 162 ms |
| download: total | 3.6 ms | 197 ms |
| download share spent in the redundant host copy | 60 % | 82 % |

The host copies run at 0.7 to 1.4 GB/s because the destination is new memory every call (page
faults on first touch of a 256 MB allocation dominate). The first download after start also pays
the pinned pool growth (`reserve_cpu`: 100 ms at 16 MB, 139 ms at 256 MB on iteration 0), and
every call pays a device-thread hand-off plus a CUDA event create/record/synchronize/destroy
(`t4a-cubecl-cuda/src/compute/sync/fence.rs:24-75`), which is the 2 to 3x fixed cost at small sizes (not separately measured).

## Requests

1. **Remove the redundant host copies.** `client.rs:232` should move `data` instead of
   cloning it, and `create_from_slice` (`client.rs:296`) should copy the caller's slice exactly
   once, directly into a pinned staging buffer (the `create_with_data` path of `command.rs`
   already does this for the pinned pool) and issue the async H2D from there. For downloads,
   expose the `Bytes` (or write straight into a caller-provided `&mut [T]`,
   `download_into`) instead of `to_vec()`.
2. **Do not fall back to `vec![0; n]` above 100 MB** (`command.rs:162`): use chunked pinned
   staging; the zero fill plus first-touch page faults cost more than the transfer.
3. **Pinned host buffers and async transfers**: an allocator for pinned (cached and
   write-combined; note write-combined is 7x slower for 4 KB copies) host memory, entry points
   that take them (`upload_from_pinned`, `download_to_pinned`), and an async variant on a
   caller-chosen stream returning an event-like token (`is_complete`, `wait`, and a way for a
   compute stream to wait on it), so independent groups can be pipelined. Measured on this GPU:
   pinned copies overlap a concurrent kernel completely (overlap 1.00) and the two copy engines
   run both directions concurrently.
4. Avoid the redundant stream synchronize before `read_one` (`memory.rs:217`) and the
   per-call fence for small transfers, or document the cost.
5. Document the transfer path and its expected rates in `docs/guides/devices-and-gpu.md`.

## Why it matters

mVMC (the many-variable variational Monte Carlo package, our Rust port
[AtelierArith/mvmc-rs](https://github.com/AtelierArith/mvmc-rs)) must keep the Metropolis
decision and the SFMT stream on the host to stay bit-compatible with the C reference, so every
Monte Carlo step moves small buffers host to device and back (walker configurations in;
Pfaffians, energies, O vectors out, 4 KB to 1 MB). The batched Pfaffian kernel of
[mvmc-rs#423](https://github.com/AtelierArith/mvmc-rs/issues/423) is 5 to 24x faster than one CPU
thread in the kernel alone, but transfers through tenferro erase the gain
([mvmc-rs#432](https://github.com/AtelierArith/mvmc-rs/issues/432)). We now bypass
`upload_tensor`/`download_tensor` for those transfers with a cudarc-based pinned/async helper;
fixing the path above would let tenferro users, including the tensor-network and AD workloads
that move results to the host every step, benefit as well.
