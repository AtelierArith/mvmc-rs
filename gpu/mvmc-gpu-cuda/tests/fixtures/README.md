# Interval sampler / Julia GPU shared-tape parity fixture

`interval_parity_j8.txt` is the reference output of the Julia GPU sampler for the
shared-tape parity test `../interval_parity.rs` (issue #547).

## Provenance

- Generator: `gen_interval_parity.jl`
- Fixture SHA-256: `ee610d99a6f0e9dc2ddc0518bcaa642d17f4add401feac2b677f5a77495079fe`
- Generator SHA-256: `073dad9ef8d0458105a48cd0be90bdfed30ab950aa587dfa39c9ad30e6bf4dfc`
- Julia 1.13.1 (`julia +1.13.1 --project=extern/Julia-mVMC/test/gpu`)
- CUDA.jl 6.4.2, CUDA runtime jll 0.25.0
- NVIDIA A100 80GB PCIe, driver 580.126.09, sm_80

## Regenerate

```sh
julia +1.13.1 --project=extern/Julia-mVMC/test/gpu \
    gpu/mvmc-gpu-cuda/tests/fixtures/gen_interval_parity.jl
```

Then run the Rust side (requires CUDA):

```sh
LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:/usr/local/cuda-12.6/targets/x86_64-linux/lib \
cargo test --manifest-path gpu/mvmc-gpu-cuda/Cargo.toml --test interval_parity -- \
    --ignored --nocapture
```

## Contract

The generator builds a small synthetic, projection-free state (`nsite=8`, `ne=4`,
`nqp=2`, one walker), a fixed device tape of `5` `UInt32` words per step in the
Julia `[word][walker][step]` layout, and runs the Julia GPU path
`gpu_sample!` -> `cached_sampling_kernel!` with `empty_destinations=true` and
`exchange_fraction=0` (`NExUpdatePath=0`). It dumps the initial `slater`,
`weights`, configuration and tape, and the final configuration and occupancy.

The Rust `IntervalSampler` builds the same initial state through
`calc_m_all_real`, replays the same tape and must reproduce the final
configuration and occupancy. Because both implementations read the same tape,
the RNG stream is out of scope: only the sampler semantics (proposal, rank-one
ratio, Metropolis decision, rank-two inverse update, rebuild window) are
compared. Expected final configuration:

```
ele = [4, 6, 0, 2, 4, 6, 1, 7]
occupancy = [1,0,1,0,1,0,1,0, 0,1,0,0,1,0,1,1]
accepted = 12
```
