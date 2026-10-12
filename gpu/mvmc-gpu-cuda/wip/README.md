# WIP: CubeCL-native interval sampler (issue #547)

`interval_cubecl.rs` is the draft of the device-resident interval sampler written as a
CubeCL `#[cube(launch_unchecked)]` kernel (the `t4a-cubecl` package tenferro 0.7.1 uses),
following the user's decision to use CubeCL rather than a hand-written `.cu`.

It is **not** part of the module tree yet and is not compiled by `cargo check`/`build`.
It does not yet pass the CubeCL frontend: one generic-bound error remains
(`usize: From<i32>` / `i32: Into<usize>`, i.e. a `usize::cast_from(<i32>)` or an `i32`
index produced by the macro expansion). The Rust runner (tenferro tensors, tape
generation, launch through `tenferro_gpu::cuda::cubecl::Session`, initial/periodic
Pfaffian rebuild) and the device-vs-host gate test are also not written yet.

Move it into `src/`, add the `cubecl` dependency, and fix the typing error to resume.
