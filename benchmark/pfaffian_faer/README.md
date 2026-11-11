# Pfaffian migration probes (#731)

These are preliminary kernel measurements, not the final adoption/E2E report.
CPU: Linux x86_64, Xeon E5-2699 v3, rustc 1.99.1
(`58cf8f9c3`, LLVM 23.1.1), system OpenBLAS 0.3.26. GPU: RTX 3060 sm_86,
CUDA 12.9.2 container image `b129bd583bbe243671d4d4b9ec18cf39b00d0037cadd098197530d2d33750918`.
Sources: the implementation following `00f7106c` on `perf/faer-pfaffian`;
t4a-faer is the registry release 0.32.0.

## CPU complete rebuild probe

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 cargo run --release -p pfapack \
  --example faer_rebuild_bench --features 'faer-backend blas-backend'
```

`cpu-rebuild-probe-20261111.csv` compares the existing BLAS-enabled
`dsktf2`/`zsktf2` + `utu2pfa` + `utu2inv` with sequential faer Pfaffian plus
blocked LU/inverse. Both reuse preallocated workspaces; the reference includes
its necessary input copy. Each row is one adaptive 250 ms timing interval
after a warmup, **not** a multi-sample median. The matrix is a deterministic
bounded skew matrix with strengthened 2x2 diagonal blocks (see the example).
It is not the random/graded workload of the linked upstream benchmark.

This probe predates adding the final inverse skew projection to the adapter;
rerun with final code for adoption decisions. The complex candidate gains
1.17–1.69x for n=64–512. Real full rebuilds do not show a reliable gain; small
sizes regress. Do not infer a real sampler speedup from the standalone faer
Pfaffian benchmark. Full rebuild integration is currently experimental behind
`mvmc-core/faer-pfaffian`; its default is not enabled.

`cpu-rebuild-samples-20261111.csv` repeats the probe five times with the final
skew projection included, on the same shared host while the GPU gate was also
running. Across the arithmetic means of those five samples, complex n=64–512
gains 1.14–1.69x. Real rebuilds remain slower (0.57–0.95x over n=16–512).
These data confirm that adopting the combined real rebuild by default needs
more optimization or a different inverse strategy; a standalone Pfaffian
benchmark is insufficient. Raw per-sample timings are retained rather than
presenting these shared-host probes as final E2E evidence.

## GPU panel probe

```sh
docker run --rm --gpus '"device=0"' \
  -v "$PWD:/work:ro" -w /work/benchmark/pfaffian_faer b129bd583bbe \
  bash -lc 'nvcc -std=c++17 -O3 --fmad=false -arch=sm_86 gpu_probe.cu \
    -o /tmp/mvmc-pf-panel-probe && /tmp/mvmc-pf-panel-probe'
```

The optional CUDA probe compiles the actual production helper source. It is
independent of Cargo builds/tests and adds no C reference runtime requirement.
`gpu-probe-20261111.csv` contains 32-plane batches and CUDA event timing for 40
launches, including device input copies but excluding allocation and host
transfers. It compares the old and panelized standalone-Pfaffian helpers. It
checks 30 bounded dense planes against the old helper, plus analytical
block-diagonal and zero planes. All observed differences were zero; the
assertion uses an explicit numerical bound, not bitwise equality.

Panel updates gain about 1.44–1.94x for real n=128–512 and 1.07–1.68x for
complex n=64–512 in this probe. Production dispatch retains the unblocked
helper below these respective sizes. The new helper uses the existing two
n-vectors and matrix scratch, with no added allocation, transfer or launch.
This is a CUDA port of the panel-update organization, not a faer GPU API call.

**Not yet established:** repeated medians, broad conditioning/nonfinite
coverage, resident Opt/PhysCal E2E performance and transfer counters, full
Julia GPU fixture gates, and the complete Pfaffian/inverse rebuild kernels.
The existing CUDA LTL/inverse rebuild kernel has not been replaced yet.
