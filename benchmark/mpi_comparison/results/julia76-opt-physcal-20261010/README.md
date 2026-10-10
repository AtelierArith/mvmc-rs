# Unified post-PR76 Opt / PhysCal measurements

Rust checkout `fd45894c6b752866b34ae91c6d038b5f4d26cfbe` (production Rust kernels unchanged from PR #507), Julia upstream PR #76 merge `02afdae0a19732c727e0d09132d9761c57d68fcb`, authoritative C `d73d06bd529d3b2573f38eb5817c4a5f52971006`. Linux x86_64 Dev Container, Julia 1.13.1. Source/package/binary identities and actual worker/BLAS metadata are archived; source hashes were verified unchanged after measurement.

Reproduction from the Linux Dev Container:

```bash
bash bench/run.sh --sites 32 64 --ranks 4 --threads 4 \
  --steps 300 --groups 100 --samples 300 --warmups 1 --reps 3 \
  --output bench-out/julia76-opt-physcal-20261010
```

Use a fresh output directory when repeating. `report.md` contains all medians and timing-scope limitations. All 36 timing records and all 2,600 PhysCal energy files plus Opt outputs passed finite/shape checks, including separate untimed observations. `validate.py` is the executed optional uv validator, not a Cargo test. Runtime metadata/logs are in `runtime-logs.tar.gz`; full inputs, outputs, projects and rebuilt C sources remain in `bench-out/julia76-opt-physcal-20261010/`.

Both ports have lower medians than C in this run. Julia L32 Opt is only 0.88% lower; together with prior paired evidence this does not prove a robust performance advantage. No additional optimization or compiler-option change was introduced for this report. Related to #496 and #497.
