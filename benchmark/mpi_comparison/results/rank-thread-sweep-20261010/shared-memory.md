# Shared-memory retry provenance

The first five completed configurations used container a33420ab7cee with Docker's default 64 MiB /dev/shm. The initial 16×1 C launch failed in MPI_Init before numerical computation: UCX requested 4,292,720 bytes per rank and exhausted shared memory. Its output is preserved in r16-t1-failed-shm64mb. No timing from that attempt is included.

A temporary container used the same image, workspace and cache volumes, with 1 GiB /dev/shm:

```bash
docker run -d --name mvmc-bench-16rank-20261010 \
  --shm-size=1g --volumes-from a33420ab7cee \
  --workdir /workspaces/mvmc-rs \
  sha256:803cdee63c3bea8d2a9604295eeb5a18352b4eb8519286b3ffdb4276458c33c4 \
  sleep infinity
docker exec mvmc-bench-16rank-20261010 bash bench/run.sh \
  --sites 32 64 --ranks 16 --threads 1 \
  --steps 300 --groups 100 --samples 320 --warmups 1 --reps 3 \
  --output bench-out/rank-thread-sweep-20261010/r16-t1
```

The MPI/UCX transport was unchanged. Identical binary SHA-256 values were checked in both containers:

| Binary | SHA-256 |
|---|---|
| MPICH libmpi.so.12.4.0 | 638c51116955894e0a8fd91b3f7091246ad3155c37141780958e39f5e102da11 |
| Native OpenBLAS libopenblas.so.0 | bfc7492adbf84a8f567720a9e1fae2afc18f3d817da233e7f4d453683485308e |
| Julia 1.13.1 executable | 54ddfc4a058c7228aa2da1c5623a9032df495fddf9b88983aa7eed9122e30b39 |

The permanent Dev Container setting is tracked separately in PR #511 / issue #510.
