# Device-resident SR step benchmark (issue #447)

See docs/design/gpu-readiness.md section 14.3 for the method and metadata (shared host, load 17-20, RTX 3060, driver 580.178.04, CUDA 12.9.2).

**Direct SR step (Gram + S/g + Cholesky solve), milliseconds**

| NPara+1 | samples | CPU 1 thread | CPU all cores | GPU end-to-end | GPU resident | e2e vs 1 thread | e2e vs all cores | resident vs all cores | GPU phases: upload / Gram / assemble / solve (ms) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1000 | 1000 | 60.4 | 29.1 | 21.2 | 19.7 | 2.9x | 1.4x | 1.5x | 1.5 / 10.7 / 0.1 / 8.8 |
| 1000 | 10000 | 403 | 74.0 | 120 | 108 | 3.4x | 0.6x | 0.7x | 11.9 / 99.5 / 0.1 / 8.3 |
| 3000 | 3000 | 1384 | 452 | 254 | 242 | 5.5x | 1.8x | 1.9x | 11.4 / 165 / 0.7 / 76.6 |
| 3000 | 10000 | 3270 | 709 | 658 | 626 | 5.0x | 1.1x | 1.1x | 31.9 / 548 / 0.7 / 76.6 |
| 5000 | 5000 | 5020 | 1156 | 1063 | 1038 | 4.7x | 1.1x | 1.1x | 25.6 / 741 / 2.1 / 294 |
| 10000 | 1000 | 13030 | 4164 | 2667 | 2656 | 4.9x | 1.6x | 1.6x | 10.9 / 564 / 17.3 / 2075 |
| 10000 | 10000 | 34404 | 9000 | 7815 | 7710 | 4.4x | 1.2x | 1.2x | 106 / 5609 / 17.1 / 2083 |

**CG SR step (50 CG iterations), milliseconds**

| NPara+1 | samples | CPU 1 thread | CPU all cores | GPU end-to-end | GPU resident | e2e vs 1 thread | e2e vs all cores | resident vs all cores | GPU phases: upload / Gram / assemble / solve (ms) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1000 | 1000 | 31.2 | 13.7 | 12.2 | 10.2 | 2.5x | 1.1x | 1.3x | 1.9 / - / - / 10.1 |
| 1000 | 10000 | 856 | 292 | 42.7 | 30.8 | 20.0x | 6.8x | 9.5x | 11.9 / - / - / 30.5 |
| 3000 | 3000 | 760 | 247 | 43.3 | 28.9 | 17.6x | 5.7x | 8.5x | 11.2 / - / - / 28.8 |
| 3000 | 10000 | 2498 | 939 | 115 | 83.5 | 21.7x | 8.1x | 11.2x | 31.8 / - / - / 83.4 |
| 5000 | 5000 | 1973 | 715 | 94.5 | 69.2 | 20.9x | 7.6x | 10.3x | 25.4 / - / - / 68.9 |
| 10000 | 1000 | 729 | 200 | 49.3 | 33.7 | 14.8x | 4.1x | 5.9x | 15.6 / - / - / 33.4 |
| 10000 | 10000 | 8476 | 2754 | 382 | 275 | 22.2x | 7.2x | 10.0x | 107 / - / - / 274 |
