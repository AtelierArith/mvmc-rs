### opt

| CPU threads | C control | Rust before | Rust final | Before/final |
|---:|---:|---:|---:|---:|
| 1 | 93.232 s | 78.911 s | 79.786 s | 0.989× |
| 4 | 29.637 s | 57.129 s | 29.226 s | 1.955× |
| 8 | 19.491 s | 48.049 s | 21.645 s | 2.220× |
| 16 | 21.424 s | 49.168 s | 22.667 s | 2.169× |

### phys

| CPU threads | C control | Rust before | Rust final | Before/final |
|---:|---:|---:|---:|---:|
| 1 | 5.833 s | 4.275 s | 4.446 s | 0.962× |
| 4 | 1.909 s | 3.544 s | 2.930 s | 1.210× |
| 8 | 1.265 s | 3.690 s | 2.281 s | 1.618× |
| 16 | 1.278 s | 2.799 s | 1.792 s | 1.562× |

### Default calc-m-all (separate snapshots)

| CPU threads | Opt before | Opt 8b861f21 | PhysCal before | PhysCal 8b861f21 |
|---:|---:|---:|---:|---:|
| 1 | 77.615 s | 77.885 s | 4.221 s | 4.222 s |
| 4 | 28.410 s | 29.240 s | 2.936 s | 2.889 s |
| 8 | 19.610 s | 19.772 s | 2.185 s | 2.229 s |
| 16 | 22.949 s | 22.632 s | 1.347 s | 1.372 s |
