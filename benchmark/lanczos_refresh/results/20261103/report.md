# Results (2026-11-03)

Xeon E5-2699 v3, Linux x86_64. Julia 1.13.1 / OpenBLAS 0.3.30 ILP64; Rust 1.99.0 / OpenBLAS 0.3.26. One warmup + three measurements, median seconds. Different timing scopes: warm Julia API vs Rust CLI process. C not rerun.

## phys_lanczos

| CPU threads | Julia | Rust |
|---:|---:|---:|
| 1 | 57.010 s | 32.644 s |
| 4 | 57.928 s | 40.895 s |
| 8 | 58.915 s | 45.421 s |
| 16 | 59.883 s | 47.280 s |

## phys_interall_lanczos1

| CPU threads | Julia | Rust |
|---:|---:|---:|
| 1 | 112.181 s | 62.813 s |
| 4 | 117.086 s | 73.294 s |
| 8 | 114.419 s | 77.397 s |
| 16 | 117.568 s | 79.188 s |

## phys_interall_lanczos2

| CPU threads | Julia | Rust |
|---:|---:|---:|
| 1 | 166.002 s | 85.371 s |
| 4 | 171.593 s | 96.929 s |
| 8 | 172.114 s | 101.337 s |
| 16 | 174.022 s | 114.624 s |

All 96 output sets / 5,760 Lanczos files have finite values and matching shapes. Fixed-thread repeats have observed maximum absolute difference 0; cross-thread same-implementation maximum is 4.929390229335695e-13. No bitwise computed-value test is used.

Rust 1-thread first measured outputs match the historical Rust numeric outputs (maximum absolute difference 0). Updated Julia differs from historical Julia by at most 3.4533487180965494e-13 across these Lanczos outputs. Ordinary mode1 Julia/Rust maximum absolute difference is 9.117151478221786e-13. InterAll modes1/2 retain pre-existing differences: QQQQ maximum absolute difference 0.4573802746838851 and ls_out 0.022787326145605746. These are not claimed as cross-language parity and are not attributed to Monte Carlo noise. See verification.json for each output family.

Host affinity/frequency were not isolated. Rust InterAll mode2 / 16 measured repetitions are 116.906653445, 114.624022685, 107.864999129 seconds; all three are retained. Host load is recorded for Rust runs. Timing differences are observations, not attribution to a specific kernel or scheduler.

Raw archive: 2,565,708 bytes, 16,518 files; every archived file hash was verified against files.sha256.json.

Extract raw.tar.gz into a new directory and run benchmark/lanczos_refresh/verify.py on that directory. Historical comparison additionally requires the original local output tree. The published verification.json includes that comparison.
