| process output | correctness | checks | failed checks |
|---|---|---:|---|
| julia-p1.txt | PASS | 13 |  |
| rust-align64-p1.txt | PASS | 35 |  |
| rust-default-p1.txt | PASS | 35 |  |

| kernel | build:variant | procs | median ns | range | net of copy |
|---|---|---:|---:|---:|---:|
| K1 | julia:copy-only | 1 | 5272 | 5272..5272 |  |
| K1 | julia:julia-simd(mVMC form) | 1 | 14974 | 14974..14974 | 9702 |
| K1 | julia:julia-turbo | 1 | 14008 | 14008..14008 | 8736 |
| K1 | rust-align64:avx2 | 1 | 18432 | 18432..18432 | 12885 |
| K1 | rust-align64:copy-only | 1 | 5547 | 5547..5547 |  |
| K1 | rust-align64:pulp-scalar | 1 | 21208 | 21208..21208 | 15662 |
| K1 | rust-align64:pulp-v3 | 1 | 15843 | 15843..15843 | 10297 |
| K1 | rust-align64:pulp-v3+Arch::new/call | 1 | 15788 | 15788..15788 | 10242 |
| K1 | rust-align64:pulp-v3-fma | 1 | 13961 | 13961..13961 | 8415 |
| K1 | rust-align64:scalar | 1 | 18892 | 18892..18892 | 13345 |
| K1 | rust-default:avx2 | 1 | 16325 | 16325..16325 | 8808 |
| K1 | rust-default:copy-only | 1 | 7517 | 7517..7517 |  |
| K1 | rust-default:pulp-scalar | 1 | 18679 | 18679..18679 | 11162 |
| K1 | rust-default:pulp-v3 | 1 | 14611 | 14611..14611 | 7094 |
| K1 | rust-default:pulp-v3+Arch::new/call | 1 | 16476 | 16476..16476 | 8960 |
| K1 | rust-default:pulp-v3-fma | 1 | 14728 | 14728..14728 | 7211 |
| K1 | rust-default:scalar | 1 | 20245 | 20245..20245 | 12728 |
| K2 | julia:copy-only | 1 | 5423 | 5423..5423 |  |
| K2 | julia:julia-simd(PfaPack form) | 1 | 79498 | 79498..79498 | 74075 |
| K2 | julia:julia-turbo | 1 | 67436 | 67436..67436 | 62013 |
| K2 | rust-align64:avx2 | 1 | 83959 | 83959..83959 | 78436 |
| K2 | rust-align64:avx2-scalar-tail | 1 | 60157 | 60157..60157 | 54634 |
| K2 | rust-align64:copy-only | 1 | 5523 | 5523..5523 |  |
| K2 | rust-align64:pulp-scalar | 1 | 90876 | 90876..90876 | 85353 |
| K2 | rust-align64:pulp-v3 | 1 | 90242 | 90242..90242 | 84720 |
| K2 | rust-align64:pulp-v3-fma | 1 | 63422 | 63422..63422 | 57900 |
| K2 | rust-align64:scalar | 1 | 91879 | 91879..91879 | 86356 |
| K2 | rust-default:avx2 | 1 | 82458 | 82458..82458 | 77073 |
| K2 | rust-default:avx2-scalar-tail | 1 | 60727 | 60727..60727 | 55342 |
| K2 | rust-default:copy-only | 1 | 5386 | 5386..5386 |  |
| K2 | rust-default:pulp-scalar | 1 | 92785 | 92785..92785 | 87400 |
| K2 | rust-default:pulp-v3 | 1 | 89512 | 89512..89512 | 84126 |
| K2 | rust-default:pulp-v3-fma | 1 | 63621 | 63621..63621 | 58235 |
| K2 | rust-default:scalar | 1 | 92611 | 92611..92611 | 87226 |
| K3 | julia:julia-plain(C order) | 1 | 13401 | 13401..13401 |  |
| K3 | julia:julia-turbo | 1 | 2116 | 2116..2116 |  |
| K3 | rust-align64:avx2-rows(exact,new) | 1 | 4426 | 4426..4426 |  |
| K3 | rust-align64:pulp-scalar-fma | 1 | 7677 | 7677..7677 |  |
| K3 | rust-align64:pulp-v3-fma | 1 | 3676 | 3676..3676 |  |
| K3 | rust-align64:scalar | 1 | 4473 | 4473..4473 |  |
| K3 | rust-default:avx2-rows(exact,new) | 1 | 4718 | 4718..4718 |  |
| K3 | rust-default:pulp-scalar-fma | 1 | 7633 | 7633..7633 |  |
| K3 | rust-default:pulp-v3-fma | 1 | 3781 | 3781..3781 |  |
| K3 | rust-default:scalar | 1 | 4872 | 4872..4872 |  |
| K4 | julia:julia-plain(per column) | 1 | 138380 | 138380..138380 |  |
| K4 | julia:julia-turbo(across columns) | 1 | 70077 | 70077..70077 |  |
| K4 | rust-align64:avx2 | 1 | 37845 | 37845..37845 |  |
| K4 | rust-align64:pulp-scalar | 1 | 113580 | 113580..113580 |  |
| K4 | rust-align64:pulp-v3 | 1 | 149229 | 149229..149229 |  |
| K4 | rust-align64:scalar | 1 | 60634 | 60634..60634 |  |
| K4 | rust-default:avx2 | 1 | 37952 | 37952..37952 |  |
| K4 | rust-default:pulp-scalar | 1 | 110301 | 110301..110301 |  |
| K4 | rust-default:pulp-v3 | 1 | 144728 | 144728..144728 |  |
| K4 | rust-default:scalar | 1 | 60870 | 60870..60870 |  |
