| kernel | build:variant | procs | median ns | range | net of copy |
|---|---|---:|---:|---:|---:|
| K1 | julia:copy-only | 5 | 5054 | 4890..5100 |  |
| K1 | julia:julia-simd(mVMC form) | 5 | 13220 | 13018..14882 | 8165 |
| K1 | julia:julia-turbo | 5 | 12692 | 12502..12792 | 7637 |
| K1 | rust-align64:avx2 | 5 | 13342 | 12691..13402 | 6960 |
| K1 | rust-align64:copy-only | 5 | 6382 | 6272..6754 |  |
| K1 | rust-align64:pulp-scalar | 5 | 17366 | 16884..17602 | 10983 |
| K1 | rust-align64:pulp-v3 | 5 | 13429 | 12884..13785 | 7047 |
| K1 | rust-align64:pulp-v3+Arch::new/call | 5 | 13743 | 13539..13940 | 7361 |
| K1 | rust-align64:pulp-v3-fma | 5 | 14582 | 14463..16264 | 8200 |
| K1 | rust-align64:scalar | 5 | 17332 | 16760..17714 | 10949 |
| K1 | rust-default:avx2 | 5 | 13261 | 12395..13716 | 6923 |
| K1 | rust-default:copy-only | 5 | 6338 | 6277..6769 |  |
| K1 | rust-default:pulp-scalar | 5 | 16856 | 16658..17226 | 10518 |
| K1 | rust-default:pulp-v3 | 5 | 13020 | 12605..15039 | 6682 |
| K1 | rust-default:pulp-v3+Arch::new/call | 5 | 13607 | 13150..14539 | 7269 |
| K1 | rust-default:pulp-v3-fma | 5 | 14634 | 14194..15946 | 8296 |
| K1 | rust-default:scalar | 5 | 17271 | 16680..17891 | 10933 |
| K2 | julia:copy-only | 5 | 4999 | 4935..5535 |  |
| K2 | julia:julia-simd(PfaPack form) | 5 | 72681 | 71484..73567 | 67683 |
| K2 | julia:julia-turbo | 5 | 62979 | 61675..64685 | 57980 |
| K2 | rust-align64:avx2 | 5 | 62955 | 62509..63009 | 57940 |
| K2 | rust-align64:avx2-scalar-tail | 5 | 63659 | 63249..64438 | 58644 |
| K2 | rust-align64:copy-only | 5 | 5015 | 4962..5332 |  |
| K2 | rust-align64:pulp-scalar | 5 | 86649 | 85755..87089 | 81634 |
| K2 | rust-align64:pulp-v3 | 5 | 67325 | 65730..68287 | 62310 |
| K2 | rust-align64:pulp-v3-fma | 5 | 61180 | 59344..62491 | 56165 |
| K2 | rust-align64:scalar | 5 | 86054 | 84928..87137 | 81038 |
| K2 | rust-default:avx2 | 5 | 62887 | 61952..63329 | 57817 |
| K2 | rust-default:avx2-scalar-tail | 5 | 62533 | 61821..63094 | 57463 |
| K2 | rust-default:copy-only | 5 | 5070 | 5042..5519 |  |
| K2 | rust-default:pulp-scalar | 5 | 84208 | 83895..85254 | 79138 |
| K2 | rust-default:pulp-v3 | 5 | 66227 | 65346..67014 | 61157 |
| K2 | rust-default:pulp-v3-fma | 5 | 61672 | 59901..62643 | 56602 |
| K2 | rust-default:scalar | 5 | 84801 | 84516..85414 | 79731 |
| K3 | julia:julia-plain(C order) | 5 | 12086 | 12062..12091 |  |
| K3 | julia:julia-turbo | 5 | 1894 | 1822..1963 |  |
| K3 | rust-align64:avx2-rows(exact,new) | 5 | 4149 | 4127..4364 |  |
| K3 | rust-align64:pulp-scalar-fma | 5 | 7151 | 7142..7240 |  |
| K3 | rust-align64:pulp-v3-fma | 5 | 3295 | 3294..3436 |  |
| K3 | rust-align64:scalar | 5 | 4233 | 4225..4522 |  |
| K3 | rust-default:avx2-rows(exact,new) | 5 | 4210 | 4178..4248 |  |
| K3 | rust-default:pulp-scalar-fma | 5 | 7735 | 7719..7749 |  |
| K3 | rust-default:pulp-v3-fma | 5 | 3301 | 3290..3340 |  |
| K3 | rust-default:scalar | 5 | 4454 | 4442..4484 |  |
| K4 | julia:julia-plain(per column) | 5 | 126647 | 126270..126831 |  |
| K4 | julia:julia-turbo(across columns) | 5 | 62513 | 60668..63508 |  |
| K4 | rust-align64:avx2 | 5 | 35400 | 35335..35667 |  |
| K4 | rust-align64:pulp-scalar | 5 | 101802 | 100141..103715 |  |
| K4 | rust-align64:pulp-v3 | 5 | 136335 | 134679..137072 |  |
| K4 | rust-align64:scalar | 5 | 56648 | 56600..56784 |  |
| K4 | rust-default:avx2 | 5 | 35454 | 35345..35558 |  |
| K4 | rust-default:pulp-scalar | 5 | 103304 | 102107..208934 |  |
| K4 | rust-default:pulp-v3 | 5 | 137169 | 135725..191678 |  |
| K4 | rust-default:scalar | 5 | 56648 | 56636..56851 |  |
