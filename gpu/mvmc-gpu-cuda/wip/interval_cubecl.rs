//! CubeCL-native device-resident interval sampler (issue #547).
//!
//! One CubeCL cube advances one walker's configuration for a whole interval of ordinary
//! hopping proposals on the device: candidate selection, the Gutzwiller/Jastrow projection
//! delta and the Metropolis decision all run on the device, so the per-hop host<->device
//! round trip that dominates the cudarc/NVRTC lock-step sampler (`k_propose`/`k_accept`
//! driven once per hop) disappears.
//!
//! This is written in CubeCL (the `t4a-cubecl` package tenferro 0.7.1 uses) rather than a
//! hand-written `.cu`, and is meant to be launched through tenferro's CubeCL session
//! (`tenferro_gpu::cuda::cubecl::Session`, obtained via `CudaExecSession::with_cubecl`), so
//! the kernel and its tensors share tenferro's CUDA runtime.
//!
//! # Buffer dtypes
//!
//! CubeCL's `cast_from` follows Rust's `From`; signed integer -> `usize` is not available,
//! so every index-like buffer is `u32` (site indices, the Gutzwiller/Jastrow parameter
//! tables with `u32::MAX` as the "no parameter" sentinel) and occupancy is `f64` (0.0/1.0).
//!
//! # Proposal contract (a separate GPU RNG contract, like the Julia GPU path)
//!
//! A fixed tape of `u32` words, 4 per hop, laid out `[word][walker][hop]`:
//!   * `w0` source electron slot: `mi = w0 % ne`
//!   * `w1` spin sector (top bit): `spin = w1 >> 31`
//!   * `w2` destination rank among the empty sites of that spin (uniform over empty sites)
//!   * `w3` acceptance uniform: `u = w3 / 2^32`
//!
//! Every hop consumes exactly 4 words and is always a valid hop, so the tape length is fixed
//! and decision independent.
//!
//! Only the real, non-FSZ, ordinary-hopping normal mode without local spins is handled; the
//! caller rejects anything else before launching.
//!
//! # Status
//!
//! Milestone 1 (this module): the kernel compiles as a `#[cube]` kernel and encodes the
//! whole interval with the cube's move/decision words shared through `mv`. The Rust runner
//! (tenferro tensors, tape generation, tenferro-session launch, initial and periodic
//! Pfaffian rebuild) and the device-vs-host gate test are the next step.

use cubecl::prelude::*;

/// Advance `max_hops` ordinary hopping proposals of every walker passed as `CUBE_POS_X`.
///
/// State per walker (resident across calls):
/// * `ele`  `[W][n]`           configuration (site index per electron slot, `u32`)
/// * `occ`  `[W][2*nsite]`     occupancy (`ele_num`, `f64` 0.0/1.0)
/// * `inv`  `[W][nqp][n*n]`    mVMC inverse (`invM = -X^-1`)
/// * `pf`   `[W][nqp]`         Pfaffians
/// * `aw`   `[W]`              accepted moves since the last rebuild (in/out)
/// * `logip` `[W]`             cached `log|inner product|` (in/out)
///
/// Outputs per walker (written by unit 0):
/// * `hops_done[W]` hops advanced (< `max_hops` when a rebuild was requested)
/// * `rebuild[W]`  1 when `aw` passed `nsite` (host must recompute before continuing)
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments, clippy::eq_op)]
pub fn interval_hop_kernel(
    ele: &mut Array<u32>,
    occ: &mut Array<f64>,
    inv: &mut Array<f64>,
    pf: &mut Array<f64>,
    aw: &mut Array<u32>,
    logip: &mut Array<f64>,
    slater: &Array<f64>,
    weights: &Array<f64>,
    gutz: &Array<u32>,
    jast: &Array<u32>,
    params: &Array<f64>,
    tape: &Array<u32>,
    hop_offset: usize,
    max_hops: usize,
    hops_done: &mut Array<u32>,
    rebuild: &mut Array<u32>,
    #[comptime] total_hops: usize,
    #[comptime] n: usize,
    #[comptime] ne: usize,
    #[comptime] nsite: usize,
    #[comptime] nqp: usize,
    #[comptime] ngutz: usize,
    #[comptime] n_jast: usize,
    #[comptime] nproj: usize,
    #[comptime] nt: usize,
) {
    let w = CUBE_POS_X as usize;
    let tid = UNIT_POS as usize;
    let n2 = 2 * nsite;

    // Shared scratch: pfnew[nqp], out, v1[n], v2[n], v3[n]; plus the per-hop move/decision
    // words (shared because only unit 0 computes them).
    let mut rs = SharedMemory::<u32>::new(n);
    let mut sh = SharedMemory::<f64>::new(nqp + 1 + 3 * n);
    // mv: 0 valid, 1 mi, 2 spin, 3 ri, 4 rj, 5 accept, 6 rebuild.
    let mut mv = SharedMemory::<u32>::new(8);

    let mut log_old = logip[w];
    let mut window = aw[w];
    let mut advanced = 0u32 as usize;
    let mut need_rebuild = false;

    let mut hop = 0u32 as usize;
    while hop < max_hops && !need_rebuild {
        let base = (w * total_hops + (hop_offset + hop)) * 4;
        if tid == 0 {
            let mi = (tape[base] % u32::cast_from(ne)) as usize;
            let spin = (tape[base + 1] >> 31u32) as usize;
            let ri = usize::cast_from(ele[w * n + mi + spin * ne]);
            // count empty sites of this spin, then take the rank-th one.
            let mut count = 0u32;
            let mut r = 0usize;
            while r < nsite {
                if occ[w * 2 * nsite + r + spin * nsite] == 0.0f64 {
                    count += 1u32;
                }
                r += 1;
            }
            let mut rj = 0usize;
            let valid = count > 0u32;
            if valid {
                let mut rank = tape[base + 2] % count;
                r = 0usize;
                while r < nsite {
                    if occ[w * 2 * nsite + r + spin * nsite] == 0.0f64 {
                        if rank == 0u32 {
                            rj = r;
                            break;
                        }
                        rank -= 1u32;
                    }
                    r += 1;
                }
            }
            mv[0usize] = if valid { 1u32 } else { 0u32 };
            mv[1usize] = u32::cast_from(mi);
            mv[2usize] = u32::cast_from(spin);
            mv[3usize] = u32::cast_from(ri);
            mv[4usize] = u32::cast_from(rj);
        }
        sync_cube();
        let valid = mv[0usize] != 0u32;
        let mi = usize::cast_from(mv[1usize]);
        let spin = usize::cast_from(mv[2usize]);
        let ri = usize::cast_from(mv[3usize]);
        let rj = usize::cast_from(mv[4usize]);
        if valid {
            if tid == 0 {
                ele[w * n + mi + spin * ne] = u32::cast_from(rj);
                occ[w * 2 * nsite + ri + spin * nsite] = 0.0f64;
                occ[w * 2 * nsite + rj + spin * nsite] = 1.0f64;
            }
            sync_cube();
            let msa = mi + spin * ne;
            let mut j = tid;
            while j < n {
                rs[j] = ele[w * n + j] + if j >= ne { u32::cast_from(nsite) } else { 0u32 };
                j += nt;
            }
            sync_cube();

            // rank-one ratio for every QP plane
            let mut q = 0usize;
            while q < nqp {
                let rsa = usize::cast_from(rs[msa]);
                let invq = w * nqp * n * n + q * n * n;
                let mut jj = tid;
                while jj < n {
                    let rsj = usize::cast_from(rs[jj]);
                    sh[nqp + 1 + 2 * n + jj] =
                        inv[invq + msa * n + jj] * slater[(q * n2 + rsa) * n2 + rsj];
                    jj += nt;
                }
                sync_cube();
                if tid == 0 {
                    let mut ratio = 0.0f64;
                    let mut k = 0usize;
                    while k < n {
                        ratio += sh[nqp + 1 + 2 * n + k];
                        k += 1;
                    }
                    sh[q] = -ratio * pf[w * nqp + q];
                }
                sync_cube();
                q += 1;
            }

            if tid == 0 {
                let mut ip_new = 0.0f64;
                let mut q2 = 0usize;
                while q2 < nqp {
                    ip_new += weights[q2] * sh[q2];
                    q2 += 1;
                }
                let abs_ip = if ip_new < 0.0f64 { -ip_new } else { ip_new };
                let log_new = abs_ip.ln();
                let proj = interval_proj_delta(
                    occ, gutz, jast, params, w, nsite, nsite * nsite, ngutz, n_jast, nproj, ri, rj,
                );
                let mut weight = (2.0f64 * (proj + log_new - log_old)).exp();
                if weight != weight || weight > 1.0e308f64 {
                    weight = -1.0f64;
                }
                let u = (tape[base + 3] as f64) / 4294967296.0f64;
                let accept = weight > u;
                mv[5usize] = if accept { 1u32 } else { 0u32 };
                if accept {
                    sh[nqp] = log_new;
                } else {
                    ele[w * n + mi + spin * ne] = u32::cast_from(ri);
                    occ[w * 2 * nsite + ri + spin * nsite] = 1.0f64;
                    occ[w * 2 * nsite + rj + spin * nsite] = 0.0f64;
                }
            }
            sync_cube();
            if mv[5usize] != 0u32 {
                let mut q3 = 0usize;
                while q3 < nqp {
                    let rsa = usize::cast_from(rs[msa]);
                    let inv_base = w * nqp * n * n + q3 * n * n;
                    // v3[j] = S[rsa][rs[j]]
                    let mut jj = tid;
                    while jj < n {
                        let rsj = usize::cast_from(rs[jj]);
                        sh[nqp + 1 + 2 * n + jj] = slater[(q3 * n2 + rsa) * n2 + rsj];
                        jj += nt;
                    }
                    sync_cube();
                    // v1[i] = sum_j (-inv[j*n+i]) * v3[j]
                    let mut i = tid;
                    while i < n {
                        let mut acc = 0.0f64;
                        let mut jv = 0usize;
                        while jv < n {
                            acc +=
                                0.0f64 - inv[inv_base + jv * n + i] * sh[nqp + 1 + 2 * n + jv];
                            jv += 1;
                        }
                        sh[nqp + 1 + i] = acc;
                        i += nt;
                    }
                    sync_cube();
                    let v1msa = sh[nqp + 1 + msa];
                    let scale = -1.0f64 / v1msa;
                    let mut i2 = tid;
                    while i2 < n {
                        sh[nqp + 1 + n + i2] = inv[inv_base + msa * n + i2] * scale;
                        i2 += nt;
                    }
                    sync_cube();
                    // rank-two update, one row per unit stride
                    let mut i3 = tid;
                    while i3 < n {
                        let vi1 = sh[nqp + 1 + i3];
                        let vi2 = sh[nqp + 1 + n + i3];
                        let rowoff = inv_base + i3 * n;
                        let mut j3 = 0usize;
                        while j3 < n {
                            let v1j = sh[nqp + 1 + j3];
                            let v2j = sh[nqp + 1 + n + j3];
                            let mut x = inv[rowoff + j3];
                            x += vi1 * v2j - v1j * vi2;
                            if j3 == msa {
                                x -= vi2;
                            }
                            if i3 == msa {
                                x += v2j;
                            }
                            inv[rowoff + j3] = x;
                            j3 += 1;
                        }
                        i3 += nt;
                    }
                    sync_cube();
                    if tid == 0 {
                        pf[w * nqp + q3] = sh[q3];
                    }
                    sync_cube();
                    q3 += 1;
                }
                if tid == 0 {
                    log_old = sh[nqp];
                    window += 1;
                    if window > u32::cast_from(nsite) {
                        mv[6usize] = 1u32;
                    }
                }
                sync_cube();
            }
        }
        advanced += 1;
        hop += 1;
        if mv[6usize] != 0u32 {
            need_rebuild = true;
        }
    }

    if tid == 0 {
        hops_done[w] = u32::cast_from(advanced);
        rebuild[w] = if need_rebuild { 1u32 } else { 0u32 };
        aw[w] = window;
        logip[w] = log_old;
    }
}

/// Gutzwiller/Jastrow projection delta of a hop `ri -> rj` using the post-hop occupancy.
///
/// Collapses `update_proj_cnt` + `log_proj_ratio`: only the touched counters change, so the
/// delta is `sum_i params[i] * (new_count[i] - old_count[i])`. `u32::MAX` is the "no
/// parameter" sentinel of the index tables.
#[cube]
#[allow(clippy::too_many_arguments)]
fn interval_proj_delta(
    occ: &Array<f64>,
    gutz: &Array<u32>,
    jast: &Array<u32>,
    params: &Array<f64>,
    w: usize,
    nsite: usize,
    jast_stride: usize,
    ngutz: usize,
    n_jast: usize,
    nproj: usize,
    ri: usize,
    rj: usize,
) -> f64 {
    let ob = w * 2 * nsite;
    let n0ri = occ[ob + ri];
    let n1ri = occ[ob + nsite + ri];
    let n0rj = occ[ob + rj];
    let n1rj = occ[ob + nsite + rj];
    let mut delta = 0.0f64;
    if ngutz > 0 {
        let ii = gutz[ri];
        if ii != u32::MAX && usize::cast_from(ii) < nproj {
            delta -= params[usize::cast_from(ii)] * (n0ri + n1ri);
        }
        let ij = gutz[rj];
        if ij != u32::MAX && usize::cast_from(ij) < nproj {
            delta += params[usize::cast_from(ij)] * (n0rj * n1rj);
        }
    }
    if n_jast > 0 {
        let offset = ngutz;
        let xi_ri = n0ri + n1ri;
        let xi_rj = n0rj + n1rj;
        let (a, b) = if ri < rj { (ri, rj) } else { (rj, ri) };
        let idx = jast[a * jast_stride + b];
        if idx != u32::MAX && offset + usize::cast_from(idx) < nproj {
            delta += params[offset + usize::cast_from(idx)] * (xi_ri - xi_rj + 1.0f64);
        }
        let mut rk = 0usize;
        while rk < nsite {
            if rk != ri && rk != rj {
                let x = occ[ob + rk] + occ[ob + nsite + rk] - 1.0f64;
                if x != 0.0f64 {
                    let (a1, b1) = if ri < rk { (ri, rk) } else { (rk, ri) };
                    let ia = jast[a1 * jast_stride + b1];
                    if ia != u32::MAX && offset + usize::cast_from(ia) < nproj {
                        delta -= params[offset + usize::cast_from(ia)] * x;
                    }
                    let (a2, b2) = if rj < rk { (rj, rk) } else { (rk, rj) };
                    let ib = jast[a2 * jast_stride + b2];
                    if ib != u32::MAX && offset + usize::cast_from(ib) < nproj {
                        delta += params[offset + usize::cast_from(ib)] * x;
                    }
                }
            }
            rk += 1;
        }
    }
    delta
}
