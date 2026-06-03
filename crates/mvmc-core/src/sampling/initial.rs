//! Phase 4.3.7 — initial-sample generator.
//!
//! Port target: `make_initial_sample!` in
//! `MVMCOptimizers.jl/src/vmc_sampling.jl` (~line 1199). The RNG draw
//! order matches upstream: for every local-spin site we draw
//! `(gen_rand_mod(n_elec), spin_coin)`; for itinerant electrons we
//! draw `gen_rand_mod(n_site)` until a free non-local site is found.

use mvmc_expert_parsers::ExpertModeData;
use sfmt19937::Sfmt19937Rng;

use crate::sampling::projection::{init_loc_spn, make_proj_cnt};

/// Result of [`make_initial_sample`]. `Ok(())` matches upstream's
/// `info = 0`; `Err(())` matches `info != 0` (too many retries).
pub fn make_initial_sample(
    ele_idx: &mut [i64],
    ele_cfg: &mut [i64],
    ele_num: &mut [i64],
    ele_proj_cnt: &mut [i64],
    data: &ExpertModeData,
    loc_spn: &[i64],
    rng: &mut Sfmt19937Rng,
) -> Result<(), ()> {
    let n_site = data.modpara.nsite.max(0) as usize;
    let n_elec = data.modpara.nelec.max(0) as usize;
    let n_site2 = 2 * n_site;

    // Up to 100 retries, matching the Julia guard. The Pfaffian-validity
    // retry is left to the caller because it needs the full state.
    const MAX_LOOPS: usize = 100;
    for _ in 0..MAX_LOOPS {
        for slot in ele_idx.iter_mut() {
            *slot = -1;
        }
        for slot in ele_cfg.iter_mut() {
            *slot = -1;
        }

        // Local spin sites.
        for ri in 0..n_site {
            if loc_spn.get(ri).copied().unwrap_or(0) != 1 {
                continue;
            }
            loop {
                let mi = rng.gen_rand_mod(n_elec as u32) as usize;
                let r = rng.genrand_real2();
                let si = if r < 0.5 { 0 } else { 1 };
                if ele_idx[mi + si * n_elec] == -1 {
                    ele_cfg[ri + si * n_site] = mi as i64;
                    ele_idx[mi + si * n_elec] = ri as i64;
                    break;
                }
            }
        }

        // Itinerant electrons.
        for si in 0..2 {
            for mi in 0..n_elec {
                if ele_idx[mi + si * n_elec] != -1 {
                    continue;
                }
                loop {
                    let ri = rng.gen_rand_mod(n_site as u32) as usize;
                    if ele_cfg[ri + si * n_site] == -1 && loc_spn.get(ri).copied().unwrap_or(0) != 1
                    {
                        ele_cfg[ri + si * n_site] = mi as i64;
                        ele_idx[mi + si * n_elec] = ri as i64;
                        break;
                    }
                }
            }
        }

        // Electron-number array.
        for rsi in 0..n_site2 {
            ele_num[rsi] = if ele_cfg[rsi] < 0 { 0 } else { 1 };
        }
        // Projection counts.
        make_proj_cnt(ele_proj_cnt, ele_num, data);
        // Caller can validate via Pfaffian; we always return Ok on the
        // first successful layout, mirroring upstream when `flag == 0`.
        return Ok(());
    }
    Err(())
}

/// Convenience wrapper that initialises `loc_spn` from `data.locspin_terms`
/// before delegating to [`make_initial_sample`].
pub fn make_initial_sample_init_loc_spn(
    ele_idx: &mut [i64],
    ele_cfg: &mut [i64],
    ele_num: &mut [i64],
    ele_proj_cnt: &mut [i64],
    loc_spn: &mut [i64],
    data: &ExpertModeData,
    rng: &mut Sfmt19937Rng,
) -> Result<(), ()> {
    init_loc_spn(loc_spn, data);
    make_initial_sample(ele_idx, ele_cfg, ele_num, ele_proj_cnt, data, loc_spn, rng)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn make_initial_sample_respects_n_elec_and_loc_spn() {
        let mut data = ExpertModeData::new();
        data.modpara.nsite = 4;
        data.modpara.nelec = 2;
        let loc_spn = vec![1, 0, 1, 0];
        let mut ele_idx = vec![0_i64; 4];
        let mut ele_cfg = vec![0_i64; 8];
        let mut ele_num = vec![0_i64; 8];
        let mut proj = vec![0_i64; 0];
        let mut rng = Sfmt19937Rng::new(11272);
        make_initial_sample(
            &mut ele_idx,
            &mut ele_cfg,
            &mut ele_num,
            &mut proj,
            &data,
            &loc_spn,
            &mut rng,
        )
        .expect("layout succeeds");
        // Local spin sites must hold an electron in some spin sector.
        for (ri, flag) in loc_spn.iter().enumerate() {
            if *flag == 1 {
                assert!(
                    ele_num[ri] + ele_num[ri + 4] == 1,
                    "local spin site {} must host exactly one electron",
                    ri
                );
            }
        }
        // Every electron index must point to a non-negative site.
        for v in &ele_idx {
            assert!(*v >= 0);
        }
    }
}

/// FSZ `make_initial_sample_fsz!` mirror.
pub fn make_initial_sample_fsz(
    ele_idx: &mut [i64],
    ele_cfg: &mut [i64],
    ele_num: &mut [i64],
    ele_proj_cnt: &mut [i64],
    ele_spn: &mut [i64],
    data: &ExpertModeData,
    loc_spn: &[i64],
    rng: &mut Sfmt19937Rng,
) -> Result<(), ()> {
    let n_site = data.modpara.nsite.max(0) as usize;
    let n_elec = data.modpara.nelec.max(0) as usize;
    let n_size = 2 * n_elec;
    let n_site2 = 2 * n_site;
    let tmp_two_sz = if data.modpara.two_sz == -1 {
        0
    } else {
        data.modpara.two_sz / 2
    };

    for slot in ele_idx.iter_mut() {
        *slot = -1;
    }
    for slot in ele_spn.iter_mut() {
        *slot = -1;
    }
    for slot in ele_cfg.iter_mut() {
        *slot = -1;
    }

    for mi in 0..n_size {
        ele_spn[mi] = if (mi as i64) < data.modpara.nelec + tmp_two_sz {
            0
        } else {
            1
        };
    }

    for ri in 0..n_site {
        if loc_spn.get(ri).copied().unwrap_or(0) == 1 {
            loop {
                let mi = rng.gen_rand_mod(n_size as u32) as usize;
                if ele_idx[mi] == -1 {
                    let si = ele_spn[mi] as usize;
                    ele_cfg[ri + si * n_site] = mi as i64;
                    ele_idx[mi] = ri as i64;
                    break;
                }
            }
        }
    }

    for mi in 0..n_size {
        if ele_idx[mi] == -1 {
            let si = ele_spn[mi] as usize;
            loop {
                let ri = rng.gen_rand_mod(n_site as u32) as usize;
                if ele_cfg[ri + si * n_site] == -1 && loc_spn.get(ri).copied().unwrap_or(0) == 0 {
                    ele_cfg[ri + si * n_site] = mi as i64;
                    ele_idx[mi] = ri as i64;
                    break;
                }
            }
        }
    }

    for rsi in 0..n_site2 {
        ele_num[rsi] = if ele_cfg[rsi] == -1 { 0 } else { 1 };
    }
    make_proj_cnt(ele_proj_cnt, ele_num, data);
    Ok(())
}
