//! Phase 4.3.3 — normal-mode candidate-move generators.
//!
//! Port targets from `MVMCOptimizers.jl/src/vmc_sampling.jl`:
//! `make_candidate_hopping`, `make_candidate_exchange`, and
//! `get_update_type`.
//!
//! The RNG draw order is part of the public contract here. `rng_mod`
//! is C-style biased modulo (`gen_rand32() % n`) and the spin coin flip
//! uses `genrand_real2()`, matching the Julia helpers at
//! `vmc_sampling.jl:53-57`.

use sfmt19937::Sfmt19937Rng;

/// Candidate update type (`@enum UpdateType HOPPING EXCHANGE LOCALSPINFLIP NONE`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UpdateType {
    /// Single-electron hopping update.
    Hopping,
    /// Opposite-spin exchange update.
    Exchange,
    /// FSZ local-spin flip update.
    LocalSpinFlip,
    /// No update.
    None,
}

/// Result of [`make_candidate_hopping`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HoppingCandidate {
    /// Electron index within its spin sector (`mi`).
    pub mi: usize,
    /// Source site (`ri`).
    pub ri: usize,
    /// Destination site (`rj`).
    pub rj: usize,
    /// Spin sector (`s`, 0 = up, 1 = down).
    pub spin: u8,
    /// Whether the candidate should be rejected before expensive work.
    pub reject: bool,
}

/// Result of [`make_candidate_exchange`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExchangeCandidate {
    /// First electron index (`mi`) in spin sector `spin`.
    pub mi: usize,
    /// First site (`ri`).
    pub ri: usize,
    /// Second electron index (`mj`) in spin sector `spin_other`.
    pub mj: usize,
    /// Second site (`rj`).
    pub rj: usize,
    /// First spin (`s`).
    pub spin: u8,
    /// Opposite spin (`t = 1 - s`).
    pub spin_other: u8,
    /// Whether the candidate should be rejected before expensive work.
    pub reject: bool,
}

/// Result of [`make_candidate_hopping_fsz`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FszHoppingCandidate {
    /// Electron index in the total electron list (`mi`).
    pub mi: usize,
    /// Source site (`ri`).
    pub ri: usize,
    /// Destination site (`rj`).
    pub rj: usize,
    /// Source spin (`s`, from `ele_spn[mi]`).
    pub spin: u8,
    /// Destination spin (`t`; random in `two_sz == -1`, else `s`).
    pub spin_to: u8,
    /// Whether the candidate should be rejected before expensive work.
    pub reject: bool,
}

/// Result of FSZ local-spin-flip candidates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LocalSpinFlipCandidate {
    /// Electron index (`mi`).
    pub mi: usize,
    /// Source site (`ri`).
    pub ri: usize,
    /// Destination site (`rj`, equal to `ri` for local spin flips).
    pub rj: usize,
    /// Source spin (`s`).
    pub spin: u8,
    /// Flipped spin (`t = 1 - s`).
    pub spin_to: u8,
    /// Whether the candidate should be rejected before expensive work.
    pub reject: bool,
}

/// Result of [`make_candidate_exchange_fsz`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FszExchangeCandidate {
    /// First electron index (`mi`).
    pub mi: usize,
    /// First site (`ri`).
    pub ri: usize,
    /// Second site (`rj`). Upstream selects `mj` internally but does
    /// not return it, so this Rust type mirrors the Julia return tuple.
    pub rj: usize,
    /// First spin (`s`).
    pub spin: u8,
    /// Whether the candidate should be rejected before expensive work.
    pub reject: bool,
}

#[inline]
fn rng_mod(rng: &mut Sfmt19937Rng, n: usize) -> usize {
    if n == 0 {
        0
    } else {
        crate::sampling::driver::trace::draw_mod(rng, n as u32) as usize
    }
}

#[inline]
fn rng_spin(rng: &mut Sfmt19937Rng) -> u8 {
    if crate::sampling::driver::trace::draw_real2(rng) < 0.5 {
        0
    } else {
        1
    }
}

/// Port of `make_candidate_hopping`.
///
/// Layouts:
/// * `ele_idx[mi + s*n_elec] = ri`.
/// * `ele_cfg[ri + s*n_site] = mi` or `-1`.
/// * `loc_spn[ri] = 1` marks local-spin sites disallowed for itinerant
///   hopping.
pub fn make_candidate_hopping(
    ele_idx: &[i64],
    ele_cfg: &[i64],
    n_site: usize,
    n_elec: usize,
    loc_spn: &[i64],
    rng: &mut Sfmt19937Rng,
) -> HoppingCandidate {
    let c = make_candidate_hopping_inner(ele_idx, ele_cfg, n_site, n_elec, loc_spn, rng);
    crate::sampling::driver::trace::record(
        1,
        &[
            c.mi as i64,
            c.ri as i64,
            c.rj as i64,
            c.spin as i64,
            i64::from(c.reject),
        ],
    );
    c
}

fn make_candidate_hopping_inner(
    ele_idx: &[i64],
    ele_cfg: &[i64],
    n_site: usize,
    n_elec: usize,
    loc_spn: &[i64],
    rng: &mut Sfmt19937Rng,
) -> HoppingCandidate {
    let mut candidate = HoppingCandidate {
        mi: 0,
        ri: 0,
        rj: 0,
        spin: 0,
        reject: false,
    };
    let icnt_max = n_site * n_site;

    // Select a non-local-spin electron. The loop cap mirrors the Julia
    // guard added to avoid infinite loops when all selected electrons are local.
    let mut icnt_electron = 0usize;
    loop {
        if icnt_electron > icnt_max {
            candidate.reject = true;
            candidate.rj = 0;
            return candidate;
        }
        icnt_electron += 1;
        candidate.mi = rng_mod(rng, n_elec);
        candidate.spin = rng_spin(rng);
        let idx = candidate.mi + (candidate.spin as usize) * n_elec;
        candidate.ri = ele_idx.get(idx).copied().unwrap_or(0).max(0) as usize;
        if loc_spn.get(candidate.ri).copied().unwrap_or(1) != 1 {
            break;
        }
    }

    // Pick an empty, non-local destination in the same spin sector.
    let mut icnt = 0usize;
    candidate.rj = rng_mod(rng, n_site);
    while ele_cfg
        .get(candidate.rj + (candidate.spin as usize) * n_site)
        .copied()
        .unwrap_or(0)
        != -1
        || loc_spn.get(candidate.rj).copied().unwrap_or(1) == 1
    {
        if icnt > icnt_max {
            candidate.reject = true;
            break;
        }
        candidate.rj = rng_mod(rng, n_site);
        icnt += 1;
    }

    candidate
}

/// Port of `make_candidate_exchange` for Sz-conserved normal mode.
pub fn make_candidate_exchange(
    ele_idx: &[i64],
    ele_cfg: &[i64],
    n_site: usize,
    n_elec: usize,
    ele_num: &[i64],
    rng: &mut Sfmt19937Rng,
) -> ExchangeCandidate {
    let c = make_candidate_exchange_inner(ele_idx, ele_cfg, n_site, n_elec, ele_num, rng);
    crate::sampling::driver::trace::record(
        2,
        &[
            c.mi as i64,
            c.ri as i64,
            c.rj as i64,
            c.spin as i64,
            c.mj as i64,
            c.spin_other as i64,
            i64::from(c.reject),
        ],
    );
    c
}

fn make_candidate_exchange_inner(
    ele_idx: &[i64],
    ele_cfg: &[i64],
    n_site: usize,
    n_elec: usize,
    ele_num: &[i64],
    rng: &mut Sfmt19937Rng,
) -> ExchangeCandidate {
    // Reject if no site has exactly one electron.
    let mut has_single = false;
    for ri in 0..n_site {
        if ele_num.get(ri).copied().unwrap_or(0) + ele_num.get(ri + n_site).copied().unwrap_or(0)
            == 1
        {
            has_single = true;
            break;
        }
    }
    if !has_single {
        return ExchangeCandidate {
            mi: 0,
            ri: 0,
            mj: 0,
            rj: 0,
            spin: 0,
            spin_other: 0,
            reject: true,
        };
    }

    let mut mi;
    let mut spin;
    let mut ri;
    loop {
        mi = rng_mod(rng, n_elec);
        spin = rng_spin(rng);
        ri = ele_idx[mi + spin as usize * n_elec] as usize;
        let opposite_spin = 1 - spin;
        if ele_cfg[ri + opposite_spin as usize * n_site] == -1 {
            break;
        }
    }

    let spin_other = 1 - spin;
    let mut mj;
    let mut rj;
    loop {
        mj = rng_mod(rng, n_elec);
        rj = ele_idx[mj + spin_other as usize * n_elec] as usize;
        if ele_cfg[rj + spin as usize * n_site] == -1 {
            break;
        }
    }

    ExchangeCandidate {
        mi,
        ri,
        mj,
        rj,
        spin,
        spin_other,
        reject: false,
    }
}

/// Port of `get_update_type`.
pub fn get_update_type(
    n_ex_update_path: i64,
    i_flg_orbital_general: i64,
    two_sz: i64,
    rng: &mut Sfmt19937Rng,
) -> UpdateType {
    let update = get_update_type_inner(n_ex_update_path, i_flg_orbital_general, two_sz, rng);
    let code = match update {
        UpdateType::Hopping => 0,
        UpdateType::Exchange => 1,
        UpdateType::LocalSpinFlip => 2,
        UpdateType::None => 3,
    };
    crate::sampling::driver::trace::record(0, &[code]);
    update
}

fn get_update_type_inner(
    n_ex_update_path: i64,
    i_flg_orbital_general: i64,
    two_sz: i64,
    rng: &mut Sfmt19937Rng,
) -> UpdateType {
    match n_ex_update_path {
        0 => UpdateType::Hopping,
        1 => {
            if crate::sampling::driver::trace::draw_real2(rng) < 0.5 {
                UpdateType::Exchange
            } else {
                UpdateType::Hopping
            }
        }
        2 => {
            if i_flg_orbital_general == 0 {
                UpdateType::Exchange
            } else if two_sz == -1 {
                if crate::sampling::driver::trace::draw_real2(rng) < 0.5 {
                    UpdateType::Exchange
                } else {
                    UpdateType::LocalSpinFlip
                }
            } else {
                UpdateType::Exchange
            }
        }
        3 => {
            if crate::sampling::driver::trace::draw_real2(rng) < 0.5 {
                UpdateType::Hopping
            } else if crate::sampling::driver::trace::draw_real2(rng) < 0.5 {
                UpdateType::Exchange
            } else {
                UpdateType::LocalSpinFlip
            }
        }
        _ => UpdateType::None,
    }
}

/// Port of `make_candidate_hopping_fsz`.
///
/// Layouts:
/// * `ele_idx[mi] = ri`.
/// * `ele_spn[mi] = s`.
/// * `ele_cfg[ri + s*n_site] = mi` or `-1`.
/// * `loc_spn[ri] = 1` marks local-spin sites disallowed for itinerant
///   hopping.
#[allow(clippy::too_many_arguments)]
pub fn make_candidate_hopping_fsz(
    ele_idx: &[i64],
    ele_cfg: &[i64],
    ele_spn: &[i64],
    loc_spn: &[i64],
    n_site: usize,
    n_size: usize,
    two_sz: i64,
    rng: &mut Sfmt19937Rng,
) -> FszHoppingCandidate {
    let c = make_candidate_hopping_fsz_inner(
        FszHoppingInput {
            ele_idx,
            ele_cfg,
            ele_spn,
            loc_spn,
            n_site,
            n_size,
            two_sz,
        },
        rng,
    );
    crate::sampling::driver::trace::record(
        3,
        &[
            c.mi as i64,
            c.ri as i64,
            c.rj as i64,
            c.spin as i64,
            c.spin_to as i64,
            i64::from(c.reject),
        ],
    );
    c
}

struct FszHoppingInput<'a> {
    ele_idx: &'a [i64],
    ele_cfg: &'a [i64],
    ele_spn: &'a [i64],
    loc_spn: &'a [i64],
    n_site: usize,
    n_size: usize,
    two_sz: i64,
}

fn make_candidate_hopping_fsz_inner(
    input: FszHoppingInput<'_>,
    rng: &mut Sfmt19937Rng,
) -> FszHoppingCandidate {
    let FszHoppingInput {
        ele_idx,
        ele_cfg,
        ele_spn,
        loc_spn,
        n_site,
        n_size,
        two_sz,
    } = input;
    let icnt_max = n_site * n_site;

    let mut mi;
    let mut spin;
    let mut ri;
    loop {
        mi = rng_mod(rng, n_size);
        spin = ele_spn.get(mi).copied().unwrap_or(0).clamp(0, 1) as u8;
        ri = ele_idx.get(mi).copied().unwrap_or(0).max(0) as usize;
        if loc_spn.get(ri).copied().unwrap_or(1) == 0 {
            break;
        }
    }

    let mut rj;
    let mut spin_to;
    let mut icnt = 0usize;
    let mut reject = false;
    loop {
        rj = rng_mod(rng, n_site);
        spin_to = if two_sz == -1 { rng_spin(rng) } else { spin };
        if icnt > icnt_max {
            reject = true;
            break;
        }
        icnt += 1;
        if ele_cfg
            .get(rj + (spin_to as usize) * n_site)
            .copied()
            .unwrap_or(0)
            == -1
            && loc_spn.get(rj).copied().unwrap_or(1) == 0
        {
            break;
        }
    }

    FszHoppingCandidate {
        mi,
        ri,
        rj,
        spin,
        spin_to,
        reject,
    }
}

/// Port of `make_candidate_local_spin_flip_conduction`.
pub fn make_candidate_local_spin_flip_conduction(
    ele_idx: &[i64],
    ele_cfg: &[i64],
    ele_spn: &[i64],
    loc_spn: &[i64],
    n_site: usize,
    n_size: usize,
    rng: &mut Sfmt19937Rng,
) -> LocalSpinFlipCandidate {
    let c = make_candidate_local_spin_flip_conduction_inner(
        ele_idx, ele_cfg, ele_spn, loc_spn, n_site, n_size, rng,
    );
    crate::sampling::driver::trace::record(
        4,
        &[
            c.mi as i64,
            c.ri as i64,
            c.rj as i64,
            c.spin as i64,
            c.spin_to as i64,
            i64::from(c.reject),
        ],
    );
    c
}

fn make_candidate_local_spin_flip_conduction_inner(
    ele_idx: &[i64],
    ele_cfg: &[i64],
    ele_spn: &[i64],
    loc_spn: &[i64],
    n_site: usize,
    n_size: usize,
    rng: &mut Sfmt19937Rng,
) -> LocalSpinFlipCandidate {
    let icnt_max = n_site * n_site;
    let mut icnt = 0usize;
    let mut reject = false;

    let mut mi;
    let mut spin;
    let mut spin_to;
    let mut ri;
    loop {
        mi = rng_mod(rng, n_size);
        spin = ele_spn.get(mi).copied().unwrap_or(0).clamp(0, 1) as u8;
        spin_to = 1 - spin;
        ri = ele_idx.get(mi).copied().unwrap_or(0).max(0) as usize;
        if icnt > icnt_max {
            reject = true;
            break;
        }
        icnt += 1;
        if loc_spn.get(ri).copied().unwrap_or(1) == 0
            && ele_cfg
                .get(ri + (spin_to as usize) * n_site)
                .copied()
                .unwrap_or(0)
                == -1
        {
            break;
        }
    }

    LocalSpinFlipCandidate {
        mi,
        ri,
        rj: ri,
        spin,
        spin_to,
        reject,
    }
}

/// Port of `make_candidate_local_spin_flip_localspin`.
pub fn make_candidate_local_spin_flip_localspin(
    ele_idx: &[i64],
    ele_spn: &[i64],
    loc_spn: &[i64],
    n_size: usize,
    rng: &mut Sfmt19937Rng,
) -> LocalSpinFlipCandidate {
    let c = make_candidate_local_spin_flip_localspin_inner(ele_idx, ele_spn, loc_spn, n_size, rng);
    crate::sampling::driver::trace::record(
        5,
        &[
            c.mi as i64,
            c.ri as i64,
            c.rj as i64,
            c.spin as i64,
            c.spin_to as i64,
            i64::from(c.reject),
        ],
    );
    c
}

fn make_candidate_local_spin_flip_localspin_inner(
    ele_idx: &[i64],
    ele_spn: &[i64],
    loc_spn: &[i64],
    n_size: usize,
    rng: &mut Sfmt19937Rng,
) -> LocalSpinFlipCandidate {
    let mut mi;
    let mut spin;
    let mut ri;
    loop {
        mi = rng_mod(rng, n_size);
        spin = ele_spn.get(mi).copied().unwrap_or(0).clamp(0, 1) as u8;
        ri = ele_idx.get(mi).copied().unwrap_or(0).max(0) as usize;
        if loc_spn.get(ri).copied().unwrap_or(0) == 1 {
            break;
        }
    }
    LocalSpinFlipCandidate {
        mi,
        ri,
        rj: ri,
        spin,
        spin_to: 1 - spin,
        reject: false,
    }
}

/// Port of `make_candidate_exchange_fsz`.
pub fn make_candidate_exchange_fsz(
    ele_idx: &[i64],
    ele_cfg: &[i64],
    ele_num: &[i64],
    ele_spn: &[i64],
    n_site: usize,
    n_size: usize,
    rng: &mut Sfmt19937Rng,
) -> FszExchangeCandidate {
    let c =
        make_candidate_exchange_fsz_inner(ele_idx, ele_cfg, ele_num, ele_spn, n_site, n_size, rng);
    crate::sampling::driver::trace::record(
        6,
        &[
            c.mi as i64,
            c.ri as i64,
            c.rj as i64,
            c.spin as i64,
            i64::from(c.reject),
        ],
    );
    c
}

fn make_candidate_exchange_fsz_inner(
    ele_idx: &[i64],
    ele_cfg: &[i64],
    ele_num: &[i64],
    ele_spn: &[i64],
    n_site: usize,
    n_size: usize,
    rng: &mut Sfmt19937Rng,
) -> FszExchangeCandidate {
    // Check that there are at least two single-occupancy sites with
    // opposite spins; otherwise exchange is impossible.
    let mut flag = true;
    let mut spn_0 = 0i64;
    let mut spn_1;
    for ri in 0..n_site {
        if ele_num.get(ri).copied().unwrap_or(0) + ele_num.get(ri + n_site).copied().unwrap_or(0)
            == 1
        {
            if spn_0 == 0 {
                spn_0 = 2 * ele_num.get(ri).copied().unwrap_or(0) - 1;
            } else {
                spn_1 = 2 * ele_num.get(ri).copied().unwrap_or(0) - 1;
                if spn_0 * spn_1 < 0 {
                    flag = false;
                    break;
                }
            }
        }
    }
    if flag {
        return FszExchangeCandidate {
            mi: 0,
            ri: 0,
            rj: 0,
            spin: 0,
            reject: true,
        };
    }

    let mut mi;
    let mut spin;
    let mut ri;
    loop {
        mi = rng_mod(rng, n_size);
        spin = ele_spn.get(mi).copied().unwrap_or(0).clamp(0, 1) as u8;
        ri = ele_idx.get(mi).copied().unwrap_or(0).max(0) as usize;
        if ele_cfg
            .get(ri + (1 - spin) as usize * n_site)
            .copied()
            .unwrap_or(0)
            == -1
        {
            break;
        }
    }

    let spin_other = 1 - spin;
    let mut mj;
    let mut rj;
    loop {
        mj = rng_mod(rng, n_size);
        rj = ele_idx.get(mj).copied().unwrap_or(0).max(0) as usize;
        if ele_cfg
            .get(rj + spin as usize * n_site)
            .copied()
            .unwrap_or(0)
            == -1
            && ele_spn.get(mj).copied().unwrap_or(-1) == spin_other as i64
        {
            break;
        }
    }

    FszExchangeCandidate {
        mi,
        ri,
        rj,
        spin,
        reject: false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn normal_state() -> (Vec<i64>, Vec<i64>, Vec<i64>, Vec<i64>) {
        let ele_idx = vec![0, 2, 4, 1, 3, 5];
        let ele_cfg = vec![0, -1, 1, -1, 2, -1, -1, 0, -1, 1, -1, 2];
        let ele_num = vec![1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1];
        let loc_spn = vec![0, 0, 0, 0, 0, 0];
        (ele_idx, ele_cfg, ele_num, loc_spn)
    }

    #[test]
    fn get_update_type_path0_is_deterministic() {
        let mut rng = Sfmt19937Rng::new(42);
        assert_eq!(get_update_type(0, 0, 0, &mut rng), UpdateType::Hopping);
    }

    #[test]
    fn hopping_candidate_is_in_bounds() {
        let (ele_idx, ele_cfg, _ele_num, loc_spn) = normal_state();
        let mut rng = Sfmt19937Rng::new(11272);
        let c = make_candidate_hopping(&ele_idx, &ele_cfg, 6, 3, &loc_spn, &mut rng);
        assert!(!c.reject);
        assert!(c.mi < 3);
        assert!(c.ri < 6);
        assert!(c.rj < 6);
        assert!(c.spin <= 1);
        assert_eq!(ele_cfg[c.rj + c.spin as usize * 6], -1);
    }

    #[test]
    fn exchange_candidate_is_opposite_spin() {
        let (ele_idx, ele_cfg, ele_num, _loc_spn) = normal_state();
        let mut rng = Sfmt19937Rng::new(11272);
        let c = make_candidate_exchange(&ele_idx, &ele_cfg, 6, 3, &ele_num, &mut rng);
        assert!(!c.reject);
        assert_eq!(c.spin_other, 1 - c.spin);
    }

    #[test]
    fn fsz_hopping_candidate_is_in_bounds() {
        let ele_idx = vec![0, 1, 2, 3, 4, 5];
        let ele_spn = vec![0, 1, 0, 1, 0, 1];
        let ele_cfg = vec![0, -1, 2, -1, 4, -1, -1, 1, -1, 3, -1, 5];
        let loc_spn = vec![0, 0, 1, 0, 1, 0];
        let mut rng = Sfmt19937Rng::new(11272);
        let c =
            make_candidate_hopping_fsz(&ele_idx, &ele_cfg, &ele_spn, &loc_spn, 6, 6, -1, &mut rng);
        assert!(c.mi < 6);
        assert!(c.ri < 6);
        assert!(c.rj < 6);
        assert!(c.spin <= 1);
        assert!(c.spin_to <= 1);
    }

    #[test]
    fn fsz_local_spin_flip_localspin_selects_local_site() {
        let ele_idx = vec![0, 1, 2, 3, 4, 5];
        let ele_spn = vec![0, 1, 0, 1, 0, 1];
        let loc_spn = vec![0, 0, 1, 0, 1, 0];
        let mut rng = Sfmt19937Rng::new(11272);
        let c = make_candidate_local_spin_flip_localspin(&ele_idx, &ele_spn, &loc_spn, 6, &mut rng);
        assert_eq!(c.ri, c.rj);
        assert_eq!(loc_spn[c.ri], 1);
        assert_eq!(c.spin_to, 1 - c.spin);
        assert!(!c.reject);
    }
}
