//! Green function calculations for VMC
//!
//! This module implements Green functions for calculating non-diagonal
//! Hamiltonian matrix elements in VMC calculations.
//!
//! Reference: mVMC/src/mVMC/greenfunction.c

use crate::Result;
use crate::monte_carlo::ElectronConfiguration;
use mvmc_physics::wavefunction::CombinedWavefunction;
use num_complex::Complex64;

/// Calculates the 1-body Green function (GreenFunc1)
///
/// Computes: <c^†_{rj,s} c_{ri,s}> = ψ(X')/ψ(X)
///
/// Where X' is the configuration after moving electron from (ri,s) to (rj,s)
///
/// # Arguments
///
/// * `ri` - Source site index
/// * `rj` - Destination site index
/// * `s` - Spin (0: up, 1: down)
/// * `ip` - Current wavefunction amplitude
/// * `config` - Current electron configuration
/// * `wavefunction` - Wavefunction object
///
/// # Returns
///
/// * Complex amplitude ratio ψ(X')/ψ(X)
///
/// Reference: mVMC/src/mVMC/greenfunction.c:50-120 (GreenFunc1)
pub fn green_func1(
    ri: usize,
    rj: usize,
    s: usize,
    ip: Complex64,
    config: &ElectronConfiguration,
    wavefunction: &CombinedWavefunction,
) -> Result<Complex64> {
    let nsite = config.nsite();

    // Boundary checks
    if ri >= nsite || rj >= nsite || s > 1 {
        return Ok(Complex64::new(0.0, 0.0));
    }

    let site_spin_idx_ri = ri + s * nsite;
    let site_spin_idx_rj = rj + s * nsite;

    // Check if move is possible
    // Reference: mVMC/src/mVMC/greenfunction.c:60-62
    if config.ele_cfg()[site_spin_idx_ri] == -1 {
        // ri is empty - cannot move electron from empty site
        return Ok(Complex64::new(0.0, 0.0));
    }
    if config.ele_cfg()[site_spin_idx_rj] != -1 {
        // rj is occupied - cannot move electron to occupied site
        return Ok(Complex64::new(0.0, 0.0));
    }

    // Create new configuration with electron moved from ri to rj
    // Reference: mVMC/src/mVMC/greenfunction.c:65-75
    let mut new_config = config.clone();
    let mut new_ele_cfg = config.ele_cfg().to_vec();
    let mut new_ele_idx = config.ele_idx().to_vec();
    let mut new_ele_num = config.ele_num().to_vec();

    // Get electron index at site ri
    let mi = new_ele_cfg[site_spin_idx_ri];

    // Move electron: ri → rj
    new_ele_cfg[site_spin_idx_ri] = -1;          // ri becomes empty
    new_ele_cfg[site_spin_idx_rj] = mi;          // rj gets electron mi
    new_ele_idx[mi as usize] = site_spin_idx_rj as i32;  // Update electron position
    new_ele_num[site_spin_idx_ri] = 0;           // ri has 0 electrons
    new_ele_num[site_spin_idx_rj] = 1;           // rj has 1 electron

    // Update configuration
    new_config.set_configuration(&new_ele_cfg, &new_ele_num)?;

    // Calculate new wavefunction amplitude
    // Reference: mVMC/src/mVMC/greenfunction.c:80-85
    let spin_config = electron_config_to_spin_config(&new_config);
    let ip_new = wavefunction.calculate(&spin_config);

    // Return amplitude ratio
    // Reference: mVMC/src/mVMC/greenfunction.c:90
    if ip.norm() < 1e-12 {
        Ok(Complex64::new(0.0, 0.0))
    } else {
        Ok(ip_new / ip)
    }
}

/// Calculates the 2-body Green function (GreenFunc2)
///
/// Computes: <c^†_{rk,sk} c^†_{rl,sl} c_{rj,sj} c_{ri,si}>
///
/// This is used for exchange coupling and other two-body operators.
///
/// # Arguments
///
/// * `ri` - First annihilation site
/// * `rj` - Second annihilation site
/// * `rk` - First creation site
/// * `rl` - Second creation site
/// * `si` - First annihilation spin (0: up, 1: down)
/// * `sj` - Second annihilation spin
/// * `sk` - First creation spin
/// * `sl` - Second creation spin
/// * `ip` - Current wavefunction amplitude
/// * `config` - Current electron configuration
/// * `wavefunction` - Wavefunction object
///
/// # Returns
///
/// * Complex amplitude ratio for the two-body operator
///
/// Reference: mVMC/src/mVMC/greenfunction.c:130-200 (GreenFunc2)
pub fn green_func2(
    ri: usize,
    rj: usize,
    rk: usize,
    rl: usize,
    si: usize,
    sj: usize,
    sk: usize,
    sl: usize,
    ip: Complex64,
    config: &ElectronConfiguration,
    wavefunction: &CombinedWavefunction,
) -> Result<Complex64> {
    let nsite = config.nsite();

    // Boundary checks
    if ri >= nsite || rj >= nsite || rk >= nsite || rl >= nsite {
        return Ok(Complex64::new(0.0, 0.0));
    }
    if si > 1 || sj > 1 || sk > 1 || sl > 1 {
        return Ok(Complex64::new(0.0, 0.0));
    }

    let site_spin_idx_ri = ri + si * nsite;
    let site_spin_idx_rj = rj + sj * nsite;
    let site_spin_idx_rk = rk + sk * nsite;
    let site_spin_idx_rl = rl + sl * nsite;

    // Check if operation is possible
    // Reference: mVMC/src/mVMC/greenfunction.c:145-155

    // ri and rj must be occupied (electrons to annihilate)
    if config.ele_cfg()[site_spin_idx_ri] == -1 {
        return Ok(Complex64::new(0.0, 0.0));
    }
    if config.ele_cfg()[site_spin_idx_rj] == -1 {
        return Ok(Complex64::new(0.0, 0.0));
    }

    // rk and rl must be empty (sites to create electrons)
    if config.ele_cfg()[site_spin_idx_rk] != -1 {
        // Unless rk is same as ri or rj (annihilation-creation on same site)
        if !(rk == ri && sk == si) && !(rk == rj && sk == sj) {
            return Ok(Complex64::new(0.0, 0.0));
        }
    }
    if config.ele_cfg()[site_spin_idx_rl] != -1 {
        if !(rl == ri && sl == si) && !(rl == rj && sl == sj) {
            return Ok(Complex64::new(0.0, 0.0));
        }
    }

    // Create new configuration
    // Reference: mVMC/src/mVMC/greenfunction.c:160-180
    let mut new_config = config.clone();
    let mut new_ele_cfg = config.ele_cfg().to_vec();
    let mut new_ele_idx = config.ele_idx().to_vec();
    let mut new_ele_num = config.ele_num().to_vec();

    // Get electron indices
    let mi = new_ele_cfg[site_spin_idx_ri];
    let mj = new_ele_cfg[site_spin_idx_rj];

    // Annihilate electrons at ri and rj
    new_ele_cfg[site_spin_idx_ri] = -1;
    new_ele_cfg[site_spin_idx_rj] = -1;
    new_ele_num[site_spin_idx_ri] = 0;
    new_ele_num[site_spin_idx_rj] = 0;

    // Create electrons at rk and rl
    new_ele_cfg[site_spin_idx_rk] = mi;
    new_ele_cfg[site_spin_idx_rl] = mj;
    new_ele_idx[mi as usize] = site_spin_idx_rk as i32;
    new_ele_idx[mj as usize] = site_spin_idx_rl as i32;
    new_ele_num[site_spin_idx_rk] = 1;
    new_ele_num[site_spin_idx_rl] = 1;

    // Update configuration
    new_config.set_configuration(&new_ele_cfg, &new_ele_num)?;

    // Calculate new wavefunction amplitude
    // Reference: mVMC/src/mVMC/greenfunction.c:185-190
    let spin_config = electron_config_to_spin_config(&new_config);
    let ip_new = wavefunction.calculate(&spin_config);

    // Calculate fermionic sign
    // For exchange: c^†_↓ c^†_↑ c_↑ c_↓ has sign from anticommutation
    // Reference: mVMC/src/mVMC/greenfunction.c:195
    let sign = calculate_fermionic_sign(ri, rj, rk, rl, si, sj, sk, sl);

    // Return amplitude ratio with sign
    if ip.norm() < 1e-12 {
        Ok(Complex64::new(0.0, 0.0))
    } else {
        Ok(sign * ip_new / ip)
    }
}

/// Converts ElectronConfiguration to spin configuration array
///
/// # Arguments
///
/// * `config` - Electron configuration
///
/// # Returns
///
/// * Spin configuration as Vec<u8> (0: empty, 1: up, 2: down, 3: both)
fn electron_config_to_spin_config(config: &ElectronConfiguration) -> Vec<u8> {
    let nsite = config.nsite();
    let mut spin_config = vec![0u8; nsite];

    for site in 0..nsite {
        let n_up = config.electron_number_up(site);
        let n_down = config.electron_number_down(site);

        spin_config[site] = match (n_up, n_down) {
            (0, 0) => 0,  // Empty
            (1, 0) => 1,  // Up spin
            (0, 1) => 2,  // Down spin
            (1, 1) => 3,  // Both spins (doubly occupied)
            _ => 0,       // Invalid - treat as empty
        };
    }

    spin_config
}

/// Calculates fermionic sign from anticommutation relations
///
/// For exchange operator S^+_i S^-_j = c^†_{i,↑} c_{i,↓} c^†_{j,↓} c_{j,↑}
/// Need to account for sign changes from anticommutation
///
/// # Arguments
///
/// * `ri`, `rj`, `rk`, `rl` - Site indices
/// * `si`, `sj`, `sk`, `sl` - Spin indices
///
/// # Returns
///
/// * Sign factor (±1.0)
///
/// Reference: mVMC/src/mVMC/greenfunction.c:195
fn calculate_fermionic_sign(
    _ri: usize,
    _rj: usize,
    _rk: usize,
    _rl: usize,
    _si: usize,
    _sj: usize,
    _sk: usize,
    _sl: usize,
) -> Complex64 {
    // For now, return +1
    // Full implementation requires counting anticommutations
    // Reference: mVMC/src/mVMC/greenfunction.c:195-200
    Complex64::new(1.0, 0.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{SiteCount, ElectronCount, TwoSz};
    use mvmc_physics::wavefunction::CombinedWavefunction;

    #[test]
    fn test_green_func1_empty_source() {
        let nsite = 4;
        let ne = 2;
        let config = ElectronConfiguration::new(
            SiteCount::new(nsite),
            ElectronCount::new(ne),
            TwoSz::new(0),
        );
        let wavefunction = CombinedWavefunction::new(nsite, ne).unwrap();
        let ip = Complex64::new(1.0, 0.0);

        // Try to move from empty site (should return 0)
        let result = green_func1(0, 1, 0, ip, &config, &wavefunction).unwrap();
        assert_eq!(result.norm(), 0.0);
    }

    #[test]
    fn test_green_func1_occupied_destination() {
        let nsite = 4;
        let ne = 2;
        let mut config = ElectronConfiguration::new(
            SiteCount::new(nsite),
            ElectronCount::new(ne),
            TwoSz::new(0),
        );

        // Place electrons at sites 0 and 1
        config.set_electron(0, 1).unwrap();  // up spin at site 0
        config.set_electron(1, 1).unwrap();  // up spin at site 1

        let wavefunction = CombinedWavefunction::new(nsite, ne).unwrap();
        let ip = Complex64::new(1.0, 0.0);

        // Try to move to occupied site (should return 0)
        let result = green_func1(0, 1, 0, ip, &config, &wavefunction).unwrap();
        assert_eq!(result.norm(), 0.0);
    }

    #[test]
    fn test_green_func2_boundary_checks() {
        let nsite = 4;
        let ne = 2;
        let config = ElectronConfiguration::new(
            SiteCount::new(nsite),
            ElectronCount::new(ne),
            TwoSz::new(0),
        );
        let wavefunction = CombinedWavefunction::new(nsite, ne).unwrap();
        let ip = Complex64::new(1.0, 0.0);

        // Out of bounds site index
        let result = green_func2(
            10, 0, 0, 1,  // ri out of bounds
            0, 0, 0, 0,
            ip,
            &config,
            &wavefunction,
        ).unwrap();
        assert_eq!(result.norm(), 0.0);

        // Invalid spin index
        let result = green_func2(
            0, 1, 0, 1,
            3, 0, 0, 0,  // si out of bounds
            ip,
            &config,
            &wavefunction,
        ).unwrap();
        assert_eq!(result.norm(), 0.0);
    }
}
