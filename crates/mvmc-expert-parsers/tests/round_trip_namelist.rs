//! Round-trip integration test against the four upstream
//! `examples/inputs/*/namelist.def` cases.
//!
//! The reference numbers are pinned to the values produced by
//! `extern/Julia-mVMC` (see `tools/dump_namelist_counts.jl` -- not
//! committed since the four cases are tiny enough to inline). The Rust
//! parser must reproduce them verbatim.

use std::path::{Path, PathBuf};

use mvmc_expert_parsers::parse_expert_mode_files;

mod support;
use support::julia_mvmc_root;

fn upstream_inputs_dir(case: &str) -> PathBuf {
    julia_mvmc_root()
        .expect("Julia-mVMC checkout required for round_trip test")
        .join("examples/inputs")
        .join(case)
}

#[derive(Debug)]
struct Expected {
    nsite: i64,
    ncond: i64,
    two_sz: i64,
    nelec: i64,
    nlocspin: i64,
    nsp_gauss_leg: i64,
    nmp_trans: i64,
    n_orbital_idx: i64,
    transfer_terms: usize,
    coulomb_intra: usize,
    coulomb_inter: usize,
    hund_terms: usize,
    exchange_terms: usize,
    locspin_terms: usize,
    gutzwiller_terms: usize,
    n_gutzwiller_idx: i64,
    jastrow_terms: usize,
    n_jastrow_idx: i64,
    orbital_terms: usize,
    i_flg_orbital_general: i64,
    i_flg_orbital_anti_parallel: i64,
    i_flg_orbital_parallel: i64,
    green_one_terms: usize,
    green_two_terms: usize,
    n_qp_trans: i64,
}

fn check(case: &str, expect: &Expected) {
    let dir = upstream_inputs_dir(case);
    let namelist = dir.join("namelist.def");
    assert!(
        Path::new(&namelist).is_file(),
        "upstream namelist not found at {}; did `git submodule update --init` run?",
        namelist.display()
    );
    let data = parse_expert_mode_files(&namelist).unwrap_or_else(|e| {
        panic!(
            "parse_expert_mode_files({}) failed: {:?}",
            namelist.display(),
            e
        )
    });

    let p = &data.modpara;
    assert_eq!(p.nsite, expect.nsite, "{case} nsite");
    assert_eq!(p.ncond, expect.ncond, "{case} ncond");
    assert_eq!(p.two_sz, expect.two_sz, "{case} two_sz");
    assert_eq!(p.nelec, expect.nelec, "{case} nelec");
    assert_eq!(p.nlocspin, expect.nlocspin, "{case} nlocspin");
    assert_eq!(
        p.nsp_gauss_leg, expect.nsp_gauss_leg,
        "{case} nsp_gauss_leg"
    );
    assert_eq!(p.nmp_trans, expect.nmp_trans, "{case} nmp_trans");
    assert_eq!(
        p.n_orbital_idx, expect.n_orbital_idx,
        "{case} n_orbital_idx"
    );

    assert_eq!(
        data.transfer_terms.len(),
        expect.transfer_terms,
        "{case} transfer_terms"
    );
    assert_eq!(
        data.coulomb_intra_terms.len(),
        expect.coulomb_intra,
        "{case} coulomb_intra"
    );
    assert_eq!(
        data.coulomb_inter_terms.len(),
        expect.coulomb_inter,
        "{case} coulomb_inter"
    );
    assert_eq!(
        data.hund_terms.len(),
        expect.hund_terms,
        "{case} hund_terms"
    );
    assert_eq!(
        data.exchange_terms.len(),
        expect.exchange_terms,
        "{case} exchange_terms"
    );
    assert_eq!(
        data.locspin_terms.len(),
        expect.locspin_terms,
        "{case} locspin_terms"
    );
    assert_eq!(
        data.gutzwiller_terms.len(),
        expect.gutzwiller_terms,
        "{case} gutzwiller_terms"
    );
    assert_eq!(
        data.n_gutzwiller_idx, expect.n_gutzwiller_idx,
        "{case} n_gutzwiller_idx"
    );
    assert_eq!(
        data.jastrow_terms.len(),
        expect.jastrow_terms,
        "{case} jastrow_terms"
    );
    assert_eq!(
        data.n_jastrow_idx, expect.n_jastrow_idx,
        "{case} n_jastrow_idx"
    );
    assert_eq!(
        data.orbital_terms.len(),
        expect.orbital_terms,
        "{case} orbital_terms"
    );
    assert_eq!(
        data.i_flg_orbital_general, expect.i_flg_orbital_general,
        "{case} i_flg_orbital_general"
    );
    assert_eq!(
        data.i_flg_orbital_anti_parallel, expect.i_flg_orbital_anti_parallel,
        "{case} i_flg_orbital_anti_parallel"
    );
    assert_eq!(
        data.i_flg_orbital_parallel, expect.i_flg_orbital_parallel,
        "{case} i_flg_orbital_parallel"
    );
    assert_eq!(
        data.green_one_terms.len(),
        expect.green_one_terms,
        "{case} green_one_terms"
    );
    assert_eq!(
        data.green_two_terms.len(),
        expect.green_two_terms,
        "{case} green_two_terms"
    );
    assert_eq!(data.n_qp_trans, expect.n_qp_trans, "{case} n_qp_trans");
}

#[test]
fn heisenberg_chain_real_round_trip() {
    check(
        "heisenberg_chain_real",
        &Expected {
            nsite: 6,
            ncond: 0,
            two_sz: 0,
            nelec: 3,
            nlocspin: 6,
            nsp_gauss_leg: 8,
            nmp_trans: -1,
            n_orbital_idx: 12,
            transfer_terms: 0,
            coulomb_intra: 0,
            coulomb_inter: 6,
            hund_terms: 6,
            exchange_terms: 6,
            locspin_terms: 6,
            gutzwiller_terms: 1,
            n_gutzwiller_idx: 1,
            jastrow_terms: 1,
            n_jastrow_idx: 1,
            orbital_terms: 36,
            i_flg_orbital_general: 0,
            i_flg_orbital_anti_parallel: 1,
            i_flg_orbital_parallel: 0,
            green_one_terms: 2,
            green_two_terms: 36,
            n_qp_trans: 2,
        },
    );
}

#[test]
fn heisenberg_chain_cmp_round_trip() {
    check(
        "heisenberg_chain_cmp",
        &Expected {
            nsite: 6,
            ncond: 0,
            two_sz: 0,
            nelec: 3,
            nlocspin: 6,
            nsp_gauss_leg: 8,
            nmp_trans: -1,
            n_orbital_idx: 12,
            transfer_terms: 0,
            coulomb_intra: 0,
            coulomb_inter: 6,
            hund_terms: 6,
            exchange_terms: 6,
            locspin_terms: 6,
            gutzwiller_terms: 1,
            n_gutzwiller_idx: 1,
            jastrow_terms: 1,
            n_jastrow_idx: 1,
            orbital_terms: 36,
            i_flg_orbital_general: 0,
            i_flg_orbital_anti_parallel: 1,
            i_flg_orbital_parallel: 0,
            green_one_terms: 2,
            green_two_terms: 36,
            n_qp_trans: 2,
        },
    );
}

#[test]
fn heisenberg_chain_fsz_round_trip() {
    check(
        "heisenberg_chain_fsz",
        &Expected {
            nsite: 6,
            ncond: 0,
            two_sz: -1,
            nelec: 3,
            nlocspin: 6,
            nsp_gauss_leg: 1,
            nmp_trans: -1,
            n_orbital_idx: 22,
            transfer_terms: 24,
            coulomb_intra: 0,
            coulomb_inter: 6,
            hund_terms: 6,
            exchange_terms: 6,
            locspin_terms: 6,
            gutzwiller_terms: 1,
            n_gutzwiller_idx: 1,
            jastrow_terms: 1,
            n_jastrow_idx: 1,
            orbital_terms: 66,
            i_flg_orbital_general: 1,
            i_flg_orbital_anti_parallel: 1,
            i_flg_orbital_parallel: 1,
            green_one_terms: 2,
            green_two_terms: 36,
            n_qp_trans: 2,
        },
    );
}

#[test]
fn hubbard_chain_real_round_trip() {
    check(
        "hubbard_chain_real",
        &Expected {
            nsite: 6,
            ncond: 6,
            two_sz: 0,
            nelec: 3,
            nlocspin: 0,
            nsp_gauss_leg: 8,
            nmp_trans: -1,
            n_orbital_idx: 12,
            transfer_terms: 24,
            coulomb_intra: 6,
            coulomb_inter: 0,
            hund_terms: 0,
            exchange_terms: 0,
            locspin_terms: 6,
            gutzwiller_terms: 2,
            n_gutzwiller_idx: 2,
            jastrow_terms: 5,
            n_jastrow_idx: 5,
            orbital_terms: 36,
            i_flg_orbital_general: 0,
            i_flg_orbital_anti_parallel: 1,
            i_flg_orbital_parallel: 0,
            green_one_terms: 12,
            green_two_terms: 36,
            n_qp_trans: 2,
        },
    );
}

/// Spot-check term-level fidelity: the first Exchange entry for the
/// Heisenberg real chain is `(0, 1, -0.5)` and the qptrans site_map
/// for mpidx 1 must be a cyclic shift by 1.
#[test]
fn heisenberg_chain_real_spot_check_values() {
    let dir = upstream_inputs_dir("heisenberg_chain_real");
    let data = parse_expert_mode_files(dir.join("namelist.def")).unwrap();

    let first_exchange = data.exchange_terms[0];
    assert_eq!(first_exchange.site1, 0);
    assert_eq!(first_exchange.site2, 1);
    assert!((first_exchange.value + 0.5).abs() < 1e-15);

    let last_exchange = data.exchange_terms.last().unwrap();
    assert_eq!(last_exchange.site1, 5);
    assert_eq!(last_exchange.site2, 0); // cyclic boundary

    // qptrans entry mpidx=1 should map site j -> (j + 1) mod 6.
    assert_eq!(data.qp_trans_entries.len(), 2);
    let shift = &data.qp_trans_entries[1];
    for j in 0..6 {
        assert_eq!(shift.site_map[j], ((j + 1) % 6) as i64, "mpidx=1 j={}", j);
        assert_eq!(shift.site_sign[j], 1);
    }
}
