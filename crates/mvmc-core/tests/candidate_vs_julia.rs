//! Phase 4.3.3 normal-mode candidate generator parity.
//!
//! Fixture is produced by `extern/Julia-mVMC/tools/dump_candidate_reference.jl`.

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;

use mvmc_core::sampling::candidate::{
    get_update_type, make_candidate_exchange, make_candidate_exchange_fsz, make_candidate_hopping,
    make_candidate_hopping_fsz, make_candidate_local_spin_flip_conduction,
    make_candidate_local_spin_flip_localspin, UpdateType,
};
use sfmt19937::Sfmt19937Rng;

const FIXTURE: &str = "../../tests/fixtures/candidate/normal.txt";

#[derive(Debug, Clone, PartialEq, Eq)]
struct ExpectedHopping {
    mi: usize,
    ri: usize,
    rj: usize,
    spin: u8,
    reject: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ExpectedExchange {
    mi: usize,
    ri: usize,
    mj: usize,
    rj: usize,
    spin: u8,
    spin_other: u8,
    reject: bool,
}

#[derive(Debug)]
struct FixtureData {
    n_site: usize,
    n_elec: usize,
    ele_idx: Vec<i64>,
    ele_cfg: Vec<i64>,
    ele_num: Vec<i64>,
    loc_spn: Vec<i64>,
    update_seed: u32,
    update_specs: Vec<(i64, i64, i64, UpdateType)>,
    hopping_seed: u32,
    hoppings: Vec<ExpectedHopping>,
    exchange_seed: u32,
    exchanges: Vec<ExpectedExchange>,
}

fn parse_i64s(rest: &str) -> Vec<i64> {
    rest.split_ascii_whitespace()
        .map(|s| s.parse::<i64>().unwrap())
        .collect()
}

fn parse_update_type(s: &str) -> UpdateType {
    match s {
        "HOPPING" => UpdateType::Hopping,
        "EXCHANGE" => UpdateType::Exchange,
        "LOCALSPINFLIP" => UpdateType::LocalSpinFlip,
        "NONE" => UpdateType::None,
        other => panic!("unknown update type: {other}"),
    }
}

fn load_fixture() -> FixtureData {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(FIXTURE);
    let file =
        File::open(&path).unwrap_or_else(|e| panic!("failed to open {}: {e}", path.display()));
    let mut n_site = None;
    let mut n_elec = None;
    let mut ele_idx = Vec::new();
    let mut ele_cfg = Vec::new();
    let mut ele_num = Vec::new();
    let mut loc_spn = Vec::new();
    let mut update_seed = None;
    let mut update_specs = Vec::new();
    let mut hopping_seed = None;
    let mut hoppings = Vec::new();
    let mut exchange_seed = None;
    let mut exchanges = Vec::new();

    for line in BufReader::new(file).lines() {
        let line = line.unwrap();
        let t = line.trim();
        if t.is_empty() || t.starts_with('#') {
            continue;
        }
        if let Some(rest) = t.strip_prefix("n_site ") {
            n_site = Some(rest.parse::<usize>().unwrap());
        } else if let Some(rest) = t.strip_prefix("n_elec ") {
            n_elec = Some(rest.parse::<usize>().unwrap());
        } else if let Some(rest) = t.strip_prefix("ele_idx ") {
            ele_idx = parse_i64s(rest);
        } else if let Some(rest) = t.strip_prefix("ele_cfg ") {
            ele_cfg = parse_i64s(rest);
        } else if let Some(rest) = t.strip_prefix("ele_num ") {
            ele_num = parse_i64s(rest);
        } else if let Some(rest) = t.strip_prefix("loc_spn ") {
            loc_spn = parse_i64s(rest);
        } else if let Some(rest) = t.strip_prefix("update_seed ") {
            update_seed = Some(rest.parse::<u32>().unwrap());
        } else if let Some(rest) = t.strip_prefix("update ") {
            let mut p = rest.split_ascii_whitespace();
            let path = p.next().unwrap().parse::<i64>().unwrap();
            let orbital_general = p.next().unwrap().parse::<i64>().unwrap();
            let two_sz = p.next().unwrap().parse::<i64>().unwrap();
            let update_type = parse_update_type(p.next().unwrap());
            update_specs.push((path, orbital_general, two_sz, update_type));
        } else if let Some(rest) = t.strip_prefix("hopping_seed ") {
            hopping_seed = Some(rest.parse::<u32>().unwrap());
        } else if let Some(rest) = t.strip_prefix("hopping ") {
            let vals = parse_i64s(rest);
            hoppings.push(ExpectedHopping {
                mi: vals[0] as usize,
                ri: vals[1] as usize,
                rj: vals[2] as usize,
                spin: vals[3] as u8,
                reject: vals[4] != 0,
            });
        } else if let Some(rest) = t.strip_prefix("exchange_seed ") {
            exchange_seed = Some(rest.parse::<u32>().unwrap());
        } else if let Some(rest) = t.strip_prefix("exchange ") {
            let vals = parse_i64s(rest);
            exchanges.push(ExpectedExchange {
                mi: vals[0] as usize,
                ri: vals[1] as usize,
                mj: vals[2] as usize,
                rj: vals[3] as usize,
                spin: vals[4] as u8,
                spin_other: vals[5] as u8,
                reject: vals[6] != 0,
            });
        }
    }
    FixtureData {
        n_site: n_site.unwrap(),
        n_elec: n_elec.unwrap(),
        ele_idx,
        ele_cfg,
        ele_num,
        loc_spn,
        update_seed: update_seed.unwrap(),
        update_specs,
        hopping_seed: hopping_seed.unwrap(),
        hoppings,
        exchange_seed: exchange_seed.unwrap(),
        exchanges,
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ExpectedFszHopping {
    mi: usize,
    ri: usize,
    rj: usize,
    spin: u8,
    spin_to: u8,
    reject: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ExpectedLocalSpinFlip {
    mi: usize,
    ri: usize,
    rj: usize,
    spin: u8,
    spin_to: u8,
    reject: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ExpectedFszExchange {
    mi: usize,
    ri: usize,
    rj: usize,
    spin: u8,
    reject: bool,
}

#[derive(Debug)]
struct FszFixtureData {
    n_site: usize,
    n_size: usize,
    ele_idx: Vec<i64>,
    ele_cfg: Vec<i64>,
    ele_num: Vec<i64>,
    ele_spn: Vec<i64>,
    loc_spn: Vec<i64>,
    hopping_seed: u32,
    hoppings: Vec<ExpectedFszHopping>,
    local_conduction_seed: u32,
    local_conduction: Vec<ExpectedLocalSpinFlip>,
    local_localspin_seed: u32,
    local_localspin: Vec<ExpectedLocalSpinFlip>,
    exchange_seed: u32,
    exchanges: Vec<ExpectedFszExchange>,
}

fn load_fsz_fixture() -> FszFixtureData {
    let path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/candidate/fsz.txt");
    let file =
        File::open(&path).unwrap_or_else(|e| panic!("failed to open {}: {e}", path.display()));
    let mut n_site = None;
    let mut n_size = None;
    let mut ele_idx = Vec::new();
    let mut ele_cfg = Vec::new();
    let mut ele_num = Vec::new();
    let mut ele_spn = Vec::new();
    let mut loc_spn = Vec::new();
    let mut hopping_seed = None;
    let mut hoppings = Vec::new();
    let mut local_conduction_seed = None;
    let mut local_conduction = Vec::new();
    let mut local_localspin_seed = None;
    let mut local_localspin = Vec::new();
    let mut exchange_seed = None;
    let mut exchanges = Vec::new();

    for line in BufReader::new(file).lines() {
        let line = line.unwrap();
        let t = line.trim();
        if t.is_empty() || t.starts_with('#') {
            continue;
        }
        if let Some(rest) = t.strip_prefix("n_site ") {
            n_site = Some(rest.parse::<usize>().unwrap());
        } else if let Some(rest) = t.strip_prefix("n_size ") {
            n_size = Some(rest.parse::<usize>().unwrap());
        } else if let Some(rest) = t.strip_prefix("ele_idx ") {
            ele_idx = parse_i64s(rest);
        } else if let Some(rest) = t.strip_prefix("ele_cfg ") {
            ele_cfg = parse_i64s(rest);
        } else if let Some(rest) = t.strip_prefix("ele_num ") {
            ele_num = parse_i64s(rest);
        } else if let Some(rest) = t.strip_prefix("ele_spn ") {
            ele_spn = parse_i64s(rest);
        } else if let Some(rest) = t.strip_prefix("loc_spn ") {
            loc_spn = parse_i64s(rest);
        } else if let Some(rest) = t.strip_prefix("hopping_fsz_seed ") {
            hopping_seed = Some(rest.parse::<u32>().unwrap());
        } else if let Some(rest) = t.strip_prefix("hopping_fsz ") {
            let vals = parse_i64s(rest);
            hoppings.push(ExpectedFszHopping {
                mi: vals[0] as usize,
                ri: vals[1] as usize,
                rj: vals[2] as usize,
                spin: vals[3] as u8,
                spin_to: vals[4] as u8,
                reject: vals[5] != 0,
            });
        } else if let Some(rest) = t.strip_prefix("local_conduction_seed ") {
            local_conduction_seed = Some(rest.parse::<u32>().unwrap());
        } else if let Some(rest) = t.strip_prefix("local_conduction ") {
            let vals = parse_i64s(rest);
            local_conduction.push(ExpectedLocalSpinFlip {
                mi: vals[0] as usize,
                ri: vals[1] as usize,
                rj: vals[2] as usize,
                spin: vals[3] as u8,
                spin_to: vals[4] as u8,
                reject: vals[5] != 0,
            });
        } else if let Some(rest) = t.strip_prefix("local_localspin_seed ") {
            local_localspin_seed = Some(rest.parse::<u32>().unwrap());
        } else if let Some(rest) = t.strip_prefix("local_localspin ") {
            let vals = parse_i64s(rest);
            local_localspin.push(ExpectedLocalSpinFlip {
                mi: vals[0] as usize,
                ri: vals[1] as usize,
                rj: vals[2] as usize,
                spin: vals[3] as u8,
                spin_to: vals[4] as u8,
                reject: vals[5] != 0,
            });
        } else if let Some(rest) = t.strip_prefix("exchange_fsz_seed ") {
            exchange_seed = Some(rest.parse::<u32>().unwrap());
        } else if let Some(rest) = t.strip_prefix("exchange_fsz ") {
            let vals = parse_i64s(rest);
            exchanges.push(ExpectedFszExchange {
                mi: vals[0] as usize,
                ri: vals[1] as usize,
                rj: vals[2] as usize,
                spin: vals[3] as u8,
                reject: vals[4] != 0,
            });
        }
    }
    FszFixtureData {
        n_site: n_site.unwrap(),
        n_size: n_size.unwrap(),
        ele_idx,
        ele_cfg,
        ele_num,
        ele_spn,
        loc_spn,
        hopping_seed: hopping_seed.unwrap(),
        hoppings,
        local_conduction_seed: local_conduction_seed.unwrap(),
        local_conduction,
        local_localspin_seed: local_localspin_seed.unwrap(),
        local_localspin,
        exchange_seed: exchange_seed.unwrap(),
        exchanges,
    }
}

#[test]
fn get_update_type_matches_julia_sfmt_sequence() {
    let f = load_fixture();
    let mut rng = Sfmt19937Rng::new(f.update_seed);
    for (idx, (path, orbital_general, two_sz, want)) in f.update_specs.iter().copied().enumerate() {
        let got = get_update_type(path, orbital_general, two_sz, &mut rng);
        assert_eq!(got, want, "update spec index {idx}");
    }
}

#[test]
fn hopping_candidates_match_julia_sfmt_sequence() {
    let f = load_fixture();
    let mut rng = Sfmt19937Rng::new(f.hopping_seed);
    for (idx, want) in f.hoppings.iter().enumerate() {
        let got = make_candidate_hopping(
            &f.ele_idx, &f.ele_cfg, f.n_site, f.n_elec, &f.loc_spn, &mut rng,
        );
        assert_eq!(got.mi, want.mi, "hop {idx} mi");
        assert_eq!(got.ri, want.ri, "hop {idx} ri");
        assert_eq!(got.rj, want.rj, "hop {idx} rj");
        assert_eq!(got.spin, want.spin, "hop {idx} spin");
        assert_eq!(got.reject, want.reject, "hop {idx} reject");
    }
}

#[test]
fn exchange_candidates_match_julia_sfmt_sequence() {
    let f = load_fixture();
    let mut rng = Sfmt19937Rng::new(f.exchange_seed);
    for (idx, want) in f.exchanges.iter().enumerate() {
        let got = make_candidate_exchange(
            &f.ele_idx, &f.ele_cfg, f.n_site, f.n_elec, &f.ele_num, &mut rng,
        );
        assert_eq!(got.mi, want.mi, "exchange {idx} mi");
        assert_eq!(got.ri, want.ri, "exchange {idx} ri");
        assert_eq!(got.mj, want.mj, "exchange {idx} mj");
        assert_eq!(got.rj, want.rj, "exchange {idx} rj");
        assert_eq!(got.spin, want.spin, "exchange {idx} spin");
        assert_eq!(got.spin_other, want.spin_other, "exchange {idx} spin_other");
        assert_eq!(got.reject, want.reject, "exchange {idx} reject");
    }
}

#[test]
fn fsz_hopping_candidates_match_julia_sfmt_sequence() {
    let f = load_fsz_fixture();
    let mut rng = Sfmt19937Rng::new(f.hopping_seed);
    for (idx, want) in f.hoppings.iter().enumerate() {
        let got = make_candidate_hopping_fsz(
            &f.ele_idx, &f.ele_cfg, &f.ele_spn, &f.loc_spn, f.n_site, f.n_size, -1, &mut rng,
        );
        assert_eq!(
            (got.mi, got.ri, got.rj, got.spin, got.spin_to, got.reject),
            (
                want.mi,
                want.ri,
                want.rj,
                want.spin,
                want.spin_to,
                want.reject
            ),
            "fsz hopping {idx}"
        );
    }
}

#[test]
fn fsz_local_spin_flip_conduction_matches_julia_sfmt_sequence() {
    let f = load_fsz_fixture();
    let mut rng = Sfmt19937Rng::new(f.local_conduction_seed);
    for (idx, want) in f.local_conduction.iter().enumerate() {
        let got = make_candidate_local_spin_flip_conduction(
            &f.ele_idx, &f.ele_cfg, &f.ele_spn, &f.loc_spn, f.n_site, f.n_size, &mut rng,
        );
        assert_eq!(
            (got.mi, got.ri, got.rj, got.spin, got.spin_to, got.reject),
            (
                want.mi,
                want.ri,
                want.rj,
                want.spin,
                want.spin_to,
                want.reject
            ),
            "local conduction {idx}"
        );
    }
}

#[test]
fn fsz_local_spin_flip_localspin_matches_julia_sfmt_sequence() {
    let f = load_fsz_fixture();
    let mut rng = Sfmt19937Rng::new(f.local_localspin_seed);
    for (idx, want) in f.local_localspin.iter().enumerate() {
        let got = make_candidate_local_spin_flip_localspin(
            &f.ele_idx, &f.ele_spn, &f.loc_spn, f.n_size, &mut rng,
        );
        assert_eq!(
            (got.mi, got.ri, got.rj, got.spin, got.spin_to, got.reject),
            (
                want.mi,
                want.ri,
                want.rj,
                want.spin,
                want.spin_to,
                want.reject
            ),
            "local localspin {idx}"
        );
    }
}

#[test]
fn fsz_exchange_candidates_match_julia_sfmt_sequence() {
    let f = load_fsz_fixture();
    let mut rng = Sfmt19937Rng::new(f.exchange_seed);
    for (idx, want) in f.exchanges.iter().enumerate() {
        let got = make_candidate_exchange_fsz(
            &f.ele_idx, &f.ele_cfg, &f.ele_num, &f.ele_spn, f.n_site, f.n_size, &mut rng,
        );
        assert_eq!(
            (got.mi, got.ri, got.rj, got.spin, got.reject),
            (want.mi, want.ri, want.rj, want.spin, want.reject),
            "fsz exchange {idx}"
        );
    }
}
