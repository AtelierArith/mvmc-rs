//! Phase 4.3.1 parity for the projection-bookkeeping helpers.
//!
//! Reads the fixtures emitted by
//! `extern/Julia-mVMC/tools/dump_projection_reference.jl` and replays
//! the same `make_proj_cnt` -> hop -> `update_proj_cnt` sequence on the
//! Rust port. The asserts demand byte-for-byte agreement on the
//! integer `proj_cnt` arrays.

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

use mvmc_core::sampling::projection::{init_loc_spn, make_proj_cnt, update_proj_cnt};
use mvmc_expert_parsers::parse_expert_mode_files;

mod support;
use support::julia_mvmc_root;

const FIXTURE_DIR: &str = "../../tests/fixtures/projection";

fn fixture_path(case: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join(FIXTURE_DIR)
        .join(format!("{case}.txt"))
}

fn namelist_path(case: &str) -> PathBuf {
    julia_mvmc_root()
        .expect("Julia-mVMC checkout required for projection_vs_julia test")
        .join("examples/inputs")
        .join(case)
        .join("namelist.def")
}

#[derive(Debug)]
struct HopStep {
    mi: usize,
    ri: i64,
    rj: i64,
    spin: u8,
    ele_num: Vec<i64>,
    proj_cnt: Vec<i64>,
}

#[derive(Debug)]
struct Fixture {
    n_site: usize,
    n_proj: usize,
    loc_spn: Vec<i64>,
    initial_ele_num: Vec<i64>,
    initial_proj_cnt: Vec<i64>,
    hops: Vec<HopStep>,
}

fn parse_ints(spec: &str) -> Vec<i64> {
    spec.split_ascii_whitespace()
        .map(|s| s.parse::<i64>().expect("int field parses"))
        .collect()
}

fn load_fixture(path: &Path) -> Fixture {
    let file = File::open(path).unwrap_or_else(|e| {
        panic!(
            "failed to open fixture {} (did you run dump_projection_reference.jl?): {}",
            path.display(),
            e
        )
    });
    let mut n_site = None;
    let mut n_proj = None;
    let mut loc_spn = Vec::new();
    let mut initial_ele_num = Vec::new();
    let mut initial_proj_cnt = Vec::new();
    let mut hops = Vec::new();
    for line in BufReader::new(file).lines() {
        let line = line.expect("read line");
        let t = line.trim();
        if t.is_empty() || t.starts_with('#') {
            continue;
        }
        if let Some(rest) = t.strip_prefix("n_site ") {
            n_site = Some(rest.parse().unwrap());
            continue;
        }
        if let Some(rest) = t.strip_prefix("n_elec ") {
            let _: usize = rest.parse().unwrap();
            continue;
        }
        if let Some(rest) = t.strip_prefix("n_proj ") {
            n_proj = Some(rest.parse().unwrap());
            continue;
        }
        if let Some(rest) = t.strip_prefix("loc_spn ") {
            loc_spn = parse_ints(rest);
            continue;
        }
        if let Some(rest) = t.strip_prefix("initial_ele_num ") {
            initial_ele_num = parse_ints(rest);
            continue;
        }
        if let Some(rest) = t.strip_prefix("initial_proj_cnt ") {
            initial_proj_cnt = parse_ints(rest);
            continue;
        }
        if let Some(rest) = t.strip_prefix("n_hops ") {
            let _: usize = rest.parse().unwrap();
            continue;
        }
        if let Some(rest) = t.strip_prefix("case ") {
            let _ = rest;
            continue;
        }
        if let Some(rest) = t.strip_prefix("hop ") {
            let mut parts = rest.split('|');
            let header = parts.next().expect("hop header");
            let ele_block = parts.next().expect("ele_num block");
            let proj_block = parts.next().expect("proj_cnt block");
            let header_ints = parse_ints(header);
            let (mi, ri, rj, spin) = (
                header_ints[0] as usize,
                header_ints[1],
                header_ints[2],
                header_ints[3] as u8,
            );
            let ele_num = parse_ints(ele_block.trim_start_matches(" ele_num").trim());
            let proj_cnt = parse_ints(proj_block.trim_start_matches(" proj_cnt").trim());
            hops.push(HopStep {
                mi,
                ri,
                rj,
                spin,
                ele_num,
                proj_cnt,
            });
            continue;
        }
    }
    Fixture {
        n_site: n_site.expect("missing n_site"),
        n_proj: n_proj.expect("missing n_proj"),
        loc_spn,
        initial_ele_num,
        initial_proj_cnt,
        hops,
    }
}

fn check(case: &str) {
    let fx = load_fixture(&fixture_path(case));
    let data = parse_expert_mode_files(&namelist_path(case)).expect("parse namelist");
    assert_eq!(data.modpara.nsite as usize, fx.n_site);

    // (1) init_loc_spn parity.
    let mut loc_spn = vec![0_i64; fx.n_site];
    init_loc_spn(&mut loc_spn, &data);
    assert_eq!(loc_spn, fx.loc_spn, "{case} loc_spn");

    // (2) make_proj_cnt parity on the synthetic ele_num.
    let mut proj_cnt = vec![0_i64; fx.n_proj];
    make_proj_cnt(&mut proj_cnt, &fx.initial_ele_num, &data);
    assert_eq!(proj_cnt, fx.initial_proj_cnt, "{case} initial proj_cnt");

    // (3) Replay the hop sequence and check the snapshot after each hop.
    let mut ele_num = fx.initial_ele_num.clone();
    let mut proj_cnt = fx.initial_proj_cnt.clone();
    for (idx, hop) in fx.hops.iter().enumerate() {
        let rsa_old = hop.ri as usize + (hop.spin as usize) * fx.n_site;
        let rsa_new = hop.rj as usize + (hop.spin as usize) * fx.n_site;
        // Apply ele_num update then incremental proj_cnt update -- matches
        // the upstream C call order in vmcmake.c:164-165.
        ele_num[rsa_old] = 0;
        ele_num[rsa_new] = 1;
        assert_eq!(ele_num, hop.ele_num, "{case} hop {idx} ele_num");
        let proj_cnt_old = proj_cnt.clone();
        update_proj_cnt(
            hop.ri,
            hop.rj,
            hop.spin,
            &mut proj_cnt,
            &proj_cnt_old,
            &ele_num,
            &data,
        );
        assert_eq!(
            proj_cnt, hop.proj_cnt,
            "{case} hop {idx} (mi={}, ri={}, rj={}, spin={}) proj_cnt",
            hop.mi, hop.ri, hop.rj, hop.spin
        );
    }
}

#[test]
fn heisenberg_chain_real() {
    check("heisenberg_chain_real");
}
#[test]
fn heisenberg_chain_cmp() {
    check("heisenberg_chain_cmp");
}
#[test]
fn heisenberg_chain_fsz() {
    check("heisenberg_chain_fsz");
}
#[test]
fn hubbard_chain_real() {
    check("hubbard_chain_real");
}
