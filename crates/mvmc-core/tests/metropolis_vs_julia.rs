//! Phase 4.3.5a parity for Metropolis accept/reject scalar logic.

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;

use mvmc_core::sampling::metropolis::metropolis_decision;
use num_complex::Complex64;
use sfmt19937::Sfmt19937Rng;

const FIXTURE: &str = "../../tests/fixtures/metropolis/decision.txt";

#[derive(Debug)]
struct Case {
    log_proj_delta: f64,
    log_rbm_delta: Complex64,
    log_ip_new: Complex64,
    log_ip_old: Complex64,
    weight: f64,
    draw: f64,
    accepted: bool,
}

fn parse_c64(s: &str) -> Complex64 {
    let mut it = s.split_ascii_whitespace();
    Complex64::new(
        it.next().unwrap().parse::<f64>().unwrap(),
        it.next().unwrap().parse::<f64>().unwrap(),
    )
}

fn load_fixture() -> (u32, Vec<Case>) {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(FIXTURE);
    let file = File::open(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let mut seed = None;
    let mut cases = Vec::new();
    for line in BufReader::new(file).lines() {
        let line = line.unwrap();
        let t = line.trim();
        if t.is_empty() || t.starts_with('#') {
            continue;
        }
        if let Some(rest) = t.strip_prefix("seed ") {
            seed = Some(rest.parse::<u32>().unwrap());
        } else if let Some(rest) = t.strip_prefix("case ") {
            let parts: Vec<&str> = rest.split('|').map(str::trim).collect();
            assert_eq!(parts.len(), 7);
            cases.push(Case {
                log_proj_delta: parts[0].parse::<f64>().unwrap(),
                log_rbm_delta: parse_c64(parts[1]),
                log_ip_new: parse_c64(parts[2]),
                log_ip_old: parse_c64(parts[3]),
                weight: parts[4].parse::<f64>().unwrap(),
                draw: parts[5].parse::<f64>().unwrap(),
                accepted: parts[6] != "0",
            });
        }
    }
    (seed.unwrap(), cases)
}

fn close(a: f64, b: f64) -> bool {
    let d = (a - b).abs();
    d <= 1e-15 || d <= 1e-13 * a.abs().max(b.abs())
}

#[test]
fn metropolis_decisions_match_julia_sfmt_sequence() {
    let (seed, cases) = load_fixture();
    let mut rng = Sfmt19937Rng::new(seed);
    for (idx, c) in cases.iter().enumerate() {
        let got = metropolis_decision(
            c.log_proj_delta,
            c.log_rbm_delta,
            c.log_ip_new,
            c.log_ip_old,
            &mut rng,
        );
        assert!(
            close(got.weight, c.weight),
            "case {idx} weight rust={} julia={}",
            got.weight,
            c.weight
        );
        assert!(
            close(got.draw, c.draw),
            "case {idx} draw rust={} julia={}",
            got.draw,
            c.draw
        );
        assert_eq!(got.accepted, c.accepted, "case {idx} accepted");
    }
}
