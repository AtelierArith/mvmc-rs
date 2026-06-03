//! Phase 4.3.5b parity for one normal hopping move scaffold.

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;

use mvmc_core::sampling::candidate::HoppingCandidate;
use mvmc_core::sampling::candidate::UpdateType;
use mvmc_core::sampling::one_move::{
    attempt_hopping_move, run_generated_hopping_mini_loop, run_generated_normal_mini_loop,
    run_hopping_mini_loop, ExchangeMoveStatus, GeneratedHoppingStepInput, GeneratedNormalCandidate,
    GeneratedNormalMoveOutcome, GeneratedNormalStepInput, HoppingMoveStatus, HoppingStepInput,
};
use mvmc_expert_parsers::{ExpertModeData, GutzwillerTerm, JastrowTerm};
use num_complex::Complex64;
use sfmt19937::Sfmt19937Rng;

const FIXTURE: &str = "../../tests/fixtures/one_move/hopping.txt";

#[derive(Debug)]
struct Case {
    label: String,
    log_ip_new: Complex64,
    status: String,
    log_proj_delta: f64,
    weight: f64,
    draw: f64,
    accepted: bool,
    ele_idx: Vec<i64>,
    ele_cfg: Vec<i64>,
    ele_num: Vec<i64>,
    proj_cnt: Vec<i64>,
    proj_new: Vec<i64>,
}

#[derive(Debug)]
struct MiniStep {
    candidate: HoppingCandidate,
    log_ip_new: Complex64,
    status: String,
    log_proj_delta: f64,
    weight: f64,
    draw: f64,
    accepted: bool,
    log_ip_old_after: Complex64,
    ele_idx: Vec<i64>,
    ele_cfg: Vec<i64>,
    ele_num: Vec<i64>,
    proj_cnt: Vec<i64>,
    proj_new: Vec<i64>,
}

#[derive(Debug)]
struct MiniFixture {
    seed: u32,
    accepted_count: usize,
    log_ip_final: Complex64,
    steps: Vec<MiniStep>,
}

#[derive(Debug)]
struct DispatchStep {
    update_type: UpdateType,
    candidate: String,
    log_ip_new: Complex64,
    status: String,
    log_proj_delta: f64,
    weight: f64,
    draw: f64,
    accepted: bool,
    log_ip_old_after: Complex64,
    ele_idx: Vec<i64>,
    ele_cfg: Vec<i64>,
    ele_num: Vec<i64>,
    proj_cnt: Vec<i64>,
    proj_new: Vec<i64>,
}

#[derive(Debug)]
struct DispatchFixture {
    seed: u32,
    accepted_count: usize,
    log_ip_final: Complex64,
    steps: Vec<DispatchStep>,
}

fn parse_i64s(s: &str) -> Vec<i64> {
    s.split_ascii_whitespace()
        .map(|x| x.parse().unwrap())
        .collect()
}

fn parse_c64(s: &str) -> Complex64 {
    let mut it = s.split_ascii_whitespace();
    Complex64::new(
        it.next().unwrap().parse::<f64>().unwrap(),
        it.next().unwrap().parse::<f64>().unwrap(),
    )
}

fn load_fixture() -> (u32, HoppingCandidate, Vec<Case>) {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(FIXTURE);
    let file = File::open(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let lines: Vec<String> = BufReader::new(file)
        .lines()
        .map(|l| l.unwrap())
        .filter(|l| {
            let t = l.trim();
            !t.is_empty() && !t.starts_with('#')
        })
        .collect();
    let mut seed = 0;
    let mut cand = HoppingCandidate {
        mi: 0,
        ri: 0,
        rj: 0,
        spin: 0,
        reject: false,
    };
    let mut cand_set = false;
    let mut cases = Vec::new();
    let mut current: Option<Case> = None;
    for line in lines {
        let t = line.trim();
        if t.starts_with("mini_loop_seed ") {
            break;
        }
        if let Some(rest) = t.strip_prefix("seed ") {
            seed = rest.parse().unwrap();
        } else if let Some(rest) = t.strip_prefix("candidate ") {
            if !cand_set {
                let xs = parse_i64s(rest);
                cand = HoppingCandidate {
                    mi: xs[0] as usize,
                    ri: xs[1] as usize,
                    rj: xs[2] as usize,
                    spin: xs[3] as u8,
                    reject: xs[4] != 0,
                };
                cand_set = true;
            }
        } else if let Some(rest) = t.strip_prefix("case ") {
            if let Some(c) = current.take() {
                cases.push(c);
            }
            current = Some(Case {
                label: rest.to_string(),
                log_ip_new: Complex64::new(0.0, 0.0),
                status: String::new(),
                log_proj_delta: 0.0,
                weight: 0.0,
                draw: 0.0,
                accepted: false,
                ele_idx: Vec::new(),
                ele_cfg: Vec::new(),
                ele_num: Vec::new(),
                proj_cnt: Vec::new(),
                proj_new: Vec::new(),
            });
        } else if let Some(c) = current.as_mut() {
            if let Some(rest) = t.strip_prefix("log_ip_new ") {
                c.log_ip_new = parse_c64(rest);
            } else if let Some(rest) = t.strip_prefix("status ") {
                c.status = rest.to_string();
            } else if let Some(rest) = t.strip_prefix("log_proj_delta ") {
                c.log_proj_delta = rest.parse().unwrap();
            } else if let Some(rest) = t.strip_prefix("weight ") {
                c.weight = rest.parse().unwrap();
            } else if let Some(rest) = t.strip_prefix("draw ") {
                c.draw = rest.parse().unwrap();
            } else if let Some(rest) = t.strip_prefix("accepted ") {
                c.accepted = rest != "0";
            } else if let Some(rest) = t.strip_prefix("ele_idx ") {
                c.ele_idx = parse_i64s(rest);
            } else if let Some(rest) = t.strip_prefix("ele_cfg ") {
                c.ele_cfg = parse_i64s(rest);
            } else if let Some(rest) = t.strip_prefix("ele_num ") {
                c.ele_num = parse_i64s(rest);
            } else if let Some(rest) = t.strip_prefix("proj_cnt ") {
                c.proj_cnt = parse_i64s(rest);
            } else if let Some(rest) = t.strip_prefix("proj_new ") {
                c.proj_new = parse_i64s(rest);
            }
        }
    }
    if let Some(c) = current.take() {
        cases.push(c);
    }
    (seed, cand, cases)
}

fn load_mini_fixture() -> MiniFixture {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(FIXTURE);
    let file = File::open(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let lines: Vec<String> = BufReader::new(file)
        .lines()
        .map(|l| l.unwrap())
        .filter(|l| {
            let t = l.trim();
            !t.is_empty() && !t.starts_with('#')
        })
        .collect();
    let mut seed = 0;
    let mut accepted_count = 0;
    let mut log_ip_final = Complex64::new(0.0, 0.0);
    let mut steps = Vec::new();
    let mut cur: Option<MiniStep> = None;
    let mut in_mini = false;
    for line in lines {
        let t = line.trim();
        if let Some(rest) = t.strip_prefix("mini_loop_seed ") {
            in_mini = true;
            seed = rest.parse().unwrap();
            continue;
        }
        if t.starts_with("generated_loop_seed ") {
            break;
        }
        if !in_mini {
            continue;
        }
        if let Some(rest) = t.strip_prefix("mini_accepted_count ") {
            accepted_count = rest.parse().unwrap();
        } else if let Some(rest) = t.strip_prefix("mini_log_ip_final ") {
            log_ip_final = parse_c64(rest);
        } else if t.starts_with("mini_step ") {
            if let Some(s) = cur.take() {
                steps.push(s);
            }
            cur = Some(MiniStep {
                candidate: HoppingCandidate {
                    mi: 0,
                    ri: 0,
                    rj: 0,
                    spin: 0,
                    reject: false,
                },
                log_ip_new: Complex64::new(0.0, 0.0),
                status: String::new(),
                log_proj_delta: 0.0,
                weight: 0.0,
                draw: 0.0,
                accepted: false,
                log_ip_old_after: Complex64::new(0.0, 0.0),
                ele_idx: Vec::new(),
                ele_cfg: Vec::new(),
                ele_num: Vec::new(),
                proj_cnt: Vec::new(),
                proj_new: Vec::new(),
            });
        } else if let Some(s) = cur.as_mut() {
            if let Some(rest) = t.strip_prefix("candidate ") {
                let xs = parse_i64s(rest);
                s.candidate = HoppingCandidate {
                    mi: xs[0] as usize,
                    ri: xs[1] as usize,
                    rj: xs[2] as usize,
                    spin: xs[3] as u8,
                    reject: xs[4] != 0,
                };
            } else if let Some(rest) = t.strip_prefix("log_ip_new ") {
                s.log_ip_new = parse_c64(rest);
            } else if let Some(rest) = t.strip_prefix("status ") {
                s.status = rest.to_string();
            } else if let Some(rest) = t.strip_prefix("log_proj_delta ") {
                s.log_proj_delta = rest.parse().unwrap();
            } else if let Some(rest) = t.strip_prefix("weight ") {
                s.weight = rest.parse().unwrap();
            } else if let Some(rest) = t.strip_prefix("draw ") {
                s.draw = rest.parse().unwrap();
            } else if let Some(rest) = t.strip_prefix("accepted ") {
                s.accepted = rest != "0";
            } else if let Some(rest) = t.strip_prefix("log_ip_old_after ") {
                s.log_ip_old_after = parse_c64(rest);
            } else if let Some(rest) = t.strip_prefix("ele_idx ") {
                s.ele_idx = parse_i64s(rest);
            } else if let Some(rest) = t.strip_prefix("ele_cfg ") {
                s.ele_cfg = parse_i64s(rest);
            } else if let Some(rest) = t.strip_prefix("ele_num ") {
                s.ele_num = parse_i64s(rest);
            } else if let Some(rest) = t.strip_prefix("proj_cnt ") {
                s.proj_cnt = parse_i64s(rest);
            } else if let Some(rest) = t.strip_prefix("proj_new ") {
                s.proj_new = parse_i64s(rest);
            }
        }
    }
    if let Some(s) = cur.take() {
        steps.push(s);
    }
    MiniFixture {
        seed,
        accepted_count,
        log_ip_final,
        steps,
    }
}

fn load_generated_fixture() -> MiniFixture {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(FIXTURE);
    let file = File::open(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let lines: Vec<String> = BufReader::new(file)
        .lines()
        .map(|l| l.unwrap())
        .filter(|l| {
            let t = l.trim();
            !t.is_empty() && !t.starts_with('#')
        })
        .collect();
    let mut seed = 0;
    let mut accepted_count = 0;
    let mut log_ip_final = Complex64::new(0.0, 0.0);
    let mut steps = Vec::new();
    let mut cur: Option<MiniStep> = None;
    let mut in_generated = false;
    for line in lines {
        let t = line.trim();
        if let Some(rest) = t.strip_prefix("generated_loop_seed ") {
            in_generated = true;
            seed = rest.parse().unwrap();
            continue;
        }
        if t.starts_with("dispatch_loop_seed ") {
            break;
        }
        if !in_generated {
            continue;
        }
        if let Some(rest) = t.strip_prefix("generated_accepted_count ") {
            accepted_count = rest.parse().unwrap();
        } else if let Some(rest) = t.strip_prefix("generated_log_ip_final ") {
            log_ip_final = parse_c64(rest);
        } else if t.starts_with("generated_step ") {
            if let Some(s) = cur.take() {
                steps.push(s);
            }
            cur = Some(MiniStep {
                candidate: HoppingCandidate {
                    mi: 0,
                    ri: 0,
                    rj: 0,
                    spin: 0,
                    reject: false,
                },
                log_ip_new: Complex64::new(0.0, 0.0),
                status: String::new(),
                log_proj_delta: 0.0,
                weight: 0.0,
                draw: 0.0,
                accepted: false,
                log_ip_old_after: Complex64::new(0.0, 0.0),
                ele_idx: Vec::new(),
                ele_cfg: Vec::new(),
                ele_num: Vec::new(),
                proj_cnt: Vec::new(),
                proj_new: Vec::new(),
            });
        } else if let Some(s) = cur.as_mut() {
            if let Some(rest) = t.strip_prefix("candidate ") {
                let xs = parse_i64s(rest);
                s.candidate = HoppingCandidate {
                    mi: xs[0] as usize,
                    ri: xs[1] as usize,
                    rj: xs[2] as usize,
                    spin: xs[3] as u8,
                    reject: xs[4] != 0,
                };
            } else if let Some(rest) = t.strip_prefix("log_ip_new ") {
                s.log_ip_new = parse_c64(rest);
            } else if let Some(rest) = t.strip_prefix("status ") {
                s.status = rest.to_string();
            } else if let Some(rest) = t.strip_prefix("log_proj_delta ") {
                s.log_proj_delta = rest.parse().unwrap();
            } else if let Some(rest) = t.strip_prefix("weight ") {
                s.weight = rest.parse().unwrap();
            } else if let Some(rest) = t.strip_prefix("draw ") {
                s.draw = rest.parse().unwrap();
            } else if let Some(rest) = t.strip_prefix("accepted ") {
                s.accepted = rest != "0";
            } else if let Some(rest) = t.strip_prefix("log_ip_old_after ") {
                s.log_ip_old_after = parse_c64(rest);
            } else if let Some(rest) = t.strip_prefix("ele_idx ") {
                s.ele_idx = parse_i64s(rest);
            } else if let Some(rest) = t.strip_prefix("ele_cfg ") {
                s.ele_cfg = parse_i64s(rest);
            } else if let Some(rest) = t.strip_prefix("ele_num ") {
                s.ele_num = parse_i64s(rest);
            } else if let Some(rest) = t.strip_prefix("proj_cnt ") {
                s.proj_cnt = parse_i64s(rest);
            } else if let Some(rest) = t.strip_prefix("proj_new ") {
                s.proj_new = parse_i64s(rest);
            }
        }
    }
    if let Some(s) = cur.take() {
        steps.push(s);
    }
    MiniFixture {
        seed,
        accepted_count,
        log_ip_final,
        steps,
    }
}

fn parse_update_type(s: &str) -> UpdateType {
    match s {
        "HOPPING" => UpdateType::Hopping,
        "EXCHANGE" => UpdateType::Exchange,
        "LOCALSPINFLIP" => UpdateType::LocalSpinFlip,
        "NONE" => UpdateType::None,
        other => panic!("unknown update type {other}"),
    }
}

fn load_dispatch_fixture() -> DispatchFixture {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(FIXTURE);
    let file = File::open(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let lines: Vec<String> = BufReader::new(file)
        .lines()
        .map(|l| l.unwrap())
        .filter(|l| {
            let t = l.trim();
            !t.is_empty() && !t.starts_with('#')
        })
        .collect();
    let mut seed = 0;
    let mut accepted_count = 0;
    let mut log_ip_final = Complex64::new(0.0, 0.0);
    let mut steps = Vec::new();
    let mut cur: Option<DispatchStep> = None;
    let mut in_dispatch = false;
    for line in lines {
        let t = line.trim();
        if let Some(rest) = t.strip_prefix("dispatch_loop_seed ") {
            in_dispatch = true;
            seed = rest.parse().unwrap();
            continue;
        }
        if !in_dispatch {
            continue;
        }
        if let Some(rest) = t.strip_prefix("dispatch_accepted_count ") {
            accepted_count = rest.parse().unwrap();
        } else if let Some(rest) = t.strip_prefix("dispatch_log_ip_final ") {
            log_ip_final = parse_c64(rest);
        } else if t.starts_with("dispatch_step ") {
            if let Some(s) = cur.take() {
                steps.push(s);
            }
            cur = Some(DispatchStep {
                update_type: UpdateType::None,
                candidate: String::new(),
                log_ip_new: Complex64::new(0.0, 0.0),
                status: String::new(),
                log_proj_delta: 0.0,
                weight: 0.0,
                draw: 0.0,
                accepted: false,
                log_ip_old_after: Complex64::new(0.0, 0.0),
                ele_idx: Vec::new(),
                ele_cfg: Vec::new(),
                ele_num: Vec::new(),
                proj_cnt: Vec::new(),
                proj_new: Vec::new(),
            });
        } else if let Some(s) = cur.as_mut() {
            if let Some(rest) = t.strip_prefix("update_type ") {
                s.update_type = parse_update_type(rest);
            } else if let Some(rest) = t.strip_prefix("candidate ") {
                s.candidate = rest.to_string();
            } else if let Some(rest) = t.strip_prefix("log_ip_new ") {
                s.log_ip_new = parse_c64(rest);
            } else if let Some(rest) = t.strip_prefix("status ") {
                s.status = rest.to_string();
            } else if let Some(rest) = t.strip_prefix("log_proj_delta ") {
                s.log_proj_delta = rest.parse().unwrap();
            } else if let Some(rest) = t.strip_prefix("weight ") {
                s.weight = rest.parse().unwrap();
            } else if let Some(rest) = t.strip_prefix("draw ") {
                s.draw = rest.parse().unwrap();
            } else if let Some(rest) = t.strip_prefix("accepted ") {
                s.accepted = rest != "0";
            } else if let Some(rest) = t.strip_prefix("log_ip_old_after ") {
                s.log_ip_old_after = parse_c64(rest);
            } else if let Some(rest) = t.strip_prefix("ele_idx ") {
                s.ele_idx = parse_i64s(rest);
            } else if let Some(rest) = t.strip_prefix("ele_cfg ") {
                s.ele_cfg = parse_i64s(rest);
            } else if let Some(rest) = t.strip_prefix("ele_num ") {
                s.ele_num = parse_i64s(rest);
            } else if let Some(rest) = t.strip_prefix("proj_cnt ") {
                s.proj_cnt = parse_i64s(rest);
            } else if let Some(rest) = t.strip_prefix("proj_new ") {
                s.proj_new = parse_i64s(rest);
            }
        }
    }
    if let Some(s) = cur.take() {
        steps.push(s);
    }
    DispatchFixture {
        seed,
        accepted_count,
        log_ip_final,
        steps,
    }
}

fn data() -> ExpertModeData {
    let mut data = ExpertModeData::new();
    data.modpara.nsite = 4;
    data.modpara.nelec = 2;
    data.n_gutzwiller_idx = 1;
    data.gutzwiller_idx = vec![0, 0, 0, 0];
    data.gutzwiller_terms.push(GutzwillerTerm {
        site: 0,
        value: Complex64::new(0.2, 0.0),
        is_complex: false,
    });
    data.n_jastrow_idx = 1;
    data.jastrow_idx = vec![
        vec![-1, 0, 0, 0],
        vec![0, -1, 0, 0],
        vec![0, 0, -1, 0],
        vec![0, 0, 0, -1],
    ];
    data.jastrow_terms.push(JastrowTerm {
        site1: 0,
        site2: 1,
        value: Complex64::new(-0.1, 0.0),
        is_complex: false,
    });
    data
}

fn close(a: f64, b: f64) -> bool {
    let d = (a - b).abs();
    d <= 1e-15 || d <= 1e-13 * a.abs().max(b.abs())
}

#[test]
fn one_hopping_move_matches_julia_accept_and_reject_cases() {
    let (seed, candidate, cases) = load_fixture();
    assert_eq!(cases.len(), 2);
    for case in cases {
        let mut ele_idx = vec![0, 1, 2, 3];
        let mut ele_cfg = vec![0, 1, -1, -1, -1, -1, 0, 1];
        let mut ele_num = vec![1, 1, 0, 0, 0, 0, 1, 1];
        let mut proj_cnt = vec![0, 0];
        let mut proj_new = vec![0, 0];
        let mut rng = Sfmt19937Rng::new(seed);
        let outcome = attempt_hopping_move(
            candidate,
            &mut ele_idx,
            &mut ele_cfg,
            &mut ele_num,
            &mut proj_cnt,
            &mut proj_new,
            &data(),
            4,
            2,
            Complex64::new(0.0, 0.0),
            case.log_ip_new,
            Complex64::new(0.0, 0.0),
            &mut rng,
        );
        let want_status = if case.status == "accepted" {
            HoppingMoveStatus::Accepted
        } else {
            HoppingMoveStatus::MetropolisRejected
        };
        assert_eq!(outcome.status, want_status, "{} status", case.label);
        assert!(close(outcome.log_proj_delta, case.log_proj_delta));
        let decision = outcome.decision.expect("metropolis decision");
        assert!(close(decision.weight, case.weight), "{} weight", case.label);
        assert!(close(decision.draw, case.draw), "{} draw", case.label);
        assert_eq!(decision.accepted, case.accepted, "{} accepted", case.label);
        assert_eq!(ele_idx, case.ele_idx, "{} ele_idx", case.label);
        assert_eq!(ele_cfg, case.ele_cfg, "{} ele_cfg", case.label);
        assert_eq!(ele_num, case.ele_num, "{} ele_num", case.label);
        assert_eq!(proj_cnt, case.proj_cnt, "{} proj_cnt", case.label);
        assert_eq!(proj_new, case.proj_new, "{} proj_new", case.label);
    }
}

#[test]
fn hopping_mini_loop_matches_julia_snapshots() {
    let mini = load_mini_fixture();
    let steps: Vec<HoppingStepInput> = mini
        .steps
        .iter()
        .map(|s| HoppingStepInput {
            candidate: s.candidate,
            log_ip_new: s.log_ip_new,
            log_rbm_delta: Complex64::new(0.0, 0.0),
        })
        .collect();
    let mut ele_idx = vec![0, 1, 2, 3];
    let mut ele_cfg = vec![0, 1, -1, -1, -1, -1, 0, 1];
    let mut ele_num = vec![1, 1, 0, 0, 0, 0, 1, 1];
    let mut proj_cnt = vec![0, 0];
    let mut proj_new = vec![0, 0];
    let mut rng = Sfmt19937Rng::new(mini.seed);
    let outcome = run_hopping_mini_loop(
        &steps,
        &mut ele_idx,
        &mut ele_cfg,
        &mut ele_num,
        &mut proj_cnt,
        &mut proj_new,
        &data(),
        4,
        2,
        Complex64::new(0.0, 0.0),
        &mut rng,
    );

    assert_eq!(outcome.accepted_count, mini.accepted_count);
    assert_eq!(outcome.log_ip_final, mini.log_ip_final);
    assert_eq!(outcome.steps.len(), mini.steps.len());

    for (idx, (got, want)) in outcome.steps.iter().zip(mini.steps.iter()).enumerate() {
        let want_status = match want.status.as_str() {
            "accepted" => HoppingMoveStatus::Accepted,
            "metropolis_rejected" => HoppingMoveStatus::MetropolisRejected,
            "candidate_rejected" => HoppingMoveStatus::CandidateRejected,
            other => panic!("unknown status {other}"),
        };
        assert_eq!(got.status, want_status, "mini step {idx} status");
        assert!(
            close(got.log_proj_delta, want.log_proj_delta),
            "mini step {idx} log_proj_delta"
        );
        match got.decision {
            Some(decision) => {
                assert!(
                    close(decision.weight, want.weight),
                    "mini step {idx} weight"
                );
                assert!(close(decision.draw, want.draw), "mini step {idx} draw");
                assert_eq!(decision.accepted, want.accepted, "mini step {idx} accepted");
            }
            None => {
                assert_eq!(want.status, "candidate_rejected");
                assert_eq!(want.weight, -1.0);
                assert_eq!(want.draw, -1.0);
            }
        }
    }

    let last = mini.steps.last().expect("mini has steps");
    assert_eq!(ele_idx, last.ele_idx);
    assert_eq!(ele_cfg, last.ele_cfg);
    assert_eq!(ele_num, last.ele_num);
    assert_eq!(proj_cnt, last.proj_cnt);
    assert_eq!(proj_new, last.proj_new);
}

#[test]
fn generated_hopping_mini_loop_matches_julia_snapshots() {
    let mini = load_generated_fixture();
    let steps: Vec<GeneratedHoppingStepInput> = mini
        .steps
        .iter()
        .map(|s| GeneratedHoppingStepInput {
            log_ip_new: s.log_ip_new,
            log_rbm_delta: Complex64::new(0.0, 0.0),
        })
        .collect();
    let mut ele_idx = vec![0, 1, 2, 3];
    let mut ele_cfg = vec![0, 1, -1, -1, -1, -1, 0, 1];
    let mut ele_num = vec![1, 1, 0, 0, 0, 0, 1, 1];
    let mut proj_cnt = vec![0, 0];
    let mut proj_new = vec![0, 0];
    let loc_spn = vec![0, 0, 0, 0];
    let mut rng = Sfmt19937Rng::new(mini.seed);
    let outcome = run_generated_hopping_mini_loop(
        &steps,
        &mut ele_idx,
        &mut ele_cfg,
        &mut ele_num,
        &mut proj_cnt,
        &mut proj_new,
        &loc_spn,
        &data(),
        4,
        2,
        Complex64::new(0.0, 0.0),
        &mut rng,
    );

    assert_eq!(outcome.accepted_count, mini.accepted_count);
    assert_eq!(outcome.log_ip_final, mini.log_ip_final);
    assert_eq!(outcome.steps.len(), mini.steps.len());

    for (idx, (got, want)) in outcome.steps.iter().zip(mini.steps.iter()).enumerate() {
        assert_eq!(
            got.candidate, want.candidate,
            "generated step {idx} candidate"
        );
        let want_status = match want.status.as_str() {
            "accepted" => HoppingMoveStatus::Accepted,
            "metropolis_rejected" => HoppingMoveStatus::MetropolisRejected,
            "candidate_rejected" => HoppingMoveStatus::CandidateRejected,
            other => panic!("unknown status {other}"),
        };
        assert_eq!(
            got.outcome.status, want_status,
            "generated step {idx} status"
        );
        assert!(
            close(got.outcome.log_proj_delta, want.log_proj_delta),
            "generated step {idx} log_proj_delta"
        );
        match got.outcome.decision {
            Some(decision) => {
                assert!(
                    close(decision.weight, want.weight),
                    "generated step {idx} weight"
                );
                assert!(close(decision.draw, want.draw), "generated step {idx} draw");
                assert_eq!(
                    decision.accepted, want.accepted,
                    "generated step {idx} accepted"
                );
            }
            None => {
                assert_eq!(want.status, "candidate_rejected");
            }
        }
    }

    let last = mini.steps.last().expect("generated mini has steps");
    assert_eq!(ele_idx, last.ele_idx);
    assert_eq!(ele_cfg, last.ele_cfg);
    assert_eq!(ele_num, last.ele_num);
    assert_eq!(proj_cnt, last.proj_cnt);
    assert_eq!(proj_new, last.proj_new);
}

fn candidate_string(candidate: &GeneratedNormalCandidate) -> String {
    match candidate {
        GeneratedNormalCandidate::Hopping(c) => format!(
            "H {} {} {} {} {}",
            c.mi,
            c.ri,
            c.rj,
            c.spin,
            if c.reject { 1 } else { 0 }
        ),
        GeneratedNormalCandidate::Exchange(c) => format!(
            "E {} {} {} {} {} {} {}",
            c.mi,
            c.ri,
            c.mj,
            c.rj,
            c.spin,
            c.spin_other,
            if c.reject { 1 } else { 0 }
        ),
        GeneratedNormalCandidate::Unsupported => "U".to_string(),
    }
}

#[test]
fn generated_normal_dispatch_loop_matches_julia_snapshots() {
    let mini = load_dispatch_fixture();
    let steps: Vec<GeneratedNormalStepInput> = mini
        .steps
        .iter()
        .map(|s| GeneratedNormalStepInput {
            log_ip_new: s.log_ip_new,
            log_rbm_delta: Complex64::new(0.0, 0.0),
        })
        .collect();
    let mut ele_idx = vec![0, 1, 2, 3];
    let mut ele_cfg = vec![0, 1, -1, -1, -1, -1, 0, 1];
    let mut ele_num = vec![1, 1, 0, 0, 0, 0, 1, 1];
    let mut proj_cnt = vec![0, 0];
    let mut proj_new = vec![0, 0];
    let loc_spn = vec![0, 0, 0, 0];
    let mut rng = Sfmt19937Rng::new(mini.seed);
    let outcome = run_generated_normal_mini_loop(
        &steps,
        &mut ele_idx,
        &mut ele_cfg,
        &mut ele_num,
        &mut proj_cnt,
        &mut proj_new,
        &loc_spn,
        &data(),
        4,
        2,
        1,
        0,
        0,
        Complex64::new(0.0, 0.0),
        &mut rng,
    );

    assert_eq!(outcome.accepted_count, mini.accepted_count);
    assert_eq!(outcome.log_ip_final, mini.log_ip_final);
    assert_eq!(outcome.steps.len(), mini.steps.len());

    for (idx, (got, want)) in outcome.steps.iter().zip(mini.steps.iter()).enumerate() {
        assert_eq!(
            got.update_type, want.update_type,
            "dispatch step {idx} update_type"
        );
        assert_eq!(
            candidate_string(&got.candidate),
            want.candidate,
            "dispatch step {idx} candidate"
        );

        let (status, log_proj_delta, decision) = match &got.outcome {
            GeneratedNormalMoveOutcome::Hopping(out) => (
                match out.status {
                    HoppingMoveStatus::Accepted => "accepted",
                    HoppingMoveStatus::MetropolisRejected => "metropolis_rejected",
                    HoppingMoveStatus::CandidateRejected => "candidate_rejected",
                },
                out.log_proj_delta,
                out.decision.as_ref(),
            ),
            GeneratedNormalMoveOutcome::Exchange(out) => (
                match out.status {
                    ExchangeMoveStatus::Accepted => "accepted",
                    ExchangeMoveStatus::MetropolisRejected => "metropolis_rejected",
                    ExchangeMoveStatus::CandidateRejected
                    | ExchangeMoveStatus::InvalidCandidate => "candidate_rejected",
                },
                out.log_proj_delta,
                out.decision.as_ref(),
            ),
            GeneratedNormalMoveOutcome::Unsupported => ("unsupported", 0.0, None),
        };
        assert_eq!(status, want.status, "dispatch step {idx} status");
        assert!(
            close(log_proj_delta, want.log_proj_delta),
            "dispatch step {idx} log_proj_delta"
        );
        match decision {
            Some(decision) => {
                assert!(
                    close(decision.weight, want.weight),
                    "dispatch step {idx} weight"
                );
                assert!(close(decision.draw, want.draw), "dispatch step {idx} draw");
                assert_eq!(
                    decision.accepted, want.accepted,
                    "dispatch step {idx} accepted bool"
                );
            }
            None => {
                assert_eq!(want.weight, -1.0);
                assert_eq!(want.draw, -1.0);
            }
        }
    }

    let last = mini.steps.last().expect("dispatch mini has steps");
    assert_eq!(ele_idx, last.ele_idx);
    assert_eq!(ele_cfg, last.ele_cfg);
    assert_eq!(ele_num, last.ele_num);
    assert_eq!(proj_cnt, last.proj_cnt);
    assert_eq!(proj_new, last.proj_new);
}
