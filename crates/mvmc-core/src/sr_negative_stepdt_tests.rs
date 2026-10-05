//! #367: a negative `DSROptStepDt` is normalized by the modpara reader
//! (readdef.c:749-755) and the direct SR update then equals C's
//! `stcOptInit` + `dposv` on fixed operands. Expected values come from
//! `tests/fixtures/negative_stepdt/c_sr.txt` (`scripts/check_negative_stepdt_c_parity.py`).
//! This test reads the fixture only; it never compiles or runs C.
use super::*;
use mvmc_expert_parsers::parsers::modpara::parse_modpara_content;
use mvmc_expert_parsers::GutzwillerTerm;

// Same operands as c_toolbox/negative_stepdt.c, Rust (n+1)x(n+1) layout.
const OO: [[f64; 4]; 4] = [
    [1.0, 0.3, -0.2, 0.1],
    [0.3, 2.09, 0.44, 0.13],
    [-0.2, 0.44, 3.04, 0.18],
    [0.1, 0.13, 0.18, 1.51],
];
const HO: [f64; 4] = [-1.5, 0.7, -0.4, 0.25];

const HEADER: &str = "--------------------\nModel_Parameters 0\n--------------------\nVMC_Cal_Parameters\n--------------------\nCDataFileHead zvo\nCParaFileHead zqp\n--------------------\n";

#[test]
fn negative_step_dt_update_matches_c_stcoptinit_dposv() {
    let text = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/negative_stepdt/c_sr.txt"),
    )
    .unwrap();
    let mut cases = 0;
    let mut lines = text.lines().filter(|l| !l.starts_with('#')).peekable();
    while let Some(line) = lines.next() {
        let f: Vec<&str> = line.split_whitespace().collect();
        assert_eq!(f[0], "case");
        let (name, raw_dt, sta_del) = (f[1], f[3], f[5]);
        let mut srflag = None;
        let mut dt = None;
        let mut header = None;
        let mut solution = None;
        while let Some(next) = lines.peek() {
            if next.starts_with("case ") {
                break;
            }
            let (key, rest) = lines.next().unwrap().split_once(' ').unwrap();
            match key {
                "srflag" => srflag = Some(rest == "1"),
                "dt" => dt = Some(rest.parse::<f64>().unwrap()),
                "header" => header = Some(rest.to_owned()),
                "solution" => {
                    solution = Some(
                        rest.split_whitespace()
                            .map(|v| v.parse::<f64>().unwrap())
                            .collect::<Vec<_>>(),
                    )
                }
                _ => {}
            }
        }
        let (srflag, dt, header, solution) = (
            srflag.unwrap(),
            dt.unwrap(),
            header.unwrap(),
            solution.unwrap(),
        );

        // Reader: the sign becomes SRFlag and the step is non-negative.
        let modpara = parse_modpara_content(&format!(
            "{HEADER}DSROptStepDt {raw_dt}\nDSROptStaDel {sta_del}\nDSROptRedCut 0.0\n"
        ))
        .unwrap();
        assert_eq!(modpara.sr_flag, srflag, "{name}: SRFlag");
        assert_eq!(modpara.dsr_opt_step_dt, dt, "{name}: step");
        assert_eq!(
            modpara.dsr_opt_step_dt.is_sign_negative(),
            dt.is_sign_negative(),
            "{name}: step sign"
        );
        assert_eq!(
            header.contains("sEigenMax"),
            srflag,
            "{name}: SRinfo header spelling"
        );

        // Solver: the update equals C's dposv solution.
        let mut data = ExpertModeData::new();
        data.modpara = modpara;
        for site in 0..3 {
            data.gutzwiller_terms.push(GutzwillerTerm {
                site,
                value: Complex64::new(1.0, 0.0),
                is_complex: false,
            });
        }
        data.optimization_flags = vec![1, 0, 1, 0, 1, 0];
        let mut state = VmcOptimizationState::zeros(3, 1, 3, 3, 1, 1, false, false);
        for (a, row) in OO.iter().enumerate() {
            for (b, value) in row.iter().enumerate() {
                state.sr_opt.sr_opt_oo_real[a * 4 + b] = *value;
            }
        }
        state.sr_opt.sr_opt_ho_real[..4].copy_from_slice(&HO);
        assert_eq!(stochastic_opt_real(&mut data, &mut state), 0, "{name}");
        for (i, term) in data.gutzwiller_terms.iter().enumerate() {
            // Small SPD 3x3 system (condition number about 2) solved by LAPACK in
            // both implementations; the only divergence can be roundoff in the
            // Cholesky solve and in `value - 1.0`, hence 1e-15 absolute (several
            // ulps of 1.0) plus 1e-13 relative.
            crate::numerical_comparison::assert_close(
                term.value.re - 1.0,
                solution[i],
                1e-15,
                1e-13,
                format!("{name}: update[{i}]"),
            );
        }
        cases += 1;
    }
    assert_eq!(cases, 5);
}

#[test]
fn negative_step_dt_sets_sr_flag_and_keeps_magnitude() {
    let a = parse_modpara_content(&format!("{HEADER}DSROptStepDt -0.25\n")).unwrap();
    let b = parse_modpara_content(&format!("{HEADER}DSROptStepDt 0.25\n")).unwrap();
    assert!(a.sr_flag && !b.sr_flag);
    assert_eq!(a.dsr_opt_step_dt, 0.25);
    assert_eq!(b.dsr_opt_step_dt, 0.25);
}
