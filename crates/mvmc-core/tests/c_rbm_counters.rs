//! Actual C readers, MakeRBMCnt and UpdateRBMCnt; no native tools at test time.
#[path = "../../../tests/support/numerical_comparison.rs"]
mod numerical_comparison;
use mvmc_core::sampling::rbm::{make_rbm_cnt, update_rbm_cnt_hopping, RbmConfig};
use mvmc_expert_parsers::{parse_expert_mode_files, parsers::rbm::SECTION_NAMES};
use num_complex::Complex64;
use std::{fs, path::PathBuf};

struct Input(PathBuf);
impl Drop for Input {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).unwrap();
    }
}

fn bits(line: &str) -> Vec<u64> {
    line.split_whitespace()
        .map(|v| u64::from_str_radix(v, 16).unwrap())
        .collect()
}

fn check_counter(values: &[Complex64], expected: Vec<u64>, context: &str) {
    // <=12 sites with additions and incremental subtraction: 64 epsilon.
    numerical_comparison::assert_values_close(
        values.iter().flat_map(|z| [z.re, z.im]),
        expected.into_iter().map(f64::from_bits),
        64.0 * f64::EPSILON,
        64.0 * f64::EPSILON,
        context,
    );
}

fn check_native_counters(update: bool) {
    let dir = Input(std::env::temp_dir().join(format!(
        "mvmc-c-rbm-counters-{}-{update}",
        std::process::id()
    )));
    fs::create_dir_all(&dir.0).unwrap();
    let mut lines = include_str!("../../../tests/fixtures/rbm/c_counters.txt")
        .lines()
        .filter(|line| !line.starts_with('#'));
    let mut tables = 0;
    let mut cases = 0;
    while let Some(header) = lines.next() {
        let header: Vec<_> = header.split_whitespace().collect();
        assert_eq!(header[0], "MODEL");
        let name = header[1];
        let nsite: usize = header[2].parse().unwrap();
        let hidden: usize = header[3].parse().unwrap();
        let count: usize = header[4].parse().unwrap();
        let widths: Vec<usize> = header[5..].iter().map(|v| v.parse().unwrap()).collect();
        fs::write(dir.0.join("modpara.def"), format!(
            "Nsite {nsite}\nNElec 1\nNMPTrans -1\nNneuronCharge {hidden}\nNneuronSpin {hidden}\nNneuronGeneral {hidden}\n"
        )).unwrap();
        let mut namelist = "ModPara modpara.def\n".to_owned();
        for (section, payload) in lines.next().unwrap().split('~').enumerate() {
            if payload == "-" {
                continue;
            }
            fs::write(
                dir.0.join(format!("section{section}.def")),
                payload.replace('|', "\n"),
            )
            .unwrap();
            namelist.push_str(&format!(
                "{} section{section}.def\n",
                SECTION_NAMES[section]
            ));
        }
        fs::write(dir.0.join("namelist.def"), namelist).unwrap();
        let mut data = parse_expert_mode_files(dir.0.join("namelist.def")).unwrap();
        assert!(
            data.input_errors.is_empty(),
            "{name}: {:?}",
            data.input_errors
        );
        assert_eq!(data.rbm_section_sizes().as_slice(), widths, "{name}");
        let values = bits(lines.next().unwrap());
        data.set_rbm_parameters(
            values
                .as_chunks::<2>()
                .0
                .iter()
                .map(|v| Complex64::new(f64::from_bits(v[0]), f64::from_bits(v[1])))
                .collect::<Vec<_>>(),
        );
        let cfg = RbmConfig::from(&data);
        for case in 0..count {
            let movement: Vec<usize> = lines
                .next()
                .unwrap()
                .split_whitespace()
                .map(|v| v.parse().unwrap())
                .collect();
            let old = bits(lines.next().unwrap());
            let new = bits(lines.next().unwrap());
            let occupation: Vec<i64> = (0..2 * nsite)
                .map(|r| ((movement[0] >> r) & 1) as i64)
                .collect();
            if update {
                // Start from the actual C counter to isolate incremental
                // arithmetic from the separate full-counter check.
                let old: Vec<Complex64> = old
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|v| Complex64::new(f64::from_bits(v[0]), f64::from_bits(v[1])))
                    .collect();
                let mut actual = vec![Complex64::new(99.0, 123.0); old.len()];
                update_rbm_cnt_hopping(
                    &mut actual,
                    &old,
                    movement[1] as i64,
                    movement[2] as i64,
                    movement[3] as u8,
                    &cfg,
                );
                check_counter(&actual, new, &format!("{name} hop {case}: {movement:?}"));
            } else {
                check_counter(
                    &make_rbm_cnt(&occupation, &cfg),
                    old,
                    &format!("{name} counter {case}: {movement:?}"),
                );
            }
            cases += 1;
        }
        tables += 1;
    }
    assert_eq!((tables, cases), (66, 4994));
}

#[test]
fn declared_rbm_counters_match_actual_c_operation_order() {
    check_native_counters(false);
}

#[test]
fn rbm_hop_updates_match_actual_c_operation_order() {
    check_native_counters(true);
}
