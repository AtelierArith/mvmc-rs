//! Complete C RBM initialization/loading arrays and next 624 native SFMT words.
use mvmc_expert_parsers::utils::parameter_init::init_parameter;
use mvmc_expert_parsers::{parse_expert_mode_files, parsers::rbm::SECTION_NAMES};
use sfmt19937::Sfmt19937Rng;
use std::{fs, path::PathBuf};

struct Input(PathBuf);
impl Drop for Input {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).unwrap();
    }
}

#[test]
fn all_declared_rbm_slots_and_following_slater_values_match_native_c_and_rng() {
    let rows: Vec<_> = include_str!("../../../tests/fixtures/rbm/c_parameters.txt")
        .lines()
        .filter(|line| !line.starts_with('#'))
        .collect();
    assert_eq!(rows.len(), 84 * 6);
    let dir =
        Input(std::env::temp_dir().join(format!("mvmc-c-rbm-parameters-{}", std::process::id())));
    fs::create_dir_all(&dir.0).unwrap();
    let maps: String = (0..3)
        .flat_map(|i| (0..3).map(move |j| format!("{i} {j} {}\n", (i * 3 + j) % 4)))
        .collect();
    fs::write(
        dir.0.join("orbital.def"),
        format!("===\nWidth 4\nComplexType 0\n===\n===\n{maps}-7 1\n-7 2\n-7 -1\n-7 3\n"),
    )
    .unwrap();
    fs::write(
        dir.0.join("gutz.def"),
        "===\nWidth 2\nComplexType 0\n===\n===\n0 0\n1 0\n2 0\n-7 3\n-7 -2\n",
    )
    .unwrap();
    for record in rows.chunks_exact(6) {
        let header: Vec<_> = record[0].split_whitespace().collect();
        let neurons: i64 = header[3].parse().unwrap();
        let seed: u32 = header[4].parse().unwrap();
        let mode = header[5];
        let widths: Vec<usize> = header[6..].iter().map(|v| v.parse().unwrap()).collect();
        fs::write(dir.0.join("modpara.def"),format!("Nsite 3\nNElec 1\nNMPTrans -1\nNneuron {}\nNneuronCharge 2\nNneuronSpin 2\nNneuronGeneral 2\n",neurons-6)).unwrap();
        // Reverse namelist order verifies canonical section offsets.
        let mut namelist = "Orbital orbital.def\nGutzwiller gutz.def\n".to_owned();
        let definitions: Vec<_> = record[1].split('~').collect();
        assert_eq!(definitions.len(), 9);
        for (section, payload) in definitions.iter().enumerate().rev() {
            if *payload == "-" {
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
        namelist.push_str("ModPara modpara.def\n");
        fs::write(dir.0.join("namelist.def"), namelist).unwrap();
        let mut data = parse_expert_mode_files(dir.0.join("namelist.def")).unwrap();
        assert!(
            data.input_errors.is_empty(),
            "{}: {:?}",
            header[0],
            data.input_errors
        );
        assert_eq!(data.rbm_section_sizes().as_slice(), widths, "{}", header[0]);
        let flags: Vec<i64> = record[3]
            .split_whitespace()
            .map(|v| v.parse().unwrap())
            .collect();
        assert_eq!(data.optimization_flags, flags, "{}", header[0]);
        let mut rng = Sfmt19937Rng::new(seed);
        init_parameter(&mut data, &mut rng);
        if mode != "init" {
            fs::write(dir.0.join("initial.def"), record[2].replace('|', "\n")).unwrap();
            assert!(
                mvmc_core::read_initial_def(&mut data, dir.0.join("initial.def")).unwrap(),
                "{}",
                header[0]
            );
        }
        let expected: Vec<u64> = record[4]
            .split_whitespace()
            .map(|v| u64::from_str_radix(v, 16).unwrap())
            .collect();
        let actual: Vec<u64> = data
            .projection_parameters()
            .into_iter()
            .chain(data.rbm_parameters())
            .chain(data.slater_params.iter().copied())
            .flat_map(|v| [v.re.to_bits(), v.im.to_bits()])
            .collect();
        assert_eq!(actual, expected, "{} complete parameter array", header[0]);
        let expected: Vec<u32> = record[5]
            .split_whitespace()
            .map(|v| v.parse().unwrap())
            .collect();
        assert_eq!(
            (0..624).map(|_| rng.gen_rand32()).collect::<Vec<_>>(),
            expected,
            "{} RNG",
            header[0]
        );
    }
}
