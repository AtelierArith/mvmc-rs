//! C preserves declared Slater slots even when no spatial mapping uses them.
use mvmc_expert_parsers::utils::parameter_init::{init_parameter, sync_modified_parameter};
use mvmc_expert_parsers::{ExpertModeData, OrbitalTerm};
use sfmt19937::Sfmt19937Rng;

fn records() -> Vec<&'static str> {
    include_str!("../../../tests/fixtures/orbital_general/c_initialization.txt")
        .lines()
        .filter(|line| !line.starts_with('#'))
        .collect()
}

fn data(header: &str) -> (ExpertModeData, Sfmt19937Rng) {
    let fields: Vec<u32> = header
        .split_whitespace()
        .map(|s| s.parse().unwrap())
        .collect();
    let mut data = ExpertModeData::new();
    data.modpara.nsite = 2;
    data.modpara.nmp_trans = 1;
    data.modpara.n_orbital_idx = 13;
    data.n_orbital_anti_parallel = 7;
    data.i_flg_orbital_anti_parallel = 1;
    data.i_flg_orbital_parallel = 1;
    data.i_flg_orbital_general = 1;
    data.orbital_terms = [
        (0, 0, 0),
        (0, 1, 1),
        (1, 0, 1),
        (1, 1, 0),
        (0, 1, 7),
        (0, 1, 8),
    ]
    .into_iter()
    .map(|(site1, site2, idx)| OrbitalTerm {
        site1,
        site2,
        idx,
        sign: 1,
        is_complex: fields[1] != 0,
    })
    .collect();
    data.optimization_flags = (0..13)
        .flat_map(|i| {
            let active = fields[2] == 0 || i < 9;
            [i64::from(active), i64::from(active && fields[1] != 0)]
        })
        .collect();
    (data, Sfmt19937Rng::new(fields[0]))
}

fn check_parameter_bits(data: &ExpertModeData, line: &str, label: &str) {
    let bits: Vec<u64> = line
        .split_whitespace()
        .map(|s| u64::from_str_radix(s, 16).unwrap())
        .collect();
    assert_eq!(bits.len(), 26);
    assert_eq!(data.slater_params.len(), 13);
    for (index, value) in data.slater_params.iter().enumerate() {
        let i = 2 * index;
        assert_eq!(
            [value.re.to_bits(), value.im.to_bits()],
            [bits[i], bits[i + 1]],
            "{label}, declared index {index}"
        );
    }
}

#[test]
fn c_initialization_consumes_every_declared_active_slot_before_sampling() {
    let rows = records();
    assert_eq!(rows.len(), 8 * 4);
    for record in rows.as_chunks::<4>().0.iter() {
        let (mut data, mut rng) = data(record[0]);
        init_parameter(&mut data, &mut rng);
        check_parameter_bits(&data, record[1], record[0]);
        let expected: Vec<u32> = record[3]
            .split_whitespace()
            .map(|s| s.parse().unwrap())
            .collect();
        assert_eq!(expected.len(), 624);
        for (i, expected) in expected.into_iter().enumerate() {
            assert_eq!(
                rng.gen_rand32(),
                expected,
                "{}, next SFMT word {i}",
                record[0]
            );
        }
    }
}

#[test]
fn c_normalization_includes_initialized_declared_slots_without_spatial_mappings() {
    let rows = records();
    assert_eq!(rows.len(), 8 * 4);
    for record in rows.as_chunks::<4>().0.iter() {
        let (mut data, mut rng) = data(record[0]);
        init_parameter(&mut data, &mut rng);
        check_parameter_bits(&data, record[1], record[0]);
        sync_modified_parameter(&mut data, false);
        check_parameter_bits(&data, record[2], record[0]);
    }
}
