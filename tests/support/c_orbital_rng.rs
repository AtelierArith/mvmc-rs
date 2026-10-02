use mvmc_expert_parsers::{utils::parameter_init::all_complex_flag, ExpertModeData};

/// C InitParameter with the same explicit active RBM prefix and Slater flags.
/// This is a kernel comparison, not C acceptance of legacy malformed inputs.
pub fn declared_slater_record(data: &ExpertModeData) -> [&'static str; 4] {
    assert_eq!(data.modpara.n_orbital_idx, 4);
    let n_proj = data.projection_layout().n_proj;
    let n_rbm = data.count_rbm_parameters();
    let active_rbm = (0..n_rbm)
        .filter(|&i| {
            data.optimization_flags
                .get(2 * (n_proj + i))
                .copied()
                .unwrap_or(false)
        })
        .count();
    let mask = (0..4).fold(0, |mask, i| {
        mask | (usize::from(
            data.optimization_flags
                .get(2 * (n_proj + n_rbm + i))
                .copied()
                .unwrap_or(true),
        ) << i)
    });
    let key = format!("{} {active_rbm} {mask}", u8::from(all_complex_flag(data)));
    let rows: Vec<_> = include_str!("../fixtures/orbital_general/c_rbm_prefix.txt")
        .lines()
        .filter(|line| !line.starts_with('#'))
        .collect();
    let record = rows
        .chunks_exact(4)
        .find(|record| record[0] == key)
        .unwrap_or_else(|| panic!("missing C InitParameter kernel case {key}"));
    record.try_into().unwrap()
}

pub fn declared_slater_rng(data: &ExpertModeData) -> Vec<u32> {
    let record = declared_slater_record(data);
    let words: Vec<_> = record[3]
        .split_whitespace()
        .map(|s| s.parse().unwrap())
        .collect();
    assert_eq!(words.len(), 624);
    words
}
