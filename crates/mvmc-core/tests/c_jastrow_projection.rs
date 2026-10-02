//! Actual-C MakeProjCnt/UpdateProjCnt expectations with directional definitions.
use mvmc_core::sampling::projection::{make_proj_cnt, update_proj_cnt};
use mvmc_expert_parsers::parsers::jastrow::parse_jastrow_content;
use mvmc_expert_parsers::ExpertModeData;

#[test]
fn parsed_directional_jastrow_counts_and_every_supplied_hop_match_c() {
    let rows: Vec<_> = include_str!("../../../tests/fixtures/jastrow/c_projection_counts.txt")
        .lines()
        .filter(|line| !line.starts_with('#'))
        .collect();
    assert_eq!(rows.len(), 60 * 3);
    let mut checked = 0;
    for record in rows.chunks_exact(3) {
        let header: Vec<_> = record[0].split_whitespace().collect();
        let nsite: usize = header[1].parse().unwrap();
        let width: usize = header[2].parse().unwrap();
        let section = parse_jastrow_content(&record[1].replace('|', "\n"), nsite as i64).unwrap();
        let mut data = ExpertModeData::new();
        data.modpara.nsite = nsite as i64;
        data.n_gutzwiller_idx = 2;
        data.gutzwiller_idx = (0..nsite).map(|i| (i % 2) as i64).collect();
        data.n_jastrow_idx = section.n_jastrow_idx;
        data.jastrow_terms = section.terms;
        data.jastrow_idx = section.idx_matrix;
        let nproj = width + 2;
        assert_eq!(data.projection_layout().n_proj, nproj);
        let mut case_count = 0;
        for row in record[2].split('|') {
            let values: Vec<i64> = row.split_whitespace().map(|v| v.parse().unwrap()).collect();
            assert_eq!(values.len(), 4 + 2 * nsite + 2 * nproj);
            let label = format!(
                "{} pattern={} hop=({},{},{})",
                header[0], values[0], values[1], values[2], values[3]
            );
            let mut occupancy = values[4..4 + 2 * nsite].to_vec();
            let expected_old = &values[4 + 2 * nsite..4 + 2 * nsite + nproj];
            let expected_new = &values[4 + 2 * nsite + nproj..];
            let mut old = vec![0; nproj];
            make_proj_cnt(&mut old, &occupancy, &data);
            assert_eq!(old, expected_old, "{label} MakeProjCnt");
            let mut next = old.clone();
            let spin = values[3];
            if spin >= 0 {
                let ri = values[1] as usize;
                let rj = values[2] as usize;
                assert_eq!(occupancy[spin as usize * nsite + ri], 1);
                assert_eq!(occupancy[spin as usize * nsite + rj], 0);
                occupancy[spin as usize * nsite + ri] -= 1;
                occupancy[spin as usize * nsite + rj] += 1;
                update_proj_cnt(
                    ri as i64, rj as i64, spin as u8, &mut next, &old, &occupancy, &data,
                );
            }
            assert_eq!(next, expected_new, "{label} UpdateProjCnt");
            let mut rebuilt = vec![0; nproj];
            make_proj_cnt(&mut rebuilt, &occupancy, &data);
            assert_eq!(rebuilt, expected_new, "{label} rebuild");
            case_count += 1;
            checked += 1;
        }
        assert_eq!(case_count, header[3].parse::<usize>().unwrap());
    }
    assert_eq!(checked, 1795);
}
