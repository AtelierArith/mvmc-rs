//! Convert unchanged Julia reference rows into C declared Slater order.
use mvmc_expert_parsers::ExpertModeData;

// Historical Julia snapshots duplicate shared mapping values. Convert their
// rows to C's declared-index order, checking every duplicate bit/string and
// requiring every declared slot to be present in the historical fixture.
pub(super) fn declared_slater_rows<T: Clone + Eq + std::fmt::Debug>(
    data: &ExpertModeData,
    mapped: &[T],
) -> Vec<T> {
    assert_eq!(mapped.len(), data.orbital_terms.len());
    let mut slots = vec![None; data.modpara.n_orbital_idx as usize];
    for (term, row) in data.orbital_terms.iter().zip(mapped) {
        let slot = &mut slots[term.idx as usize];
        if let Some(previous) = slot {
            assert_eq!(previous, row);
        } else {
            *slot = Some(row.clone());
        }
    }
    slots
        .into_iter()
        .map(|row| row.expect("historical fixture must cover every declared slot"))
        .collect()
}

pub(super) fn declared_output(data: &ExpertModeData, name: &str, historical: String) -> String {
    let prefix = data.gutzwiller_terms.len() + data.jastrow_terms.len();
    match name {
        "zvo_var.dat" => historical
            .lines()
            .map(|line| {
                let rows: Vec<String> = line.split_inclusive("0.0 ").map(str::to_owned).collect();
                assert_eq!(rows.len(), 2 + prefix + data.orbital_terms.len());
                rows[..2 + prefix]
                    .iter()
                    .cloned()
                    .chain(declared_slater_rows(data, &rows[2 + prefix..]))
                    .collect::<String>()
                    + "\n"
            })
            .collect(),
        "zqp_opt.dat" => {
            let rows: Vec<String> = historical
                .split_inclusive('\n')
                .map(str::to_owned)
                .collect();
            assert_eq!(rows.len(), prefix + data.orbital_terms.len());
            rows[..prefix]
                .iter()
                .cloned()
                .chain(declared_slater_rows(data, &rows[prefix..]))
                .collect()
        }
        "zqp_orbital_opt.dat" => {
            let rows: Vec<String> = historical
                .split_inclusive('\n')
                .map(str::to_owned)
                .collect();
            assert_eq!(rows.len(), 4 + data.orbital_terms.len());
            let mapped: Vec<String> = rows[4..]
                .iter()
                .enumerate()
                .map(|(i, row)| {
                    let (index, value) = row.split_once(' ').unwrap();
                    assert_eq!(index.parse::<usize>().unwrap(), i);
                    value.to_owned()
                })
                .collect();
            let mut out = rows[0].clone()
                + &format!("NOrbitalIdx {}\n", data.modpara.n_orbital_idx)
                + &rows[2]
                + &rows[3];
            for (index, row) in declared_slater_rows(data, &mapped).iter().enumerate() {
                out.push_str(&format!("{index} {row}"));
            }
            out
        }
        _ => historical,
    }
}
