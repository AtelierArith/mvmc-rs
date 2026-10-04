//! C readdef.c uses keyword slots rather than textual namelist order.
use mvmc_expert_parsers::parse_expert_mode_files;
use std::fs;

fn definition(header: &str, width: usize, rows: &str) -> String {
    format!("===\n{header} {width}\nComplexType 0\n===\n===\n{rows}")
}

fn integers(line: &str) -> Vec<i64> {
    line.split_whitespace()
        .map(|v| v.parse().unwrap())
        .collect()
}

fn permutations(items: &mut [usize], start: usize, result: &mut Vec<Vec<usize>>) {
    if start == items.len() {
        result.push(items.to_vec());
        return;
    }
    for i in start..items.len() {
        items.swap(start, i);
        permutations(items, start + 1, result);
        items.swap(start, i);
    }
}

#[test]
fn every_namelist_permutation_preserves_c_ap_parallel_layout_and_boundary_signs() {
    let dir = std::env::temp_dir().join(format!("mvmc-c-order-{}", std::process::id()));
    fs::create_dir_all(&dir).unwrap();
    let ap_flags: String = (0..7)
        .map(|i| format!("{i} {}\n", i64::from(i % 3 != 0)))
        .collect();
    fs::write(
        dir.join("ap.def"),
        definition(
            "NOrbitalIdx",
            7,
            &format!("0 0 0 -1\n0 1 1 1\n1 0 1 -1\n1 1 0 1\n{ap_flags}"),
        ),
    )
    .unwrap();
    fs::write(
        dir.join("p.def"),
        definition("NOrbitalParallel", 3, "0 1 0 -1\n0 1\n1 0\n2 1\n"),
    )
    .unwrap();
    fs::write(
        dir.join("locspin.def"),
        definition("NlocalSpin", 1, "0 1\n1 0\n"),
    )
    .unwrap();
    fs::write(
        dir.join("qp.def"),
        definition("NQPTrans", 1, "0 1\n0 0 1 -1\n0 1 0 1\n"),
    )
    .unwrap();
    let oracle = include_str!("../../../tests/fixtures/orbital_general/c_order.txt");
    let mut lines = oracle.lines().filter(|line| !line.starts_with('#'));
    let mut orders = Vec::new();
    permutations(&mut [0, 1, 2, 3, 4], 0, &mut orders);
    let mut cases = 0;
    while let Some(header) = lines.next() {
        let fields: Vec<_> = header.split_whitespace().collect();
        let kinds: Vec<_> = lines.next().unwrap().split_whitespace().collect();
        let counts = integers(lines.next().unwrap());
        let indices = integers(lines.next().unwrap());
        let signs = integers(lines.next().unwrap());
        let flags = integers(lines.next().unwrap());
        let antiperiodic = fields[1] == "1";
        fs::write(
            dir.join("modpara.def"),
            format!(
                "Nsite 2\nNElec 1\nNMPTrans {}\n",
                if antiperiodic { -1 } else { 1 }
            ),
        )
        .unwrap();
        let paths = ["modpara.def", "locspin.def", "ap.def", "p.def", "qp.def"];
        for order in &orders {
            let namelist: String = order
                .iter()
                .map(|&i| format!("{} {}\n", kinds[i], paths[i]))
                .collect();
            fs::write(dir.join("namelist.def"), &namelist).unwrap();
            let data = parse_expert_mode_files(dir.join("namelist.def"))
                .unwrap_or_else(|error| panic!("{header} {order:?}: {error}"));
            assert_eq!(
                data.namelist
                    .iter()
                    .map(|(kind, _)| kind.as_str())
                    .collect::<Vec<_>>(),
                order.iter().map(|&i| kinds[i]).collect::<Vec<_>>(),
                "retain input metadata"
            );
            assert!(
                data.input_errors.is_empty(),
                "{header} {order:?}: {:?}",
                data.input_errors
            );
            assert_eq!(data.modpara.nsite, 2, "{header} {order:?}");
            assert_eq!(data.modpara.nlocspin, 1, "{header} {order:?}");
            assert_eq!(
                data.n_orbital_anti_parallel, counts[0],
                "{header} {order:?}"
            );
            assert_eq!(data.modpara.n_orbital_idx, counts[1], "{header} {order:?}");
            assert_eq!(
                data.orbital_idx_matrix
                    .unwrap()
                    .into_iter()
                    .flatten()
                    .collect::<Vec<_>>(),
                indices,
                "{header} {order:?}"
            );
            assert_eq!(
                data.orbital_sgn_matrix
                    .unwrap()
                    .into_iter()
                    .flatten()
                    .collect::<Vec<_>>(),
                signs,
                "{header} {order:?}"
            );
            // #21 tracks C's distinct imaginary-component initialization.
            // This ordering regression covers AP/P real-flag offsets.
            assert_eq!(
                data.optimization_flags
                    .iter()
                    .step_by(2)
                    .copied()
                    .collect::<Vec<_>>(),
                flags.iter().step_by(2).copied().collect::<Vec<_>>(),
                "{header} {order:?}"
            );
            assert_eq!(data.qp_trans_entries.len(), 1, "{header} {order:?}");
            assert_eq!(
                data.qp_trans_entries[0].site_map,
                [1, 0],
                "{header} {order:?}"
            );
            assert_eq!(
                data.qp_trans_entries[0].site_sign,
                [-1, 1],
                "raw signs retained for BC lookup"
            );
            cases += 1;
        }
    }
    assert_eq!(cases, 480);
    fs::remove_dir_all(dir).unwrap();
}
