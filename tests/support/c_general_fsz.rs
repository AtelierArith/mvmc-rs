//! Independently generated native-C FSZ expectations; no C execution here.
use crate::numerical_comparison;
use crate::slater_derivative::{slater_elm_diff_fsz_with_scratch, SlaterDerivativeScratch};
use crate::slater_update::update_slater_elm_fsz;
use crate::VmcOptimizationState;
use mvmc_expert_parsers::utils::qp_weight::init_qp_weight;
use mvmc_expert_parsers::{parse_expert_mode_files, QPTransEntry};
use num_complex::Complex64;
use std::fs;

#[test]
fn six_column_general_slater_and_derivatives_match_actual_c_fsz_kernels() {
    let rows: Vec<_> = include_str!("../fixtures/orbital_general/c_general_kernels.txt")
        .lines()
        .filter(|line| !line.starts_with('#'))
        .collect();
    assert_eq!(rows.len(), 104 * 8);
    let dir = std::env::temp_dir().join(format!("mvmc-c-general-fsz-{}", std::process::id()));
    fs::create_dir_all(&dir).unwrap();
    let complexes = |text: &str| {
        let values: Vec<_> = text
            .split_whitespace()
            .map(|s| f64::from_bits(u64::from_str_radix(s, 16).unwrap()))
            .collect();
        values
            .as_chunks::<2>()
            .0
            .iter()
            .map(|z| Complex64::new(z[0], z[1]))
            .collect::<Vec<_>>()
    };
    for record in rows.as_chunks::<8>().0.iter() {
        let header: Vec<_> = record[0].split_whitespace().collect();
        let nsite: usize = header[1].parse().unwrap();
        let anti = header[2] == "1";
        let width: usize = header[3].parse().unwrap();
        fs::write(dir.join("general.def"), record[1].replace('|', "\n")).unwrap();
        fs::write(
            dir.join("modpara.def"),
            format!(
                "Nsite {nsite}\nNElec 2\nNSPGaussLeg 1\nNMPTrans {}2\n",
                if anti { "-" } else { "" }
            ),
        )
        .unwrap();
        fs::write(
            dir.join("namelist.def"),
            "ModPara modpara.def\nOrbitalGeneral general.def\n",
        )
        .unwrap();
        let mut data = parse_expert_mode_files(dir.join("namelist.def")).unwrap();
        assert!(
            data.input_errors.is_empty(),
            "{} {:?}",
            header[0],
            data.input_errors
        );
        data.slater_params = complexes(record[2]);
        data.n_qp_trans = 2;
        data.para_qp_trans = vec![Complex64::new(1.0, 0.0), Complex64::new(-0.375, 0.0)];
        data.qp_trans_entries = (0..2)
            .map(|qp| QPTransEntry {
                weight: data.para_qp_trans[qp],
                site_map: (0..nsite).map(|i| ((i + qp) % nsite) as i64).collect(),
                site_sign: (0..nsite)
                    .map(|i| if anti && qp == 1 && i % 2 == 0 { -1 } else { 1 })
                    .collect(),
            })
            .collect();
        init_qp_weight(&mut data);
        let mut state = VmcOptimizationState::zeros(nsite, 2, 0, width, 2, 1, true, true);
        state
            .slater_matrix
            .pf_m
            .copy_from_slice(&complexes(record[3]));
        let inverse = complexes(record[4]);
        for qp in 0..2 {
            state
                .slater_matrix
                .inv_m
                .qp_matrix_slice_mut(qp)
                .copy_from_slice(&inverse[qp * 16..(qp + 1) * 16]);
        }
        update_slater_elm_fsz(&mut data, &mut state);
        let expected_slater = complexes(record[6]);
        // QP Slater construction is a short signed, weighted sum.
        numerical_comparison::assert_values_close(
            state
                .slater_matrix
                .slater_elm
                .as_slice()
                .iter()
                .flat_map(|z| [z.re, z.im]),
            expected_slater.iter().flat_map(|z| [z.re, z.im]),
            16.0 * f64::EPSILON,
            16.0 * f64::EPSILON,
            format!("{} Slater", header[0]),
        );
        let mut actual = vec![Complex64::new(0.0, 0.0); 2 * width];
        slater_elm_diff_fsz_with_scratch(
            &mut actual,
            complexes(record[5])[0],
            &[0, 1, 0, 1],
            &[0, 0, 1, 1],
            &data,
            &state.slater_matrix,
            &mut SlaterDerivativeScratch::new(),
        );
        let expected = complexes(record[7]);
        // Existing 1e-14 derivative budget, now componentwise and scale aware.
        numerical_comparison::assert_values_close(
            actual.iter().flat_map(|z| [z.re, z.im]),
            expected.iter().flat_map(|z| [z.re, z.im]),
            1e-14,
            1e-14,
            format!("{} derivatives", header[0]),
        );
        // Unmapped derivative slots are initialized/copied zeros, not a
        // cancellation result: retain their exact storage contract.
        for slot in 0..width {
            if !data
                .orbital_terms
                .iter()
                .any(|term| term.idx as usize == slot)
            {
                assert_eq!(actual[2 * slot], Complex64::new(0.0, 0.0));
                assert_eq!(actual[2 * slot + 1], Complex64::new(0.0, 0.0));
            }
        }
        // The generated geometry only references slots 0..3 in synthetic
        // cases; C and Rust return zero derivatives for every unmapped slot.
        assert_eq!(actual.len(), expected.len());
    }
    fs::remove_dir_all(dir).unwrap();
}
