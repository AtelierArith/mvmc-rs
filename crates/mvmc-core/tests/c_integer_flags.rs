//! C-derived component selection exercised through both Rust SR solvers.
#[path = "../../../tests/support/numerical_comparison.rs"]
mod numerical_comparison;
use mvmc_core::{sr, sr_cg, VmcOptimizationState};
use mvmc_expert_parsers::parse_expert_mode_files;
use num_complex::Complex64;
use std::fs;

#[test]
fn direct_and_cg_sr_update_exactly_the_components_selected_by_c() {
    let rows: Vec<_> =
        include_str!("../../../tests/fixtures/optimization_flags/c_integer_flags.txt")
            .lines()
            .filter(|line| !line.starts_with('#'))
            .collect();
    assert_eq!(rows.len(), 70 * 7);
    let dir = std::env::temp_dir().join(format!("mvmc-c-integer-sr-{}", std::process::id()));
    fs::create_dir_all(&dir).unwrap();
    for record in rows.as_chunks::<7>().0.iter() {
        let header: Vec<_> = record[0].split_whitespace().collect();
        let complex = header[5] != "0";
        fs::write(dir.join("modpara.def"), concat!("--------------------\nModel_Parameters 0\n--------------------\nVMC_Cal_Parameters\n--------------------\nCDataFileHead zvo\nCParaFileHead zqp\n--------------------\n", "Nsite 2\nNe 1\nNMPTrans -1\n")).unwrap();
        fs::write(dir.join("ap.def"), record[1].replace('|', "\n")).unwrap();
        let mut namelist = "ModPara modpara.def\nOrbitalAntiParallel ap.def\n".to_owned();
        if !record[2].is_empty() {
            fs::write(dir.join("p.def"), record[2].replace('|', "\n")).unwrap();
            namelist.push_str("OrbitalParallel p.def\n");
        }
        fs::write(dir.join("namelist.def"), namelist).unwrap();
        let mut data = parse_expert_mode_files(dir.join("namelist.def")).unwrap();
        assert!(data.input_errors.is_empty());
        let n = data.count_variational_parameters();
        data.modpara.nvmc_sample = 128;
        data.modpara.dsr_opt_red_cut = 0.0;
        data.modpara.dsr_opt_sta_del = 0.0;
        data.modpara.dsr_opt_step_dt = 0.5;
        data.slater_params = (0..n)
            .map(|i| Complex64::new((20 + i) as f64, if complex { 5.0 } else { 0.0 }))
            .collect();
        let mut expected = data.slater_params.clone();
        for field in record[6].split_whitespace() {
            let component: usize = field.parse().unwrap();
            if component.is_multiple_of(2) {
                expected[component / 2].re -= 1.0;
            } else {
                assert!(complex);
                expected[component / 2].im -= 1.0;
            }
        }
        let mut state = VmcOptimizationState::zeros(2, 1, 0, n, 1, 128, complex, false);
        state.energy.wc = Complex64::new(128.0, 0.0);
        let size = if complex { 2 * (n + 1) } else { n + 1 };
        // Paired +/-8 samples give zero means and identity real covariance:
        // 2*8^2/128=1 exactly. Complex component pairs differ by a factor i.
        for i in 0..n {
            if complex {
                for component in [2 * i, 2 * i + 1] {
                    state.sr_opt.sr_opt_oo[(component + 2) * size + component + 2] =
                        Complex64::new(1.0, 0.0);
                    state.sr_opt.sr_opt_ho[component + 2] = Complex64::new(1.0, 0.0);
                }
                for (sample, sign) in [(i, 1.0), (i + n, -1.0)] {
                    state.sr_opt.sr_opt_o_store[sample * size + 2 * i + 2] =
                        Complex64::new(sign * 8.0, 0.0);
                    state.sr_opt.sr_opt_o_store[sample * size + 2 * i + 3] =
                        Complex64::new(0.0, sign * 8.0);
                }
            } else {
                state.sr_opt.sr_opt_oo_real[(i + 1) * size + i + 1] = 1.0;
                state.sr_opt.sr_opt_ho_real[i + 1] = 1.0;
                state.sr_opt.sr_opt_o_store_real[i * size + i + 1] = 8.0;
                state.sr_opt.sr_opt_o_store_real[(i + n) * size + i + 1] = -8.0;
            }
        }
        for cg in [false, true] {
            let mut actual = data.clone();
            let info = if cg {
                sr_cg::stochastic_opt_cg(&mut actual, &state, None).unwrap()
            } else if complex {
                sr::stochastic_opt_complex(&mut actual, &mut state)
            } else {
                sr::stochastic_opt_real(&mut actual, &mut state)
            };
            assert_eq!(info, 0, "{} cg={cg}", header[0]);
            // Identity covariance: condition one, with a single step/update.
            numerical_comparison::assert_values_close(
                actual.slater_params.iter().flat_map(|z| [z.re, z.im]),
                expected.iter().flat_map(|z| [z.re, z.im]),
                16.0 * f64::EPSILON,
                16.0 * f64::EPSILON,
                format!("{} cg={cg}", header[0]),
            );
            assert_eq!(actual.optimization_flags, data.optimization_flags);
        }
    }
    fs::remove_dir_all(dir).unwrap();
}
