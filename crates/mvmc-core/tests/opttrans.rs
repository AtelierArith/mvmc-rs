use mvmc_core::{qp, read_initial_def, read_opt_para_file, ExpertModeData};
use num_complex::Complex64;
use std::path::PathBuf;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/opttrans")
}

#[test]
fn full_record_loads_opttrans_after_all_other_factors() {
    let mut data = ExpertModeData::new();
    data.opt_trans = vec![Complex64::new(1.0, 0.0); 2];
    let path = std::env::temp_dir().join(format!("mvmc-opttrans-red-{}.def", std::process::id()));
    std::fs::write(&path, "1 2 3 4 5 6 0.7 -0.8 9.9 0.9 0.2 9.9").unwrap();
    let result = read_opt_para_file(&mut data, &path);
    std::fs::remove_file(path).unwrap();
    assert_eq!(result.unwrap(), 2);
    assert_eq!(
        data.opt_trans,
        vec![Complex64::new(0.7, -0.8), Complex64::new(0.9, 0.2)]
    );
}

#[test]
fn data_weight_initialization_and_refresh_include_opttrans_sectors() {
    let mut data = ExpertModeData::new();
    data.modpara.nsp_gauss_leg = 1;
    data.modpara.nmp_trans = 2;
    data.para_qp_trans = vec![Complex64::new(1.0, 0.0), Complex64::new(0.5, 0.0)];
    data.opt_trans = vec![Complex64::new(0.25, 0.0), Complex64::new(0.75, 0.0)];
    qp::init_qp_weight(&mut data);
    assert_eq!(data.qp_weights.as_ref().unwrap().qp_full_weight.len(), 4);
    data.opt_trans[1] = Complex64::new(0.5, -0.25);
    qp::update_qp_weight_for(&mut data);
    assert_eq!(
        data.qp_weights.unwrap().qp_full_weight[2],
        data.opt_trans[1]
    );
}

fn model(name: &str) -> ExpertModeData {
    let base = if matches!(name, "short_opt" | "long_opt" | "empty_opt") {
        "layout"
    } else {
        name
    };
    let mut data =
        mvmc_expert_parsers::parse_expert_mode_files(root().join(format!("namelist_{base}.def")))
            .unwrap();
    match name {
        "empty_opt" => data.opt_trans.clear(),
        "short_opt" => data.opt_trans.truncate(1),
        "long_opt" => data.opt_trans.push(Complex64::new(0.5, -0.25)),
        _ => {}
    }
    data
}

fn bits(values: impl IntoIterator<Item = Complex64>, expected: &str, label: &str) {
    let actual: Vec<_> = values
        .into_iter()
        .flat_map(|v| [v.re.to_bits(), v.im.to_bits()])
        .collect();
    let expected: Vec<_> = expected
        .split_whitespace()
        .map(|v| u64::from_str_radix(v, 16).unwrap())
        .collect();
    assert_eq!(actual, expected, "{label}");
}

fn values(data: &mut ExpertModeData) -> Vec<Complex64> {
    let mut values = data.projection_parameters();
    data.visit_rbm_terms_mut(|_, term| values.push(term.value()));
    values.extend(data.orbital_terms.iter().map(|t| t.value));
    values.extend(data.opt_trans.iter().copied());
    values
}

#[test]
fn full_record_loader_counts_errors_values_and_atomicity_match_julia() {
    let text = std::fs::read_to_string(root().join("loaders.txt")).unwrap();
    let mut lines = text.lines().filter(|line| !line.starts_with('#'));
    let mut cases = 0;
    while let Some(header) = lines.next() {
        let fields: Vec<_> = header.split_whitespace().collect();
        let mut data = model(fields[0]);
        let file = format!("load_{}_{}.def", fields[0], fields[1]);
        let path = root().join(&file);
        let (result, error) = match fields[2] {
            "initial" => (
                i64::from(read_initial_def(&mut data, &path).unwrap()),
                String::new(),
            ),
            "optimized" => match read_opt_para_file(&mut data, &path) {
                Ok(n) => (n as i64, String::new()),
                Err(e) => (-1, e.replace(path.to_str().unwrap(), &file)),
            },
            _ => panic!("unknown loader: {header}"),
        };
        assert_eq!(result, fields[3].parse::<i64>().unwrap(), "{header}");
        assert_eq!(error, lines.next().unwrap(), "{header}");
        bits(values(&mut data), lines.next().unwrap(), header);
        bits(data.para_qp_opt_trans, lines.next().unwrap(), header);
        cases += 1;
    }
    assert_eq!(cases, 132);
}

#[test]
fn initialized_updated_and_resized_sector_weights_match_julia_bits() {
    let text = std::fs::read_to_string(root().join("weights.txt")).unwrap();
    let mut lines = text.lines().filter(|line| !line.starts_with('#'));
    let mut data = ExpertModeData::new();
    let mut cases = 0;
    while let Some(header) = lines.next() {
        let fields: Vec<_> = header.split_whitespace().collect();
        match fields[4] {
            "initial" => {
                data = ExpertModeData::new();
                data.modpara.nsp_gauss_leg = fields[0].parse().unwrap();
                data.modpara.nsp_stot = fields[1].parse().unwrap();
                data.modpara.nmp_trans = fields[2].parse().unwrap();
                data.para_qp_trans = vec![Complex64::new(1.0, 0.25), Complex64::new(-0.5, -0.125)];
                data.opt_trans = match fields[3] {
                    "empty" => vec![],
                    "one" => vec![Complex64::new(0.75, -0.0)],
                    "zero" => vec![Complex64::new(0.0, 0.0), Complex64::new(-0.0, -0.0)],
                    "complex" => vec![Complex64::new(0.25, -0.125), Complex64::new(-0.75, 0.5)],
                    _ => panic!("unknown mode: {header}"),
                };
                qp::init_qp_weight(&mut data);
            }
            "replace" => {
                data.opt_trans = vec![Complex64::new(0.5, 0.25), Complex64::new(-0.125, -0.75)]
            }
            "grow" => data.opt_trans.push(Complex64::new(1.5, -0.5)),
            "shrink" => data.opt_trans.truncate(1),
            "clear" => data.opt_trans.clear(),
            "repeat" => {}
            _ => panic!("unknown phase: {header}"),
        }
        if fields[4] != "initial" {
            qp::update_qp_weight_for(&mut data);
        }
        let weights = data.qp_weights.as_ref().unwrap();
        for values in [
            &weights.qp_full_weight,
            &weights.qp_fix_weight,
            &weights.spgl_cos,
            &weights.spgl_sin,
            &weights.spgl_cos_sin,
            &weights.spgl_cos_cos,
            &weights.spgl_sin_sin,
        ] {
            bits(values.iter().copied(), lines.next().unwrap(), header);
        }
        cases += 1;
    }
    assert_eq!(cases, 576);
}
