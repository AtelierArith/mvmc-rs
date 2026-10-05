//! C run-log files (issue #346): `_time_`, `_SRinfo` and `NFileFlushInterval`.
//!
//! Expected text comes from `tests/fixtures/time_srinfo_c/`, produced by the actual C
//! `InitFile`/`OutputTime`/`FlushFile`/`CloseFile` and SRinfo `fprintf` statements
//! (see its PROVENANCE.md). Printed literals are compared as text; no computed
//! floating-point result is compared bitwise.

use std::fs;
use std::path::{Path, PathBuf};

use mvmc_core::output_files::{
    c_ctime_now, format_sr_info_row, format_time_line, sr_info_header, RunFiles, RunKind, SrInfoRow,
};
use mvmc_core::ExpertModeData;

const FIXTURE_CTIME: &str = "Tue Nov 14 22:13:20 2023\n";

fn fixture(path: &str) -> String {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/time_srinfo_c");
    fs::read_to_string(root.join(path)).unwrap_or_else(|e| panic!("{path}: {e}"))
}

fn scratch(label: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("mvmc-run-log-{label}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

#[allow(clippy::too_many_arguments)]
fn row(
    n_smat: i64,
    opt_num: i64,
    cut_num: i64,
    s_diag_max: f64,
    s_diag_min: f64,
    r_max: f64,
    i_max: i64,
    cg_info: Option<i64>,
) -> SrInfoRow {
    SrInfoRow {
        n_para: 48,
        n_smat,
        opt_num,
        cut_num,
        s_diag_max,
        s_diag_min,
        r_max,
        i_max,
        cg_info,
    }
}

#[test]
fn sr_info_rows_and_headers_match_c_printf_output() {
    // The four rows per step are the literals passed to the C probe's row_* helpers.
    let step_rows = [
        row(40, 4, 4, 3.10304e-2, 0.0, 7.84187e-3, 13, None),
        row(40, 4, 4, 3.53637e-2, 0.0, -5.5165e-3, 43, Some(35)),
        row(1, 0, 0, 1.0e100, -1.0e-300, 123456.789, 99999, None),
        row(12345, 100000, 7, -0.0, 2.5, -0.0, -1, None),
    ];
    let mut expected = sr_info_header(false).to_string();
    for _ in 0..4 {
        for r in &step_rows {
            expected += &format_sr_info_row(r);
        }
    }
    assert_eq!(expected, fixture("paraopt/custom_SRinfo.dat"));

    let diag = sr_info_header(true).to_string()
        + &format_sr_info_row(&row(3, 1, 2, 4.5, 0.25, -1.0, 4, None));
    assert_eq!(diag, fixture("diag/zvo_SRinfo.dat"));
}

#[test]
fn time_rows_match_c_output_except_ctime() {
    let counters: [[i64; 6]; 5] = [
        [0, 0, 0, 0, 0, 0],
        [1000, 437, 250, 100, 0, 0],
        [123456789, 61728394, 3, 1, 7, 7],
        [2147483647, 1, 0, 5, 9, 4],
        [10, 10, 10, 10, 10, 10],
    ];
    let expected: String = counters
        .iter()
        .enumerate()
        .map(|(step, c)| format_time_line(step as i64, c, FIXTURE_CTIME))
        .collect();
    assert_eq!(expected, fixture("paraopt/custom_time_007.dat"));

    let physcal = [[0, 0, 0, 0, 0, 0], [20, 5, 0, 0, 0, 0], [20, 5, 0, 0, 0, 0]];
    let expected: String = physcal
        .iter()
        .enumerate()
        .map(|(step, c)| format_time_line(step as i64, c, FIXTURE_CTIME))
        .collect();
    assert_eq!(expected, fixture("physcal/zvo_time_-01.dat"));
}

#[test]
fn ctime_has_c_layout() {
    let text = c_ctime_now();
    assert_eq!(text.len(), 25, "{text:?}");
    assert!(text.ends_with('\n'));
    let fields: Vec<&str> = text.split_whitespace().collect();
    assert_eq!(fields.len(), 5, "{text:?}");
    assert!(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"].contains(&fields[0]));
    assert_eq!(fields[3].split(':').count(), 3);
}

#[test]
fn init_creates_c_named_files_and_header() {
    let mut data = ExpertModeData::new();
    data.modpara.c_data_file_head = "custom".into();
    data.modpara.n_data_idx_start = 7;
    let dir = scratch("init");
    RunFiles::init(&data, &dir, RunKind::ParaOpt, None)
        .unwrap()
        .close()
        .unwrap();
    assert_eq!(
        fs::read_to_string(dir.join("custom_time_007.dat")).unwrap(),
        ""
    );
    assert_eq!(
        fs::read_to_string(dir.join("custom_SRinfo.dat")).unwrap(),
        fixture("diag/zvo_SRinfo.dat")
            .replace("sEigenMax  sEigenMin", "sDiagMax  sDiagMin")
            .lines()
            .next()
            .unwrap()
            .to_string()
            + "\n"
    );

    // PhysCal writes only the time file; C formats the signed index with %03d.
    let mut data = ExpertModeData::new();
    data.modpara.n_data_idx_start = -1;
    let dir = scratch("init-physcal");
    RunFiles::init(&data, &dir, RunKind::PhysCal, None)
        .unwrap()
        .close()
        .unwrap();
    let mut names: Vec<_> = fs::read_dir(&dir)
        .unwrap()
        .map(|e| e.unwrap().file_name().into_string().unwrap())
        .collect();
    names.sort();
    assert_eq!(names, ["zvo_time_-01.dat"]);

    // Truncation, as C's "w" mode: stale files from an earlier run are replaced.
    fs::write(dir.join("zvo_time_-01.dat"), "stale\n").unwrap();
    RunFiles::init(&data, &dir, RunKind::PhysCal, None)
        .unwrap()
        .close()
        .unwrap();
    assert_eq!(
        fs::read_to_string(dir.join("zvo_time_-01.dat")).unwrap(),
        ""
    );
}

#[test]
fn sr_info_header_follows_sr_flag_for_direct_and_cg_writers() {
    // RunFiles owns the single header writer used by the direct and CG solvers;
    // `sr_flag` is C's SRFlag (set when DSROptStepDt is negative).
    for (sr_flag, header) in [
        (false, "sDiagMax  sDiagMin"),
        (true, "sEigenMax  sEigenMin"),
    ] {
        let mut data = ExpertModeData::new();
        data.modpara.sr_flag = sr_flag;
        let dir = scratch(&format!("header-{sr_flag}"));
        RunFiles::init(&data, &dir, RunKind::ParaOpt, None)
            .unwrap()
            .close()
            .unwrap();
        let text = fs::read_to_string(dir.join("zvo_SRinfo.dat")).unwrap();
        assert_eq!(
            text,
            format!("#Npara Msize optCut diagCut {header}    absRmax       imax\n")
        );
        fs::remove_dir_all(dir).unwrap();
    }
}

#[test]
fn flush_interval_must_be_a_natural_number() {
    let data = ExpertModeData::new();
    let dir = scratch("flush-invalid");
    for bad in [0, -3] {
        assert!(RunFiles::init(&data, &dir, RunKind::ParaOpt, Some(bad)).is_err());
    }
}

#[test]
fn flush_file_flushes_only_every_interval_steps() {
    let data = ExpertModeData::new();
    let dir = scratch("flush");
    let path = dir.join("zvo_time_000.dat");
    let mut files = RunFiles::init(&data, &dir, RunKind::ParaOpt, Some(3)).unwrap();
    let counter = [0_i64; 6];
    let lines = |p: &Path| fs::read_to_string(p).unwrap().lines().count();
    let mut visible = Vec::new();
    for step in 0..7 {
        files.output_time(step, &counter).unwrap();
        files.flush_file(step).unwrap();
        visible.push(lines(&path));
    }
    // C: step % 3 == 0 flushes at steps 0, 3 and 6; rows between flushes stay buffered.
    assert_eq!(visible, [1, 1, 1, 4, 4, 4, 7]);
    assert_eq!(files.flush_count(), 3);
    files.output_time(7, &counter).unwrap();
    assert_eq!(lines(&path), 7);
    files.close().unwrap();
    assert_eq!(lines(&path), 8);

    // Default interval 1 flushes every step.
    let dir = scratch("flush-default");
    let path = dir.join("zvo_time_000.dat");
    let mut files = RunFiles::init(&data, &dir, RunKind::ParaOpt, None).unwrap();
    for step in 0..3 {
        files.output_time(step, &counter).unwrap();
        files.flush_file(step).unwrap();
        assert_eq!(lines(&path), step as usize + 1);
    }
}

#[test]
fn para_opt_run_writes_c_time_and_srinfo_files() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures");
    for (namelist, mode) in [
        (root.join("dh2/production_real/namelist.def"), "real"),
        (root.join("c_orbital_inputs/namelist_dh2_cmp.def"), "cmp"),
    ] {
        let dir = scratch(&format!("run-{mode}"));
        let steps = 3;
        let summary = mvmc_core::run_para_opt_from_namelist(
            &namelist,
            mvmc_core::RunConfig {
                nsmp: Some(steps),
                output_dir: Some(dir.clone()),
                file_flush_interval: Some(2),
                binary_output: false,
                ..mvmc_core::RunConfig::new(steps, mode)
            },
        )
        .unwrap();
        let data = mvmc_expert_parsers::parse_expert_mode_files(&namelist).unwrap();
        let index = data.modpara.n_data_idx_start;
        let time = fs::read_to_string(summary.output_dir.join(format!("zvo_time_{index:03}.dat")))
            .unwrap();
        let rows: Vec<&str> = time.lines().collect();
        // C writes OutputTime(0..NSROptItrStep-1) and a final OutputTime(NSROptItrStep).
        assert_eq!(rows.len(), steps as usize + 1, "{mode}\n{time}");
        for (step, line) in rows.iter().enumerate() {
            assert!(line.starts_with(&format!("{step:05}  ")), "{line}");
            let (body, ctime) = line.split_once(": ").unwrap();
            assert_eq!(ctime.split_whitespace().count(), 5, "{line}");
            if step == 0 {
                assert_eq!(
                    body,
                    "00000  acc_hop acc_ex  acc_lsf n_hop    n_ex      n_lsf   "
                );
                continue;
            }
            // %.5lf ratios in [0,1] followed by left-justified %-8d counters.
            // 5-digit step, two spaces, three 7-wide ratios, then 8+1, 8+2 and 8 wide counters.
            assert!(body.len() >= 58, "{body:?}");
            let fields: Vec<&str> = body.split_whitespace().collect();
            assert_eq!(fields.len(), 7, "{body:?}");
            for ratio in &fields[1..4] {
                assert_eq!(ratio.split('.').nth(1).unwrap().len(), 5, "{body:?}");
                let value: f64 = ratio.parse().unwrap();
                assert!((0.0..=1.0).contains(&value), "{body:?}");
            }
            for count in &fields[4..7] {
                count.parse::<i32>().unwrap();
            }
        }
        // Hubbard-type inputs attempt hops every step; counters are sampler statistics
        // reduced after the previous step, so row 1 already carries a positive count.
        let n_hop: i64 = rows[1].split_whitespace().nth(4).unwrap().parse().unwrap();
        assert!(n_hop > 0, "{}", rows[1]);

        let info = fs::read_to_string(summary.output_dir.join("zvo_SRinfo.dat")).unwrap();
        let mut lines = info.lines();
        assert_eq!(
            format!("{}\n", lines.next().unwrap()),
            sr_info_header(false)
        );
        let body: Vec<&str> = lines.collect();
        assert_eq!(body.len(), steps as usize, "{info}");
        for line in body {
            assert!(!line.contains(','), "direct rows have no CG info: {line}");
            let fields: Vec<&str> = line.split_whitespace().collect();
            assert_eq!(fields.len(), 8, "{line}");
            assert_eq!(
                fields[0].parse::<usize>().unwrap(),
                data.count_variational_parameters(),
                "{line}"
            );
            let n_smat: usize = fields[1].parse().unwrap();
            assert!(n_smat > 0);
            let imax: usize = fields[7].parse().unwrap();
            if mode == "real" {
                // C expands real parameters into (re, im) pairs; the real part is even.
                assert_eq!(imax % 2, 0, "{line}");
            }
        }
    }
}
