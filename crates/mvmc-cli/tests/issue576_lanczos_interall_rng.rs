//! Compare sampled configurations and SFMT state against the independent C probe.
use std::{collections::HashMap, fs, path::Path, process::Command};

#[test]
fn interall_lanczos_preserves_native_c_sampling_and_rng() {
    let root =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/native_c_lanczos_576");
    for (name, mode) in [
        ("interall_real1", "real"),
        ("interall_real2", "real"),
        ("interall_complex1", "cmp"),
        ("interall_complex2", "cmp"),
        ("interall_nodal1", "real"),
        ("interall_nodal2", "real"),
    ] {
        let fixture = root.join(name);
        let temp = std::env::temp_dir().join(format!("mvmc-576-{name}-{}", std::process::id()));
        fs::create_dir(&temp).unwrap();
        let trace = temp.join("trace");
        let result = Command::new(env!("CARGO_BIN_EXE_mvmc"))
            .arg(fixture.join("inputs/namelist.def"))
            .arg("--physcal")
            .arg(fixture.join("zqp_opt.dat"))
            .args(["--seed", "1", "--mode", mode, "--out-dir"])
            .arg(temp.join("output"))
            .arg("--physcal-trace")
            .arg(&trace)
            .env("OMP_NUM_THREADS", "1")
            .env("OPENBLAS_NUM_THREADS", "1")
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "{name}: {}",
            String::from_utf8_lossy(&result.stderr)
        );
        let native = fs::read_to_string(fixture.join("native-state/state_dump_000.txt")).unwrap();
        let native: HashMap<_, Vec<_>> = native
            .lines()
            .map(|line| {
                let mut words = line.split_whitespace();
                (words.next().unwrap(), words.collect())
            })
            .collect();
        for (key, file) in [
            ("ele_idx", "ele_idx.txt"),
            ("ele_cfg", "ele_cfg.txt"),
            ("ele_num", "ele_num.txt"),
            ("ele_proj_cnt", "ele_proj_cnt.txt"),
            ("ele_spn", "ele_spn.txt"),
            ("draws", "draw-count.txt"),
            ("next624", "next624.txt"),
        ] {
            let text = fs::read_to_string(trace.join("sample-0").join(file)).unwrap();
            assert_eq!(
                text.split_whitespace().collect::<Vec<_>>(),
                native[key],
                "{name}: {key}"
            );
        }
        let counter = fs::read_to_string(trace.join("sample-0/counter.txt")).unwrap();
        assert_eq!(
            &counter.split_whitespace().collect::<Vec<_>>()[..6],
            native["counter"],
            "{name}: counter"
        );
        fs::remove_dir_all(temp).unwrap();
    }
}
