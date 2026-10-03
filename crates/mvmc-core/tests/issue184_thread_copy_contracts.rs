//! Original threading copy assertions M0838..M0842, not all 96 assertions.
//! Rust copies a common slice prefix; Julia's explicit n is represented by
//! slicing the source. Exact copies are not computed floating-point parity.

use mvmc_core::threading::{
    copy_complex_realpart, copy_real_to_complex, inner_thread_config, start_observation,
};
use num_complex::Complex64;

#[test]
fn original_copy_prefix_contracts_match_workers_one_two_four() {
    for workers in [1, 2, 4] {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--ignored", "--exact", "copy_contract_child", "--nocapture"])
            .env("ISSUE184_COPY_CHILD", "1")
            .env("MVMC_RS_INNER_THREADS", workers.to_string())
            .env("MVMC_RS_INNER_THRESHOLD", "32")
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "workers={workers}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        eprint!("{}", String::from_utf8_lossy(&output.stderr));
    }
}

fn observed_copy(items: usize, operation: impl FnOnce()) {
    let observer = start_observation();
    operation();
    let snapshot = observer.finish();
    assert_eq!(
        snapshot.serial_entry_items + snapshot.parallel_entry_items,
        items
    );
    assert_eq!(snapshot.executed_qp_items, 0);
    assert_eq!(snapshot.executed_term_items, 0);
    let config = inner_thread_config();
    if config.threads > 1 && items >= 32 {
        assert_eq!(snapshot.parallel_entry_items, items);
        assert_eq!(snapshot.serial_entry_items, 0);
        assert!(!snapshot.worker_ids.is_empty());
        assert!(snapshot
            .worker_ids
            .iter()
            .all(|&worker| worker < config.threads));
    } else {
        assert_eq!(snapshot.parallel_entry_items, 0);
        assert_eq!(snapshot.worker_entries, 0);
    }
    eprintln!(
        "copy workers={} items={items}: {snapshot:?}",
        config.threads
    );
}

#[test]
#[ignore = "internal isolated copy process; ISSUE184_COPY_CHILD required"]
fn copy_contract_child() {
    assert_eq!(std::env::var("ISSUE184_COPY_CHILD").as_deref(), Ok("1"));
    let real: Vec<_> = (1..=128).map(|i| (0.03 * i as f64).sin()).collect();
    let complex: Vec<_> = (1..=128)
        .map(|i| Complex64::new((0.05 * i as f64).cos(), (0.07 * i as f64).sin()))
        .collect();
    let sentinel = Complex64::new(99.0, 99.0);
    let mut complex_dst = vec![sentinel; 128];
    observed_copy(128, || copy_real_to_complex(&mut complex_dst, &real));
    assert_eq!(
        complex_dst,
        real.iter()
            .map(|&x| Complex64::new(x, 0.0))
            .collect::<Vec<_>>()
    );
    let mut real_dst = vec![99.0; 128];
    observed_copy(128, || copy_complex_realpart(&mut real_dst, &complex));
    assert_eq!(real_dst, complex.iter().map(|x| x.re).collect::<Vec<_>>());

    let mut short = vec![Complex64::new(0.0, 0.0); 4];
    observed_copy(4, || copy_real_to_complex(&mut short, &real));
    assert_eq!(short, complex_dst[..4]);
    let mut long = vec![sentinel; 132];
    observed_copy(128, || copy_real_to_complex(&mut long, &real));
    assert_eq!(long[..128], complex_dst);
    assert_eq!(long[128..], [sentinel; 4]);
    let mut long_real = vec![99.0; 132];
    observed_copy(128, || copy_complex_realpart(&mut long_real, &complex));
    assert_eq!(long_real[..128], real_dst);
    assert_eq!(long_real[128..], [99.0; 4]);
    observed_copy(0, || copy_real_to_complex(&mut long, &[]));
    assert_eq!(long[128..], [sentinel; 4]);
}
