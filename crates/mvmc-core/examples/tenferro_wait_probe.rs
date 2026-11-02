//! Measure idle CPU after one tenferro Gram operation; see benchmark/tenferro_wait_policy.
use mvmc_core::sr_backend::{SrStages, TenferroSr};
use std::io::Write;
use std::time::Duration;

fn main() {
    let n = 256;
    let samples = 300;
    let store: Vec<f64> = (0..n * samples).map(|i| (i % 101) as f64 / 101.0).collect();
    let mut out = vec![0.0; n * n];
    let mut stage = TenferroSr::new_cpu().expect("tenferro CPU stage");
    stage.gram_real(&store, n, samples, &mut out).unwrap();
    println!("READY {}", std::process::id());
    std::io::stdout().flush().unwrap();
    std::thread::sleep(Duration::from_secs(3));
}
