//! Observe idle CPU occupancy while a tenferro stage exists.
//! Run with `RAYON_NUM_THREADS=1` and `RAYON_NUM_THREADS=4`, respectively;
//! observe the printed PID with `ps -p <pid> -o %cpu,nlwp,args` during the wait.
use mvmc_core::stage_backend::StageBackend;
use std::io::Write;
use std::time::Duration;
fn main() {
    let _stage = StageBackend::tenferro_cpu().expect("tenferro CPU stage");
    println!("READY {}", std::process::id());
    std::io::stdout().flush().unwrap();
    std::thread::sleep(Duration::from_secs(10));
}
