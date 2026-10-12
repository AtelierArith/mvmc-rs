//! Explicit standalone RNG generation/refill benchmark. No sampler speedup claim.
use mvmc_core::sampling::rng::{SamplingRng, XorwowRng};
use mvmc_gpu_cuda::xorwow::DeviceXorwow;
use sfmt19937::Sfmt19937Rng;
use std::time::Instant;
fn main() -> Result<(), String> {
    let device = std::env::var("MVMC_RS_XORWOW_DEVICE")
        .unwrap_or_else(|_| "1".into())
        .parse()
        .map_err(|e| format!("{e}"))?;
    println!("device,lanes,words_per_lane,repetitions,generation_us,download_us,words_per_s,cpu_xorwow_us,cpu_sfmt_us");
    for (lanes, words, reps) in [(1, 4096, 200), (256, 4096, 100), (1024, 4096, 30)] {
        let mut rng = DeviceXorwow::new(device, 11272, lanes, words)?;
        for _ in 0..5 {
            rng.generate(words)?;
            rng.synchronize()?;
        }
        let start = Instant::now();
        for _ in 0..reps {
            rng.generate(words)?;
        }
        rng.synchronize()?;
        let generation = start.elapsed().as_secs_f64() / reps as f64;
        let mut host = Vec::with_capacity(lanes * words);
        rng.download_into(&mut host)?;
        let start = Instant::now();
        for _ in 0..reps {
            rng.download_into(&mut host)?;
            std::hint::black_box(&host);
        }
        let transfer = start.elapsed().as_secs_f64() / reps as f64;
        fn cpu_time(rng: &mut impl SamplingRng, words: usize, reps: usize) -> f64 {
            let start = Instant::now();
            let mut checksum = 0u32;
            for _ in 0..reps {
                for _ in 0..words {
                    checksum ^= rng.gen_rand32();
                }
            }
            std::hint::black_box(checksum);
            start.elapsed().as_secs_f64() / reps as f64
        }
        let cpu_xorwow = cpu_time(&mut XorwowRng::new(11272), lanes * words, reps);
        let cpu_sfmt = cpu_time(&mut Sfmt19937Rng::new(11272), lanes * words, reps);
        println!(
            "{device},{lanes},{words},{reps},{:.3},{:.3},{:.0},{:.3},{:.3}",
            generation * 1e6,
            transfer * 1e6,
            (lanes * words) as f64 / generation,
            cpu_xorwow * 1e6,
            cpu_sfmt * 1e6
        );
    }
    Ok(())
}
