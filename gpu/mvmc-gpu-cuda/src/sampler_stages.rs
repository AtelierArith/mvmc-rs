//! Production [`SamplerStages`] on the CUDA device-resident lock-step sampler (issue #517).
//!
//! The service is the standalone [`CudaSamplerService`] of issue #434 with the pinned
//! asynchronous transfer path. `sample_real` runs the single production chain `(data, state,
//! rng)` through [`run_lockstep_real_one_inline`]: the host driver, the SFMT draws and the
//! sampled configurations are the C path's, only the Pfaffian stages live on the device. The
//! inline runner drives the service on the walker's own thread (no channel, reply slot or wake
//! per hop; issue #517 milestone 4). With `MVMC_C_TIMER=1` the walker's sampler sections land in
//! the run's C timer, so the device run reports the same `zvo_CalcTimer.dat` breakdown as the
//! C-order path; `MVMC_RS_SAMPLER_PROFILE=1` additionally prints the per-call lock-step and
//! service attribution on stderr.
//! `MVMC_RS_CUDA_RNG=xorwow` opts into device-generated word batches; proposal control
//! still runs on the host and downloads a batch every 4096 consumed words.
//!
//! The device chain is the serial production chain. A multi-process reducer (MPI QP split or a
//! collective sampling sum) has no device implementation and is a hard [`StageError::Unsupported`],
//! never a silent C-order execution. Complex and FSZ modes are rejected by the runner before this
//! stage is reached (`sampler_backend::check_supported_mode`).

use mvmc_core::c_timer::CTimer;
use mvmc_core::device_sampler::{
    run_lockstep_real_one_inline, LockstepOptions, LockstepStats, WalkerRun,
};
use mvmc_core::reducer::Reducer;
use mvmc_core::sampler_backend::{SamplerStages, SamplerTiming};
use mvmc_core::sampling::driver::SampleStats;
use mvmc_core::sampling::rng::SamplingRng;
use mvmc_core::stage_backend::StageError;
use mvmc_core::state::VmcOptimizationState;
use mvmc_core::ExpertModeData;
use sfmt19937::Sfmt19937Rng;

use crate::device_sampler::{CudaSamplerService, PinnedTransfer};

/// Environment variable that turns on the per-call lock-step/service attribution on stderr.
pub const SAMPLER_PROFILE_VARIABLE: &str = "MVMC_RS_SAMPLER_PROFILE";

fn profile_enabled() -> bool {
    std::env::var_os(SAMPLER_PROFILE_VARIABLE).is_some()
}

/// How long the parked production walker spins for a reply before parking. Kept for the channel
/// runner (`run_lockstep_real`); the inline single-walker runner of issue #517 milestone 4 never
/// parks, so it ignores `spin_us`.
const SPIN_US: u64 = 200;

/// [`SamplerStages`] implementation on one CUDA device.
pub struct CudaSamplerStages {
    service: CudaSamplerService<PinnedTransfer>,
    /// Snapshot of the service's cumulative timings after the previous call (the profile print
    /// reports per-call deltas).
    last_timings: crate::device_sampler::ServiceTimings,
    rng: Option<crate::xorwow::GpuSamplingRng>,
    xorwow: bool,
    seed: Option<u64>,
    device: usize,
}

impl CudaSamplerStages {
    /// Compile the kernels and open the service on `device`.
    pub fn new(device: usize) -> Result<Self, String> {
        let xorwow = match std::env::var("MVMC_RS_CUDA_RNG").as_deref() {
            Ok("xorwow") => true,
            Ok("sfmt") | Err(_) => false,
            Ok(value) => {
                return Err(format!(
                    "unsupported MVMC_RS_CUDA_RNG={value}; use sfmt or xorwow"
                ))
            }
        };
        let seed = std::env::var("MVMC_RS_CUDA_RNG_SEED")
            .ok()
            .map(|s| {
                s.parse::<u64>()
                    .map_err(|e| format!("invalid CUDA RNG seed: {e}"))
            })
            .transpose()?;
        Self::with_rng(device, xorwow, seed)
    }

    /// Select GPU XORWOW (`true`) or CPU SFMT (`false`) explicitly. With no GPU seed,
    /// the first series seeds XORWOW from two words of the caller's initialized SFMT.
    pub fn with_rng(device: usize, xorwow: bool, seed: Option<u64>) -> Result<Self, String> {
        Ok(Self {
            service: CudaSamplerService::new_pinned(device)?,
            last_timings: crate::device_sampler::ServiceTimings::default(),
            rng: None,
            xorwow,
            seed,
            device,
        })
    }

    /// Words consumed from GPU-generated batches so far (prefetch does not count).
    pub fn gpu_words_consumed(&self) -> Option<u128> {
        self.rng.as_ref().map(|r| r.words_consumed())
    }

    /// Nonconsuming diagnostic preview of the next GPU stream words. Returns `false`
    /// before XORWOW initialization or when SFMT is selected. Crossing a prefetched
    /// batch boundary downloads the six-word device state for a CPU-only preview.
    pub fn gpu_words_preview(&self, out: &mut [u32]) -> bool {
        if let Some(rng) = &self.rng {
            rng.dump_rand32(out);
            true
        } else {
            false
        }
    }

    /// The underlying service (transfer path label, resident-table downloads for the gates).
    pub fn service(&self) -> &CudaSamplerService<PinnedTransfer> {
        &self.service
    }

    /// Mutable service access (diagnostics and gate downloads).
    pub fn service_mut(&mut self) -> &mut CudaSamplerService<PinnedTransfer> {
        &mut self.service
    }
}

fn run_series(
    service: &mut CudaSamplerService<PinnedTransfer>,
    data: &ExpertModeData,
    state: &mut VmcOptimizationState,
    rng: &mut impl SamplingRng,
    timing: &mut dyn SamplerTiming,
    reducer: &dyn Reducer,
) -> Result<(WalkerRun, LockstepStats), String> {
    let opts = LockstepOptions {
        spin_us: Some(SPIN_US),
        ..LockstepOptions::default()
    };
    if timing.enabled() {
        let mut timer = CTimer::<true>::new();
        timer.diagnostics = timing.diagnostics();
        let result =
            run_lockstep_real_one_inline(data, state, rng, reducer, service, opts, &mut timer);
        timing.merge_elapsed(&timer.elapsed_ns);
        result
    } else {
        let mut timer = CTimer::<false>::new();
        timer.diagnostics = timing.diagnostics();
        run_lockstep_real_one_inline(data, state, rng, reducer, service, opts, &mut timer)
    }
}

impl SamplerStages for CudaSamplerStages {
    fn begin_run(&mut self) {
        self.rng = None;
    }

    fn label(&self) -> String {
        format!(
            "cuda-sampler:{}:rng={}",
            self.service.transfer_name(),
            if self.xorwow { "xorwow" } else { "sfmt" }
        )
    }

    fn provider(&self) -> String {
        "device-resident lock-step sampler (issue #434) with pinned-async transfers".to_string()
    }

    fn sample_real(
        &mut self,
        data: &ExpertModeData,
        state: &mut VmcOptimizationState,
        rng: &mut Sfmt19937Rng,
        timing: &mut dyn SamplerTiming,
        reducer: &dyn Reducer,
    ) -> Result<SampleStats, StageError> {
        // The device chain is the serial production chain: MPI QP splits and collective sampling
        // sums have no device implementation. Hard error, never a CPU fallback.
        mvmc_core::sampler_backend::check_serial_reducer(reducer)?;
        let profile = profile_enabled() || self.service.profile;
        if profile {
            self.service.profile = true;
        }
        let t0 = std::time::Instant::now();
        if self.xorwow && self.rng.is_none() {
            // Honor the initialized SFMT seed when no explicit GPU seed was supplied.
            // Consume two u32 words, avoiding SFMT's alignment-sensitive gen_rand64.
            let seed = self.seed.unwrap_or_else(|| {
                u64::from(rng.gen_rand32()) | (u64::from(rng.gen_rand32()) << 32)
            });
            self.rng = Some(
                crate::xorwow::GpuSamplingRng::new(self.device, seed, 4096)
                    .map_err(StageError::Failed)?,
            );
        }
        // The walker's sampler sections (candidate, projection counters, the proposal round, the
        // acceptance, the recompute) are recorded like the C-order path's; the service thread's
        // attribution is printed when the profile variable is set.
        let result = if let Some(device_rng) = &mut self.rng {
            run_series(&mut self.service, data, state, device_rng, timing, reducer)
        } else {
            run_series(&mut self.service, data, state, rng, timing, reducer)
        };
        if profile {
            let wall = t0.elapsed().as_secs_f64();
            let now = self.service.timings;
            let timings = crate::device_sampler::ServiceTimings {
                stage_s: now.stage_s - self.last_timings.stage_s,
                upload_s: now.upload_s - self.last_timings.upload_s,
                launch_s: now.launch_s - self.last_timings.launch_s,
                wait_s: now.wait_s - self.last_timings.wait_s,
                slow_wait_s: now.slow_wait_s - self.last_timings.slow_wait_s,
                passes: now.passes - self.last_timings.passes,
                slow_batches: now.slow_batches - self.last_timings.slow_batches,
                dev_accept_s: now.dev_accept_s - self.last_timings.dev_accept_s,
                dev_propose_s: now.dev_propose_s - self.last_timings.dev_propose_s,
                dev_slow_s: now.dev_slow_s - self.last_timings.dev_slow_s,
            };
            self.last_timings = now;
            let stats: &LockstepStats = match &result {
                Ok((_, stats)) => stats,
                Err(_) => &LockstepStats::default(),
            };
            eprintln!(
                "sampler-profile wall_s={wall:.6} passes={} proposals={} accepts={} begins={} \
                 recomputes={} recv_s={:.6} round_s={:.6} deliver_s={:.6} | service stage_s={:.6} \
                 upload_s={:.6} launch_s={:.6} wait_s={:.6} slow_wait_s={:.6} slow_batches={} \
                 dev_accept_s={:.6} dev_propose_s={:.6} dev_slow_s={:.6}",
                stats.passes,
                stats.proposals,
                stats.accepts,
                stats.begins,
                stats.recomputes,
                stats.recv_s,
                stats.round_s,
                stats.deliver_s,
                timings.stage_s,
                timings.upload_s,
                timings.launch_s,
                timings.wait_s,
                timings.slow_wait_s,
                timings.slow_batches,
                timings.dev_accept_s,
                timings.dev_propose_s,
                timings.dev_slow_s,
            );
        }
        let (run, _stats) = result.map_err(StageError::Failed)?;
        run.stats.map_err(StageError::Failed)
    }
}
