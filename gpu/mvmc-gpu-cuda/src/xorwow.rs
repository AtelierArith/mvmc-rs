//! Device-resident XORWOW state and reusable word buffers. No cuRAND dependency.
//! Output is draw-major, lane-minor. Lane `i` uses `curand_init(seed+i,0,0)`.
//! This is not cuRAND host-library stream ordering or its subsequence spacing.
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::compile_ptx;
use mvmc_core::sampling::rng::{SamplingRng, XorwowRng};
use std::sync::Arc;

fn cu(e: impl std::fmt::Debug) -> String {
    format!("{e:?}")
}

/// State is initialized once and advances on every `generate` call.
pub struct DeviceXorwow {
    stream: Arc<CudaStream>,
    kernel: CudaFunction,
    state: CudaSlice<u32>,
    words: CudaSlice<u32>,
    lanes: usize,
    capacity: usize,
    generated: usize,
}

impl DeviceXorwow {
    pub fn new(
        device: usize,
        seed: u64,
        lanes: usize,
        words_per_lane: usize,
    ) -> Result<Self, String> {
        if lanes == 0
            || words_per_lane == 0
            || lanes > i32::MAX as usize
            || words_per_lane > i32::MAX as usize
        {
            return Err("XORWOW dimensions must be positive and fit i32".into());
        }
        let size = lanes
            .checked_mul(words_per_lane)
            .ok_or("XORWOW buffer size overflow")?;
        let ctx = CudaContext::new(device).map_err(cu)?;
        let stream = ctx.default_stream();
        let module = ctx
            .load_module(compile_ptx(include_str!("xorwow.cu")).map_err(cu)?)
            .map_err(cu)?;
        let kernel = module.load_function("xorwow_words").map_err(cu)?;
        let states: Vec<u32> = (0..lanes)
            .flat_map(|i| XorwowRng::new(seed.wrapping_add(i as u64)).state())
            .collect();
        let state = stream.clone_htod(&states).map_err(cu)?;
        let words = stream.alloc_zeros(size).map_err(cu)?;
        Ok(Self {
            stream,
            kernel,
            state,
            words,
            lanes,
            capacity: words_per_lane,
            generated: 0,
        })
    }
    /// Enqueue generation; output and states remain on the device.
    pub fn generate(&mut self, words_per_lane: usize) -> Result<&CudaSlice<u32>, String> {
        if words_per_lane == 0 || words_per_lane > self.capacity {
            return Err("XORWOW request exceeds buffer capacity".into());
        }
        let lanes = self.lanes as i32;
        let n = words_per_lane as i32;
        let mut launch = self.stream.launch_builder(&self.kernel);
        launch
            .arg(&mut self.state)
            .arg(&mut self.words)
            .arg(&lanes)
            .arg(&n);
        unsafe { launch.launch(LaunchConfig::for_num_elems(self.lanes as u32)) }.map_err(cu)?;
        self.generated = words_per_lane * self.lanes;
        Ok(&self.words)
    }
    pub fn synchronize(&self) -> Result<(), String> {
        self.stream.synchronize().map_err(cu)
    }
    /// Explicit diagnostic / host-control boundary; generation itself performs no download.
    pub fn download(&self) -> Result<Vec<u32>, String> {
        self.stream
            .clone_dtoh(&self.words.slice(..self.generated))
            .map_err(cu)
    }
    /// Download into a reusable host buffer (no allocation once capacity suffices).
    pub fn download_into(&self, out: &mut Vec<u32>) -> Result<(), String> {
        out.resize(self.generated, 0);
        self.stream
            .memcpy_dtoh(&self.words.slice(..self.generated), out.as_mut_slice())
            .map_err(cu)
    }
    pub fn states(&self) -> Result<Vec<u32>, String> {
        self.stream.clone_dtoh(&self.state).map_err(cu)
    }
}

/// GPU-generated batched words for the current host-driven sampler. Buffers and state persist
/// between sampling series. This amortizes transfers; proposal control still runs on the host.
pub struct GpuSamplingRng {
    device: DeviceXorwow,
    buffer: Vec<u32>,
    cursor: usize,
    consumed: u128,
    batch: usize,
}
impl GpuSamplingRng {
    pub fn new(device: usize, seed: u64, batch: usize) -> Result<Self, String> {
        Ok(Self {
            device: DeviceXorwow::new(device, seed, 1, batch)?,
            buffer: Vec::with_capacity(batch),
            cursor: 0,
            consumed: 0,
            batch,
        })
    }
}
impl SamplingRng for GpuSamplingRng {
    fn gen_rand32(&mut self) -> u32 {
        if self.cursor == self.buffer.len() {
            self.device
                .generate(self.batch)
                .expect("XORWOW device generation failed");
            self.device
                .download_into(&mut self.buffer)
                .expect("XORWOW word download failed");
            self.cursor = 0;
        }
        let word = self.buffer[self.cursor];
        self.cursor += 1;
        self.consumed += 1;
        word
    }
    fn words_consumed(&self) -> u128 {
        self.consumed
    }
    fn dump_rand32(&self, out: &mut [u32]) {
        let take = out.len().min(self.buffer.len() - self.cursor);
        out[..take].copy_from_slice(&self.buffer[self.cursor..self.cursor + take]);
        if take < out.len() {
            let state: [u32; 6] = self
                .device
                .states()
                .expect("XORWOW state download failed")
                .try_into()
                .expect("one stream");
            XorwowRng::from_state(state).dump_rand32(&mut out[take..]);
        }
    }
}
