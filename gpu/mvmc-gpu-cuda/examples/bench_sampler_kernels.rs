//! Kernel-only timing of the fast lane of the device-resident sampler (issue #517).
//!
//! Times the three fast-lane kernels (`k_accept`, `k_commit_moves`, `k_propose`) and one full
//! round emulation (staging upload + accept + commit + propose + result download through the
//! `PinnedTransfer` path), in the production launch geometry of `CudaSamplerService::round`:
//! one block per (item, qp), `n = 2 * Ne`, `NQP = 8`.
//!
//! Unlike the `ServiceTimings` events of the service, these measurements are taken on an idle
//! device with the stream drained before every repetition, so the event elapsed is the kernel
//! latency and the host span is the launch/enqueue path cost.
//!
//! ```text
//! cd gpu/mvmc-gpu-cuda
//! cargo run --release --example bench_sampler_kernels [n ...]
//! ```

use std::sync::Arc;
use std::time::Instant;

use cudarc::driver::{
    sys::CUevent_flags, CudaContext, CudaEvent, CudaFunction, CudaSlice, DevicePtr, LaunchConfig,
    PushKernelArg,
};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};

use mvmc_gpu_cuda::device_sampler::{PinnedTransfer, TransferPath};
use mvmc_gpu_cuda::transfer::TransferStream;

const SRC_SAMPLER: &str = include_str!("../src/sampler_kernels.cu");

/// Experimental `k_accept` variants (issue #517 sampler-hop profiling): the same arithmetic as
/// `k_accept` with a division-free row loop, and a two-phase split (vector phase + update phase)
/// that spreads one plane over `split` blocks. Only the hopping branch is implemented; the
/// benchmark drives it with one-electron pending moves.
const SRC_VARIANTS: &str = r#"
#define SLT(tab, qp, row, col, n2) ((tab)[(((size_t)(qp) * (n2)) + (row)) * (n2) + (col)])
__device__ inline int cand_site2(const int* ele_w, const int* mv, int j) {
  if (j == mv[0]) return mv[1];
  if (mv[3] >= 0 && j == mv[3]) return mv[4];
  return ele_w[j];
}

// row-loop variant of the hop update: one block per (item, qp), one warp per row group,
// consecutive lanes along the row (fully coalesced, no integer division).
extern "C" __global__ void k_accept_v2(const int* acc_w, const double* const* slater_tab,
                                       const int* ele, double* inv, double* pf,
                                       const double* pfnew, const int* pend,
                                       int n, int ne, int nsite, int nqp) {
  extern __shared__ char smem_raw[];
  int* rs = reinterpret_cast<int*>(smem_raw);
  double* vec = reinterpret_cast<double*>(rs + ((n + 1) & ~1));
  double* v3 = vec;
  double* v1 = vec + n;
  double* v2 = vec + 2 * n;
  const int item = blockIdx.x / nqp, qp = blockIdx.x % nqp;
  const int w = acc_w[item];
  const int* pd = pend + (size_t)w * 8;
  const int mv[6] = {pd[0], pd[1], pd[2], pd[3], pd[4], pd[5]};
  const int* ele_w = ele + (size_t)w * n;
  const double* tab = slater_tab[w];
  const int n2 = 2 * nsite;
  double* invq = inv + ((size_t)w * nqp + qp) * (size_t)n * n;
  const int tid = threadIdx.x, nt = blockDim.x;
  for (int j = tid; j < n; j += nt) rs[j] = cand_site2(ele_w, mv, j) + (j >= ne ? nsite : 0);
  __syncthreads();
  const int msa = (mv[3] >= 0) ? mv[3] : mv[0];
  const int site = (mv[3] >= 0) ? mv[4] : mv[1];
  const int rsa = site + (msa >= ne ? nsite : 0);
  for (int j = tid; j < n; j += nt) v3[j] = SLT(tab, qp, rsa, rs[j], n2);
  __syncthreads();
  for (int i = tid; i < n; i += nt) {
    double acc = 0.0;
#pragma unroll 8
    for (int j = 0; j < n; ++j) acc += -invq[(size_t)j * n + i] * v3[j];
    v1[i] = acc;
  }
  __syncthreads();
  const double tmp = v1[msa];
  const double inv_vec1_a = -1.0 / tmp;
  for (int i = tid; i < n; i += nt) v2[i] = invq[(size_t)msa * n + i] * inv_vec1_a;
  __syncthreads();
  const int warp = tid >> 5, lane = tid & 31, nwarp = nt >> 5;
  for (int i = warp; i < n; i += nwarp) {
    const double vi1 = v1[i], vi2 = v2[i];
    const int rowoff = i * n;
    for (int j = lane; j < n; j += 32) {
      const int e = rowoff + j;
      double x = invq[e];
      x += vi1 * v2[j] - v1[j] * vi2;
      if (j == msa) x -= vi2;
      if (i == msa) x += v2[j];
      invq[e] = x;
    }
  }
  if (tid == 0) pf[(size_t)w * nqp + qp] = pfnew[(size_t)w * nqp + qp];
}

// phase A: gather + v1 rows (split over rows), v1 written to global
extern "C" __global__ void k_accept_vec(const int* acc_w, const double* const* slater_tab,
                                        const int* ele, const double* inv, const int* pend,
                                        double* vec_out, int n, int ne, int nsite, int nqp,
                                        int split) {
  extern __shared__ char smem_raw[];
  int* rs = reinterpret_cast<int*>(smem_raw);
  double* v3 = reinterpret_cast<double*>(rs + ((n + 1) & ~1));
  const int plane = blockIdx.x / split, s = blockIdx.x % split;
  const int item = plane / nqp, qp = plane % nqp;
  const int w = acc_w[item];
  const int* pd = pend + (size_t)w * 8;
  const int mv[6] = {pd[0], pd[1], pd[2], pd[3], pd[4], pd[5]};
  const int* ele_w = ele + (size_t)w * n;
  const double* tab = slater_tab[w];
  const int n2 = 2 * nsite;
  const double* invq = inv + ((size_t)w * nqp + qp) * (size_t)n * n;
  const int tid = threadIdx.x, nt = blockDim.x;
  for (int j = tid; j < n; j += nt) rs[j] = cand_site2(ele_w, mv, j) + (j >= ne ? nsite : 0);
  __syncthreads();
  const int msa = (mv[3] >= 0) ? mv[3] : mv[0];
  const int site = (mv[3] >= 0) ? mv[4] : mv[1];
  const int rsa = site + (msa >= ne ? nsite : 0);
  for (int j = tid; j < n; j += nt) v3[j] = SLT(tab, qp, rsa, rs[j], n2);
  __syncthreads();
  const int row_lo = (int)((long)s * n / split), row_hi = (int)((long)(s + 1) * n / split);
  double* v1 = vec_out + (size_t)plane * (size_t)n;
  for (int i = row_lo + tid; i < row_hi; i += nt) {
    double acc = 0.0;
#pragma unroll 8
    for (int j = 0; j < n; ++j) acc += -invq[(size_t)j * n + i] * v3[j];
    v1[i] = acc;
  }
}

// phase B: v2 from the global v1[msa] and the rank-two update of the split's rows
extern "C" __global__ void k_accept_upd(const int* acc_w, double* inv, const int* pend,
                                        const double* vec_in, double* pf, const double* pfnew,
                                        int n, int ne, int nsite, int nqp, int split) {
  extern __shared__ char smem_raw[];
  double* sh = reinterpret_cast<double*>(smem_raw);
  double* v1 = sh;         // n
  double* v2 = sh + n;     // n
  const int plane = blockIdx.x / split, s = blockIdx.x % split;
  const int item = plane / nqp, qp = plane % nqp;
  const int w = acc_w[item];
  const int* pd = pend + (size_t)w * 8;
  const int msa = (pd[3] >= 0) ? pd[3] : pd[0];
  const int tid = threadIdx.x, nt = blockDim.x;
  const double* vec1 = vec_in + (size_t)plane * (size_t)n;
  double* invq = inv + ((size_t)w * nqp + qp) * (size_t)n * n;
  const double inv_vec1_a = -1.0 / vec1[msa];
  for (int i = tid; i < n; i += nt) v1[i] = vec1[i];
  __syncthreads();
  for (int i = tid; i < n; i += nt) v2[i] = invq[(size_t)msa * n + i] * inv_vec1_a;
  __syncthreads();
  const int row_lo = (int)((long)s * n / split), row_hi = (int)((long)(s + 1) * n / split);
  const int warp = tid >> 5, lane = tid & 31, nwarp = nt >> 5;
  for (int i = row_lo + warp; i < row_hi; i += nwarp) {
    const double vi1 = v1[i], vi2 = v2[i];
    const int rowoff = i * n;
    for (int j = lane; j < n; j += 32) {
      const int e = rowoff + j;
      double x = invq[e];
      x += vi1 * v2[j] - v1[j] * vi2;
      if (j == msa) x -= vi2;
      if (i == msa) x += v2[j];
      invq[e] = x;
    }
  }
  if (tid == 0 && s == 0) pf[(size_t)w * nqp + qp] = pfnew[(size_t)w * nqp + qp];
}
"#;

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

/// Deterministic LCG.
struct Lcg(u64);
impl Lcg {
    fn uniform(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    }
}

struct Engine {
    ctx: Arc<CudaContext>,
    propose: CudaFunction,
    accept: CudaFunction,
    commit: CudaFunction,
    accept_v2: CudaFunction,
    accept_vec: CudaFunction,
    accept_upd: CudaFunction,
}

impl Engine {
    fn new() -> Self {
        let ctx = CudaContext::new(0).expect("device");
        let ptx = compile_ptx_with_opts(
            SRC_SAMPLER,
            CompileOptions {
                fmad: Some(false),
                arch: Some("compute_80"),
                ..Default::default()
            },
        )
        .expect("NVRTC");
        let module = ctx.load_module(ptx).expect("module");
        let vptx = compile_ptx_with_opts(
            SRC_VARIANTS,
            CompileOptions {
                fmad: Some(false),
                arch: Some("compute_80"),
                ..Default::default()
            },
        )
        .expect("NVRTC variants");
        let vmodule = ctx.load_module(vptx).expect("variant module");
        Self {
            ctx,
            propose: module.load_function("k_propose").expect("k_propose"),
            accept: module.load_function("k_accept").expect("k_accept"),
            commit: module
                .load_function("k_commit_moves")
                .expect("k_commit_moves"),
            accept_v2: vmodule.load_function("k_accept_v2").expect("k_accept_v2"),
            accept_vec: vmodule.load_function("k_accept_vec").expect("k_accept_vec"),
            accept_upd: vmodule.load_function("k_accept_upd").expect("k_accept_upd"),
        }
    }
}

struct Case {
    n: usize,
    ne: usize,
    nsite: usize,
    nqp: usize,
    ctx: Arc<CudaContext>,
    slater: CudaSlice<f64>,
    slater_ptrs: CudaSlice<u64>,
    ele: CudaSlice<i32>,
    inv: CudaSlice<f64>,
    pf: CudaSlice<f64>,
    pfnew: CudaSlice<f64>,
    pend: CudaSlice<i32>,
    dstage: CudaSlice<f64>,
    dout: CudaSlice<f64>,
    acc: CudaSlice<i32>,
    vecs: CudaSlice<f64>,
    inv_host: Vec<f64>,
}

impl Case {
    fn new(ctx: &Arc<CudaContext>, n: usize, nsite: usize, nqp: usize) -> Self {
        let s = ctx.default_stream();
        let ne = n / 2;
        let n2 = 2 * nsite;
        let mut rng = Lcg(0x5170_0000 + n as u64);
        let host_slater: Vec<f64> = (0..nqp * n2 * n2).map(|_| rng.uniform()).collect();
        let slater = s.clone_htod(&host_slater).expect("slater");
        let ptr = {
            let (p, _g) = slater.device_ptr(&s);
            s.clone_htod(&vec![p; 8]).expect("slater ptrs")
        };
        let ele: Vec<i32> = (0..n).map(|j| ((j / 2) + 1) as i32).collect();
        let inv_host: Vec<f64> = (0..nqp * n * n).map(|_| rng.uniform()).collect();
        let pf_host: Vec<f64> = (0..nqp).map(|_| rng.uniform()).collect();
        let pend_host: Vec<i32> = vec![3, 7, 0, -1, 0, 0, 0, 0];
        Self {
            n,
            ne,
            nsite,
            nqp,
            ctx: ctx.clone(),
            slater,
            slater_ptrs: ptr,
            ele: s.clone_htod(&ele).expect("ele"),
            inv: s.clone_htod(&inv_host).expect("inv"),
            pf: s.clone_htod(&pf_host).expect("pf"),
            pfnew: s.alloc_zeros(nqp).expect("pfnew"),
            pend: s.clone_htod(&pend_host).expect("pend"),
            dstage: s.alloc_zeros(1024).expect("stage"),
            dout: s.alloc_zeros(1024).expect("out"),
            acc: s.clone_htod(&[0i32; 8]).expect("acc"),
            vecs: s.alloc_zeros(8 * n).expect("vecs"),
            inv_host,
        }
    }

    fn slater_shared(&self) -> u32 {
        (((self.n + 1) & !1) * 4 + (4 * self.n + 16) * 8) as u32
    }

    fn event(&self) -> CudaEvent {
        self.ctx
            .new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
            .expect("event")
    }

    fn time_launch(
        &mut self,
        engine: &Engine,
        reps: usize,
        mut launch: impl FnMut(&mut Self, &Engine),
    ) -> (f64, f64) {
        let s = self.ctx.default_stream();
        launch(self, engine);
        s.synchronize().expect("sync");
        let mut dev = Vec::new();
        let mut host = Vec::new();
        for _ in 0..reps {
            let e0 = self.event();
            e0.record(&s).expect("record");
            let t0 = Instant::now();
            launch(self, engine);
            host.push(t0.elapsed().as_secs_f64());
            let e1 = self.event();
            e1.record(&s).expect("record");
            s.synchronize().expect("sync");
            dev.push(f64::from(e0.elapsed_ms(&e1).unwrap_or(0.0)) * 1e-3);
        }
        (median(dev), median(host))
    }

    fn launch_propose(&mut self, engine: &Engine, np: usize, profile: bool) {
        let s = self.ctx.default_stream();
        let n_i = self.n as i32;
        let (ne_i, ns_i, nq_i) = (self.ne as i32, self.nsite as i32, self.nqp as i32);
        let count = np as i32;
        let nt = self.n.next_power_of_two().clamp(32, 256) as u32;
        let shared = (((self.n + 1) & !1) * 4 + (8 * self.n + 8) * 8) as u32;
        let out_off = 0i32;
        let mut b = s.launch_builder(&engine.propose);
        b.arg(&self.dstage)
            .arg(&count)
            .arg(&self.slater_ptrs)
            .arg(&self.ele)
            .arg(&self.inv)
            .arg(&self.pf)
            .arg(&mut self.pfnew)
            .arg(&mut self.pend)
            .arg(&mut self.dout)
            .arg(&out_off)
            .arg(&n_i)
            .arg(&ne_i)
            .arg(&ns_i)
            .arg(&nq_i);
        // SAFETY: kernel ABI of k_propose.
        unsafe {
            b.launch(LaunchConfig {
                grid_dim: ((np * self.nqp) as u32, 1, 1),
                block_dim: (nt, 1, 1),
                shared_mem_bytes: shared,
            })
        }
        .expect("propose");
        if profile {
            let _ = &self.slater;
        }
    }

    fn launch_accept(&mut self, engine: &Engine, na: usize) {
        self.launch_accept_nt(engine, na, self.accept_block());
    }

    fn accept_block(&self) -> u32 {
        self.n.next_power_of_two().clamp(32, 256) as u32
    }

    fn launch_accept_nt(&mut self, engine: &Engine, na: usize, nt: u32) {
        let s = self.ctx.default_stream();
        let n_i = self.n as i32;
        let (ne_i, ns_i, nq_i) = (self.ne as i32, self.nsite as i32, self.nqp as i32);
        let shared = self.slater_shared();
        let mut b = s.launch_builder(&engine.accept);
        b.arg(&self.acc)
            .arg(&self.slater_ptrs)
            .arg(&self.ele)
            .arg(&mut self.inv)
            .arg(&mut self.pf)
            .arg(&self.pfnew)
            .arg(&self.pend)
            .arg(&n_i)
            .arg(&ne_i)
            .arg(&ns_i)
            .arg(&nq_i);
        // SAFETY: kernel ABI of k_accept.
        unsafe {
            b.launch(LaunchConfig {
                grid_dim: ((na * self.nqp) as u32, 1, 1),
                block_dim: (nt, 1, 1),
                shared_mem_bytes: shared,
            })
        }
        .expect("accept");
    }

    fn launch_commit(&mut self, engine: &Engine, na: usize) {
        let s = self.ctx.default_stream();
        let n_i = self.n as i32;
        let count = na as i32;
        let mut b = s.launch_builder(&engine.commit);
        b.arg(&self.acc)
            .arg(&count)
            .arg(&mut self.ele)
            .arg(&self.pend)
            .arg(&n_i);
        // SAFETY: kernel ABI of k_commit_moves.
        unsafe {
            b.launch(LaunchConfig {
                grid_dim: ((na as u32).div_ceil(64), 1, 1),
                block_dim: (64, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .expect("commit");
    }

    /// Division-free row-loop variant of the accept kernel (same arithmetic).
    fn launch_accept_v2(&mut self, engine: &Engine, na: usize, nt: u32) {
        let s = self.ctx.default_stream();
        let n_i = self.n as i32;
        let (ne_i, ns_i, nq_i) = (self.ne as i32, self.nsite as i32, self.nqp as i32);
        let shared = (((self.n + 1) & !1) * 4 + 3 * self.n * 8) as u32;
        let mut b = s.launch_builder(&engine.accept_v2);
        b.arg(&self.acc)
            .arg(&self.slater_ptrs)
            .arg(&self.ele)
            .arg(&mut self.inv)
            .arg(&mut self.pf)
            .arg(&self.pfnew)
            .arg(&self.pend)
            .arg(&n_i)
            .arg(&ne_i)
            .arg(&ns_i)
            .arg(&nq_i);
        // SAFETY: kernel ABI of k_accept_v2.
        unsafe {
            b.launch(LaunchConfig {
                grid_dim: ((na * self.nqp) as u32, 1, 1),
                block_dim: (nt, 1, 1),
                shared_mem_bytes: shared,
            })
        }
        .expect("accept_v2");
    }

    /// Two-phase accept: vector phase (split rows of the GEMV) + update phase.
    fn launch_accept_split(&mut self, engine: &Engine, na: usize, split: usize, nt: u32) {
        let s = self.ctx.default_stream();
        let n_i = self.n as i32;
        let (ne_i, ns_i, nq_i) = (self.ne as i32, self.nsite as i32, self.nqp as i32);
        let split_i = split as i32;
        let planes = na * self.nqp;
        {
            let shared = (((self.n + 1) & !1) * 4 + self.n * 8) as u32;
            let mut b = s.launch_builder(&engine.accept_vec);
            b.arg(&self.acc)
                .arg(&self.slater_ptrs)
                .arg(&self.ele)
                .arg(&self.inv)
                .arg(&self.pend)
                .arg(&mut self.vecs)
                .arg(&n_i)
                .arg(&ne_i)
                .arg(&ns_i)
                .arg(&nq_i)
                .arg(&split_i);
            // SAFETY: kernel ABI of k_accept_vec.
            unsafe {
                b.launch(LaunchConfig {
                    grid_dim: ((planes * split) as u32, 1, 1),
                    block_dim: (nt, 1, 1),
                    shared_mem_bytes: shared,
                })
            }
            .expect("accept_vec");
        }
        {
            let shared = (2 * self.n * 8) as u32;
            let mut b = s.launch_builder(&engine.accept_upd);
            b.arg(&self.acc)
                .arg(&mut self.inv)
                .arg(&self.pend)
                .arg(&self.vecs)
                .arg(&mut self.pf)
                .arg(&self.pfnew)
                .arg(&n_i)
                .arg(&ne_i)
                .arg(&ns_i)
                .arg(&nq_i)
                .arg(&split_i);
            // SAFETY: kernel ABI of k_accept_upd.
            unsafe {
                b.launch(LaunchConfig {
                    grid_dim: ((planes * split) as u32, 1, 1),
                    block_dim: (nt, 1, 1),
                    shared_mem_bytes: shared,
                })
            }
            .expect("accept_upd");
        }
    }

    /// Bit-level checksum of a device buffer (FNV-1a over the bits).
    fn checksum(v: &[f64]) -> u64 {
        let mut h = 0xcbf2_9ce4_8422_2325u64;
        for &x in v {
            h ^= x.to_bits();
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
            h ^= h >> 29;
        }
        h
    }

    /// One accept on a fresh copy of the resident inverse; returns the inverse as the host sees
    /// it (mVMC convention, no pad slots).
    fn accept_result(&mut self, engine: &Engine, run: impl Fn(&mut Self, &Engine)) -> Vec<f64> {
        let s = self.ctx.default_stream();
        {
            let mut view = self.inv.slice_mut(..self.inv_host.len());
            s.memcpy_htod(&self.inv_host, &mut view).expect("reset");
        }
        run(self, engine);
        s.synchronize().expect("sync");
        s.clone_dtoh(&self.inv).expect("download")
    }

    /// One full round emulation: pinned upload of the staging ints + accept + commit + propose +
    /// pinned download, exactly the service's fast lane for one walker.
    fn time_round(
        &mut self,
        engine: &Engine,
        reps: usize,
        na: usize,
        np: usize,
    ) -> (f64, f64, f64, f64, f64) {
        let s = self.ctx.default_stream();
        let ts = Arc::new(TransferStream::new(&self.ctx).expect("stream"));
        let mut xfer = PinnedTransfer::new(&self.ctx, ts.clone());
        let mut out = Vec::new();
        self.launch_propose(engine, np, false);
        s.synchronize().expect("sync");
        let mut dev = Vec::new();
        let mut host = Vec::new();
        let (mut t_up, mut t_launch, mut t_dl) = (0.0, 0.0, 0.0);
        for _ in 0..reps {
            let t0 = Instant::now();
            let stage = vec![0.0f64; 32];
            xfer.upload(&mut self.dstage, &stage).expect("upload");
            t_up += t0.elapsed().as_secs_f64();
            let e0 = self.event();
            e0.record(&s).expect("record");
            let t1 = Instant::now();
            if na > 0 {
                self.launch_accept(engine, na);
                self.launch_commit(engine, na);
            }
            self.launch_propose(engine, np, false);
            t_launch += t1.elapsed().as_secs_f64();
            let e1 = self.event();
            e1.record(&s).expect("record");
            let t2 = Instant::now();
            xfer.download_start(&self.dout, np * self.nqp)
                .expect("download_start");
            dev.push(f64::from(e0.elapsed_ms(&e1).unwrap_or(0.0)) * 1e-3);
            xfer.download_finish(true, &mut out).expect("finish");
            t_dl += t2.elapsed().as_secs_f64();
            host.push(t0.elapsed().as_secs_f64());
        }
        (
            median(dev),
            median(host),
            t_up / reps as f64,
            t_launch / reps as f64,
            t_dl / reps as f64,
        )
    }
}

fn main() {
    mvmc_gpu_cuda::install();
    let engine = Engine::new();
    let sizes: Vec<u64> = std::env::args()
        .skip(1)
        .filter_map(|a| a.parse().ok())
        .collect();
    let sizes: Vec<usize> = if sizes.is_empty() {
        vec![256, 448]
    } else {
        sizes.into_iter().map(|v| v as usize).collect()
    };
    println!(
        "# fast-lane kernels of the device sampler (issue #517); medians, microseconds; \
         deep idle device, stream drained per rep"
    );
    println!("n,kind,items,device_us,host_enqueue_us");
    for n in sizes {
        // Hubbard chain at half filling: Nsite = n, Ne = n / 2.
        let nsite = n;
        let mut case = Case::new(&engine.ctx, n, nsite, 8);
        for items in [1usize, 2, 4] {
            let (d, h) = case.time_launch(&engine, 50, |c, e| c.launch_accept(e, items));
            println!("{n},k_accept,{items},{:.2},{:.2}", d * 1e6, h * 1e6);
        }
        let nt = 512u32;
        let (d, h) = case.time_launch(&engine, 50, |c, e| c.launch_accept_nt(e, 1, nt));
        println!("{n},k_accept_nt{nt},1,{:.2},{:.2}", d * 1e6, h * 1e6);
        for nt in [256u32, 512, 1024] {
            let (d, h) = case.time_launch(&engine, 50, |c, e| c.launch_accept_v2(e, 1, nt));
            println!("{n},k_accept_v2_nt{nt},1,{:.2},{:.2}", d * 1e6, h * 1e6);
        }
        for (split, nt) in [
            (1usize, 256u32),
            (4, 256),
            (4, 512),
            (8, 256),
            (8, 512),
            (16, 256),
        ] {
            let (d, h) =
                case.time_launch(&engine, 50, |c, e| c.launch_accept_split(e, 1, split, nt));
            println!(
                "{n},k_accept_split{split}_nt{nt},1,{:.2},{:.2}",
                d * 1e6,
                h * 1e6
            );
        }
        let items = 1usize;
        let (d, h) = case.time_launch(&engine, 50, |c, e| c.launch_commit(e, items));
        println!("{n},k_commit_moves,{items},{:.2},{:.2}", d * 1e6, h * 1e6);
        let (d, h) = case.time_launch(&engine, 50, |c, e| c.launch_propose(e, items, true));
        println!("{n},k_propose,{items},{:.2},{:.2}", d * 1e6, h * 1e6);
        // bit-exactness of the variants: identical inverse (and Pfaffian) after one accept
        let reference = case.accept_result(&engine, |c, e| c.launch_accept(e, 1));
        let pf_ref = case.ctx.default_stream().clone_dtoh(&case.pf).expect("pf");
        macro_rules! check {
            ($name:expr, $run:expr) => {{
                let got = case.accept_result(&engine, $run);
                let max = got
                    .iter()
                    .zip(&reference)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f64, f64::max);
                let pf_got = case.ctx.default_stream().clone_dtoh(&case.pf).expect("pf");
                let pf_max = pf_got
                    .iter()
                    .zip(&pf_ref)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f64, f64::max);
                println!("{n},bitdiff_{},0,{max:.3e},{pf_max:.3e}", $name);
            }};
        }
        check!("accept_v2", |c: &mut Case, e: &Engine| c
            .launch_accept_v2(e, 1, 256));
        check!("split1", |c: &mut Case, e: &Engine| c
            .launch_accept_split(e, 1, 1, 256));
        check!("split8", |c: &mut Case, e: &Engine| c
            .launch_accept_split(e, 1, 8, 256));
        // exchange branch (two-electron pending move): bit checksum of the resident inverse
        {
            let s = case.ctx.default_stream();
            let exch: Vec<i32> = vec![3, 7, 0, 5, 11, 1, 0, 0];
            let dev = s.clone_htod(&exch).expect("pend");
            let old = s.clone_dtoh(&case.pend).expect("pend old");
            s.memcpy_dtod(&dev, &mut case.pend.slice_mut(..8))
                .expect("pend set");
            let inv = case.accept_result(&engine, |c, e| c.launch_accept(e, 1));
            println!(
                "{n},exchange_checksum,0,{:#x},oldpend={:?}",
                Case::checksum(&inv),
                old
            );
        }
        for (na, np) in [(0usize, 1usize), (1, 1)] {
            let (dev, host, up, launch, dl) = case.time_round(&engine, 30, na, np);
            println!("{n},round_hit{na},{np},{:.2},{:.2}", dev * 1e6, host * 1e6);
            println!(
                "{n},round_host_split,{np},{:.2},{:.2}",
                up * 1e6,
                launch * 1e6
            );
            println!("{n},round_download,{np},{:.2},0", dl * 1e6);
        }
    }
}
