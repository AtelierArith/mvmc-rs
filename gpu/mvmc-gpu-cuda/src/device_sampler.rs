//! CUDA [`DeviceService`] of the device-resident lock-step sampler (issue #434).
//!
//! The service owns one cudarc context, the resident tables (inverses `[W][NQP][n*n]` in mVMC's
//! `invM = -X^-1` convention, Pfaffians, configurations, Slater tables) and **two lanes**, each
//! a CUDA stream with its own staging buffers, pinned transfer buffers and scratch planes:
//!
//! * the **fast lane** runs the accepted-move updates and the proposal ratios of every round and
//!   is waited for (one upload, a few kernels, one download of the `NQP`-vector per proposal);
//! * the **slow lanes** run the initial table construction and the periodic recomputation
//!   (Slater planes assembled on the device, then the batched Pfaffian/inverse of #423,
//!   `pfaffian_batched.cu`). It is asynchronous: its latency (hundreds of microseconds, the
//!   serial LTL^T steps) overlaps the fast lane's rounds instead of stalling every walker. The
//!   slow lane waits (on the device, by event) for the round's accepted moves.
//!
//! There are [`SLOW_LANES`] slow lanes, one slow batch in flight each; a recompute batch of a
//! few walkers occupies only a few blocks of the device, so several run concurrently. When all
//! lanes are busy the oldest batch is completed first.
//!
//! Compute and transfers both go through cudarc, instead of tenferro's raw session: the resident
//! buffers are owned by a single allocator and the transfer helper of #432 ([`crate::transfer`])
//! addresses them directly. The kernels are plain NVRTC sources, as in `pfaffian.rs`, and could
//! be moved to `CudaExecSession::with_raw` unchanged (the gate test
//! `pinned_helper_raw_copies_work_on_tenferro_raw_session_addresses` checks that the `*_raw`
//! copies work on addresses owned by a tenferro raw session).
//! [`TransferPath`] is the seam for the host-device copies: [`PinnedTransfer`] (pinned pool and
//! asynchronous copies of #432) and [`PageableTransfer`] (cudarc pageable copies, the baseline).

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::Instant;

use cudarc::driver::{
    CudaContext, CudaEvent, CudaFunction, CudaSlice, DevicePtr, DevicePtrMut, LaunchConfig,
    PushKernelArg,
};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};
use mvmc_core::device_sampler::{DeviceService, FastBatch, Geometry, SlowBatch, SlowReply};
use mvmc_core::state::SlaterElmFlat;

use crate::transfer::{PinnedBuf, PinnedKind, PinnedPool, TransferEvent, TransferStream};

const SRC_PFAFFIAN: &str = include_str!("pfaffian_batched.cu");
const SRC_SAMPLER: &str = include_str!("sampler_kernels.cu");

fn cu(e: impl std::fmt::Debug) -> String {
    format!("{e:?}")
}

// ---------------------------------------------------------------------------------------------
// Transfer path
// ---------------------------------------------------------------------------------------------

/// Host-device copies of one lane. Uploads are stream-ordered before the kernels that follow;
/// a download is started (enqueued) and later finished (data on the host).
pub trait TransferPath {
    /// Label for reports.
    fn name(&self) -> &str;
    /// Copy `data` into the front of `dev` (in stream order with the following kernels).
    fn upload(&mut self, dev: &mut CudaSlice<f64>, data: &[f64]) -> Result<(), String>;
    /// Enqueue the copy of the first `n` doubles of `dev` to the host.
    fn download_start(&mut self, dev: &CudaSlice<f64>, n: usize) -> Result<(), String>;
    /// Finish the download started last: with `block` wait for it; otherwise return `false`
    /// when it has not completed. On `true`, `out` holds the data.
    fn download_finish(&mut self, block: bool, out: &mut Vec<f64>) -> Result<bool, String>;
}

/// Pinned pool + asynchronous copies (the #432 helper).
pub struct PinnedTransfer {
    ts: Arc<TransferStream>,
    pool: PinnedPool,
    up: Option<PinnedBuf>,
    down: Option<PinnedBuf>,
    event: Option<TransferEvent>,
    down_n: usize,
}

impl PinnedTransfer {
    /// Transfer path on the stream of `ts`.
    pub fn new(ctx: &Arc<CudaContext>, ts: Arc<TransferStream>) -> Self {
        Self {
            ts,
            pool: PinnedPool::new(ctx),
            up: None,
            down: None,
            event: None,
            down_n: 0,
        }
    }
}

impl TransferPath for PinnedTransfer {
    fn name(&self) -> &str {
        "pinned-async"
    }

    fn upload(&mut self, dev: &mut CudaSlice<f64>, data: &[f64]) -> Result<(), String> {
        let bytes = data.len() * 8;
        let need = bytes.max(64).next_power_of_two();
        if self.up.as_ref().is_none_or(|b| b.len_bytes() < need) {
            if let Some(old) = self.up.take() {
                self.pool.put(old);
            }
            self.up = Some(
                self.pool
                    .take(need, PinnedKind::for_upload(bytes))
                    .map_err(cu)?,
            );
        }
        let host = self.up.as_mut().expect("pinned upload buffer");
        host.as_f64_mut()[..data.len()].copy_from_slice(data);
        let (ptr, _g) = dev.device_ptr_mut(self.ts.stream());
        // SAFETY: `host` is pinned and stays alive and unmodified until the next upload of this
        // lane, which happens after the kernels that depend on this copy and the download behind
        // them completed (the lane's stream is in-order and its owner waits for the download
        // before the next round); `dev` holds at least `bytes` (callers size the staging buffer).
        let pending = unsafe {
            self.ts
                .upload_raw(host.as_f64().as_ptr().cast::<u8>(), ptr, bytes)
        }
        .map_err(cu)?;
        // SAFETY: see above; the event is dropped without being waited for.
        drop(unsafe { pending.detach() });
        Ok(())
    }

    fn download_start(&mut self, dev: &CudaSlice<f64>, n: usize) -> Result<(), String> {
        let bytes = n * 8;
        let need = bytes.max(64).next_power_of_two();
        if self.down.as_ref().is_none_or(|b| b.len_bytes() < need) {
            if let Some(old) = self.down.take() {
                self.pool.put(old);
            }
            self.down = Some(self.pool.take(need, PinnedKind::Cached).map_err(cu)?);
        }
        let host = self.down.as_mut().expect("pinned download buffer");
        let (ptr, _g) = dev.device_ptr(self.ts.stream());
        // SAFETY: `host` is pinned, exclusively ours until `download_finish` observed the event.
        let pending = unsafe {
            self.ts
                .download_raw(ptr, host.as_f64_mut().as_mut_ptr().cast(), bytes)
        }
        .map_err(cu)?;
        // SAFETY: the returned event is stored and waited for before the buffer is read.
        self.event = Some(unsafe { pending.detach() });
        self.down_n = n;
        Ok(())
    }

    fn download_finish(&mut self, block: bool, out: &mut Vec<f64>) -> Result<bool, String> {
        let Some(ev) = self.event.as_ref() else {
            return Err("no download in flight".to_string());
        };
        if block {
            ev.wait_host().map_err(cu)?;
        } else if !ev.is_complete() {
            return Ok(false);
        }
        self.event = None;
        let host = self.down.as_ref().expect("pinned download buffer");
        out.clear();
        out.extend_from_slice(&host.as_f64()[..self.down_n]);
        Ok(true)
    }
}

/// Pageable cudarc copies (the baseline; each copy blocks the host).
pub struct PageableTransfer {
    ts: Arc<TransferStream>,
    buf: Vec<f64>,
}

impl PageableTransfer {
    /// Transfer path on the stream of `ts`.
    pub fn new(ts: Arc<TransferStream>) -> Self {
        Self {
            ts,
            buf: Vec::new(),
        }
    }
}

impl TransferPath for PageableTransfer {
    fn name(&self) -> &str {
        "pageable"
    }
    fn upload(&mut self, dev: &mut CudaSlice<f64>, data: &[f64]) -> Result<(), String> {
        let mut view = dev.slice_mut(..data.len());
        self.ts.stream().memcpy_htod(data, &mut view).map_err(cu)
    }
    fn download_start(&mut self, dev: &CudaSlice<f64>, n: usize) -> Result<(), String> {
        self.buf.clear();
        self.buf.resize(n, 0.0);
        let view = dev.slice(..n);
        self.ts
            .stream()
            .memcpy_dtoh(&view, &mut self.buf)
            .map_err(cu)?;
        self.ts.stream().synchronize().map_err(cu)
    }
    fn download_finish(&mut self, _block: bool, out: &mut Vec<f64>) -> Result<bool, String> {
        out.clear();
        out.extend_from_slice(&self.buf);
        Ok(true)
    }
}

// ---------------------------------------------------------------------------------------------
// The service
// ---------------------------------------------------------------------------------------------

/// Accumulated wall and device time of the service phases (seconds).
#[derive(Debug, Clone, Copy, Default)]
pub struct ServiceTimings {
    /// Host staging of the request buffers.
    pub stage_s: f64,
    /// Upload enqueue (pinned) or copy (pageable).
    pub upload_s: f64,
    /// Kernel launch enqueue.
    pub launch_s: f64,
    /// Waiting for the fast-lane download (device execution not hidden by the host).
    pub wait_s: f64,
    /// Time the host blocked on the slow lane.
    pub slow_wait_s: f64,
    /// Rounds.
    pub passes: usize,
    /// Slow batches completed.
    pub slow_batches: usize,
    /// Device time of the accepted-move kernels (CUDA events).
    pub dev_accept_s: f64,
    /// Device time of the proposal kernel.
    pub dev_propose_s: f64,
    /// Device time of the slow lane's pipelines (assemble + Pfaffian + commit).
    pub dev_slow_s: f64,
}

struct Scratch {
    planes: usize,
    a: CudaSlice<f64>,
    w: CudaSlice<f64>,
    out: CudaSlice<f64>,
    pf: CudaSlice<f64>,
    st: CudaSlice<i32>,
}

struct Resident {
    geom: Geometry,
    ele: CudaSlice<i32>,
    inv: CudaSlice<f64>,
    pf: CudaSlice<f64>,
    pfnew: CudaSlice<f64>,
    pend: CudaSlice<i32>,
    slater_ptrs: CudaSlice<u64>,
    slater_host: Vec<u64>,
}

struct Funcs {
    propose: CudaFunction,
    accept: CudaFunction,
    commit_moves: CudaFunction,
    assemble: CudaFunction,
    commit_recompute: CudaFunction,
    pfinv: CudaFunction,
}

struct Lane<T: TransferPath> {
    ts: Arc<TransferStream>,
    xfer: T,
    dstage: Option<CudaSlice<f64>>,
    dout: Option<CudaSlice<f64>>,
    scratch: Option<Scratch>,
}

struct SlowInflight {
    seq: u64,
    begins: Vec<usize>,
    recomputes: Vec<usize>,
    nq: usize,
    o_bfail: usize,
    o_rfail: usize,
    o_rpf: usize,
    out_len: usize,
    ev0: Option<CudaEvent>,
    ev1: Option<CudaEvent>,
}

/// Number of concurrent slow lanes (recompute batches in flight).
pub const SLOW_LANES: usize = 8;

struct SlowLane<T: TransferPath> {
    lane: Lane<T>,
    inflight: Option<SlowInflight>,
    out: Vec<f64>,
}

/// CUDA implementation of [`DeviceService`].
pub struct CudaSamplerService<T: TransferPath> {
    ctx: Arc<CudaContext>,
    funcs: Funcs,
    fast: Lane<T>,
    slows: Vec<SlowLane<T>>,
    seq: u64,
    res: Option<Resident>,
    slater: HashMap<u64, CudaSlice<f64>>,
    completed: VecDeque<SlowReply>,
    fast_out: Vec<f64>,
    /// Phase timings of all rounds so far.
    pub timings: ServiceTimings,
    /// Record CUDA events around the phases for the device-time fields of [`ServiceTimings`]
    /// (costs a few microseconds per round).
    pub profile: bool,
}

impl CudaSamplerService<PinnedTransfer> {
    /// Service on `device` with the pinned asynchronous transfer path.
    pub fn new_pinned(device: usize) -> Result<Self, String> {
        Self::with_path(device, PinnedTransfer::new)
    }
}

impl CudaSamplerService<PageableTransfer> {
    /// Service on `device` with pageable copies.
    pub fn new_pageable(device: usize) -> Result<Self, String> {
        Self::with_path(device, |_, ts| PageableTransfer::new(ts))
    }
}

fn ints_to_f64_bits(ints: &[i32]) -> Vec<f64> {
    let mut v = vec![0.0f64; ints.len().div_ceil(2)];
    // SAFETY: f64 storage reinterpreted as i32 pairs; both are plain data.
    let dst = unsafe { std::slice::from_raw_parts_mut(v.as_mut_ptr().cast::<i32>(), v.len() * 2) };
    dst[..ints.len()].copy_from_slice(ints);
    v
}

fn record(ctx: &Arc<CudaContext>, ts: &TransferStream) -> Result<CudaEvent, String> {
    let e = ctx
        .new_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))
        .map_err(cu)?;
    e.record(ts.stream()).map_err(cu)?;
    Ok(e)
}

impl<T: TransferPath> CudaSamplerService<T> {
    /// Compile the kernels and build the service; `make` creates the transfer path of a lane.
    pub fn with_path(
        device: usize,
        mut make: impl FnMut(&Arc<CudaContext>, Arc<TransferStream>) -> T,
    ) -> Result<Self, String> {
        let ctx = CudaContext::new(device).map_err(cu)?;
        // SAFETY: every cross-stream dependency is ordered by this module with events.
        unsafe { crate::transfer::disable_event_tracking(&ctx) };
        let src = format!("{SRC_PFAFFIAN}\n{SRC_SAMPLER}");
        let ptx = compile_ptx_with_opts(
            &src,
            CompileOptions {
                fmad: Some(false),
                arch: Some("compute_80"),
                ..Default::default()
            },
        )
        .map_err(|e| format!("NVRTC: {e:?}"))?;
        let module = ctx.load_module(ptx).map_err(cu)?;
        let f = |name: &str| module.load_function(name).map_err(cu);
        let funcs = Funcs {
            propose: f("k_propose")?,
            accept: f("k_accept")?,
            commit_moves: f("k_commit_moves")?,
            assemble: f("k_assemble")?,
            commit_recompute: f("k_commit_recompute")?,
            pfinv: f("pfinv_f64")?,
        };
        let mut lane = |ctx: &Arc<CudaContext>| -> Result<Lane<T>, String> {
            let ts = Arc::new(TransferStream::new(ctx).map_err(cu)?);
            let xfer = make(ctx, ts.clone());
            Ok(Lane {
                ts,
                xfer,
                dstage: None,
                dout: None,
                scratch: None,
            })
        };
        let fast = lane(&ctx)?;
        let mut slows = Vec::new();
        for _ in 0..SLOW_LANES {
            slows.push(SlowLane {
                lane: lane(&ctx)?,
                inflight: None,
                out: Vec::new(),
            });
        }
        Ok(Self {
            ctx,
            funcs,
            fast,
            slows,
            seq: 0,
            res: None,
            slater: HashMap::new(),
            completed: VecDeque::new(),
            fast_out: Vec::new(),
            timings: ServiceTimings::default(),
            profile: false,
        })
    }

    /// Name of the transfer path.
    pub fn transfer_name(&self) -> &str {
        self.fast.xfer.name()
    }

    /// The cudarc context.
    pub fn context(&self) -> &Arc<CudaContext> {
        &self.ctx
    }

    /// Copy walker `w`'s resident inverse table (mVMC convention) and Pfaffians to the host.
    pub fn download_walker(&mut self, w: usize) -> Result<(Vec<f64>, Vec<f64>), String> {
        for i in 0..self.slows.len() {
            self.finish_slow(i, true)?;
        }
        let res = self.res.as_ref().ok_or("service not prepared")?;
        let nq = res.geom.n_qp;
        let nn = res.geom.n_size() * res.geom.n_size();
        let s = self.fast.ts.stream();
        s.synchronize().map_err(cu)?;
        let inv = s
            .clone_dtoh(&res.inv.slice(w * nq * nn..(w + 1) * nq * nn))
            .map_err(cu)?;
        let pf = s
            .clone_dtoh(&res.pf.slice(w * nq..(w + 1) * nq))
            .map_err(cu)?;
        s.synchronize().map_err(cu)?;
        Ok((inv, pf))
    }

    /// Complete the batch in slow lane `i` and queue its reply; with `block == false` returns
    /// `Ok(false)` when it has not completed (or the lane is idle).
    fn finish_slow(&mut self, i: usize, block: bool) -> Result<bool, String> {
        let sl = &mut self.slows[i];
        if sl.inflight.is_none() {
            return Ok(false);
        }
        let t = Instant::now();
        let mut out = std::mem::take(&mut sl.out);
        let done = sl.lane.xfer.download_finish(block, &mut out)?;
        if !done {
            sl.out = out;
            return Ok(false);
        }
        if block {
            self.timings.slow_wait_s += t.elapsed().as_secs_f64();
        }
        let inf = sl.inflight.take().expect("in flight");
        let nq = inf.nq;
        debug_assert_eq!(out.len(), inf.out_len);
        let mut reply = SlowReply::default();
        for (k, &w) in inf.begins.iter().enumerate() {
            if out[inf.o_bfail + k] != 0.0 {
                sl.out = out;
                return Err(format!("initial recompute failed for walker {w}"));
            }
            reply.begins.push(w);
        }
        for (k, &w) in inf.recomputes.iter().enumerate() {
            let failed = out[inf.o_rfail + k] != 0.0;
            let pf = if failed {
                Vec::new()
            } else {
                out[inf.o_rpf + k * nq..inf.o_rpf + (k + 1) * nq].to_vec()
            };
            reply.recomputes.push((w, failed, pf));
        }
        sl.out = out;
        self.timings.slow_batches += 1;
        if let (Some(a), Some(b)) = (&inf.ev0, &inf.ev1) {
            self.timings.dev_slow_s += f64::from(a.elapsed_ms(b).unwrap_or(0.0)) * 1e-3;
        }
        self.completed.push_back(reply);
        Ok(true)
    }

    /// A free slow lane; when all are busy, completes the oldest batch (blocking) first.
    fn free_slow_lane(&mut self) -> Result<usize, String> {
        if let Some(i) = self.slows.iter().position(|s| s.inflight.is_none()) {
            return Ok(i);
        }
        let oldest = self
            .slows
            .iter()
            .enumerate()
            .min_by_key(|(_, s)| s.inflight.as_ref().map_or(u64::MAX, |f| f.seq))
            .map(|(i, _)| i)
            .expect("at least one slow lane");
        self.finish_slow(oldest, true)?;
        Ok(oldest)
    }
}

/// Grow-only device buffer of at least `len` doubles.
fn ensure_f64(
    slot: &mut Option<CudaSlice<f64>>,
    ts: &TransferStream,
    len: usize,
) -> Result<(), String> {
    if slot.as_ref().is_none_or(|b| b.len() < len) {
        *slot = Some(
            ts.stream()
                .alloc_zeros(len.max(64).next_power_of_two())
                .map_err(cu)?,
        );
    }
    Ok(())
}

/// Launch assemble + batched Pfaffian + commit for `items` configurations staged on `lane`.
/// Offsets are in staging ints (walker/ele), staging doubles (pf_set) and output doubles.
#[allow(clippy::too_many_arguments)]
fn launch_recompute<T: TransferPath>(
    lane: &mut Lane<T>,
    res: &mut Resident,
    funcs: &Funcs,
    items: usize,
    pf_set_off: Option<usize>,
    walker_off: usize,
    ele_off: usize,
    out_pf_off: usize,
    out_failed_off: usize,
) -> Result<(), String> {
    let g = res.geom;
    let (n, ne, ns, nq) = (g.n_size(), g.n_elec, g.n_site, g.n_qp);
    let nn = n * n;
    let planes = items * nq;
    let stream = lane.ts.stream().clone();
    if lane.scratch.as_ref().is_none_or(|s| s.planes < planes) {
        let cap = planes.max(8).next_power_of_two();
        lane.scratch = None;
        lane.scratch = Some(Scratch {
            planes: cap,
            a: stream.alloc_zeros(cap * nn).map_err(cu)?,
            w: stream.alloc_zeros(cap * nn).map_err(cu)?,
            out: stream.alloc_zeros(cap * nn).map_err(cu)?,
            pf: stream.alloc_zeros(cap).map_err(cu)?,
            st: stream.alloc_zeros(cap).map_err(cu)?,
        });
    }
    let sc = lane.scratch.as_mut().expect("scratch");
    let stage = lane.dstage.as_ref().expect("stage buffer");
    let dout = lane.dout.as_mut().expect("out buffer");
    let base = {
        let (p, _g) = stage.device_ptr(&stream);
        p
    };
    let walker_ptr = base + (walker_off * 4) as u64;
    let ele_ptr = base + (ele_off * 4) as u64;
    let pf_set_ptr: u64 = pf_set_off.map_or(0, |o| base + (o * 8) as u64);
    let (n_i, ne_i, ns_i, nq_i) = (n as i32, ne as i32, ns as i32, nq as i32);
    let (off_pf, off_failed) = (out_pf_off as i32, out_failed_off as i32);

    let mut b = stream.launch_builder(&funcs.assemble);
    b.arg(&walker_ptr)
        .arg(&ele_ptr)
        .arg(&res.slater_ptrs)
        .arg(&mut sc.a)
        .arg(&n_i)
        .arg(&ne_i)
        .arg(&ns_i)
        .arg(&nq_i);
    // SAFETY: kernel ABI (const int*, const int*, const double* const*, double*, int x4); the
    // pointers address live device buffers sized for `items` planes.
    unsafe {
        b.launch(LaunchConfig {
            grid_dim: (planes as u32, 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        })
    }
    .map_err(cu)?;
    let nt = n.next_power_of_two().clamp(32, 256);
    let shared = (n * 8 + nt * 8 + nt * 4 + n * 8) as u32;
    let mut b = stream.launch_builder(&funcs.pfinv);
    b.arg(&mut sc.a)
        .arg(&mut sc.w)
        .arg(&mut sc.out)
        .arg(&mut sc.pf)
        .arg(&mut sc.st)
        .arg(&n_i);
    // SAFETY: kernel ABI (double* x4, int*, int); one block per plane.
    unsafe {
        b.launch(LaunchConfig {
            grid_dim: (planes as u32, 1, 1),
            block_dim: (nt as u32, 1, 1),
            shared_mem_bytes: shared,
        })
    }
    .map_err(cu)?;
    let mut b = stream.launch_builder(&funcs.commit_recompute);
    b.arg(&walker_ptr)
        .arg(&ele_ptr)
        .arg(&sc.out)
        .arg(&sc.pf)
        .arg(&sc.st)
        .arg(&pf_set_ptr)
        .arg(&mut res.inv)
        .arg(&mut res.pf)
        .arg(&mut res.ele)
        .arg(dout)
        .arg(&off_pf)
        .arg(&off_failed)
        .arg(&n_i)
        .arg(&nq_i);
    // SAFETY: kernel ABI as declared in sampler_kernels.cu; one block per item.
    unsafe {
        b.launch(LaunchConfig {
            grid_dim: (items as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        })
    }
    .map_err(cu)?;
    Ok(())
}

impl<T: TransferPath> DeviceService for CudaSamplerService<T> {
    fn register_slater(
        &mut self,
        key: u64,
        _geom: Geometry,
        slater: &SlaterElmFlat<f64>,
    ) -> Result<(), String> {
        let st = self.fast.ts.stream();
        let dev = st.clone_htod(slater.as_slice()).map_err(cu)?;
        st.synchronize().map_err(cu)?;
        self.slater.insert(key, dev);
        Ok(())
    }

    fn prepare(&mut self, geom: Geometry, walkers: usize) -> Result<(), String> {
        let st = self.fast.ts.stream();
        let nn = geom.n_size() * geom.n_size();
        self.res = Some(Resident {
            geom,
            ele: st.alloc_zeros(walkers * geom.n_size()).map_err(cu)?,
            inv: st.alloc_zeros(walkers * geom.n_qp * nn).map_err(cu)?,
            pf: st.alloc_zeros(walkers * geom.n_qp).map_err(cu)?,
            pfnew: st.alloc_zeros(walkers * geom.n_qp).map_err(cu)?,
            pend: st.alloc_zeros(walkers * 8).map_err(cu)?,
            slater_ptrs: st.alloc_zeros(walkers).map_err(cu)?,
            slater_host: vec![0; walkers],
        });
        st.synchronize().map_err(cu)?;
        Ok(())
    }

    fn round(&mut self, fast: &FastBatch, slow: &SlowBatch) -> Result<Vec<f64>, String> {
        let t_stage = Instant::now();
        let g = self.res.as_ref().ok_or("service not prepared")?.geom;
        let (n, nq) = (g.n_size(), g.n_qp);
        let (na, np) = (fast.accepts.len(), fast.proposes.len());
        let (ne_i, ns_i, nq_i, n_i) = (g.n_elec as i32, g.n_site as i32, nq as i32, n as i32);

        // ---- fast lane staging: ints [accepts | proposes] --------------------------------------
        let mut ints: Vec<i32> = fast.accepts.iter().map(|&w| w as i32).collect();
        let off_acc = 0usize;
        let off_prop = (ints.len() + 1) & !1;
        ints.resize(off_prop, 0);
        for p in &fast.proposes {
            let b = p.b;
            ints.extend([
                p.walker as i32,
                p.a.slot as i32,
                p.a.site as i32,
                i32::from(p.a.spin),
                b.map_or(-1, |m| m.slot as i32),
                b.map_or(0, |m| m.site as i32),
                b.map_or(0, |m| i32::from(m.spin)),
                0,
            ]);
        }
        if ints.len() % 2 == 1 {
            ints.push(0);
        }
        let stage = ints_to_f64_bits(&ints);
        let out_len = (np * nq).max(1);
        ensure_f64(&mut self.fast.dstage, &self.fast.ts, stage.len().max(1))?;
        ensure_f64(&mut self.fast.dout, &self.fast.ts, out_len)?;
        self.timings.stage_s += t_stage.elapsed().as_secs_f64();

        let t_up = Instant::now();
        if !stage.is_empty() {
            let dst = self.fast.dstage.as_mut().expect("stage");
            self.fast.xfer.upload(dst, &stage)?;
        }
        self.timings.upload_s += t_up.elapsed().as_secs_f64();

        // ---- fast lane: accepted moves --------------------------------------------------------
        let t_launch = Instant::now();
        let stream = self.fast.ts.stream().clone();
        let e_start = if self.profile {
            Some(record(&self.ctx, &self.fast.ts)?)
        } else {
            None
        };
        if na > 0 {
            let res = self.res.as_mut().expect("res");
            let stage_dev = self.fast.dstage.as_ref().expect("stage");
            let (base, _g) = stage_dev.device_ptr(&stream);
            let acc_ptr = base + (off_acc * 4) as u64;
            let count = na as i32;
            let shared = (((n + 1) & !1) * 4 + (4 * n + 16) * 8) as u32;
            let nt = n.next_power_of_two().clamp(32, 256) as u32;
            let mut b = stream.launch_builder(&self.funcs.accept);
            b.arg(&acc_ptr)
                .arg(&res.slater_ptrs)
                .arg(&res.ele)
                .arg(&mut res.inv)
                .arg(&mut res.pf)
                .arg(&res.pfnew)
                .arg(&res.pend)
                .arg(&n_i)
                .arg(&ne_i)
                .arg(&ns_i)
                .arg(&nq_i);
            // SAFETY: kernel ABI of k_accept; one block per (accepted walker, qp).
            unsafe {
                b.launch(LaunchConfig {
                    grid_dim: ((na * nq) as u32, 1, 1),
                    block_dim: (nt, 1, 1),
                    shared_mem_bytes: shared,
                })
            }
            .map_err(cu)?;
            let mut b = stream.launch_builder(&self.funcs.commit_moves);
            b.arg(&acc_ptr)
                .arg(&count)
                .arg(&mut res.ele)
                .arg(&res.pend)
                .arg(&n_i);
            // SAFETY: kernel ABI of k_commit_moves; one thread per accepted walker.
            unsafe {
                b.launch(LaunchConfig {
                    grid_dim: ((na as u32).div_ceil(64), 1, 1),
                    block_dim: (64, 1, 1),
                    shared_mem_bytes: 0,
                })
            }
            .map_err(cu)?;
        }
        let e_acc = if self.profile || slow.walkers() > 0 {
            Some(record(&self.ctx, &self.fast.ts)?)
        } else {
            None
        };
        self.timings.launch_s += t_launch.elapsed().as_secs_f64();

        // ---- slow lane: begin / recompute, asynchronous ---------------------------------------
        if slow.walkers() > 0 {
            let li = self.free_slow_lane()?;
            self.seq += 1;
            let seq = self.seq;
            let t_stage = Instant::now();
            let (nb, nr) = (slow.begins.len(), slow.recomputes.len());
            let mut ints: Vec<i32> = Vec::new();
            let off_bw = ints.len();
            ints.extend(slow.begins.iter().map(|b| b.walker as i32));
            let off_be = ints.len();
            for b in &slow.begins {
                ints.extend(b.ele_idx.iter().map(|&v| v as i32));
            }
            let off_rw = ints.len();
            ints.extend(slow.recomputes.iter().map(|r| r.walker as i32));
            let off_re = ints.len();
            for r in &slow.recomputes {
                ints.extend(r.ele_idx.iter().map(|&v| v as i32));
            }
            if ints.len() % 2 == 1 {
                ints.push(0);
            }
            let mut sstage = ints_to_f64_bits(&ints);
            let off_pfset = sstage.len();
            for b in &slow.begins {
                sstage.extend_from_slice(&b.pf);
            }
            let o_bfail = 0usize;
            let o_bpf = o_bfail + nb;
            let o_rfail = o_bpf + nb * nq;
            let o_rpf = o_rfail + nr;
            let s_out_len = (o_rpf + nr * nq).max(1);
            {
                let l = &mut self.slows[li].lane;
                ensure_f64(&mut l.dstage, &l.ts, sstage.len().max(1))?;
                ensure_f64(&mut l.dout, &l.ts, s_out_len)?;
            }
            // slater pointers of the walkers begun in this batch
            let mut ptrs_dirty = false;
            {
                let res = self.res.as_mut().expect("res");
                for b in &slow.begins {
                    let dev = self
                        .slater
                        .get(&b.slater)
                        .ok_or("unregistered Slater table")?;
                    let (p, _g) = dev.device_ptr(self.slows[li].lane.ts.stream());
                    res.slater_host[b.walker] = p;
                    ptrs_dirty = true;
                }
            }
            self.timings.stage_s += t_stage.elapsed().as_secs_f64();

            let t_up = Instant::now();
            // the slow lane waits (on the device) for this round's accepted moves
            if let Some(e) = e_acc.as_ref() {
                self.slows[li].lane.ts.stream().wait(e).map_err(cu)?;
            }
            if ptrs_dirty {
                let res = self.res.as_mut().expect("res");
                self.slows[li]
                    .lane
                    .ts
                    .stream()
                    .memcpy_htod(&res.slater_host, &mut res.slater_ptrs)
                    .map_err(cu)?;
            }
            {
                let l = &mut self.slows[li].lane;
                let dst = l.dstage.as_mut().expect("stage");
                l.xfer.upload(dst, &sstage)?;
            }
            self.timings.upload_s += t_up.elapsed().as_secs_f64();

            let t_launch = Instant::now();
            let ev0 = if self.profile {
                Some(record(&self.ctx, &self.slows[li].lane.ts)?)
            } else {
                None
            };
            {
                let res = self.res.as_mut().expect("res");
                if nb > 0 {
                    launch_recompute(
                        &mut self.slows[li].lane,
                        res,
                        &self.funcs,
                        nb,
                        Some(off_pfset),
                        off_bw,
                        off_be,
                        o_bpf,
                        o_bfail,
                    )?;
                }
                if nr > 0 {
                    launch_recompute(
                        &mut self.slows[li].lane,
                        res,
                        &self.funcs,
                        nr,
                        None,
                        off_rw,
                        off_re,
                        o_rpf,
                        o_rfail,
                    )?;
                }
            }
            let ev1 = if self.profile {
                Some(record(&self.ctx, &self.slows[li].lane.ts)?)
            } else {
                None
            };
            {
                let l = &mut self.slows[li].lane;
                let dout = l.dout.as_ref().expect("slow out");
                l.xfer.download_start(dout, s_out_len)?;
            }
            self.timings.launch_s += t_launch.elapsed().as_secs_f64();
            self.slows[li].inflight = Some(SlowInflight {
                seq,
                begins: slow.begins.iter().map(|b| b.walker).collect(),
                recomputes: slow.recomputes.iter().map(|r| r.walker).collect(),
                nq,
                o_bfail,
                o_rfail,
                o_rpf,
                out_len: s_out_len,
                ev0,
                ev1,
            });
        }

        // ---- fast lane: proposals -------------------------------------------------------------
        let mut pf_out = Vec::new();
        let ms = |a: &CudaEvent, b: &CudaEvent| f64::from(a.elapsed_ms(b).unwrap_or(0.0)) * 1e-3;
        if np > 0 {
            let t_launch = Instant::now();
            let res = self.res.as_mut().expect("res");
            let stage_dev = self.fast.dstage.as_ref().expect("stage");
            let dout = self.fast.dout.as_mut().expect("out");
            let (base, _g) = stage_dev.device_ptr(&stream);
            let prop_ptr = base + (off_prop * 4) as u64;
            let count = np as i32;
            let out_off = 0i32;
            let nt = n.next_power_of_two().clamp(32, 256) as u32;
            let shared = (((n + 1) & !1) * 4 + (8 * n + 8) * 8) as u32;
            let mut b = stream.launch_builder(&self.funcs.propose);
            b.arg(&prop_ptr)
                .arg(&count)
                .arg(&res.slater_ptrs)
                .arg(&res.ele)
                .arg(&res.inv)
                .arg(&res.pf)
                .arg(&mut res.pfnew)
                .arg(&mut res.pend)
                .arg(dout)
                .arg(&out_off)
                .arg(&n_i)
                .arg(&ne_i)
                .arg(&ns_i)
                .arg(&nq_i);
            // SAFETY: kernel ABI of k_propose; one block per (proposal, qp).
            unsafe {
                b.launch(LaunchConfig {
                    grid_dim: ((np * nq) as u32, 1, 1),
                    block_dim: (nt, 1, 1),
                    shared_mem_bytes: shared,
                })
            }
            .map_err(cu)?;
            let e_end = if self.profile {
                Some(record(&self.ctx, &self.fast.ts)?)
            } else {
                None
            };
            let dout = self.fast.dout.as_ref().expect("out");
            self.fast.xfer.download_start(dout, out_len)?;
            self.timings.launch_s += t_launch.elapsed().as_secs_f64();
            let t_wait = Instant::now();
            let mut out = std::mem::take(&mut self.fast_out);
            self.fast.xfer.download_finish(true, &mut out)?;
            self.timings.wait_s += t_wait.elapsed().as_secs_f64();
            pf_out = out[..np * nq].to_vec();
            self.fast_out = out;
            if let (Some(a), Some(b), Some(c)) = (&e_start, &e_acc, &e_end) {
                self.timings.dev_accept_s += ms(a, b);
                self.timings.dev_propose_s += ms(b, c);
            }
        }
        self.timings.passes += 1;
        Ok(pf_out)
    }

    fn poll_slow(&mut self, block: bool) -> Result<Vec<SlowReply>, String> {
        let mut any_inflight = false;
        let mut progressed = false;
        for i in 0..self.slows.len() {
            if self.slows[i].inflight.is_some() {
                any_inflight = true;
                progressed |= self.finish_slow(i, false)?;
            }
        }
        if block && any_inflight && !progressed && self.completed.is_empty() {
            let oldest = self
                .slows
                .iter()
                .enumerate()
                .filter(|(_, s)| s.inflight.is_some())
                .min_by_key(|(_, s)| s.inflight.as_ref().map_or(u64::MAX, |f| f.seq))
                .map(|(i, _)| i);
            if let Some(i) = oldest {
                self.finish_slow(i, true)?;
            }
        }
        Ok(self.completed.drain(..).collect())
    }
}
