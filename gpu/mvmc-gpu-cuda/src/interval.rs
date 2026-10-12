//! Device-resident interval sampler runner (issue #547).
//!
//! Drives the raw CUDA `k_interval_hop` kernel through the vendored
//! [`tenferro_ext::raw_exec`] helper: one `with_raw_exec` scope compiles (once) and launches
//! the kernel on tenferro's captured stream against resident [`DeviceBuffer`]s, so a whole
//! interval round runs on the device with a single host synchronization.
//!
//! One CUDA block per walker. Each walker advances from its own `hop_off[w]` until it reaches
//! `total_hops` or its accepted-move window passes `nsite`; the host then batches every walker
//! that requested a rebuild into one device `interval_assemble` + `pfinv_f64` + `interval_commit`
//! launch ([`IntervalSampler::rebuild`]) and relaunches. The recomputation stays on the device,
//! which is what makes walker batching pay: a single block is latency-bound.
//!
//! The kernel's proposal/acceptance contract is in `interval_kernels.cu`. Only the real,
//! non-FSZ, ordinary-hopping mode without local spins is represented.

use tenferro_ext::raw_exec::{with_raw_exec, Arg, DeviceBuffer};
use tenferro_tensor::BackendSession;

/// The NVRTC module: the batched Pfaffian (`pfinv_f64`), the interval kernel and the rebuild
/// helpers, compiled once per process/thread.
pub const INTERVAL_MODULE: &str = concat!(
    include_str!("pfaffian_batched.cu"),
    "\n",
    include_str!("interval_kernels.cu"),
    "\n",
    include_str!("interval_rebuild.cu"),
);

/// Kernels loaded from [`INTERVAL_MODULE`].
pub const INTERVAL_KERNELS: &[&str] = &[
    "k_interval_hop",
    "interval_assemble",
    "interval_commit",
    "pfinv_f64",
];

/// Geometry shared by every interval call of one series.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IntervalGeometry {
    /// Independent walkers (CUDA blocks).
    pub walkers: usize,
    /// Pfaffian matrix side (`2 * ne`).
    pub n: usize,
    /// Electrons per spin.
    pub ne: usize,
    /// Lattice sites.
    pub nsite: usize,
    /// QP planes.
    pub nqp: usize,
    /// Gutzwiller parameter count (leading projection block).
    pub ngutz: usize,
    /// Total projection parameter count.
    pub nproj: usize,
}

impl IntervalGeometry {
    /// `2 * nsite`.
    pub fn n2(&self) -> usize {
        2 * self.nsite
    }

    /// Dynamic shared-memory bytes of one interval block.
    pub fn shared_bytes(&self) -> u32 {
        let rs = ((self.n + 1) & !1) * 4;
        // Ratio scratch is reused by the update after the acceptance barrier.
        let scratch = (self.block() as usize / 32)
            .min(self.nqp)
            .max(3 * self.qp_groups())
            * self.n;
        let doubles = (self.nqp + 1 + scratch) * 8;
        (rs + doubles + 8 * 4) as u32
    }

    /// Independent QP updates per block, each using a whole number of warps.
    /// Tile the scratch to stay within CUDA's default 48 KiB shared-memory limit.
    pub fn qp_groups(&self) -> usize {
        let fixed = ((self.n + 1) & !1) * 4 + (self.nqp + 1) * 8 + 8 * 4;
        let mut groups = 1;
        while groups * 2 <= self.nqp
            && groups * 2 <= self.block() as usize / 32
            && fixed + 3 * groups * 2 * self.n * 8 <= 48 * 1024
        {
            groups *= 2;
        }
        groups
    }

    /// Threads per block, covering rows and allowing a warp per QP, `[32, 256]`.
    pub fn block(&self) -> u32 {
        self.n
            .max(self.nqp.saturating_mul(32))
            .clamp(32, 256)
            .next_power_of_two() as u32
    }
}

/// Proposal/word-consumption contract. Production uses CStream; JuliaTape is an
/// explicitly separate five-word reference fixture contract.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProposalContract {
    /// C raw modulo destination retries and conditional acceptance draws.
    CStream,
    /// Julia GPU's five-word tape with empty-destination rank selection.
    JuliaTape,
}

impl ProposalContract {
    /// Fixed preview capacity. A C preview holds at least one worst-case proposal:
    /// two source/spin words, Nsite^2+2 destination attempts (the last rejects),
    /// or Nsite^2+1 destinations plus one acceptance word. This guarantees progress
    /// when an incomplete proposal is rolled back and retried after refill.
    pub fn words_per_walker(self, geom: IntervalGeometry, hops: usize) -> Result<usize, String> {
        let words = match self {
            Self::JuliaTape => hops.checked_mul(5),
            Self::CStream => geom
                .nsite
                .checked_mul(geom.nsite)
                .and_then(|n| n.checked_add(4))
                .zip(hops.checked_mul(8))
                .map(|(minimum, typical)| minimum.max(typical)),
        }
        .ok_or("interval tape size overflow")?;
        if words > i32::MAX as usize {
            return Err("interval tape exceeds CUDA index range".into());
        }
        Ok(words)
    }
}

/// Why a device pass stopped advancing a walker.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IntervalStop {
    /// The requested block is complete.
    Complete,
    /// Recompute this walker's resident tables, then continue on the same preview.
    Rebuild,
    /// Replace the preview starting at the reported consumed word count.
    TapeExhausted,
}

/// Progress of one walker in a single device launch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IntervalProgress {
    pub hops_done: i32,
    pub stop: IntervalStop,
    pub accepts: i32,
}

/// Host state uploaded by [`IntervalSampler::prepare`].
pub struct IntervalHost<'a> {
    /// Configuration `[walkers][n]` (site index per electron slot).
    pub ele: &'a [i32],
    /// mVMC inverse `[walkers][nqp][n*n]` (`invM = -X^-1`).
    pub inv: &'a [f64],
    /// Pfaffians `[walkers][nqp]`.
    pub pf: &'a [f64],
    /// Slater table `[nqp][n2][n2]`.
    pub slater: &'a [f64],
    /// QP weights `[nqp]`.
    pub weights: &'a [f64],
    /// Gutzwiller parameter index per site `[nsite]` (`-1` = none).
    pub gutz: &'a [i32],
    /// Jastrow parameter index per (a, b) upper-triangle pair `[nsite*nsite]` (`-1` = none).
    pub jast: &'a [i32],
    /// Projection parameter values `[nproj]`.
    pub params: &'a [f64],
}

/// Host-visible results of all walkers, in walker-major order. The inverse remains resident.
pub struct IntervalResults {
    /// Electron sites `[walkers][n]`.
    pub ele: Vec<i32>,
    /// Pfaffians `[walkers][nqp]`.
    pub pf: Vec<f64>,
    /// Accepted moves since each walker's last rebuild.
    pub aw: Vec<i32>,
    /// Cached logarithm of the absolute overlap, per walker.
    pub logip: Vec<f64>,
    /// Words consumed from the current preview, per walker (not lifetime totals).
    pub words: Vec<i32>,
}

/// Resident device state of one interval series.
pub struct IntervalSampler {
    geom: IntervalGeometry,
    total_hops: usize,
    proposal: ProposalContract,
    tape_words: usize,
    word_off: DeviceBuffer,
    ele: DeviceBuffer,
    occ: DeviceBuffer,
    inv: DeviceBuffer,
    pf: DeviceBuffer,
    aw: DeviceBuffer,
    logip: DeviceBuffer,
    slater: DeviceBuffer,
    weights: DeviceBuffer,
    gutz: DeviceBuffer,
    jast: DeviceBuffer,
    params: DeviceBuffer,
    tape: DeviceBuffer,
    hop_off: DeviceBuffer,
    hops_done: DeviceBuffer,
    rebuild: DeviceBuffer,
    accepts: DeviceBuffer,
    // Rebuild scratch (sized for all walkers).
    rec_walker: DeviceBuffer,
    planes: DeviceBuffer,
    pfinv_w: DeviceBuffer,
    pfinv_out: DeviceBuffer,
    pfinv_pf: DeviceBuffer,
    pfinv_st: DeviceBuffer,
}

fn f64_log_ip(pf: &[f64], weights: &[f64]) -> f64 {
    let mut ip = 0.0f64;
    for (w, p) in weights.iter().zip(pf) {
        ip += w * p;
    }
    ip.abs().ln()
}

fn occupancy(ele: &[i32], geom: &IntervalGeometry) -> Vec<i32> {
    let mut occ = vec![0i32; geom.walkers * 2 * geom.nsite];
    for w in 0..geom.walkers {
        for spin in 0..2usize {
            for j in 0..geom.ne {
                let site = ele[w * geom.n + j + spin * geom.ne] as usize;
                occ[w * 2 * geom.nsite + site + spin * geom.nsite] = 1;
            }
        }
    }
    occ
}

/// Occupancy (`ele_num`, `[site] up then [site] down`) of one walker's configuration.
fn walker_occupancy(ele: &[i32], geom: &IntervalGeometry) -> Vec<i32> {
    let mut occ = vec![0i32; 2 * geom.nsite];
    for spin in 0..2usize {
        for j in 0..geom.ne {
            let site = ele[j + spin * geom.ne] as usize;
            occ[site + spin * geom.nsite] = 1;
        }
    }
    occ
}

impl IntervalSampler {
    /// Allocate resident buffers and upload `host` with an explicit proposal contract.
    /// Julia tapes are interleaved by hop; C previews are contiguous per walker.
    pub fn prepare<B: BackendSession + ?Sized>(
        backend: &mut B,
        geom: IntervalGeometry,
        host: &IntervalHost<'_>,
        total_hops: usize,
        proposal: ProposalContract,
    ) -> Result<Self, String> {
        Self::prepare_buffers(backend, geom, host, total_hops, proposal, true)
    }

    /// Allocate an initial configuration whose tables will be factored on the device.
    pub(crate) fn prepare_for_rebuild<B: BackendSession + ?Sized>(
        backend: &mut B,
        geom: IntervalGeometry,
        host: &IntervalHost<'_>,
        total_hops: usize,
        proposal: ProposalContract,
    ) -> Result<Self, String> {
        Self::prepare_buffers(backend, geom, host, total_hops, proposal, false)
    }

    fn prepare_buffers<B: BackendSession + ?Sized>(
        backend: &mut B,
        geom: IntervalGeometry,
        host: &IntervalHost<'_>,
        total_hops: usize,
        proposal: ProposalContract,
        upload_tables: bool,
    ) -> Result<Self, String> {
        if geom.ne == 0 || geom.ne > geom.nsite || geom.n != 2 * geom.ne || geom.nqp == 0 {
            return Err("unsupported interval geometry".into());
        }
        let tape_words = proposal.words_per_walker(geom, total_hops)?;
        let (w, n, nsite, nqp) = (geom.walkers, geom.n, geom.nsite, geom.nqp);
        let n2 = geom.n2();
        let nplanes = w * nqp;
        let occ_host = occupancy(host.ele, &geom);
        let logip_host: Vec<f64> = if upload_tables {
            (0..w)
                .map(|k| f64_log_ip(&host.pf[k * nqp..(k + 1) * nqp], host.weights))
                .collect()
        } else {
            vec![0.0; w]
        };
        let zeros_i32 = vec![0i32; w];

        with_raw_exec(backend, "mvmc.interval.prepare", |exec| {
            let ele = exec.alloc_bytes(w * n * 4)?;
            let occ = exec.alloc_bytes(w * 2 * nsite * 4)?;
            let inv = exec.alloc_bytes(w * nqp * n * n * 8)?;
            let pf = exec.alloc_bytes(w * nqp * 8)?;
            let aw = exec.alloc_bytes(w * 4)?;
            let logip = exec.alloc_bytes(w * 8)?;
            let slater = exec.alloc_bytes(nqp * n2 * n2 * 8)?;
            let weights = exec.alloc_bytes(nqp * 8)?;
            let gutz = exec.alloc_bytes(nsite.max(1) * 4)?;
            let jast = exec.alloc_bytes((nsite * nsite).max(1) * 4)?;
            let params = exec.alloc_bytes(geom.nproj.max(1) * 8)?;
            let tape = exec.alloc_bytes((w * tape_words * 4).max(1))?;
            let word_off = exec.alloc_bytes(w.max(1) * 4)?;
            let hop_off = exec.alloc_bytes(w.max(1) * 4)?;
            let hops_done = exec.alloc_bytes(w.max(1) * 4)?;
            let rebuild = exec.alloc_bytes(w.max(1) * 4)?;
            let accepts = exec.alloc_bytes(w.max(1) * 4)?;
            let rec_walker = exec.alloc_bytes(w.max(1) * 4)?;
            let planes_nn = nplanes.max(1) * n * n;
            let plane_buf = exec.alloc_bytes(planes_nn * 8)?;
            let pfinv_w = exec.alloc_bytes(planes_nn * 8)?;
            let pfinv_out = exec.alloc_bytes(planes_nn * 8)?;
            let pfinv_pf = exec.alloc_bytes(nplanes.max(1) * 8)?;
            let pfinv_st = exec.alloc_bytes(nplanes.max(1) * 4)?;
            unsafe {
                exec.upload(exec.ptr(&ele)?, host.ele)?;
                exec.upload(exec.ptr(&occ)?, &occ_host)?;
                if upload_tables {
                    exec.upload(exec.ptr(&inv)?, host.inv)?;
                    exec.upload(exec.ptr(&pf)?, host.pf)?;
                } else {
                    exec.zero_bytes(exec.ptr(&inv)?, w * nqp * n * n * 8)?;
                    exec.zero_bytes(exec.ptr(&pf)?, w * nqp * 8)?;
                }
                exec.upload(exec.ptr(&aw)?, &zeros_i32)?;
                exec.upload(exec.ptr(&logip)?, &logip_host)?;
                exec.upload(exec.ptr(&slater)?, host.slater)?;
                exec.upload(exec.ptr(&weights)?, host.weights)?;
                exec.upload(exec.ptr(&gutz)?, host.gutz)?;
                exec.upload(exec.ptr(&jast)?, host.jast)?;
                exec.upload(exec.ptr(&params)?, host.params)?;
                exec.upload(exec.ptr(&hop_off)?, &zeros_i32)?;
                exec.upload(exec.ptr(&word_off)?, &zeros_i32)?;
                exec.upload(exec.ptr(&hops_done)?, &zeros_i32)?;
                exec.upload(exec.ptr(&rebuild)?, &zeros_i32)?;
                exec.upload(exec.ptr(&accepts)?, &zeros_i32)?;
            }
            Ok(Self {
                geom,
                total_hops,
                proposal,
                tape_words,
                word_off,
                ele,
                occ,
                inv,
                pf,
                aw,
                logip,
                slater,
                weights,
                gutz,
                jast,
                params,
                tape,
                hop_off,
                hops_done,
                rebuild,
                accepts,
                rec_walker,
                planes: plane_buf,
                pfinv_w,
                pfinv_out,
                pfinv_pf,
                pfinv_st,
            })
        })
        .map_err(|e| e.to_string())
    }

    /// Geometry of this sampler.
    pub fn geometry(&self) -> IntervalGeometry {
        self.geom
    }

    /// Upload a fresh preview and reset its word cursor, keeping proposal offsets.
    /// Layout is `[walker][word]` for CStream, or `[hop][walker][5]` for JuliaTape.
    pub fn upload_tape<B: BackendSession + ?Sized>(
        &self,
        backend: &mut B,
        tape: &[u32],
    ) -> Result<(), String> {
        if tape.len() != self.geom.walkers * self.tape_words {
            return Err("interval tape length does not match the proposal contract".into());
        }
        let zeros = vec![0i32; self.geom.walkers];
        with_raw_exec(backend, "mvmc.interval.tape", |exec| unsafe {
            exec.upload(exec.ptr(&self.tape)?, tape)?;
            exec.upload(exec.ptr(&self.word_off)?, &zeros)
        })
        .map_err(|e| e.to_string())
    }

    /// Reset every walker to `hop_off = 0` (call before a fresh sweep over the same tape).
    pub fn reset_offsets<B: BackendSession + ?Sized>(&self, backend: &mut B) -> Result<(), String> {
        let zeros = vec![0i32; self.geom.walkers];
        with_raw_exec(backend, "mvmc.interval.reset", |exec| unsafe {
            exec.upload(exec.ptr(&self.hop_off)?, &zeros)
        })
        .map_err(|e| e.to_string())
    }

    /// Re-upload the inverse/Pfaffian state (a host-rebuild fallback or a benchmark reset):
    /// the tables, the accepted-move window (`aw`) and the cached `log|inner product|`.
    pub fn upload_tables<B: BackendSession + ?Sized>(
        &self,
        backend: &mut B,
        inv: &[f64],
        pf: &[f64],
        aw: &[i32],
        logip: &[f64],
    ) -> Result<(), String> {
        with_raw_exec(backend, "mvmc.interval.tables", |exec| {
            unsafe {
                exec.upload(exec.ptr(&self.inv)?, inv)?;
                exec.upload(exec.ptr(&self.pf)?, pf)?;
                exec.upload(exec.ptr(&self.aw)?, aw)?;
                exec.upload(exec.ptr(&self.logip)?, logip)?;
            }
            Ok(())
        })
        .map_err(|e| e.to_string())
    }

    /// Advance until block completion, a rebuild, or C-preview exhaustion. A refill
    /// retries the incomplete proposal without having changed its configuration or RNG cursor.
    pub fn run<B: BackendSession + ?Sized>(
        &self,
        backend: &mut B,
    ) -> Result<Vec<IntervalProgress>, String> {
        let g = self.geom;
        let mut hd = vec![0i32; g.walkers];
        let mut rb = vec![0i32; g.walkers];
        let mut ac = vec![0i32; g.walkers];
        with_raw_exec(backend, "mvmc.interval.run", |exec| {
            let module = exec.module(INTERVAL_MODULE, INTERVAL_KERNELS)?;
            let f = module.function("k_interval_hop")?;
            let i = |v: usize| Arg::I(v as i32);
            let p = |b: &DeviceBuffer| exec.ptr(b).map(Arg::P);
            let args = [
                p(&self.ele)?,
                p(&self.occ)?,
                p(&self.inv)?,
                p(&self.pf)?,
                p(&self.aw)?,
                p(&self.logip)?,
                p(&self.slater)?,
                p(&self.weights)?,
                p(&self.gutz)?,
                p(&self.jast)?,
                p(&self.params)?,
                p(&self.tape)?,
                i(self.total_hops),
                p(&self.hop_off)?,
                i(g.n),
                i(g.ne),
                i(g.nsite),
                i(g.nqp),
                i(g.qp_groups()),
                i(g.ngutz),
                i(g.nproj),
                i(if self.proposal == ProposalContract::CStream {
                    self.tape_words
                } else {
                    0
                }),
                p(&self.word_off)?,
                p(&self.hops_done)?,
                p(&self.rebuild)?,
                p(&self.accepts)?,
            ];
            // SAFETY: argument order/dtypes match `k_interval_hop`; buffers stay resident and
            // outlive the scope, which synchronizes before returning.
            unsafe {
                exec.launch(
                    f,
                    (g.walkers as u32, 1, 1),
                    g.block(),
                    g.shared_bytes(),
                    &args,
                )?;
            }
            // SAFETY: the buffers hold `walkers` i32 each.
            unsafe {
                exec.download(&mut hd, exec.ptr(&self.hops_done)?)?;
                exec.download(&mut rb, exec.ptr(&self.rebuild)?)?;
                exec.download(&mut ac, exec.ptr(&self.accepts)?)?;
            }
            Ok(())
        })
        .map_err(|e| e.to_string())?;
        hd.into_iter()
            .zip(rb)
            .zip(ac)
            .map(|((hops_done, reason), accepts)| {
                let stop = match reason {
                    0 => IntervalStop::Complete,
                    1 => IntervalStop::Rebuild,
                    2 => IntervalStop::TapeExhausted,
                    _ => return Err(format!("unknown interval stop reason {reason}")),
                };
                Ok(IntervalProgress {
                    hops_done,
                    stop,
                    accepts,
                })
            })
            .collect()
    }

    /// Rebuild the inverse/Pfaffian of the listed walkers on the device (assemble + batched
    /// Pfaffian + commit) and reset their accepted-move windows.
    pub fn rebuild<B: BackendSession + ?Sized>(
        &self,
        backend: &mut B,
        walkers: &[usize],
    ) -> Result<(), String> {
        if self.try_rebuild(backend, walkers)? {
            Ok(())
        } else {
            Err("interval rebuild has a singular or nonfinite Pfaffian plane".into())
        }
    }

    /// Factor into scratch, committing only when every requested plane succeeded.
    /// Initialization may retain the C setup on a numeric failure; runtime CUDA errors
    /// remain errors, and an unsuccessful factor never overwrites resident tables.
    pub(crate) fn try_rebuild<B: BackendSession + ?Sized>(
        &self,
        backend: &mut B,
        walkers: &[usize],
    ) -> Result<bool, String> {
        if walkers.is_empty() {
            return Ok(true);
        }
        let g = self.geom;
        let items = walkers.len();
        let planes = items * g.nqp;
        let rec: Vec<i32> = walkers.iter().map(|&w| w as i32).collect();
        let (nt, shared) = crate::pfaffian::launch_geometry(g.n, 8, 1);
        with_raw_exec(backend, "mvmc.interval.rebuild", |exec| {
            let module = exec.module(INTERVAL_MODULE, INTERVAL_KERNELS)?;
            let i = |v: usize| Arg::I(v as i32);
            let p = |b: &DeviceBuffer| exec.ptr(b).map(Arg::P);
            // SAFETY: `rec_walker` holds `items` i32.
            unsafe { exec.upload(exec.ptr(&self.rec_walker)?, &rec)? };
            let assemble = module.function("interval_assemble")?;
            // SAFETY: all planes reference resident `ele`/`slater`, `planes` scratch is sized
            // for `items*nqp` planes, and the scope synchronizes before returning.
            unsafe {
                exec.launch(
                    assemble,
                    (planes as u32, 1, 1),
                    128,
                    0,
                    &[
                        p(&self.rec_walker)?,
                        p(&self.ele)?,
                        p(&self.slater)?,
                        p(&self.planes)?,
                        i(g.n),
                        i(g.ne),
                        i(g.nsite),
                        i(g.nqp),
                    ],
                )?;
            }
            let pfinv = module.function("pfinv_f64")?;
            // SAFETY: `pfinv_f64` reads/writes `items*nqp` planes; geometry from
            // `launch_geometry`.
            unsafe {
                exec.launch(
                    pfinv,
                    (planes as u32, 1, 1),
                    nt,
                    shared,
                    &[
                        p(&self.planes)?,
                        p(&self.pfinv_w)?,
                        p(&self.pfinv_out)?,
                        p(&self.pfinv_pf)?,
                        p(&self.pfinv_st)?,
                        i(g.n),
                    ],
                )?;
            }
            let mut statuses = vec![0i32; planes];
            // SAFETY: pfinv_st holds at least the requested plane count.
            unsafe {
                exec.download(&mut statuses, exec.ptr(&self.pfinv_st)?)?;
            }
            exec.synchronize()?;
            if statuses.iter().any(|&info| info != 0) {
                if std::env::var_os("MVMC_INTERVAL_TIMING").is_some() {
                    eprintln!(
                        "[interval] rebuild failed: {} / {} planes, first {:?}",
                        statuses.iter().filter(|&&info| info != 0).count(),
                        planes,
                        statuses
                            .iter()
                            .enumerate()
                            .filter(|(_, info)| **info != 0)
                            .take(8)
                            .collect::<Vec<_>>()
                    );
                }
                return Ok(false);
            }
            let commit = module.function("interval_commit")?;
            // SAFETY: `interval_commit` writes the resident tables of `items` walkers.
            unsafe {
                exec.launch(
                    commit,
                    (items as u32, 1, 1),
                    128,
                    0,
                    &[
                        p(&self.rec_walker)?,
                        p(&self.pfinv_out)?,
                        p(&self.pfinv_pf)?,
                        p(&self.weights)?,
                        p(&self.inv)?,
                        p(&self.pf)?,
                        p(&self.logip)?,
                        p(&self.aw)?,
                        i(g.n),
                        i(g.nqp),
                    ],
                )?;
            }
            Ok(true)
        })
        .map_err(|e| e.to_string())
    }

    /// Download walker `w`'s configuration.
    pub fn download_ele<B: BackendSession + ?Sized>(
        &self,
        backend: &mut B,
        w: usize,
    ) -> Result<Vec<i32>, String> {
        let g = self.geom;
        let mut out = vec![0i32; g.n];
        with_raw_exec(backend, "mvmc.interval.ele", |exec| {
            let ptr = exec.ptr(&self.ele)?;
            // SAFETY: `ele` holds `walkers * n` i32.
            unsafe { exec.download(&mut out, ptr + (w * g.n * 4) as u64)? };
            Ok(())
        })
        .map_err(|e| e.to_string())?;
        Ok(out)
    }

    /// Download walker `w`'s Pfaffians.
    pub fn download_pf<B: BackendSession + ?Sized>(
        &self,
        backend: &mut B,
        w: usize,
    ) -> Result<Vec<f64>, String> {
        let g = self.geom;
        let mut out = vec![0f64; g.nqp];
        with_raw_exec(backend, "mvmc.interval.pf", |exec| {
            let ptr = exec.ptr(&self.pf)?;
            // SAFETY: `pf` holds `walkers * nqp` f64.
            unsafe { exec.download(&mut out, ptr + (w * g.nqp * 8) as u64)? };
            Ok(())
        })
        .map_err(|e| e.to_string())?;
        Ok(out)
    }

    /// Upload walker `w`'s resident sampling state into its own slot: configuration, the
    /// occupancy derived from it, inverse, Pfaffians, accepted-move window and cached
    /// `log|IP|`. Used by a group coordinator to stage each walker before a batched interval
    /// run; `inv` is the walker's `nqp*n*n` inverse and `pf` its `nqp` Pfaffians.
    #[allow(clippy::too_many_arguments)]
    pub fn upload_walker<B: BackendSession + ?Sized>(
        &self,
        backend: &mut B,
        w: usize,
        ele: &[i32],
        inv: &[f64],
        pf: &[f64],
        aw: i32,
        logip: f64,
    ) -> Result<(), String> {
        let g = self.geom;
        let occ = walker_occupancy(ele, &g);
        with_raw_exec(backend, "mvmc.interval.upload_walker", |exec| {
            // SAFETY: every pointer is offset into a buffer sized for `walkers` slots and the
            // slice lengths match one slot; the scope synchronizes before returning.
            unsafe {
                exec.upload(exec.ptr(&self.ele)? + (w * g.n * 4) as u64, ele)?;
                exec.upload(exec.ptr(&self.occ)? + (w * 2 * g.nsite * 4) as u64, &occ)?;
                exec.upload(
                    exec.ptr(&self.inv)? + (w * g.nqp * g.n * g.n * 8) as u64,
                    inv,
                )?;
                exec.upload(exec.ptr(&self.pf)? + (w * g.nqp * 8) as u64, pf)?;
                exec.upload(exec.ptr(&self.aw)? + (w * 4) as u64, &[aw])?;
                exec.upload(exec.ptr(&self.logip)? + (w * 8) as u64, &[logip])?;
            }
            Ok(())
        })
        .map_err(|e| e.to_string())
    }

    /// Download the complete group's results with four transfers and one stream synchronization.
    pub fn download_all<B: BackendSession + ?Sized>(
        &self,
        backend: &mut B,
    ) -> Result<IntervalResults, String> {
        let g = self.geom;
        let mut out = IntervalResults {
            ele: vec![0; g.walkers * g.n],
            pf: vec![0.0; g.walkers * g.nqp],
            aw: vec![0; g.walkers],
            logip: vec![0.0; g.walkers],
            words: vec![0; g.walkers],
        };
        with_raw_exec(backend, "mvmc.interval.download_all", |exec| {
            // SAFETY: each destination has exactly the resident buffer's element count.
            unsafe {
                exec.download(&mut out.ele, exec.ptr(&self.ele)?)?;
                exec.download(&mut out.pf, exec.ptr(&self.pf)?)?;
                exec.download(&mut out.aw, exec.ptr(&self.aw)?)?;
                exec.download(&mut out.logip, exec.ptr(&self.logip)?)?;
                exec.download(&mut out.words, exec.ptr(&self.word_off)?)?;
            }
            Ok(())
        })
        .map_err(|e| e.to_string())?;
        Ok(out)
    }

    /// Download walker `w`'s resident sampling state after an interval run: configuration,
    /// Pfaffians, accepted-move window and cached `log|IP|`.
    pub fn download_walker<B: BackendSession + ?Sized>(
        &self,
        backend: &mut B,
        w: usize,
    ) -> Result<(Vec<i32>, Vec<f64>, i32, f64), String> {
        let g = self.geom;
        let mut ele = vec![0i32; g.n];
        let mut pf = vec![0f64; g.nqp];
        let mut aw = vec![0i32; 1];
        let mut logip = vec![0f64; 1];
        with_raw_exec(backend, "mvmc.interval.download_walker", |exec| {
            // SAFETY: offsets index one walker slot of each resident buffer.
            unsafe {
                exec.download(&mut ele, exec.ptr(&self.ele)? + (w * g.n * 4) as u64)?;
                exec.download(&mut pf, exec.ptr(&self.pf)? + (w * g.nqp * 8) as u64)?;
                exec.download(&mut aw, exec.ptr(&self.aw)? + (w * 4) as u64)?;
                exec.download(&mut logip, exec.ptr(&self.logip)? + (w * 8) as u64)?;
            }
            Ok(())
        })
        .map_err(|e| e.to_string())?;
        Ok((ele, pf, aw[0], logip[0]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tenferro_gpu::cuda::{cuda_devices, CudaBackend};
    use tenferro_tensor::BackendSessionHost;

    #[test]
    #[ignore = "requires CUDA"]
    fn failed_rebuild_preserves_all_resident_tables() {
        let devices = cuda_devices().unwrap();
        let mut backend = CudaBackend::new(devices[0].id()).unwrap();
        let geom = IntervalGeometry {
            walkers: 2,
            n: 2,
            ne: 1,
            nsite: 2,
            nqp: 1,
            ngutz: 0,
            nproj: 0,
        };
        // Walker 0 selects a nonsingular plane; walker 1 selects a zero plane.
        // Resident sentinels deliberately distinguish the old and attempted new state.
        let mut slater = vec![0.0; 16];
        slater[2] = 2.0;
        slater[8] = -2.0;
        let inv = vec![11.0; 8];
        let host = IntervalHost {
            ele: &[0, 0, 1, 1],
            inv: &inv,
            pf: &[3.0, 5.0],
            slater: &slater,
            weights: &[1.0],
            gutz: &[-1; 2],
            jast: &[-1; 4],
            params: &[0.0],
        };
        let sampler = backend
            .with_backend_session(|s| {
                IntervalSampler::prepare(s, geom, &host, 1, ProposalContract::JuliaTape)
            })
            .unwrap();
        backend
            .with_backend_session(|s| {
                with_raw_exec(s, "test.set_windows", |exec| unsafe {
                    exec.upload(exec.ptr(&sampler.aw)?, &[7i32, 9])
                })
            })
            .unwrap();
        let before = backend
            .with_backend_session(|s| sampler.download_all(s))
            .unwrap();
        let error = backend
            .with_backend_session(|s| sampler.rebuild(s, &[0, 1]))
            .unwrap_err();
        assert!(error.contains("singular or nonfinite"), "{error}");
        let after = backend
            .with_backend_session(|s| sampler.download_all(s))
            .unwrap();
        assert_eq!(after.ele, before.ele);
        assert_eq!(after.pf, before.pf); // exact unchanged-buffer identity
        assert_eq!(after.aw, before.aw);
        assert_eq!(after.logip, before.logip);
        let mut inverse_after = vec![0.0f64; 8];
        backend
            .with_backend_session(|s| {
                with_raw_exec(s, "test.inverse", |exec| unsafe {
                    exec.download(&mut inverse_after, exec.ptr(&sampler.inv)?)
                })
            })
            .unwrap();
        assert_eq!(inverse_after, inv);
    }
}
