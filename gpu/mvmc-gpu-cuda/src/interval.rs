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
        let doubles = (self.nqp + 1 + 3 * self.n) * 8;
        (rs + doubles + 8 * 4) as u32
    }

    /// Threads per interval block: the smallest power of two covering `n`, `[32, 256]`.
    pub fn block(&self) -> u32 {
        let mut b = 32usize;
        while b < self.n {
            b *= 2;
        }
        b.clamp(32, 256) as u32
    }
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

/// Resident device state of one interval series.
pub struct IntervalSampler {
    geom: IntervalGeometry,
    total_hops: usize,
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

impl IntervalSampler {
    /// Allocate the resident buffers and upload `host`. `total_hops` is the tape length in hops
    /// per walker (uploaded with [`Self::upload_tape`]).
    pub fn prepare<B: BackendSession + ?Sized>(
        backend: &mut B,
        geom: IntervalGeometry,
        host: &IntervalHost<'_>,
        total_hops: usize,
    ) -> Result<Self, String> {
        let (w, n, nsite, nqp) = (geom.walkers, geom.n, geom.nsite, geom.nqp);
        let n2 = geom.n2();
        let nplanes = w * nqp;
        let occ_host = occupancy(host.ele, &geom);
        let logip_host: Vec<f64> = (0..w)
            .map(|k| f64_log_ip(&host.pf[k * nqp..(k + 1) * nqp], host.weights))
            .collect();
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
            let tape = exec.alloc_bytes((w * total_hops * 4 * 4).max(1))?;
            let hop_off = exec.alloc_bytes(w.max(1) * 4)?;
            let hops_done = exec.alloc_bytes(w.max(1) * 4)?;
            let rebuild = exec.alloc_bytes(w.max(1) * 4)?;
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
                exec.upload(exec.ptr(&inv)?, host.inv)?;
                exec.upload(exec.ptr(&pf)?, host.pf)?;
                exec.upload(exec.ptr(&aw)?, &zeros_i32)?;
                exec.upload(exec.ptr(&logip)?, &logip_host)?;
                exec.upload(exec.ptr(&slater)?, host.slater)?;
                exec.upload(exec.ptr(&weights)?, host.weights)?;
                exec.upload(exec.ptr(&gutz)?, host.gutz)?;
                exec.upload(exec.ptr(&jast)?, host.jast)?;
                exec.upload(exec.ptr(&params)?, host.params)?;
                exec.upload(exec.ptr(&hop_off)?, &zeros_i32)?;
                exec.upload(exec.ptr(&hops_done)?, &zeros_i32)?;
                exec.upload(exec.ptr(&rebuild)?, &zeros_i32)?;
            }
            Ok(Self {
                geom,
                total_hops,
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

    /// Upload a `u32` tape of `walkers * total_hops * 4` words (walker-major, 4 words per hop).
    pub fn upload_tape<B: BackendSession + ?Sized>(
        &self,
        backend: &mut B,
        tape: &[u32],
    ) -> Result<(), String> {
        with_raw_exec(backend, "mvmc.interval.tape", |exec| {
            unsafe { exec.upload(exec.ptr(&self.tape)?, tape) }
        })
        .map_err(|e| e.to_string())
    }

    /// Reset every walker to `hop_off = 0` (call before a fresh sweep over the same tape).
    pub fn reset_offsets<B: BackendSession + ?Sized>(&self, backend: &mut B) -> Result<(), String> {
        let zeros = vec![0i32; self.geom.walkers];
        with_raw_exec(backend, "mvmc.interval.reset", |exec| {
            unsafe { exec.upload(exec.ptr(&self.hop_off)?, &zeros) }
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

    /// Advance every walker until it reaches `total_hops` or requests a rebuild. Returns the
    /// per-walker `(hops_done, rebuild)`.
    pub fn run<B: BackendSession + ?Sized>(
        &self,
        backend: &mut B,
    ) -> Result<Vec<(i32, i32)>, String> {
        let g = self.geom;
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
                i(g.ngutz),
                i(g.nproj),
                p(&self.hops_done)?,
                p(&self.rebuild)?,
            ];
            // SAFETY: argument order/dtypes match `k_interval_hop`; buffers stay resident and
            // outlive the scope, which synchronizes before returning.
            unsafe {
                exec.launch(f, (g.walkers as u32, 1, 1), g.block(), g.shared_bytes(), &args)?;
            }
            let mut hd = vec![0i32; g.walkers];
            let mut rb = vec![0i32; g.walkers];
            // SAFETY: the buffers hold `walkers` i32 each.
            unsafe {
                exec.download(&mut hd, exec.ptr(&self.hops_done)?)?;
                exec.download(&mut rb, exec.ptr(&self.rebuild)?)?;
            }
            Ok(hd.into_iter().zip(rb).collect())
        })
        .map_err(|e| e.to_string())
    }

    /// Rebuild the inverse/Pfaffian of the listed walkers on the device (assemble + batched
    /// Pfaffian + commit) and reset their accepted-move windows.
    pub fn rebuild<B: BackendSession + ?Sized>(
        &self,
        backend: &mut B,
        walkers: &[usize],
    ) -> Result<(), String> {
        if walkers.is_empty() {
            return Ok(());
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
            Ok(())
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
}
