//! Device-resident interval sampler runner (issue #547).
//!
//! Drives the raw CUDA `k_interval_hop` kernel through the vendored
//! [`tenferro_ext::raw_exec`] helper: one `with_raw_exec` scope compiles (once) and launches
//! the kernel on tenferro's captured stream against resident [`DeviceBuffer`]s, so a whole
//! interval segment runs on the device with a single host synchronization. It replaces the
//! cudarc lock-step sampler's per-hop host round trip.
//!
//! The kernel's proposal/acceptance contract is in `interval_kernels.cu`: a fixed tape of
//! `u32` words (4 per hop), empty-destination selection, the Gutzwiller/Jastrow projection
//! delta and the Metropolis decision all on the device.
//!
//! Only the real, non-FSZ, ordinary-hopping mode without local spins is represented; the
//! caller must reject anything else before building an [`IntervalSampler`].

use tenferro_ext::raw_exec::{with_raw_exec, Arg, DeviceBuffer};
use tenferro_tensor::BackendSession;

/// The NVRTC source of the interval kernel (self-contained; see the `SLT` guard).
pub const INTERVAL_SOURCE: &str = include_str!("interval_kernels.cu");

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

    /// Dynamic shared-memory bytes of one CUDA block (`n` ints + the double scratch + 8 ints).
    pub fn shared_bytes(&self) -> u32 {
        let rs = ((self.n + 1) & !1) * 4;
        let doubles = (self.nqp + 1 + 3 * self.n) * 8;
        (rs + doubles + 8 * 4) as u32
    }

    /// Threads per block: the smallest power of two covering `n`, clamped to `[32, 256]`.
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
    hops_done: DeviceBuffer,
    rebuild: DeviceBuffer,
}

fn f64_log_ip(pf: &[f64], weights: &[f64]) -> f64 {
    let mut ip = 0.0f64;
    for (w, p) in weights.iter().zip(pf) {
        ip += w * p;
    }
    if ip < 0.0 {
        -ip
    } else {
        ip
    }
    .ln()
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
    /// Allocate the resident buffers and upload `host`. `total_hops` is the tape length in
    /// hops (the tape itself is uploaded with [`Self::upload_tape`]).
    pub fn prepare<B: BackendSession + ?Sized>(
        backend: &mut B,
        geom: IntervalGeometry,
        host: &IntervalHost<'_>,
        total_hops: usize,
    ) -> Result<Self, String> {
        let (w, n, nsite, nqp) = (geom.walkers, geom.n, geom.nsite, geom.nqp);
        let n2 = geom.n2();
        let occ_host = occupancy(host.ele, &geom);
        let logip_host: Vec<f64> = (0..w)
            .map(|k| f64_log_ip(&host.pf[k * nqp..(k + 1) * nqp], host.weights))
            .collect();
        let aw_host = vec![0i32; w];
        let zero_i32 = vec![0i32; w];

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
            let hops_done = exec.alloc_bytes(w.max(1) * 4)?;
            let rebuild = exec.alloc_bytes(w.max(1) * 4)?;
            unsafe {
                exec.upload(exec.ptr(&ele)?, host.ele)?;
                exec.upload(exec.ptr(&occ)?, &occ_host)?;
                exec.upload(exec.ptr(&inv)?, host.inv)?;
                exec.upload(exec.ptr(&pf)?, host.pf)?;
                exec.upload(exec.ptr(&aw)?, &aw_host)?;
                exec.upload(exec.ptr(&logip)?, &logip_host)?;
                exec.upload(exec.ptr(&slater)?, host.slater)?;
                exec.upload(exec.ptr(&weights)?, host.weights)?;
                exec.upload(exec.ptr(&gutz)?, host.gutz)?;
                exec.upload(exec.ptr(&jast)?, host.jast)?;
                exec.upload(exec.ptr(&params)?, host.params)?;
                exec.upload(exec.ptr(&hops_done)?, &zero_i32)?;
                exec.upload(exec.ptr(&rebuild)?, &zero_i32)?;
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
                hops_done,
                rebuild,
            })
        })
        .map_err(|e| e.to_string())
    }

    /// Geometry of this sampler.
    pub fn geometry(&self) -> IntervalGeometry {
        self.geom
    }

    /// Upload a `u32` tape of `total_hops * 4 * walkers` words.
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

    /// Run `[hop_offset, hop_offset + max_hops)` hops. Returns `(hops_done, rebuild)` of walker
    /// 0; `rebuild != 0` means the accepted-move window passed `nsite` and the caller must
    /// recompute the tables before continuing.
    pub fn run<B: BackendSession + ?Sized>(
        &self,
        backend: &mut B,
        hop_offset: usize,
        max_hops: usize,
    ) -> Result<(i32, i32), String> {
        let g = self.geom;
        if hop_offset + max_hops > self.total_hops {
            return Err(format!(
                "interval range {}..{} exceeds tape length {}",
                hop_offset,
                hop_offset + max_hops,
                self.total_hops
            ));
        }
        with_raw_exec(backend, "mvmc.interval.run", |exec| {
            let module = exec.module(INTERVAL_SOURCE, &["k_interval_hop"])?;
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
                i(hop_offset),
                i(max_hops),
                i(g.n),
                i(g.ne),
                i(g.nsite),
                i(g.nqp),
                i(g.ngutz),
                i(g.nproj),
                p(&self.hops_done)?,
                p(&self.rebuild)?,
            ];
            // SAFETY: argument order and dtypes match `k_interval_hop`; every buffer stays
            // resident for `'self` and outlives the scope, which synchronizes before returning.
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
            Ok((hd[0], rb[0]))
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
