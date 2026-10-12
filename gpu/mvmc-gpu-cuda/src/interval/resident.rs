//! Single-owner walker ensemble: device RNG and rebuild control, no host walker teams.

use super::{
    IntervalGeometry, IntervalHost, IntervalResults, IntervalSampler, INTERVAL_KERNELS,
    INTERVAL_MODULE,
};
use tenferro_ext::raw_exec::{with_raw_exec, Arg, DeviceBuffer, DeviceRng, RawExec, RawModule};
use tenferro_tensor::BackendSession;

#[cfg(test)]
mod tests;

/// Resident sampling state owned by one Opt run, independent of other runs on the
/// same device. A call to `advance` generates the Julia GPU tape and completes all
/// hops, including rebuilds, with zero explicit H2D/D2H copies. It ends with the
/// raw scope's stream barrier; it does not download configurations or inverses.
///
/// Numerical rebuild failures are latched on device. Call `check_status` before
/// consuming results on the host (or use `snapshot`, which checks automatically).
/// This sampling primitive does not yet provide the resident observable/SR pipeline.
pub struct ResidentInterval {
    sampler: IntervalSampler,
    rng: DeviceRng,
    error: DeviceBuffer,
    total_accepts: DeviceBuffer,
}

impl ResidentInterval {
    /// Upload initial state once, seed one ensemble-owned cuRAND generator once.
    pub fn prepare<B: BackendSession + ?Sized>(
        backend: &mut B,
        geom: IntervalGeometry,
        host: &IntervalHost<'_>,
        hops: usize,
        seed: u64,
    ) -> Result<Self, String> {
        validate(geom, host, hops)?;
        let sampler = IntervalSampler::prepare(backend, geom, host, hops, 0.0)?;
        with_raw_exec(backend, "mvmc.resident.prepare", |exec| {
            let rng = DeviceRng::new(exec, seed)?;
            let error = exec.alloc_bytes(4)?;
            let total_accepts = exec.alloc_bytes(geom.walkers * 4)?;
            // SAFETY: newly allocated buffers have exactly these sizes.
            unsafe {
                exec.zero_bytes(exec.ptr(&error)?, 4)?;
                exec.zero_bytes(exec.ptr(&total_accepts)?, geom.walkers * 4)?;
            }
            Ok(Self {
                sampler,
                rng,
                error,
                total_accepts,
            })
        })
        .map_err(|e| e.to_string())
    }

    /// Generate a fresh tape on the device and advance the entire ensemble. No
    /// transfer of random words, rebuild decisions or intermediate configurations.
    pub fn advance<B: BackendSession + ?Sized>(&mut self, backend: &mut B) -> Result<(), String> {
        with_raw_exec(backend, "mvmc.resident.advance", |exec| {
            let words = self.sampler.geom.walkers * self.sampler.total_hops * 5;
            self.rng.fill_u32(exec, &mut self.sampler.tape, words)?;
            self.enqueue(exec)
        })
        .map_err(|e| e.to_string())
    }

    /// Explicit supplied-tape seam for independent Julia parity checks. Does not
    /// advance this object's RNG. Use `advance` to avoid the upload.
    pub fn advance_with_tape<B: BackendSession + ?Sized>(
        &mut self,
        backend: &mut B,
        tape: &[u32],
    ) -> Result<(), String> {
        let words = self.sampler.geom.walkers * self.sampler.total_hops * 5;
        if tape.len() != words {
            return Err(format!("expected {words} tape words, got {}", tape.len()));
        }
        with_raw_exec(backend, "mvmc.resident.shared_tape", |exec| {
            // SAFETY: exact tape size checked above.
            unsafe {
                exec.upload(exec.ptr(&self.sampler.tape)?, tape)?;
            }
            self.enqueue(exec)
        })
        .map_err(|e| e.to_string())
    }

    fn enqueue(&self, exec: &RawExec<'_, '_>) -> tenferro_tensor::Result<()> {
        let s = &self.sampler;
        let g = s.geom;
        let module = exec.module(INTERVAL_MODULE, INTERVAL_KERNELS)?;
        let p = |b: &DeviceBuffer| exec.ptr(b).map(Arg::P);
        let i = |n: usize| Arg::I(n as i32);
        // Starting window <= nsite. The first rebuild may be after one hop;
        // every later one needs >= nsite+1 hops. One final launch finishes any
        // remainder. Masked kernels are no-ops once each walker has completed.
        let rounds = 1 + s.total_hops.div_ceil(g.nsite + 1);
        // SAFETY: dimensions validated at construction, buffers remain resident
        // through all launches; one stream orders every producer and consumer.
        unsafe {
            exec.zero_bytes(exec.ptr(&s.hop_off)?, g.walkers * 4)?;
            exec.zero_bytes(exec.ptr(&self.total_accepts)?, g.walkers * 4)?;
            for _ in 0..rounds {
                exec.launch(
                    module.function("interval_resident_hop")?,
                    (g.walkers as u32, 1, 1),
                    g.block(),
                    g.shared_bytes(),
                    &[
                        p(&s.ele)?,
                        p(&s.occ)?,
                        p(&s.inv)?,
                        p(&s.pf)?,
                        p(&s.aw)?,
                        p(&s.logip)?,
                        p(&s.slater)?,
                        p(&s.weights)?,
                        p(&s.gutz)?,
                        p(&s.jast)?,
                        p(&s.params)?,
                        p(&s.tape)?,
                        i(s.total_hops),
                        p(&s.hop_off)?,
                        i(g.n),
                        i(g.ne),
                        i(g.nsite),
                        i(g.nqp),
                        i(g.qp_groups()),
                        i(g.ngutz),
                        i(g.nproj),
                        p(&s.hops_done)?,
                        p(&s.rebuild)?,
                        p(&s.accepts)?,
                        p(&self.total_accepts)?,
                        p(&self.error)?,
                    ],
                )?;
                self.enqueue_rebuild(exec, &module)?;
            }
        }
        Ok(())
    }

    fn enqueue_rebuild(
        &self,
        exec: &RawExec<'_, '_>,
        module: &RawModule,
    ) -> tenferro_tensor::Result<()> {
        let s = &self.sampler;
        let g = s.geom;
        let p = |b: &DeviceBuffer| exec.ptr(b).map(Arg::P);
        let i = |n: usize| Arg::I(n as i32);
        let (nt, shared) = crate::pfaffian::launch_geometry(g.n, 8, 1);
        // SAFETY: all buffers and dimensions are owned/validated by this ensemble.
        unsafe {
            exec.launch(
                module.function("interval_resident_factor")?,
                ((g.walkers * g.nqp) as u32, 1, 1),
                nt,
                shared,
                &[
                    p(&s.rebuild)?,
                    p(&self.error)?,
                    p(&s.ele)?,
                    p(&s.slater)?,
                    p(&s.planes)?,
                    p(&s.pfinv_w)?,
                    p(&s.pfinv_out)?,
                    p(&s.pfinv_pf)?,
                    p(&s.pfinv_st)?,
                    i(g.n),
                    i(g.ne),
                    i(g.nsite),
                    i(g.nqp),
                ],
            )?;
            exec.launch(
                module.function("interval_resident_commit")?,
                (g.walkers as u32, 1, 1),
                128,
                0,
                &[
                    p(&s.rebuild)?,
                    p(&self.error)?,
                    p(&s.pfinv_st)?,
                    p(&s.pfinv_out)?,
                    p(&s.pfinv_pf)?,
                    p(&s.weights)?,
                    p(&s.inv)?,
                    p(&s.pf)?,
                    p(&s.logip)?,
                    p(&s.aw)?,
                    i(g.n),
                    i(g.nqp),
                ],
            )?;
        }
        Ok(())
    }

    /// One four-byte read at an explicit observation boundary. Error is sticky;
    /// a failed ensemble cannot silently resume and publish partial results.
    pub fn check_status<B: BackendSession + ?Sized>(
        &mut self,
        backend: &mut B,
    ) -> Result<(), String> {
        let mut error = [0i32];
        with_raw_exec(backend, "mvmc.resident.status", |exec| {
            // SAFETY: one i32 status word.
            unsafe { exec.download(&mut error, exec.ptr(&self.error)?) }
        })
        .map_err(|e| e.to_string())?;
        if error[0] != 0 {
            return Err(
                "resident interval rebuild has a singular/nonfinite plane or zero overlap".into(),
            );
        }
        Ok(())
    }

    /// Explicit diagnostic/result boundary; ordinary sampling keeps these resident.
    pub fn snapshot<B: BackendSession + ?Sized>(
        &mut self,
        backend: &mut B,
    ) -> Result<IntervalResults, String> {
        self.check_status(backend)?;
        self.sampler.download_all(backend)
    }

    /// Download acceptance counts of the most recent call, for parity/diagnostics.
    pub fn accepted<B: BackendSession + ?Sized>(
        &mut self,
        backend: &mut B,
    ) -> Result<Vec<i32>, String> {
        self.check_status(backend)?;
        let mut out = vec![0i32; self.sampler.geom.walkers];
        with_raw_exec(backend, "mvmc.resident.accepted", |exec| {
            // SAFETY: one i32 per walker.
            unsafe { exec.download(&mut out, exec.ptr(&self.total_accepts)?) }
        })
        .map_err(|e| e.to_string())?;
        Ok(out)
    }
}

fn validate(g: IntervalGeometry, h: &IntervalHost<'_>, hops: usize) -> Result<(), String> {
    let fail = || "invalid resident interval dimensions or initial state".to_string();
    if g.walkers == 0
        || g.ne == 0
        || g.nqp == 0
        || g.nqp > 48 * 1024 / 8
        || g.ne > g.nsite
        || Some(g.n) != g.ne.checked_mul(2)
        || g.ngutz > g.nproj
        || g.n > crate::pfaffian::MAX_N
        || g.nsite > (i32::MAX as usize / 2)
        || [g.walkers, g.n, g.nsite, g.nqp, g.nproj, hops]
            .iter()
            .any(|&v| v > i32::MAX as usize)
    {
        return Err(fail());
    }
    let product = |dims: &[usize]| {
        dims.iter()
            .try_fold(1usize, |a, &b| a.checked_mul(b))
            .ok_or_else(fail)
    };
    let wn = product(&[g.walkers, g.n])?;
    let wq = product(&[g.walkers, g.nqp])?;
    let inverse = product(&[wq, g.n, g.n])?;
    let slater = product(&[g.nqp, 2, g.nsite, 2, g.nsite])?;
    if g.shared_bytes() > 48 * 1024 {
        return Err("resident interval exceeds the CUDA shared-memory limit".into());
    }
    for len in [inverse, slater, product(&[g.walkers, hops, 5])?] {
        product(&[len, 8])?;
    }
    if wq > i32::MAX as usize
        || h.ele.len() != wn
        || h.inv.len() != inverse
        || h.pf.len() != wq
        || h.slater.len() != slater
        || h.weights.len() != g.nqp
        || h.gutz.len() != g.nsite
        || h.jast.len() != product(&[g.nsite, g.nsite])?
        || h.params.len() < g.nproj
        || h.params.len() > g.nproj.max(1)
        || h.gutz.iter().any(|&v| v < -1 || v >= g.ngutz as i32)
        || h.jast
            .iter()
            .any(|&v| v < -1 || v >= (g.nproj - g.ngutz) as i32)
    {
        return Err(fail());
    }
    for spin in h.ele.chunks_exact(g.ne) {
        let mut seen = vec![false; g.nsite];
        for &site in spin {
            if site < 0 || site as usize >= g.nsite || seen[site as usize] {
                return Err(fail());
            }
            seen[site as usize] = true;
        }
    }
    if h.inv
        .iter()
        .chain(h.slater)
        .chain(h.params)
        .chain(h.weights)
        .any(|v| !v.is_finite())
        || h.pf.iter().any(|v| !v.is_finite() || *v == 0.0)
        || h.pf.chunks_exact(g.nqp).any(|pf| {
            let ip = pf.iter().zip(h.weights).fold(0.0, |ip, (p, w)| ip + p * w);
            !ip.is_finite() || ip == 0.0
        })
    {
        return Err(
            "resident interval requires finite initial tables and nonzero amplitudes".into(),
        );
    }
    Ok(())
}
