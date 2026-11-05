//! Single-owner walker ensemble: device RNG and rebuild control, no host walker teams.

use super::{
    IntervalGeometry, IntervalHost, IntervalResults, IntervalSampler, INTERVAL_KERNELS,
    INTERVAL_MODULE,
};
use tenferro_ext::raw_exec::{with_raw_exec, Arg, DeviceBuffer, DeviceRng, RawExec, RawModule};
use tenferro_tensor::Engine;

#[cfg(test)]
mod tests;

/// Stage that latched the sticky resident device status word (#573), or the
/// host-side initialization retry limit. The first failing stage wins; every
/// later resident kernel observes the word and publishes nothing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResidentFailure {
    /// Configuration initialization did not find finite nonzero planes/overlaps.
    Initialization,
    /// A walker rebuild produced a singular/nonfinite QP plane or overlap.
    Rebuild,
    /// Nonfinite local energy or PhysCal observable.
    Energy,
    /// The SR Cholesky solve failed.
    Sr,
    /// The SR parameter update was nonfinite.
    Parameters,
    /// A walker did not finish its scheduled hops within the enqueued rounds.
    Schedule,
    /// A log-derivative was nonfinite.
    Derivatives,
    /// The SR-CG increment was nonfinite (C `StochasticOptCG` info=1).
    Cg,
    /// A status word outside `resident_status.cuh`.
    Unknown(i32),
}

impl ResidentFailure {
    /// Decode the device word; `None` means success. Values match `resident_status.cuh`.
    pub fn from_status(code: i32) -> Option<Self> {
        Some(match code {
            0 => return None,
            1 => Self::Rebuild,
            2 => Self::Energy,
            3 => Self::Sr,
            4 => Self::Parameters,
            5 => Self::Schedule,
            6 => Self::Derivatives,
            7 => Self::Cg,
            other => Self::Unknown(other),
        })
    }

    /// Device status word, or `None` for the host-side initialization stage.
    pub fn code(self) -> Option<i32> {
        match self {
            Self::Initialization => None,
            Self::Rebuild => Some(1),
            Self::Energy => Some(2),
            Self::Sr => Some(3),
            Self::Parameters => Some(4),
            Self::Schedule => Some(5),
            Self::Derivatives => Some(6),
            Self::Cg => Some(7),
            Self::Unknown(code) => Some(code),
        }
    }
}

impl std::fmt::Display for ResidentFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let stage = match self {
            Self::Initialization => {
                "initialization: no finite nonzero QP planes after 100 attempts"
            }
            Self::Rebuild => "rebuild: singular/nonfinite QP plane or zero overlap",
            Self::Energy => "energy: nonfinite local energy or observable",
            Self::Sr => "SR: Cholesky solve failed",
            Self::Parameters => "parameters: nonfinite update",
            Self::Schedule => "schedule: unfinished hops after the enqueued rounds",
            Self::Derivatives => "derivatives: nonfinite log-derivative",
            Self::Cg => "SR-CG: nonfinite parameter increment",
            Self::Unknown(_) => "unknown stage",
        };
        match self.code() {
            Some(code) => write!(f, "device status {code} ({stage})"),
            None => write!(f, "{stage}"),
        }
    }
}

impl std::error::Error for ResidentFailure {}

/// Resident sampling state owned by one Opt run, independent of other runs on the
/// same device. A call to `advance` generates the Julia GPU tape and completes all
/// hops, including rebuilds, with zero explicit H2D/D2H copies. It ends with the
/// raw scope's stream barrier; it does not download configurations or inverses.
///
/// Numerical rebuild failures are latched on device. Call `check_status` before
/// consuming results on the host (or use `snapshot`, which checks automatically).
/// Use [`crate::opt`] for the complete resident observable/SR pipeline.
pub struct ResidentInterval {
    pub(crate) sampler: IntervalSampler,
    pub(crate) rng: DeviceRng,
    pub(crate) error: DeviceBuffer,
    total_accepts: DeviceBuffer,
    fallback: DeviceBuffer,
    delayed: DeviceBuffer,
    /// Per-(walker, QP) validity, i32 0/1 (#573). Invalid planes carry no
    /// usable inverse; their walker uses full-Pfaffian fallbacks.
    pub(crate) valid: DeviceBuffer,
    pub(crate) delayed_updates: bool,
    /// Local-spin mask `[nsite]` (Julia `localspins`); `None` when there are none.
    pub(crate) localspins: Option<DeviceBuffer>,
    /// Julia GPU exchange probability (tape word 3).
    pub(crate) exchange_fraction: f64,
    /// Doublon-holon neighbor tables (DH2 `[n2][nsite][2]`, then DH4
    /// `[n4][nsite][4]`); `None` without DH factors (#665).
    pub(crate) dh: Option<DeviceBuffer>,
    /// DH2 and DH4 table counts.
    pub(crate) dh_tables: (usize, usize),
    /// RBM tables, coefficients and per-walker hidden fields (#667).
    pub(crate) rbm: Option<ResidentRbm>,
}

/// Device RBM state of one ensemble (`resident_rbm.cuh`). The tables and the
/// neuron count are immutable; `weights` mirrors the RBM block of the
/// parameter vector (refreshed by the Opt gauge), and `theta[w][h]` is each
/// walker's hidden field (C `rbmCnt[NRBM_PhysLayerIdx + h]`), rebuilt from the
/// configuration by C MakeRBMCnt and advanced by C UpdateRBMCnt on accepts.
pub(crate) struct ResidentRbm {
    pub tables: DeviceBuffer,
    pub weights: DeviceBuffer,
    pub theta: DeviceBuffer,
    pub nneuron: usize,
}

impl ResidentInterval {
    /// Upload initial state once, seed one ensemble-owned cuRAND generator once.
    pub fn prepare<B: Engine + ?Sized>(
        backend: &mut B,
        geom: IntervalGeometry,
        host: &IntervalHost<'_>,
        hops: usize,
        seed: u64,
    ) -> Result<Self, String> {
        validate(geom, host, hops)?;
        let sampler = IntervalSampler::prepare(backend, geom, host, hops, 0.0)?;
        Self::from_sampler(backend, sampler, seed)
    }

    pub(crate) fn from_sampler<B: Engine + ?Sized>(
        backend: &mut B,
        sampler: IntervalSampler,
        seed: u64,
    ) -> Result<Self, String> {
        let geom = sampler.geom;
        with_raw_exec(backend, "mvmc.resident.prepare", |exec| {
            let rng = DeviceRng::new(exec, seed)?;
            let error = exec.alloc_bytes(4)?;
            let total_accepts = exec.alloc_bytes(geom.walkers * 4)?;
            // Setup tables are validated (or later initialized) as all valid.
            let valid = exec.alloc_bytes(geom.walkers * geom.nqp * 4)?;
            // SAFETY: newly allocated buffers have exactly these sizes.
            unsafe {
                exec.zero_bytes(exec.ptr(&error)?, 4)?;
                exec.zero_bytes(exec.ptr(&total_accepts)?, geom.walkers * 4)?;
                exec.upload(exec.ptr(&valid)?, &vec![1i32; geom.walkers * geom.nqp])?;
            }
            Ok(Self {
                sampler,
                rng,
                error,
                total_accepts,
                fallback: exec.alloc_bytes(geom.walkers * 4)?,
                // n>=128 uses four queued (u,v) pairs plus two dots per pair
                // and one ratio. This is smaller than the validated inverse.
                delayed: exec.alloc_bytes(if geom.n >= 128 {
                    geom.walkers * geom.nqp * (8 * geom.n + 9) * 8
                } else {
                    1
                })?,
                delayed_updates: true,
                valid,
                localspins: None,
                exchange_fraction: 0.0,
                dh: None,
                dh_tables: (0, 0),
                rbm: None,
            })
        })
        .map_err(|e| e.to_string())
    }

    /// Select the Julia GPU proposal mix: local-spin sites (`1` entries of
    /// `localspins`, one per site) and the exchange probability. Uploads the
    /// mask once at setup, only when a local spin exists.
    pub(crate) fn set_moves<B: Engine + ?Sized>(
        &mut self,
        backend: &mut B,
        localspins: &[i32],
        exchange_fraction: f64,
    ) -> Result<(), String> {
        if localspins.len() != self.sampler.geom.nsite
            || localspins.iter().any(|&x| x != 0 && x != 1)
            || !(0.0..=1.0).contains(&exchange_fraction)
        {
            return Err("invalid resident proposal configuration".into());
        }
        self.exchange_fraction = exchange_fraction;
        self.localspins = if localspins.contains(&1) {
            Some(
                with_raw_exec(backend, "mvmc.resident.localspins", |exec| {
                    let b = exec.alloc_bytes(localspins.len() * 4)?;
                    // SAFETY: allocated with exactly the mask size.
                    unsafe { exec.upload(exec.ptr(&b)?, localspins)? };
                    Ok(b)
                })
                .map_err(|e| e.to_string())?,
            )
        } else {
            None
        };
        Ok(())
    }

    /// Upload the doublon-holon neighbor tables once at setup (C `projection.c`
    /// counters). The last `6*n2 + 10*n4` projection parameters are the DH
    /// block. Without tables the sampler keeps its Gutzwiller/Jastrow path.
    pub(crate) fn set_doublon_holon<B: Engine + ?Sized>(
        &mut self,
        backend: &mut B,
        tables: &[i32],
        n2: usize,
        n4: usize,
    ) -> Result<(), String> {
        let g = self.sampler.geom;
        if Some(tables.len()) != (2 * n2 + 4 * n4).checked_mul(g.nsite)
            || 6 * n2 + 10 * n4 > g.nproj - g.ngutz
            || tables.iter().any(|&r| r < 0 || r as usize >= g.nsite)
        {
            return Err("invalid resident doublon-holon tables".into());
        }
        self.dh_tables = (n2, n4);
        self.dh = if tables.is_empty() {
            None
        } else {
            Some(
                with_raw_exec(backend, "mvmc.resident.dh", |exec| {
                    let b = exec.alloc_bytes(tables.len() * 4)?;
                    // SAFETY: allocated with exactly the table size.
                    unsafe { exec.upload(exec.ptr(&b)?, tables)? };
                    Ok(b)
                })
                .map_err(|e| e.to_string())?,
            )
        };
        Ok(())
    }

    /// Upload the RBM index tables and coefficients once at setup and allocate
    /// the per-walker hidden fields. Without RBM terms nothing is allocated and
    /// the sampler keeps its projection-only path.
    pub(crate) fn set_rbm<B: Engine + ?Sized>(
        &mut self,
        backend: &mut B,
        tables: &[i32],
        weights: &[f64],
        nneuron: usize,
    ) -> Result<(), String> {
        let g = self.sampler.geom;
        if tables.is_empty() {
            self.rbm = None;
            return Ok(());
        }
        if tables.len() != 4 * g.nsite + 2 * nneuron + 2 * g.nsite * nneuron
            || weights.is_empty()
            || weights.iter().any(|x| !x.is_finite())
            || tables.iter().any(|&i| i < -1 || i >= weights.len() as i32)
            || g.walkers.checked_mul(nneuron.max(1)).is_none()
        {
            return Err("invalid resident RBM tables".into());
        }
        self.rbm = Some(
            with_raw_exec(backend, "mvmc.resident.rbm", |exec| {
                let t = exec.alloc_bytes(tables.len() * 4)?;
                let w = exec.alloc_bytes(weights.len() * 8)?;
                let theta = exec.alloc_bytes(g.walkers * nneuron.max(1) * 8)?;
                // SAFETY: allocated with exactly the table/coefficient sizes;
                // theta is written by the MakeRBMCnt kernel before any read.
                unsafe {
                    exec.upload(exec.ptr(&t)?, tables)?;
                    exec.upload(exec.ptr(&w)?, weights)?;
                    exec.zero_bytes(exec.ptr(&theta)?, g.walkers * nneuron.max(1) * 8)?;
                }
                Ok(ResidentRbm {
                    tables: t,
                    weights: w,
                    theta,
                    nneuron,
                })
            })
            .map_err(|e| e.to_string())?,
        );
        Ok(())
    }

    /// RBM kernel arguments: tables, coefficients, hidden fields (null
    /// without RBM) and the neuron count.
    pub(crate) fn rbm_args(&self, exec: &RawExec<'_, '_>) -> tenferro_tensor::Result<[Arg; 4]> {
        Ok(match &self.rbm {
            Some(r) => [
                Arg::P(exec.ptr(&r.tables)?),
                Arg::P(exec.ptr(&r.weights)?),
                Arg::P(exec.ptr(&r.theta)?),
                Arg::I(r.nneuron as i32),
            ],
            None => [Arg::P(0), Arg::P(0), Arg::P(0), Arg::I(0)],
        })
    }

    /// Device address of the DH tables, or null when there are none.
    pub(crate) fn dh_ptr(&self, exec: &RawExec<'_, '_>) -> tenferro_tensor::Result<Arg> {
        Ok(Arg::P(match &self.dh {
            Some(b) => exec.ptr(b)?,
            None => 0,
        }))
    }

    /// Device address of the local-spin mask, or null when there are none.
    pub(crate) fn localspin_ptr(&self, exec: &RawExec<'_, '_>) -> tenferro_tensor::Result<Arg> {
        Ok(Arg::P(match &self.localspins {
            Some(b) => exec.ptr(b)?,
            None => 0,
        }))
    }

    /// Generate a fresh tape on the device and advance the entire ensemble. No
    /// transfer of random words, rebuild decisions or intermediate configurations.
    pub fn advance<B: Engine + ?Sized>(&mut self, backend: &mut B) -> Result<(), String> {
        with_raw_exec(backend, "mvmc.resident.advance", |exec| {
            let words = self.sampler.geom.walkers * self.sampler.total_hops * 5;
            self.rng.fill_u32(exec, &mut self.sampler.tape, words)?;
            self.enqueue(exec)
        })
        .map_err(|e| e.to_string())
    }

    /// Explicit supplied-tape seam for independent Julia parity checks. Does not
    /// advance this object's RNG. Use `advance` to avoid the upload.
    pub fn advance_with_tape<B: Engine + ?Sized>(
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
        self.enqueue_hops(exec, self.sampler.total_hops)
    }

    pub(crate) fn enqueue_random(
        &mut self,
        exec: &RawExec<'_, '_>,
        hops: usize,
    ) -> tenferro_tensor::Result<()> {
        assert!(hops <= self.sampler.total_hops);
        self.rng.fill_u32(
            exec,
            &mut self.sampler.tape,
            self.sampler.geom.walkers * hops * 5,
        )?;
        self.enqueue_hops(exec, hops)
    }

    fn enqueue_hops(&self, exec: &RawExec<'_, '_>, hops: usize) -> tenferro_tensor::Result<()> {
        let s = &self.sampler;
        let g = s.geom;
        let module = exec.module(INTERVAL_MODULE, INTERVAL_KERNELS)?;
        let p = |b: &DeviceBuffer| exec.ptr(b).map(Arg::P);
        let i = |n: usize| Arg::I(n as i32);
        // Starting window <= nsite. The first rebuild may be after one hop;
        // every later one needs >= nsite+1 hops. One final launch finishes any
        // remainder. Masked kernels are no-ops once each walker has completed.
        let rounds = 1 + hops.div_ceil(g.nsite + 1);
        let (threads, groups, shared) = launch_geometry(g);
        // SAFETY: dimensions validated at construction, buffers remain resident
        // through all launches; one stream orders every producer and consumer.
        unsafe {
            exec.zero_bytes(exec.ptr(&s.hop_off)?, g.walkers * 4)?;
            exec.zero_bytes(exec.ptr(&self.total_accepts)?, g.walkers * 4)?;
            for _ in 0..rounds {
                exec.launch(
                    module.function("interval_resident_hop")?,
                    (g.walkers as u32, 1, 1),
                    threads,
                    shared,
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
                        i(hops),
                        p(&s.hop_off)?,
                        i(g.n),
                        i(g.ne),
                        i(g.nsite),
                        i(g.nqp),
                        i(groups),
                        i(g.ngutz),
                        i(g.nproj),
                        p(&s.hops_done)?,
                        p(&s.rebuild)?,
                        p(&s.accepts)?,
                        p(&self.total_accepts)?,
                        p(&self.error)?,
                        p(&self.delayed)?,
                        p(&s.pfinv_out)?,
                        Arg::I(i32::from(self.delayed_updates)),
                        p(&self.valid)?,
                        p(&s.planes)?,
                        p(&s.pfinv_w)?,
                        self.localspin_ptr(exec)?,
                        Arg::D(self.exchange_fraction),
                        self.dh_ptr(exec)?,
                        i(self.dh_tables.0),
                        i(self.dh_tables.1),
                    ]
                    .into_iter()
                    .chain(self.rbm_args(exec)?)
                    .collect::<Vec<_>>(),
                )?;
                self.enqueue_rebuild(exec, &module)?;
            }
            exec.launch(
                module.function("interval_resident_complete")?,
                (g.walkers.div_ceil(128) as u32, 1, 1),
                128,
                0,
                &[p(&s.hop_off)?, i(hops), p(&self.error)?, i(g.walkers)],
            )?;
        }
        Ok(())
    }

    pub(crate) fn enqueue_rebuild(
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
            // The panel, diagonal and permutations share at most 48 KiB. Smaller
            // geometries retain the established generic path.
            let capacity = ((48 * 1024 - 12 * g.n) / (32 * g.n)).min(4);
            let paired = g.n >= 128 && capacity > 0;
            if paired {
                exec.launch(
                    module.function("interval_paired_factor")?,
                    ((g.walkers * g.nqp) as u32, 1, 1),
                    128,
                    (32 * g.n * capacity + 12 * g.n) as u32,
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
                        i(capacity),
                    ],
                )?;
                exec.launch(
                    module.function("interval_paired_fallback")?,
                    (g.walkers.div_ceil(128) as u32, 1, 1),
                    128,
                    0,
                    &[
                        p(&s.rebuild)?,
                        p(&s.pfinv_st)?,
                        p(&s.pfinv_pf)?,
                        p(&s.weights)?,
                        p(&self.fallback)?,
                        i(g.walkers),
                        i(g.nqp),
                    ],
                )?;
            }
            exec.launch(
                module.function("interval_resident_factor")?,
                ((g.walkers * g.nqp) as u32, 1, 1),
                nt,
                shared,
                &[
                    p(if paired { &self.fallback } else { &s.rebuild })?,
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
                    p(&self.valid)?,
                    i(g.n),
                    i(g.nqp),
                ],
            )?;
        }
        Ok(())
    }

    /// One four-byte read at an explicit observation boundary. Error is sticky;
    /// a failed ensemble cannot silently resume and publish partial results.
    pub fn status<B: Engine + ?Sized>(
        &mut self,
        backend: &mut B,
    ) -> Result<Option<ResidentFailure>, String> {
        let mut error = [0i32];
        with_raw_exec(backend, "mvmc.resident.status", |exec| {
            // SAFETY: one i32 status word.
            unsafe { exec.download(&mut error, exec.ptr(&self.error)?) }
        })
        .map_err(|e| e.to_string())?;
        Ok(ResidentFailure::from_status(error[0]))
    }

    /// [`Self::status`] as a hard error for result boundaries.
    pub fn check_status<B: Engine + ?Sized>(&mut self, backend: &mut B) -> Result<(), String> {
        match self.status(backend)? {
            Some(failure) => Err(format!("resident interval failed: {failure}")),
            None => Ok(()),
        }
    }

    /// Explicit diagnostic/result boundary; ordinary sampling keeps these resident.
    pub fn snapshot<B: Engine + ?Sized>(
        &mut self,
        backend: &mut B,
    ) -> Result<IntervalResults, String> {
        self.check_status(backend)?;
        self.sampler.download_all(backend)
    }

    /// Download acceptance counts of the most recent call, for parity/diagnostics.
    pub fn accepted<B: Engine + ?Sized>(&mut self, backend: &mut B) -> Result<Vec<i32>, String> {
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

// Four warps balance QP cooperation with the number of resident walker blocks.
// The ratio scratch is reused by updates; the four queued sites follow mvw[8].
fn launch_geometry(g: IntervalGeometry) -> (u32, usize, u32) {
    let threads = g.block().min(128);
    let mut groups = g.qp_groups().min(threads as usize / 32);
    loop {
        let scratch = (threads as usize / 32).min(g.nqp).max(3 * groups) * g.n;
        let shared = ((g.n + 1) & !1) * 4 + (g.nqp + 1 + scratch) * 8 + 12 * 4;
        if shared <= 48 * 1024 || groups == 1 {
            return (threads, groups, shared as u32);
        }
        groups /= 2;
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
    if g.shared_bytes() > 48 * 1024 || launch_geometry(g).2 > 48 * 1024 {
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
