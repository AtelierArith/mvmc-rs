//! Device-resident SR-CG (`NSRCG != 0`, #670, #694): C `StochasticOptCG_real`
//! (`stcopt_cg_impl.c`) on the resident derivative matrix, without host round
//! trips: plain and DiagScale CG (`useDiagScale`, `NSRCG 2`), the
//! `NSRCGFallback` rerun and `RescaleSmat`. See `cg.cu` for the kernels and
//! their C source lines.
//!
//! The device decides the C stopping rule. The host enqueues an upper bound of
//! iterations (`NSROptCGMaxIter`, or the parameter count when it is 0, since
//! `nSmat <= NPara`); after the stop flag is set, each remaining iteration costs
//! only the launch of kernels that return at once. With `NSRCGFallback` a
//! second bounded solve of the other variant is enqueued; the device runs it
//! only after a failed first solve, as C does.
use super::{i, State};
use tenferro_ext::raw_exec::{at8, Arg, DeviceBuffer, RawExec, RawModule};

/// Device CG state, allocated only for Opt with `NSRCG != 0`.
pub(super) struct CgBuffers {
    /// `[g, d, r, q, sdiag, <O>]`, each `np` long (masked to the active set).
    pub vec: DeviceBuffer,
    /// 1 for an active (optimized, not diagonal-cut) parameter.
    pub active: DeviceBuffer,
    /// `[delta, threshold, alpha, <O>.x, 1/Wc, rescaleOO, rescaleO]`.
    pub st: DeviceBuffer,
    /// `[stop, iterations, max iterations, nSmat, C status, first status,
    /// fallback ran, rescaled]`.
    pub ist: DeviceBuffer,
    /// Per-sample product `O^T v`.
    pub y: DeviceBuffer,
    /// `<OO>_ii`, kept for `RescaleSmat`.
    pub oodiag: DeviceBuffer,
    /// `delta` after initialization and each iteration: `limit + 1` entries
    /// for the first solve, then as many for the fallback solve.
    pub trace: DeviceBuffer,
    /// C 1.4.0 SR-CG info of each step (the last `zqp_SRinfo.dat` column):
    /// the iteration count, or `-(iter+1)` after a breakdown or
    /// non-convergence of the first solve.
    pub iterations: DeviceBuffer,
    /// Status of the `NSRCGFallback` solve of each step, `i32::MIN` when it did
    /// not run.
    pub fallback: DeviceBuffer,
    /// Enqueued iterations per solve.
    pub limit: usize,
}

const BLOCK: u32 = 1024;

/// Enqueued iteration bound: C's `max_iter` when given, else `NPara >= nSmat`.
pub(super) fn iteration_limit(max_iter: i64, np: usize) -> Result<usize, String> {
    let limit = if max_iter > 0 {
        usize::try_from(max_iter).map_err(|_| "NSROptCGMaxIter out of range")?
    } else {
        np
    };
    if limit >= i32::MAX as usize {
        return Err("NSROptCGMaxIter exceeds the resident GPU index range".into());
    }
    Ok(limit)
}

/// Slater block `[start, end)` that `Rescale4SRCG_real` rescales, or `None`
/// where C returns before rescaling for any sample (`NProj <= 0`,
/// `NProj >= NPara` or `NSlater <= 0`, stcopt_cg_impl.c:787-789). Para is
/// `Proj | RBM | Slater | OptTrans` (`NProjBF = 0`).
pub(super) fn rescale_block(
    nproj: usize,
    nrbm: usize,
    nopt: usize,
    np: usize,
) -> Option<(usize, usize)> {
    let start = nproj + nrbm;
    let end = np.checked_sub(nopt)?;
    (nproj > 0 && nproj < np && end > start).then_some((start, end))
}

impl CgBuffers {
    pub fn allocate(
        exec: &RawExec<'_, '_>,
        np: usize,
        walkers: usize,
        steps: usize,
        limit: usize,
        fallback: bool,
    ) -> tenferro_tensor::Result<Self> {
        let alloc = |count: usize, size: usize| exec.alloc_bytes((count * size).max(1));
        Ok(Self {
            vec: alloc(6 * np, 8)?,
            active: alloc(np, 4)?,
            st: alloc(8, 8)?,
            ist: alloc(8, 4)?,
            y: alloc(walkers, 8)?,
            oodiag: alloc(np, 8)?,
            trace: alloc((limit + 1) * (1 + usize::from(fallback)), 8)?,
            iterations: alloc(steps, 4)?,
            fallback: alloc(steps, 4)?,
            limit,
        })
    }
}

impl State {
    /// One SR-CG step: moments, `RescaleSmat`, C diagonal cut and gradient,
    /// the selected CG variant with the 20-iteration residual refresh, the
    /// optional fallback solve, and the finiteness check. The increment is
    /// left in `rhs` for `opt_update`, with `info` cleared.
    pub(super) fn solve_cg(
        &self,
        exec: &RawExec<'_, '_>,
        module: &RawModule,
        step: usize,
    ) -> tenferro_tensor::Result<()> {
        let b = &self.buffers;
        let cg =
            b.cg.as_ref()
                .expect("CG buffers are allocated for NSRCG != 0");
        let m = &self.model;
        let mp = &m.data.modpara;
        let np = m.np;
        let w = self.resident.sampler.geom.walkers;
        let p = |b: &DeviceBuffer| exec.ptr(b).map(Arg::P);
        let vec = exec.ptr(&cg.vec)?;
        let at = |k: usize| Arg::P(at8(vec, k * np));
        let max_iter = i32::try_from(mp.nsr_opt_cg_max_iter.max(0))
            .map_err(|_| super::gpu_error("NSROptCGMaxIter out of range"))?;
        // stcopt_cg_impl.c:362 precon_factor, evaluated once on the host as in C.
        let factor = Arg::D(1.0 + mp.dsr_opt_sta_del);
        let diag = mp.use_diag_scale;
        // SAFETY: every buffer was allocated with the extents used below (np,
        // np+1 for mean, 6*np for vec, walkers*np for derivatives, (limit+1) per solve for
        // trace, steps for iterations, fallback and history).
        unsafe {
            exec.launch(
                module.function("cg_moments")?,
                ((np + 1).div_ceil(128) as u32, 1, 1),
                128,
                0,
                &[
                    p(&b.mean)?,
                    p(&b.variance)?,
                    p(&b.ho)?,
                    p(&cg.oodiag)?,
                    p(&b.derivatives)?,
                    p(&b.local_energy)?,
                    i(np),
                    i(w),
                ],
            )?;
            // stcopt_cg.c:58-61: Rescale4SRCG_real before StochasticOptCG_real.
            if let Some((start, end)) = mp
                .rescale_smat
                .then(|| rescale_block(m.nproj, m.nrbm, m.nopt, np))
                .flatten()
            {
                exec.launch(
                    module.function("cg_rescale")?,
                    (1, 1, 1),
                    BLOCK,
                    0,
                    &[
                        p(&b.mean)?,
                        p(&b.variance)?,
                        p(&b.ho)?,
                        p(&cg.oodiag)?,
                        p(&b.parameters)?,
                        p(&cg.st)?,
                        p(&cg.ist)?,
                        i(m.nproj),
                        i(start),
                        i(end),
                        p(&self.resident.error)?,
                    ],
                )?;
                exec.launch(
                    module.function("cg_rescale_samples")?,
                    (w as u32, 1, 1),
                    256,
                    0,
                    &[
                        p(&b.derivatives)?,
                        p(&cg.st)?,
                        p(&cg.ist)?,
                        i(np),
                        i(start),
                        i(end),
                    ],
                )?;
            }
            exec.launch(
                module.function("cg_prepare")?,
                (1, 1, 1),
                BLOCK,
                0,
                &[
                    p(&b.rhs)?,
                    at(0),
                    p(&cg.active)?,
                    p(&cg.st)?,
                    p(&cg.ist)?,
                    p(&cg.trace)?,
                    p(&b.history)?,
                    p(&b.mean)?,
                    p(&b.variance)?,
                    p(&b.ho)?,
                    p(&b.flags)?,
                    i(np),
                    i(w),
                    i(step),
                    Arg::D(mp.dsr_opt_red_cut),
                    Arg::D(mp.dsr_opt_step_dt),
                    Arg::D(mp.dsr_opt_cg_tol),
                    Arg::I(max_iter),
                    p(&self.resident.error)?,
                    i(usize::from(diag)),
                ],
            )?;
            // q = S v into `out`, for v = d (ordinary) or x (refresh, DiagScale start).
            let product = |v: Arg, out: Arg| -> tenferro_tensor::Result<()> {
                exec.launch(
                    module.function("cg_ty")?,
                    ((w + 1) as u32, 1, 1),
                    256,
                    0,
                    &[
                        p(&cg.y)?,
                        p(&cg.st)?,
                        p(&cg.ist)?,
                        p(&b.derivatives)?,
                        v,
                        p(&cg.active)?,
                        at(5),
                        i(np),
                        i(w),
                    ],
                )?;
                exec.launch(
                    module.function("cg_oy")?,
                    (np.div_ceil(128) as u32, 1, 1),
                    128,
                    0,
                    &[
                        out,
                        p(&cg.st)?,
                        p(&cg.ist)?,
                        p(&b.derivatives)?,
                        p(&cg.y)?,
                        v,
                        p(&cg.active)?,
                        at(5),
                        at(4),
                        i(np),
                        i(w),
                        Arg::D(mp.dsr_opt_sta_del),
                    ],
                )
            };
            // The Main of one variant after cg_prepare or cg_fallback.
            let solve = |diag: bool, trace: Arg| -> tenferro_tensor::Result<()> {
                let variant = i(usize::from(diag));
                if diag {
                    // r = g - S x with x = 0 (stcopt_cg_impl.c:376-381).
                    product(p(&b.rhs)?, at(2))?;
                    exec.launch(
                        module.function("cg_diag_init")?,
                        (1, 1, 1),
                        BLOCK,
                        0,
                        &[
                            at(0),
                            p(&cg.active)?,
                            p(&cg.st)?,
                            p(&cg.ist)?,
                            trace,
                            i(np),
                            factor,
                        ],
                    )?;
                }
                for iteration in 1..=cg.limit {
                    product(at(1), at(3))?;
                    if iteration % 20 == 0 {
                        exec.launch(
                            module.function("cg_alpha")?,
                            (1, 1, 1),
                            BLOCK,
                            0,
                            &[
                                p(&b.rhs)?,
                                at(0),
                                p(&cg.active)?,
                                p(&cg.st)?,
                                p(&cg.ist)?,
                                i(np),
                            ],
                        )?;
                        product(p(&b.rhs)?, at(2))?;
                        exec.launch(
                            module.function("cg_refresh")?,
                            (1, 1, 1),
                            BLOCK,
                            0,
                            &[
                                at(0),
                                p(&cg.active)?,
                                p(&cg.st)?,
                                p(&cg.ist)?,
                                trace,
                                i(np),
                                variant,
                                factor,
                            ],
                        )?;
                    } else {
                        exec.launch(
                            module.function("cg_iterate")?,
                            (1, 1, 1),
                            BLOCK,
                            0,
                            &[
                                p(&b.rhs)?,
                                at(0),
                                p(&cg.active)?,
                                p(&cg.st)?,
                                p(&cg.ist)?,
                                trace,
                                i(np),
                                variant,
                                factor,
                            ],
                        )?;
                    }
                }
                Ok(())
            };
            let trace = exec.ptr(&cg.trace)?;
            solve(diag, Arg::P(trace))?;
            if mp.nsrcg_fallback {
                let second = Arg::P(at8(trace, cg.limit + 1));
                exec.launch(
                    module.function("cg_fallback")?,
                    (1, 1, 1),
                    BLOCK,
                    0,
                    &[
                        p(&b.rhs)?,
                        at(0),
                        p(&cg.active)?,
                        p(&cg.st)?,
                        p(&cg.ist)?,
                        second,
                        i(np),
                        i(usize::from(!diag)),
                        p(&self.resident.error)?,
                    ],
                )?;
                solve(!diag, second)?;
            }
            exec.launch(
                module.function("cg_finish")?,
                (1, 1, 1),
                BLOCK,
                0,
                &[
                    p(&b.rhs)?,
                    p(&cg.active)?,
                    p(&b.info)?,
                    p(&cg.ist)?,
                    p(&cg.iterations)?,
                    p(&cg.fallback)?,
                    i(step),
                    i(np),
                    p(&self.resident.error)?,
                    i(usize::from(mp.nsrcg_abort_on_fail)),
                ],
            )
        }
    }
}
