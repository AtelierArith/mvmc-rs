//! Device-resident SR-CG (`NSRCG != 0`, #670): C `StochasticOptCG_real`
//! (`stcopt_cg_impl.c`) on the resident derivative matrix, without host round
//! trips. See `cg.cu` for the kernels and their C source lines.
//!
//! The device decides the C stopping rule. The host enqueues an upper bound of
//! iterations (`NSROptCGMaxIter`, or the parameter count when it is 0, since
//! `nSmat <= NPara`); after the stop flag is set, each remaining iteration costs
//! only the launch of kernels that return at once.
use super::{i, State};
use tenferro_ext::raw_exec::{at8, Arg, DeviceBuffer, RawExec, RawModule};

/// Device CG state, allocated only for Opt with `NSRCG != 0`.
pub(super) struct CgBuffers {
    /// `[g, d, r, q, sdiag, <O>]`, each `np` long (masked to the active set).
    pub vec: DeviceBuffer,
    /// 1 for an active (optimized, not diagonal-cut) parameter.
    pub active: DeviceBuffer,
    /// `[delta, threshold, alpha, <O>.x, 1/Wc]`.
    pub st: DeviceBuffer,
    /// `[stop, iterations, max iterations, nSmat]`.
    pub ist: DeviceBuffer,
    /// Per-sample product `O^T v`.
    pub y: DeviceBuffer,
    /// `delta` after initialization and each iteration of the latest solve.
    pub trace: DeviceBuffer,
    /// C iteration count (`StochasticOptCG_Main` return value) of each step.
    pub iterations: DeviceBuffer,
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

impl CgBuffers {
    pub fn allocate(
        exec: &RawExec<'_, '_>,
        np: usize,
        walkers: usize,
        steps: usize,
        limit: usize,
    ) -> tenferro_tensor::Result<Self> {
        let alloc = |count: usize, size: usize| exec.alloc_bytes((count * size).max(1));
        Ok(Self {
            vec: alloc(6 * np, 8)?,
            active: alloc(np, 4)?,
            st: alloc(8, 8)?,
            ist: alloc(4, 4)?,
            y: alloc(walkers, 8)?,
            trace: alloc(limit + 1, 8)?,
            iterations: alloc(steps, 4)?,
            limit,
        })
    }
}

impl State {
    /// One SR-CG step: moments, C diagonal cut and gradient, the CG loop with
    /// the 20-iteration residual refresh, and the finiteness check. The
    /// increment is left in `rhs` for `opt_update`, with `info` cleared.
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
        // SAFETY: every buffer was allocated with the extents used below (np,
        // np+1 for mean, 6*np for vec, walkers*np for derivatives, limit+1 for
        // trace, steps for iterations and history).
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
                    p(&b.derivatives)?,
                    p(&b.local_energy)?,
                    i(np),
                    i(w),
                ],
            )?;
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
                ],
            )?;
            // q = S v into `out`, for v = d (ordinary) or x (refresh).
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
                            p(&cg.trace)?,
                            i(np),
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
                            p(&cg.trace)?,
                            i(np),
                        ],
                    )?;
                }
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
                    i(step),
                    i(np),
                    p(&self.resident.error)?,
                ],
            )
        }
    }
}
