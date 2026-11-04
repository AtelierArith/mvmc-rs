//! The tenferro measurement Pfaffian stage (issue #613).
//!
//! tenferro 0.9 has no skew-symmetric factorization or Pfaffian operation. This module
//! supplies one as a tenferro *extension operation*: [`PfaffianInverseOp`] is a multi-output
//! node of a tenferro program whose CPU executor runs the pure-Rust PfaPack kernel with the C
//! operation order. [`TenferroPfaffian`] is the [`PfaffianStages`] implementation selected by
//! `MVMC_RS_MEASURE_PF_BACKEND=tenferro`: it compiles and prepares the program once per
//! `(n, planes)` shape on its own tenferro `Runtime`/`CpuEngine`, binds the caller's planes
//! zero-copy as the program input and returns the program outputs.
//!
//! # Operation contract
//!
//! The input is a compact column-major `[n, n, NQP, B]` tensor of skew-symmetric planes
//! (`f64` or `Complex64`, `n` even and at least 2; like PfaPack, only the strict upper triangle
//! is read). The three outputs are
//!
//! * `pf [NQP, B]`: the Pfaffian of each plane,
//! * `inv [n, n, NQP, B]`: the true inverse `X^-1` (mVMC's stored `invM` is its negation),
//! * `status [NQP, B]` (`i32`): `0` success, `k > 0` zero pivot at one-based row `k`, `-1`
//!   non-finite Pfaffian.
//!
//! Inverse planes whose status is not `0` are zero-filled; a zero-pivot plane reports
//! `pf = 0`. The real executor is the production C-order sequence
//! ([`crate::stage_backend::COrderPfaffian`], `dsktf2`/`utu2pfa_real`/`utu2inv_real`) and the
//! complex executor is `zsktf2`/`utu2pfa_complex`/`utu2inv_complex`, so each plane is computed
//! with the scalar C operation order. Independent planes are distributed over the bounded
//! inner Rayon pool exactly like the C-order stage; no other thread team is used.
//!
//! There is no silent fallback: unsupported dtypes or shapes are typed errors.

use std::any::Any;
use std::collections::HashMap;
use std::hash::Hasher;
use std::sync::Arc;

use num_complex::Complex64;
use pfapack::{utu2inv_complex, utu2pfa_complex, zsktf2, PivotIndex1Based, SqMat};
use tenferro_cpu::CpuEngine;
use tenferro_runtime::extension::{
    apply, define_extension_runtime, ExtensionAliasDeclaration, ExtensionEffectDeclaration,
    ExtensionExecutionContext, ExtensionOp, ExtensionShapeContext,
};
use tenferro_runtime::{
    DType, EngineId, ExecutionEngine, GraphCompiler, Prepared, Runtime, SymDim, Tensor,
    TracedTensor,
};
use tenferro_tensor::{Engine, TensorRead, TensorView};

use crate::stage_backend::{PfInvBatch, PfaffianStages, PlaneOutcome, StageError};

/// Operation family shared by every executor of [`PfaffianInverseOp`].
pub const FAMILY: &str = "mvmc.pfaffian-inverse.v1";

/// Batched Pfaffian and true inverse with per-plane `i32` status (see the module docs).
#[derive(Clone, Debug)]
pub struct PfaffianInverseOp;

impl ExtensionOp for PfaffianInverseOp {
    fn family_id(&self) -> &'static str {
        FAMILY
    }
    fn payload_hash(&self, hasher: &mut dyn Hasher) {
        hasher.write_u8(0);
    }
    fn payload_eq(&self, other: &dyn ExtensionOp) -> bool {
        other.as_any().is::<Self>()
    }
    fn clone_arc(&self) -> Arc<dyn ExtensionOp> {
        Arc::new(self.clone())
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn input_count(&self) -> usize {
        1
    }
    fn output_count(&self) -> usize {
        3
    }
    fn semantic_effects(&self) -> ExtensionEffectDeclaration<'_> {
        ExtensionEffectDeclaration::Declared(&[])
    }
    fn semantic_aliases(&self) -> ExtensionAliasDeclaration<'_> {
        ExtensionAliasDeclaration::AllFresh
    }
    fn infer_output_meta(
        &self,
        ctx: &mut ExtensionShapeContext<'_>,
    ) -> tenferro_tensor::Result<Vec<(DType, Vec<SymDim>)>> {
        let dtype = ctx.input_dtype(0)?;
        let shape = ctx.input_shape(0)?.to_vec();
        if shape.len() != 4 {
            return Err(invalid("shape", "expected [n,n,NQP,B]"));
        }
        if let (Some(rows), Some(cols)) = (shape[0].constant_value(), shape[1].constant_value()) {
            if rows != cols || rows < 2 || !rows.is_multiple_of(2) {
                return Err(invalid("shape", "expected an even square matrix side >= 2"));
            }
        }
        if !matches!(dtype, DType::F64 | DType::C64) {
            return Err(invalid("dtype", "expected f64 or complex f64"));
        }
        let planes = shape[2..].to_vec();
        Ok(vec![
            (dtype, planes.clone()),
            (dtype, shape),
            (DType::I32, planes),
        ])
    }
}

fn invalid(field: &'static str, message: &'static str) -> tenferro_tensor::Error {
    tenferro_tensor::Error::invalid_argument(FAMILY, field, message)
}

/// The CPU executor of [`PfaffianInverseOp`], for installation on a runtime builder whose CPU
/// Engine was registered with `engine_id`.
///
/// # Errors
///
/// The tenferro configuration error of the module definition.
pub fn cpu_extension_module(
    engine_id: EngineId,
) -> Result<Arc<dyn tenferro_runtime::ExtensionModule>, tenferro_runtime::RuntimeConfigError> {
    runtime_impl::extension_module::<CpuEngine>(engine_id)
}

mod runtime_impl {
    use super::*;

    define_extension_runtime! {
        runtime = PfaffianInverseRuntime,
        family_id = FAMILY,
        op_type = PfaffianInverseOp,
        execute_reads = execute_reads,
    }
}

fn execute_reads<B: Engine + ?Sized + 'static>(
    _op: &PfaffianInverseOp,
    inputs: &[TensorRead<'_>],
    _ctx: &mut ExtensionExecutionContext<'_, B>,
) -> tenferro_tensor::Result<Vec<Tensor>> {
    let input = &inputs[0];
    let shape = input.shape().to_vec();
    if shape.len() != 4 || shape[0] != shape[1] || shape[0] < 2 || !shape[0].is_multiple_of(2) {
        return Err(invalid("shape", "expected even square [n,n,NQP,B]"));
    }
    let n = shape[0];
    let planes = shape[2]
        .checked_mul(shape[3])
        .ok_or_else(|| invalid("shape", "plane count overflows"))?;
    n.checked_mul(n)
        .and_then(|nn| nn.checked_mul(planes))
        .ok_or_else(|| invalid("shape", "tensor size overflows"))?;
    let plane_shape = shape[2..].to_vec();
    match input.dtype() {
        DType::F64 => {
            // The output inverse is a fresh tensor (`AllFresh`): copy, then factorize in place.
            let mut inv = input.as_slice::<f64>()?.to_vec();
            let out = crate::stage_backend::c_order_pfaffian_in_place(&mut inv, n, planes)
                .map_err(backend_error)?;
            let status = encode_status(&out.outcome, &mut inv, n);
            Ok(vec![
                Tensor::from_vec_col_major(plane_shape.clone(), out.pf)?,
                Tensor::from_vec_col_major(shape, inv)?,
                Tensor::from_vec_col_major(plane_shape, status)?,
            ])
        }
        DType::C64 => {
            let mut inv = input.as_slice::<Complex64>()?.to_vec();
            let (pf, outcome) = complex_planes_in_place(&mut inv, n, planes);
            let status = encode_status(&outcome, &mut inv, n);
            Ok(vec![
                Tensor::from_vec_col_major(plane_shape.clone(), pf)?,
                Tensor::from_vec_col_major(shape, inv)?,
                Tensor::from_vec_col_major(plane_shape, status)?,
            ])
        }
        other => Err(backend_error(format!("unsupported dtype {other:?}"))),
    }
}

fn backend_error(error: impl std::fmt::Display) -> tenferro_tensor::Error {
    tenferro_tensor::Error::backend_source(FAMILY, std::io::Error::other(error.to_string()))
}

/// Status codes of the operation; zero-fills the inverse of every failed plane.
fn encode_status<T: Copy + Default>(outcome: &[PlaneOutcome], inv: &mut [T], n: usize) -> Vec<i32> {
    let nn = n * n;
    outcome
        .iter()
        .enumerate()
        .map(|(p, outcome)| match *outcome {
            PlaneOutcome::Ok => 0,
            PlaneOutcome::ZeroPivot { row } => {
                inv[p * nn..(p + 1) * nn].fill(T::default());
                i32::try_from(row).unwrap_or(i32::MAX)
            }
            PlaneOutcome::NonFinite => {
                inv[p * nn..(p + 1) * nn].fill(T::default());
                -1
            }
        })
        .collect()
}

type ComplexPlaneScratch = (Vec<PivotIndex1Based>, Vec<Complex64>, Vec<Complex64>);
thread_local! {
    static COMPLEX_PLANE_SCRATCH: std::cell::RefCell<ComplexPlaneScratch> = const {
        std::cell::RefCell::new((Vec::new(), Vec::new(), Vec::new()))
    };
}

/// `zsktf2`, `utu2pfa_complex`, `utu2inv_complex` per plane (the complex PfaPack sequence).
fn complex_planes_in_place(
    x: &mut [Complex64],
    n: usize,
    planes: usize,
) -> (Vec<Complex64>, Vec<PlaneOutcome>) {
    let nn = n * n;
    let zero = Complex64::new(0.0, 0.0);
    let mut results = vec![(zero, PlaneOutcome::Ok); planes];
    crate::threading::qp_update(
        &mut results,
        x,
        nn,
        nn,
        0,
        planes,
        // Complex arithmetic is roughly four real multiply-adds per element operation.
        4 * crate::pfaffian::pfaffian_qp_cost_ns(n),
        || (),
        |_, _, slot, a| {
            COMPLEX_PLANE_SCRATCH.with_borrow_mut(|(pivots, vt, work)| {
                pivots.resize(n, PivotIndex1Based(0));
                let result = {
                    let mut view = SqMat::new(&mut *a, n);
                    zsktf2(&mut view, pivots).map(|_| utu2pfa_complex(&view, pivots))
                };
                *slot = match result {
                    Err(row) => (zero, PlaneOutcome::ZeroPivot { row }),
                    Ok(value) if !(value.re.is_finite() && value.im.is_finite()) => {
                        (value, PlaneOutcome::NonFinite)
                    }
                    Ok(value) => {
                        vt.resize(n - 1, zero);
                        work.resize(nn, zero);
                        let mut view = SqMat::new(&mut *a, n);
                        let mut m_work = SqMat::new(work, n);
                        utu2inv_complex(&mut view, pivots, vt, &mut m_work);
                        (value, PlaneOutcome::Ok)
                    }
                };
            })
        },
    );
    results.into_iter().unzip()
}

/// Decode a status code of the operation.
pub fn outcome_from_code(code: i32) -> PlaneOutcome {
    match code {
        0 => PlaneOutcome::Ok,
        c if c > 0 => PlaneOutcome::ZeroPivot { row: c as usize },
        _ => PlaneOutcome::NonFinite,
    }
}

fn stage_error(error: impl std::fmt::Display) -> StageError {
    StageError::Failed(format!("tenferro Pfaffian: {error}"))
}

/// Counters of one [`TenferroPfaffian`] stage.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct TenferroPfaffianStats {
    /// Batched stage calls that ran the program.
    pub calls: u64,
    /// Planes evaluated by those calls.
    pub planes: u64,
    /// Programs compiled and prepared (one per distinct `(n, planes)` shape).
    pub prepared: u64,
}

/// [`PfaffianStages`] through a prepared tenferro program of [`PfaffianInverseOp`].
///
/// The stage owns a serial `CpuEngine` and a `Runtime` with the CPU executor installed. The
/// engine only dispatches the program; the plane loop runs on the inner Rayon pool, so the
/// stage reserves no tenferro worker team.
pub struct TenferroPfaffian {
    runtime: Runtime,
    engine: CpuEngine,
    prepared: HashMap<(usize, usize), Prepared>,
    stats: TenferroPfaffianStats,
}

impl std::fmt::Debug for TenferroPfaffian {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TenferroPfaffian")
            .field("prepared_shapes", &self.prepared.len())
            .field("stats", &self.stats)
            .finish()
    }
}

impl TenferroPfaffian {
    /// Build the runtime, register its CPU engine and install the executor.
    ///
    /// # Errors
    ///
    /// [`StageError::Failed`] when tenferro rejects the configuration.
    pub fn new_cpu() -> Result<Self, StageError> {
        let engine = CpuEngine::serial();
        let engine_id = tenferro_cpu::runtime_engine_id().map_err(stage_error)?;
        let mut builder = Runtime::builder();
        builder
            .register_engine(
                tenferro_cpu::runtime_engine_registration_with_id(&engine, engine_id.clone())
                    .map_err(stage_error)?,
            )
            .map_err(stage_error)?;
        builder
            .install_extension_module(cpu_extension_module(engine_id).map_err(stage_error)?)
            .map_err(stage_error)?;
        let runtime = builder.build().map_err(stage_error)?;
        Ok(Self {
            runtime,
            engine,
            prepared: HashMap::new(),
            stats: TenferroPfaffianStats::default(),
        })
    }

    /// Counters since construction.
    pub fn stats(&self) -> TenferroPfaffianStats {
        self.stats
    }

    fn prepared_for(&mut self, n: usize, planes: usize) -> Result<&Prepared, StageError> {
        if !self.prepared.contains_key(&(n, planes)) {
            let shape = [n, n, planes, 1];
            // `GraphCompiler::compile_many` (the multi-output entry) needs every input to carry
            // a default tensor, so the input is a zero leaf of the exact shape. Each run binds
            // the caller's planes in its place (`Runtime::run` inputs); the default is never
            // read. It is retained with the prepared program (one plane batch per shape).
            let leaf = Tensor::from_vec_col_major(shape.to_vec(), vec![0.0_f64; n * n * planes])
                .map_err(stage_error)?;
            let input = TracedTensor::from_tensor_concrete_shape(leaf).map_err(stage_error)?;
            let outputs = apply(Arc::new(PfaffianInverseOp), &[&input]).map_err(stage_error)?;
            let mut compiler = GraphCompiler::new();
            let program = compiler
                .compile_many(&[&outputs[0], &outputs[1], &outputs[2]])
                .map_err(stage_error)?;
            let prepared = self
                .runtime
                .prepare(&program, &[], &[self.engine.target()])
                .map_err(stage_error)?;
            self.prepared.insert((n, planes), prepared);
            self.stats.prepared += 1;
        }
        Ok(&self.prepared[&(n, planes)])
    }
}

impl PfaffianStages for TenferroPfaffian {
    fn label(&self) -> String {
        "tenferro-cpu".to_string()
    }

    fn provider(&self) -> String {
        format!("tenferro-runtime prepared {FAMILY} extension (pfapack C order, CpuEngine)")
    }

    fn pfaffian_inverse_batch(
        &mut self,
        x: &[f64],
        n: usize,
        planes: usize,
    ) -> Result<PfInvBatch, StageError> {
        let expected = n
            .checked_mul(n)
            .and_then(|nn| nn.checked_mul(planes))
            .ok_or_else(|| stage_error("n*n*planes overflows"))?;
        if x.len() != expected {
            return Err(stage_error(format!(
                "buffer length {} != n*n*planes = {expected}",
                x.len()
            )));
        }
        if n < 2 || !n.is_multiple_of(2) {
            return Err(StageError::Unsupported(format!(
                "tenferro Pfaffian needs an even matrix side >= 2, got {n}"
            )));
        }
        if planes == 0 {
            return Ok(PfInvBatch {
                pf: Vec::new(),
                inv: Vec::new(),
                outcome: Vec::new(),
            });
        }
        self.prepared_for(n, planes)?;
        let prepared = &self.prepared[&(n, planes)];
        let shape = [n, n, planes, 1];
        let view = TensorView::f64(&shape, x).map_err(stage_error)?;
        let engine: &mut dyn ExecutionEngine = &mut self.engine;
        let outputs = self
            .runtime
            .run(
                &mut [&mut *engine],
                prepared,
                &[TensorRead::from_view(view)],
            )
            .map_err(stage_error)?;
        let mut tensors = Vec::with_capacity(3);
        for output in outputs {
            tensors.push(output.into_tensor(&mut *engine).map_err(stage_error)?);
        }
        let [pf, inv, status] = <[Tensor; 3]>::try_from(tensors)
            .map_err(|t| stage_error(format!("expected 3 outputs, got {}", t.len())))?;
        let (_, pf) = pf
            .into_vec_col_major::<f64>()
            .map_err(|e| stage_error(format!("{e:?}")))?;
        let (_, inv) = inv
            .into_vec_col_major::<f64>()
            .map_err(|e| stage_error(format!("{e:?}")))?;
        let outcome = status
            .as_slice::<i32>()
            .map_err(stage_error)?
            .iter()
            .copied()
            .map(outcome_from_code)
            .collect();
        self.stats.calls += 1;
        self.stats.planes += planes as u64;
        Ok(PfInvBatch { pf, inv, outcome })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stage_backend::COrderPfaffian;

    /// Deterministic skew-symmetric planes from a 64-bit LCG (values in `[-1, 1)`).
    fn skew_planes(n: usize, planes: usize, seed: u64) -> Vec<f64> {
        let mut state = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
        };
        let mut x = vec![0.0; n * n * planes];
        for p in 0..planes {
            let a = &mut x[p * n * n..(p + 1) * n * n];
            for j in 0..n {
                for i in 0..j {
                    let v = next();
                    a[i + j * n] = v;
                    a[j + i * n] = -v;
                }
            }
        }
        x
    }

    /// Bound of one compared value: the tenferro executor runs the C-order PfaPack sequence
    /// itself (same scalar operation order), so no roundoff difference is expected; allow two
    /// ulps of the plane's largest reference magnitude so the check does not rely on bitwise
    /// equality of computed floating-point values (#190).
    fn assert_close(label: &str, got: &[f64], want: &[f64]) {
        assert_eq!(got.len(), want.len(), "{label}: length");
        let scale = want
            .iter()
            .fold(0.0_f64, |m, v| m.max(v.abs()))
            .max(f64::MIN_POSITIVE);
        let bound = 2.0 * f64::EPSILON * scale;
        for (k, (g, w)) in got.iter().zip(want).enumerate() {
            assert!(
                (g - w).abs() <= bound,
                "{label}[{k}]: {g} vs {w} (bound {bound:e})"
            );
        }
    }

    #[test]
    fn tenferro_pfaffian_matches_the_c_order_stage() {
        let mut stage = TenferroPfaffian::new_cpu().unwrap();
        for (n, planes) in [(2, 1), (4, 3), (8, 5), (16, 2), (30, 4), (64, 3)] {
            let x = skew_planes(n, planes, (n * 31 + planes) as u64);
            let want = COrderPfaffian
                .pfaffian_inverse_batch(&x, n, planes)
                .unwrap();
            let got = stage.pfaffian_inverse_batch(&x, n, planes).unwrap();
            assert_eq!(got.outcome, want.outcome, "n={n}");
            assert!(got.outcome.iter().all(|o| *o == PlaneOutcome::Ok));
            assert_close(&format!("pf n={n}"), &got.pf, &want.pf);
            for p in 0..planes {
                let r = p * n * n..(p + 1) * n * n;
                assert_close(
                    &format!("inv n={n} p={p}"),
                    &got.inv[r.clone()],
                    &want.inv[r],
                );
            }
        }
    }

    #[test]
    fn prepared_program_is_reused_per_shape() {
        let mut stage = TenferroPfaffian::new_cpu().unwrap();
        let x = skew_planes(6, 2, 7);
        for _ in 0..3 {
            stage.pfaffian_inverse_batch(&x, 6, 2).unwrap();
        }
        // The reused program reads each run's input, not the default tensor it was built with.
        let other = skew_planes(6, 2, 8);
        let want = COrderPfaffian.pfaffian_inverse_batch(&other, 6, 2).unwrap();
        let got = stage.pfaffian_inverse_batch(&other, 6, 2).unwrap();
        assert_close("reused pf", &got.pf, &want.pf);
        assert_close("reused inv", &got.inv, &want.inv);
        stage.pfaffian_inverse_batch(&x[..36], 6, 1).unwrap();
        let stats = stage.stats();
        assert_eq!(stats.prepared, 2);
        assert_eq!(stats.calls, 5);
        assert_eq!(stats.planes, 9);
    }

    #[test]
    fn singular_and_non_finite_planes_report_the_c_order_status() {
        let n = 4;
        let mut x = skew_planes(n, 3, 11);
        // Plane 1: exactly zero (pfapack reports the first zero pivot column).
        x[n * n..2 * n * n].fill(0.0);
        // Plane 2: overflowing entries give a non-finite Pfaffian.
        for v in &mut x[2 * n * n..] {
            *v *= 1e300;
        }
        let want = COrderPfaffian.pfaffian_inverse_batch(&x, n, 3).unwrap();
        let got = TenferroPfaffian::new_cpu()
            .unwrap()
            .pfaffian_inverse_batch(&x, n, 3)
            .unwrap();
        assert_eq!(got.outcome, want.outcome);
        assert_eq!(got.outcome[0], PlaneOutcome::Ok);
        assert!(matches!(got.outcome[1], PlaneOutcome::ZeroPivot { .. }));
        assert_eq!(got.outcome[2], PlaneOutcome::NonFinite);
        assert_eq!(got.pf[1], 0.0);
        // Failed planes are zero-filled by the operation contract.
        assert!(got.inv[n * n..].iter().all(|&v| v == 0.0));
        assert_close("pf ok", &got.pf[..1], &want.pf[..1]);
        assert_close("inv ok", &got.inv[..n * n], &want.inv[..n * n]);
    }

    #[test]
    fn invalid_shapes_are_typed_errors_and_empty_batches_succeed() {
        let mut stage = TenferroPfaffian::new_cpu().unwrap();
        assert!(matches!(
            stage.pfaffian_inverse_batch(&[0.0; 9], 3, 1),
            Err(StageError::Unsupported(_))
        ));
        assert!(matches!(
            stage.pfaffian_inverse_batch(&[0.0; 15], 4, 1),
            Err(StageError::Failed(_))
        ));
        let empty = stage.pfaffian_inverse_batch(&[], 4, 0).unwrap();
        assert!(empty.pf.is_empty() && empty.inv.is_empty() && empty.outcome.is_empty());
        assert_eq!(stage.stats().calls, 0);
    }
}
