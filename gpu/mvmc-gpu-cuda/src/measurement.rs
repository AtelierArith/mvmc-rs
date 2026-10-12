//! Device-resident measurement service (issue #518 milestone 2).
//!
//! One [`ResidentDevice`] per CUDA device owns the measurement tables of a series:
//!
//! * the Slater table `S` (uploaded once per content version, issue #518),
//! * the per-sample skew planes and their factorization (`stage A`: assemble from `S` and the
//!   electron sites, batched `pfinv_f64`, inverse planes kept on the device),
//! * the per-sample electron rows and the small per-sample kernel inputs,
//! * the input-fixed orbital/weight tables of the resident stage B.
//!
//! Two stage objects share the device through an `Arc<Mutex<..>>`: [`ResidentStageA`] fills the
//! tables in `stage_a_resident` ([`mvmc_core::stage_backend::PfaffianStages`]) and
//! [`ResidentMeasurementStage`] consumes them in `measure_sample` — the resident local-energy
//! ratio kernel and Slater-derivative kernels
//! ([`mvmc_core::measurement_backend::ResidentMeasurementOps`]). The host orchestration of both
//! stages stays in `mvmc-core`; this module implements the device operations.
//!
//! Per sample inside a series the only transfers are the sample's own electron rows (a few KB),
//! the small kernel inputs (term indices) and the outputs (one local energy plus the derivative
//! block). The inverse planes cross the link only when the caller asked for host tables
//! (PhysCal Green functions, Lanczos, or a host measurement stage B).
//!
//! No silent fallback: a missing device, an unsupported geometry or a failed kernel is a typed
//! [`StageError`].

use std::sync::{Arc, Mutex};

use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};
use mvmc_core::c_timer::CTimer;
use mvmc_core::measurement_backend::{
    ResidentGreenTwo, ResidentMeasurementOps, ResidentRatioTerms, ResidentSlaterDiff,
    TenferroMeasurement,
};
use mvmc_core::stage_backend::{
    MeasurementInput, MeasurementOutput, MeasurementStages, MeasurementTiming, PfInvBatch,
    PfaffianStages, PlaneOutcome, ResidentInputRequest, ResidentStageAInput, ResidentStageAOutput,
    ResidentStats, StageError,
};
use mvmc_core::state::VmcOptimizationState;
use mvmc_core::ExpertModeData;
use mvmc_gpu::stages::BatchedStages;

use crate::pfaffian::launch_geometry;

const SRC_PFAFFIAN: &str = include_str!("pfaffian_batched.cu");
const SRC_MEASUREMENT: &str = include_str!("measurement_kernels.cu");

fn cu(e: impl std::fmt::Debug) -> StageError {
    StageError::Failed(format!("{e:?}"))
}

fn unsupported<T>(reason: impl std::fmt::Display) -> Result<T, StageError> {
    Err(StageError::Unsupported(reason.to_string()))
}

/// Fixed geometry of one resident series.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Geom {
    n_site: usize,
    n_elec: usize,
    n_qp: usize,
    n_sp: usize,
    n_slater: usize,
}

impl Geom {
    fn n(&self) -> usize {
        2 * self.n_elec
    }
    fn n2(&self) -> usize {
        2 * self.n_site
    }
}

struct Funcs {
    assemble: CudaFunction,
    pfinv: CudaFunction,
    pfinv_wide: CudaFunction,
    ratio: CudaFunction,
    green_two: CudaFunction,
    rows: CudaFunction,
    fold: CudaFunction,
}

/// Grow-only stage A buffers.
struct StageBuffers {
    cap: usize,
    n: usize,
    n_qp: usize,
    ele: CudaSlice<i32>,
    planes: CudaSlice<f64>,
    work: CudaSlice<f64>,
    out: CudaSlice<f64>,
    pf: CudaSlice<f64>,
    st: CudaSlice<i32>,
}

fn ensure_f64(
    slot: &mut Option<CudaSlice<f64>>,
    stream: &Arc<CudaStream>,
    len: usize,
) -> Result<(), StageError> {
    if slot.as_ref().is_none_or(|b| b.len() < len.max(1)) {
        *slot = Some(stream.alloc_zeros(len.max(1)).map_err(cu)?);
    }
    Ok(())
}

fn ensure_i32(
    slot: &mut Option<CudaSlice<i32>>,
    stream: &Arc<CudaStream>,
    len: usize,
) -> Result<(), StageError> {
    if slot.as_ref().is_none_or(|b| b.len() < len.max(1)) {
        *slot = Some(stream.alloc_zeros(len.max(1)).map_err(cu)?);
    }
    Ok(())
}

fn upload_f64(
    slot: &mut Option<CudaSlice<f64>>,
    stream: &Arc<CudaStream>,
    data: &[f64],
) -> Result<(), StageError> {
    if data.is_empty() {
        return Ok(());
    }
    ensure_f64(slot, stream, data.len())?;
    let dev = slot.as_mut().expect("table");
    stream
        .memcpy_htod(data, &mut dev.slice_mut(..data.len()))
        .map_err(cu)
}

fn upload_i32(
    slot: &mut Option<CudaSlice<i32>>,
    stream: &Arc<CudaStream>,
    data: &[i32],
) -> Result<(), StageError> {
    if data.is_empty() {
        return Ok(());
    }
    ensure_i32(slot, stream, data.len())?;
    let dev = slot.as_mut().expect("table");
    stream
        .memcpy_htod(data, &mut dev.slice_mut(..data.len()))
        .map_err(cu)
}

fn ensure_stage(
    slot: &mut Option<StageBuffers>,
    stream: &Arc<CudaStream>,
    geom: Geom,
    slots: usize,
) -> Result<(), StageError> {
    let rebuild = match slot.as_ref() {
        Some(stage) => stage.n != geom.n() || stage.n_qp != geom.n_qp || stage.cap < slots,
        None => true,
    };
    if rebuild {
        let cap = slots.max(1).next_power_of_two();
        let nn = geom.n() * geom.n();
        *slot = Some(StageBuffers {
            cap,
            n: geom.n(),
            n_qp: geom.n_qp,
            ele: stream.alloc_zeros(cap * geom.n()).map_err(cu)?,
            planes: stream.alloc_zeros(cap * geom.n_qp * nn).map_err(cu)?,
            work: stream.alloc_zeros(cap * geom.n_qp * nn).map_err(cu)?,
            out: stream.alloc_zeros(cap * geom.n_qp * nn).map_err(cu)?,
            pf: stream.alloc_zeros(cap * geom.n_qp).map_err(cu)?,
            st: stream.alloc_zeros(cap * geom.n_qp).map_err(cu)?,
        });
    }
    Ok(())
}

/// Device state of the resident measurement (one CUDA device).
pub struct ResidentDevice {
    device: usize,
    stream: Arc<CudaStream>,
    funcs: Funcs,
    geom: Option<Geom>,
    /// Slater table `[qp][n2*n2]` and the content version it was uploaded from.
    slater: Option<CudaSlice<f64>>,
    slater_version: u64,
    /// Input-fixed tables of the Slater derivative: CSR per `(translated site, orbital)`.
    orbital_offsets: Option<CudaSlice<i32>>,
    orbital_trj: Option<CudaSlice<i32>>,
    orbital_sgn: Option<CudaSlice<i32>>,
    qp_weight: Option<CudaSlice<f64>>,
    cossin: Option<CudaSlice<f64>>,
    coscos: Option<CudaSlice<f64>>,
    sinsin: Option<CudaSlice<f64>>,
    stage: Option<StageBuffers>,
    /// Slots whose tables are resident from the last stage A call.
    resident_slots: usize,
    /// Per-sample kernel inputs/outputs.
    ratio_msa: Option<CudaSlice<i32>>,
    ratio_rsa: Option<CudaSlice<i32>>,
    ratio_ri: Option<CudaSlice<i32>>,
    ratio_out: Option<CudaSlice<f64>>,
    diff_tri: Option<CudaSlice<i32>>,
    diff_sgn: Option<CudaSlice<i32>>,
    diff_inv_trans: Option<CudaSlice<i32>>,
    diff_inv_sgn: Option<CudaSlice<i32>>,
    diff_rows: Option<CudaSlice<f64>>,
    diff_acc: Option<CudaSlice<f64>>,
    /// Per-job arrays of the two-body Green reduction (issue #517 milestone 5).
    green_msa: Option<CudaSlice<i32>>,
    green_msb: Option<CudaSlice<i32>>,
    green_rsa: Option<CudaSlice<i32>>,
    green_rsb: Option<CudaSlice<i32>>,
    green_out: Option<CudaSlice<f64>>,
    stats: ResidentStats,
}

impl ResidentDevice {
    /// Compile the kernels and open the device state.
    pub fn new(device: usize) -> Result<Self, String> {
        let ctx = CudaContext::new(device).map_err(|e| format!("{e:?}"))?;
        let stream = ctx.default_stream();
        let src = format!("{SRC_PFAFFIAN}\n{SRC_MEASUREMENT}");
        let ptx = compile_ptx_with_opts(
            &src,
            CompileOptions {
                fmad: Some(false),
                arch: Some("compute_80"),
                ..Default::default()
            },
        )
        .map_err(|e| format!("NVRTC: {e:?}"))?;
        let module = ctx.load_module(ptx).map_err(|e| format!("{e:?}"))?;
        let f = |name: &str| {
            module
                .load_function(name)
                .map_err(|e| format!("{name}: {e:?}"))
        };
        let funcs = Funcs {
            assemble: f("km_assemble")?,
            pfinv: f("pfinv_f64")?,
            pfinv_wide: f("pfinv_f64_wide")?,
            ratio: f("km_ratio_terms")?,
            green_two: f("km_green_two")?,
            rows: f("km_slater_rows")?,
            fold: f("km_slater_fold")?,
        };
        Ok(Self {
            device,
            stream,
            funcs,
            geom: None,
            slater: None,
            slater_version: u64::MAX,
            orbital_offsets: None,
            orbital_trj: None,
            orbital_sgn: None,
            qp_weight: None,
            cossin: None,
            coscos: None,
            sinsin: None,
            stage: None,
            resident_slots: 0,
            ratio_msa: None,
            ratio_rsa: None,
            ratio_ri: None,
            ratio_out: None,
            diff_tri: None,
            diff_sgn: None,
            diff_inv_trans: None,
            diff_inv_sgn: None,
            diff_rows: None,
            diff_acc: None,
            green_msa: None,
            green_msb: None,
            green_rsa: None,
            green_rsb: None,
            green_out: None,
            stats: ResidentStats::default(),
        })
    }

    /// Device ordinal.
    pub fn device(&self) -> usize {
        self.device
    }

    /// Make the device context current on the calling thread. The shared backend is guarded by
    /// a mutex, but its methods can run on different threads (the production runner and the
    /// gates), and a cudarc context is only current on the thread that created it.
    fn bind(&self) -> Result<(), StageError> {
        self.stream.context().bind_to_thread().map_err(cu)
    }

    fn geom(&self) -> Result<Geom, StageError> {
        self.geom.ok_or_else(|| {
            StageError::Failed("resident measurement: prepare_input was not called".into())
        })
    }

    fn download_f64(
        &self,
        dev: &CudaSlice<f64>,
        len: usize,
        out: &mut Vec<f64>,
    ) -> Result<(), StageError> {
        out.clear();
        out.resize(len, 0.0);
        self.stream.memcpy_dtoh(&dev.slice(..len), out).map_err(cu)
    }

    fn download_i32(
        &self,
        dev: &CudaSlice<i32>,
        len: usize,
        out: &mut Vec<i32>,
    ) -> Result<(), StageError> {
        out.clear();
        out.resize(len, 0);
        self.stream.memcpy_dtoh(&dev.slice(..len), out).map_err(cu)
    }

    /// Build the per-`(translated site, orbital)` CSR of the input-fixed orbital matrices.
    #[allow(clippy::type_complexity)]
    fn build_orbital_csr(
        request: &ResidentInputRequest<'_>,
    ) -> Result<(Vec<i32>, Vec<i32>, Vec<i32>), StageError> {
        let (n_site, n_slater) = (request.n_site, request.n_slater);
        if n_site == 0 || n_slater == 0 {
            return Ok((Vec::new(), Vec::new(), Vec::new()));
        }
        if request.orbital_idx.len() < n_site || request.orbital_sgn.len() < n_site {
            return unsupported("resident measurement: incomplete orbital matrices");
        }
        let mut counts = vec![0usize; n_site * n_slater];
        let mut entries: Vec<(usize, usize, i32, i32)> = Vec::new();
        for tri in 0..n_site {
            let idx_row = &request.orbital_idx[tri];
            let sgn_row = &request.orbital_sgn[tri];
            if idx_row.len() < n_site || sgn_row.len() < n_site {
                return unsupported("resident measurement: orbital matrices shorter than Nsite");
            }
            for trj in 0..n_site {
                let o = idx_row[trj];
                if o < 0 || o as usize >= n_slater {
                    continue;
                }
                counts[tri * n_slater + o as usize] += 1;
                entries.push((tri, o as usize, trj as i32, sgn_row[trj] as i32));
            }
        }
        let mut offsets = vec![0i32; n_site * (n_slater + 1)];
        for tri in 0..n_site {
            let mut acc = 0i32;
            for o in 0..n_slater {
                offsets[tri * (n_slater + 1) + o] = acc;
                acc += counts[tri * n_slater + o] as i32;
            }
            offsets[tri * (n_slater + 1) + n_slater] = acc;
        }
        let mut cursor: Vec<usize> = (0..n_site * n_slater)
            .map(|k| offsets[(k / n_slater) * (n_slater + 1) + (k % n_slater)] as usize)
            .collect();
        let mut list_trj = vec![0i32; entries.len()];
        let mut list_sgn = vec![0i32; entries.len()];
        for (tri, o, trj, sgn) in entries {
            let k = cursor[tri * n_slater + o];
            cursor[tri * n_slater + o] += 1;
            let at = tri * n_site + k;
            list_trj[at] = trj;
            list_sgn[at] = sgn;
        }
        Ok((offsets, list_trj, list_sgn))
    }

    /// Stage A: assemble the requested samples from the resident Slater table, factorize them
    /// on the device and keep the inverse planes resident.
    fn stage_a_run(
        &mut self,
        input: &ResidentStageAInput<'_>,
    ) -> Result<ResidentStageAOutput, StageError> {
        self.bind()?;
        match self.geom {
            Some(geom)
                if input.n_site != geom.n_site
                    || input.n_elec != geom.n_elec
                    || input.n_qp != geom.n_qp =>
            {
                return unsupported(format!(
                    "resident measurement: geometry {} sites / {} electrons / {} QP does not \
                     match the prepared {} / {} / {}",
                    input.n_site, input.n_elec, input.n_qp, geom.n_site, geom.n_elec, geom.n_qp
                ));
            }
            // A run that only selects the resident stage A (a host stage B) never calls
            // `prepare_input`; the geometry is then taken from the request.
            None => {
                self.geom = Some(Geom {
                    n_site: input.n_site,
                    n_elec: input.n_elec,
                    n_qp: input.n_qp,
                    n_sp: 1,
                    n_slater: 0,
                });
            }
            Some(_) => {}
        }
        let geom = self.geom()?;
        if input.slots == 0 {
            return Ok(ResidentStageAOutput {
                pf: Vec::new(),
                inv: None,
                outcome: Vec::new(),
            });
        }
        let n = geom.n();
        let nn = n * n;
        let planes = input.slots * geom.n_qp;
        if input.ele_idx.len() != input.slots * n {
            return Err(StageError::Failed(format!(
                "resident measurement: ele_idx length {} != slots*n = {}",
                input.ele_idx.len(),
                input.slots * n
            )));
        }
        if self.slater.is_none() || self.slater_version != input.slater.version() {
            let n2 = geom.n2();
            let expected = geom.n_qp * n2 * n2;
            let data = input.slater.as_slice();
            if data.len() != expected {
                return Err(StageError::Failed(format!(
                    "resident measurement: Slater table length {} != {}",
                    data.len(),
                    expected
                )));
            }
            ensure_f64(&mut self.slater, &self.stream, expected)?;
            let dev = self.slater.as_mut().expect("slater");
            self.stream
                .memcpy_htod(data, &mut dev.slice_mut(..expected))
                .map_err(cu)?;
            self.slater_version = input.slater.version();
            self.stats.slater_uploads += 1;
            self.stats.slater_upload_bytes += (expected * 8) as u64;
        }
        ensure_stage(&mut self.stage, &self.stream, geom, input.slots)?;
        let mut stage = self.stage.take().expect("stage buffers");
        let ele_i32: Vec<i32> = input
            .ele_idx
            .iter()
            .map(|&v| {
                if v < 0 || v > i32::MAX as i64 {
                    i32::MIN
                } else {
                    v as i32
                }
            })
            .collect();
        let prepare = self
            .stream
            .memcpy_htod(&ele_i32, &mut stage.ele.slice_mut(..input.slots * n));
        if let Err(error) = prepare {
            self.stage = Some(stage);
            return Err(cu(error));
        }
        // Latency-bound small batches (the sampler's 8-plane recompute shape and a measurement
        // batch that fills only a few SMs) use the wide-block entry point; see
        // `pfaffian::launch_geometry_wide`.
        let wide = crate::pfaffian::use_wide_f64(n, 8, 1);
        let (nt, shared) = if wide {
            crate::pfaffian::launch_geometry_wide(n)
        } else {
            launch_geometry(n, 8, 1)
        };
        let (n_i, ne_i, ns_i, nq_i) = (
            n as i32,
            geom.n_elec as i32,
            geom.n_site as i32,
            geom.n_qp as i32,
        );
        let result = (|| -> Result<(), StageError> {
            let slater = self.slater.as_ref().expect("slater uploaded");
            let mut b = self.stream.launch_builder(&self.funcs.assemble);
            b.arg(slater)
                .arg(&stage.ele)
                .arg(&mut stage.planes)
                .arg(&n_i)
                .arg(&ne_i)
                .arg(&ns_i)
                .arg(&nq_i);
            // SAFETY: kernel ABI (const double*, const int*, double*, int x4); one block per
            // (slot, qp) plane, buffers sized for `slots * n_qp` planes.
            unsafe {
                b.launch(LaunchConfig {
                    grid_dim: (planes as u32, 1, 1),
                    block_dim: (128, 1, 1),
                    shared_mem_bytes: 0,
                })
            }
            .map_err(cu)?;
            let mut b = self.stream.launch_builder(if wide {
                &self.funcs.pfinv_wide
            } else {
                &self.funcs.pfinv
            });
            b.arg(&mut stage.planes)
                .arg(&mut stage.work)
                .arg(&mut stage.out)
                .arg(&mut stage.pf)
                .arg(&mut stage.st)
                .arg(&n_i);
            // SAFETY: kernel ABI (double* x4, int*, int); one block per plane, dynamic shared
            // memory from `pfaffian::launch_geometry`.
            unsafe {
                b.launch(LaunchConfig {
                    grid_dim: (planes as u32, 1, 1),
                    block_dim: (nt, 1, 1),
                    shared_mem_bytes: shared,
                })
            }
            .map_err(cu)?;
            self.stream.synchronize().map_err(cu)
        })();
        if let Err(error) = result {
            self.stage = Some(stage);
            return Err(error);
        }
        let mut st = Vec::new();
        self.download_i32(&stage.st, planes, &mut st)?;
        let outcome: Vec<PlaneOutcome> = st
            .iter()
            .map(|&code| match code {
                0 => PlaneOutcome::Ok,
                c if c > 0 => PlaneOutcome::ZeroPivot { row: c as usize },
                _ => PlaneOutcome::NonFinite,
            })
            .collect();
        let mut pf = Vec::new();
        self.download_f64(&stage.pf, planes, &mut pf)?;
        let inv = if input.download_inv {
            let mut raw = Vec::new();
            self.download_f64(&stage.out, planes * nn, &mut raw)?;
            // mVMC invM = -X^-1 (calc_m_all_real's sign flip).
            for value in raw.iter_mut() {
                *value = -*value;
            }
            self.stats.inv_download_planes += planes as u64;
            self.stats.inv_download_bytes += (planes * nn * 8) as u64;
            Some(raw)
        } else {
            None
        };
        self.stage = Some(stage);
        self.stats.stage_a_calls += 1;
        self.stats.planes_factorized += planes as u64;
        self.stats.pf_download_bytes += (planes * 8 + planes * 4) as u64;
        self.resident_slots = input.slots;
        Ok(ResidentStageAOutput { pf, inv, outcome })
    }
}

impl ResidentMeasurementOps for ResidentDevice {
    fn prepare_input(&mut self, request: &ResidentInputRequest<'_>) -> Result<(), StageError> {
        self.bind()?;
        let geom = Geom {
            n_site: request.n_site,
            n_elec: request.n_elec,
            n_qp: request.n_qp,
            n_sp: request.n_sp_gauss_leg,
            n_slater: request.n_slater,
        };
        if geom.n_site == 0 || geom.n_elec == 0 || geom.n_qp == 0 || geom.n_sp == 0 {
            return unsupported("resident measurement: empty geometry");
        }
        if !geom.n_qp.is_multiple_of(geom.n_sp) {
            return unsupported("resident measurement: NQP must be a multiple of NSPGaussLeg");
        }
        // A new geometry invalidates the resident tables; every sweep also starts without
        // resident slots (the sweep's own stage A fills them), so a left-over table from an
        // earlier series can never be consumed by the resident stage B.
        self.resident_slots = 0;
        if self.geom != Some(geom) {
            self.slater = None;
            self.slater_version = u64::MAX;
            self.stage = None;
            self.ratio_msa = None;
            self.ratio_rsa = None;
            self.ratio_ri = None;
            self.ratio_out = None;
            self.diff_tri = None;
            self.diff_sgn = None;
            self.diff_inv_trans = None;
            self.diff_inv_sgn = None;
            self.diff_rows = None;
            self.diff_acc = None;
            self.green_msa = None;
            self.green_msb = None;
            self.green_rsa = None;
            self.green_rsb = None;
            self.green_out = None;
        }
        self.geom = Some(geom);
        let (offsets, trj, sgn) = Self::build_orbital_csr(request)?;
        upload_i32(&mut self.orbital_offsets, &self.stream, &offsets)?;
        upload_i32(&mut self.orbital_trj, &self.stream, &trj)?;
        upload_i32(&mut self.orbital_sgn, &self.stream, &sgn)?;
        upload_f64(&mut self.qp_weight, &self.stream, request.qp_weight)?;
        upload_f64(&mut self.cossin, &self.stream, request.spgl_cos_sin)?;
        upload_f64(&mut self.coscos, &self.stream, request.spgl_cos_cos)?;
        upload_f64(&mut self.sinsin, &self.stream, request.spgl_sin_sin)?;
        Ok(())
    }

    fn stage_a(
        &mut self,
        input: &ResidentStageAInput<'_>,
    ) -> Result<ResidentStageAOutput, StageError> {
        self.stage_a_run(input)
    }

    fn has_slot(&self, slot: usize) -> bool {
        self.geom.is_some() && slot < self.resident_slots
    }

    fn ratio_terms(
        &mut self,
        slot: usize,
        input: &ResidentRatioTerms<'_>,
    ) -> Result<Vec<f64>, StageError> {
        self.bind()?;
        let geom = self.geom()?;
        if !self.has_slot(slot) {
            return unsupported(format!(
                "resident measurement: slot {slot} has no resident tables from stage A"
            ));
        }
        let n = geom.n();
        let terms = input.msa.len();
        if terms == 0 {
            return Ok(Vec::new());
        }
        if input.rsa.len() != terms || input.ri.len() != terms {
            return Err(StageError::Failed(
                "resident measurement: inconsistent active-term arrays".into(),
            ));
        }
        if input.ele_idx.len() != n {
            return Err(StageError::Failed(format!(
                "resident measurement: ele_idx length {} != n = {n}",
                input.ele_idx.len()
            )));
        }
        // The sample's own electron rows move (the only per-sample configuration traffic).
        let mut stage = self.stage.take().expect("stage buffers");
        let ele_i32: Vec<i32> = input.ele_idx.iter().map(|&v| v as i32).collect();
        let msa: Vec<i32> = input.msa.iter().map(|&v| v as i32).collect();
        let rsa: Vec<i32> = input.rsa.iter().map(|&v| v as i32).collect();
        let ri: Vec<i32> = input.ri.iter().map(|&v| v as i32).collect();
        let prepare = (|| -> Result<(), StageError> {
            self.stream
                .memcpy_htod(&ele_i32, &mut stage.ele.slice_mut(slot * n..(slot + 1) * n))
                .map_err(cu)?;
            upload_i32(&mut self.ratio_msa, &self.stream, &msa)?;
            upload_i32(&mut self.ratio_rsa, &self.stream, &rsa)?;
            upload_i32(&mut self.ratio_ri, &self.stream, &ri)?;
            ensure_f64(&mut self.ratio_out, &self.stream, terms)
        })();
        if let Err(error) = prepare {
            self.stage = Some(stage);
            return Err(error);
        }
        let (ele0, plane0) = ((slot * n) as i32, (slot * geom.n_qp) as i32);
        let (terms_i, n_i, ne_i, ns_i, nq_i) = (
            terms as i32,
            n as i32,
            geom.n_elec as i32,
            geom.n_site as i32,
            geom.n_qp as i32,
        );
        let weights = self.qp_weight.as_ref().expect("qp weights");
        let slater = self.slater.as_ref().expect("slater uploaded");
        let msa_dev = self.ratio_msa.as_ref().expect("msa");
        let rsa_dev = self.ratio_rsa.as_ref().expect("rsa");
        let ri_dev = self.ratio_ri.as_ref().expect("ri");
        let mut out = self.ratio_out.take().expect("ratio out");
        let result = (|| -> Result<(), StageError> {
            let mut b = self.stream.launch_builder(&self.funcs.ratio);
            b.arg(slater)
                .arg(&stage.out)
                .arg(&stage.pf)
                .arg(&stage.ele)
                .arg(msa_dev)
                .arg(rsa_dev)
                .arg(ri_dev)
                .arg(weights)
                .arg(&mut out)
                .arg(&ele0)
                .arg(&plane0)
                .arg(&terms_i)
                .arg(&n_i)
                .arg(&ne_i)
                .arg(&ns_i)
                .arg(&nq_i);
            // SAFETY: kernel ABI (const double* x3, const int* x4, const double*, double*,
            // int x6); one block per active term, dynamic shared memory = n doubles.
            unsafe {
                b.launch(LaunchConfig {
                    grid_dim: (terms as u32, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: (n * 8) as u32,
                })
            }
            .map_err(cu)?;
            self.stream.synchronize().map_err(cu)
        })();
        self.stage = Some(stage);
        if let Err(error) = result {
            self.ratio_out = Some(out);
            return Err(error);
        }
        let mut values = Vec::new();
        self.download_f64(&out, terms, &mut values)?;
        self.ratio_out = Some(out);
        self.stats.ratio_tables += 1;
        Ok(values)
    }

    fn slater_diff(
        &mut self,
        slot: usize,
        input: &ResidentSlaterDiff<'_>,
    ) -> Result<Vec<f64>, StageError> {
        self.bind()?;
        let geom = self.geom()?;
        if !self.has_slot(slot) {
            return unsupported(format!(
                "resident measurement: slot {slot} has no resident tables from stage A"
            ));
        }
        let (n, n_site, n_slater) = (geom.n(), geom.n_site, geom.n_slater);
        let (ntrans, nsp) = (input.n_trans, input.n_sp_gauss_leg);
        if ntrans == 0 || n_slater == 0 {
            return Ok(vec![0.0; n_slater]);
        }
        if input.tri.len() != ntrans * n || input.sgn.len() != ntrans * n {
            return Err(StageError::Failed(
                "resident measurement: inconsistent sector translation tables".into(),
            ));
        }
        // Invert the per-sector translation: for each translation site, the electron slot and
        // its sign, separately per spin block (up and down electrons can share a site). The
        // kernel needs msj given trj.
        let mut inv_trans = vec![-1i32; 2 * ntrans * n_site];
        let mut inv_sgn = vec![0i32; 2 * ntrans * n_site];
        for trans in 0..ntrans {
            for msi in 0..n {
                let trj = input.tri[trans * n + msi];
                if trj < 0 || trj as usize >= n_site {
                    continue;
                }
                let variant = usize::from(msi >= geom.n_elec);
                let at = (variant * ntrans + trans) * n_site + trj as usize;
                inv_trans[at] = msi as i32;
                inv_sgn[at] = input.sgn[trans * n + msi] as i32;
            }
        }
        let tri_i32: Vec<i32> = input.tri.iter().map(|&v| v as i32).collect();
        let sgn_i32: Vec<i32> = input.sgn.iter().map(|&v| v as i32).collect();
        let rows_len = ntrans * n * n_slater;
        let prepare = (|| -> Result<(), StageError> {
            upload_i32(&mut self.diff_tri, &self.stream, &tri_i32)?;
            upload_i32(&mut self.diff_sgn, &self.stream, &sgn_i32)?;
            upload_i32(&mut self.diff_inv_trans, &self.stream, &inv_trans)?;
            upload_i32(&mut self.diff_inv_sgn, &self.stream, &inv_sgn)?;
            ensure_f64(&mut self.diff_rows, &self.stream, rows_len)?;
            ensure_f64(&mut self.diff_acc, &self.stream, n_slater)?;
            Ok(())
        })();
        prepare?;
        let plane0 = (slot * geom.n_qp) as i32;
        let (n_i, ne_i, nq_i, nsp_i, ntrans_i, nsla_i) = (
            n as i32,
            geom.n_elec as i32,
            geom.n_qp as i32,
            nsp as i32,
            ntrans as i32,
            n_slater as i32,
        );
        let nsite_i = n_site as i32;
        let weights = self.qp_weight.as_ref().expect("qp weights");
        let cossin = self.cossin.as_ref().expect("cossin");
        let coscos = self.coscos.as_ref().expect("coscos");
        let sinsin = self.sinsin.as_ref().expect("sinsin");
        let offsets = self.orbital_offsets.as_ref().expect("orbital offsets");
        let list_trj = self.orbital_trj.as_ref().expect("orbital trj");
        let list_sgn = self.orbital_sgn.as_ref().expect("orbital sgn");
        let tri_dev = self.diff_tri.as_ref().expect("tri");
        let sgn_dev = self.diff_sgn.as_ref().expect("sgn");
        let it_dev = self.diff_inv_trans.as_ref().expect("inv_trans");
        let is_dev = self.diff_inv_sgn.as_ref().expect("inv_sgn");
        let stage = self.stage.take().expect("stage buffers");
        let mut rows = self.diff_rows.take().expect("diff rows");
        let mut acc = self.diff_acc.take().expect("diff acc");
        let result = (|| -> Result<(), StageError> {
            let mut b = self.stream.launch_builder(&self.funcs.rows);
            b.arg(&stage.out)
                .arg(&stage.pf)
                .arg(weights)
                .arg(cossin)
                .arg(coscos)
                .arg(sinsin)
                .arg(offsets)
                .arg(list_trj)
                .arg(list_sgn)
                .arg(tri_dev)
                .arg(sgn_dev)
                .arg(it_dev)
                .arg(is_dev)
                .arg(&mut rows)
                .arg(&plane0)
                .arg(&n_i)
                .arg(&ne_i)
                .arg(&nsite_i)
                .arg(&nq_i)
                .arg(&nsp_i)
                .arg(&ntrans_i)
                .arg(&nsla_i);
            // SAFETY: kernel ABI as declared in measurement_kernels.cu; one block per
            // (trans, msi) row, `rows` sized `ntrans * n * n_slater`.
            unsafe {
                b.launch(LaunchConfig {
                    grid_dim: ((ntrans * n) as u32, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })
            }
            .map_err(cu)?;
            let mut b = self.stream.launch_builder(&self.funcs.fold);
            b.arg(&rows)
                .arg(&mut acc)
                .arg(&n_i)
                .arg(&ntrans_i)
                .arg(&nsla_i);
            // SAFETY: kernel ABI (const double*, double*, int x3).
            unsafe {
                b.launch(LaunchConfig {
                    grid_dim: (n_slater.div_ceil(256) as u32, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })
            }
            .map_err(cu)?;
            self.stream.synchronize().map_err(cu)
        })();
        self.stage = Some(stage);
        self.diff_rows = Some(rows);
        self.diff_acc = Some(acc);
        result?;
        let mut values = Vec::new();
        let acc = self.diff_acc.as_ref().expect("diff acc");
        self.download_f64(acc, n_slater, &mut values)?;
        self.stats.slater_diffs += 1;
        Ok(values)
    }

    fn green_two(
        &mut self,
        slot: usize,
        input: &ResidentGreenTwo<'_>,
    ) -> Result<Vec<f64>, StageError> {
        self.green_two_run(slot, input)
    }

    fn stats(&self) -> Option<ResidentStats> {
        Some(self.stats)
    }
}

impl ResidentDevice {
    /// Two-body Green reductions (issue #517 milestone 5): one block per active job, each block
    /// looping the QP planes over the resident inverse table and Slater table.
    fn green_two_run(
        &mut self,
        slot: usize,
        input: &ResidentGreenTwo<'_>,
    ) -> Result<Vec<f64>, StageError> {
        self.bind()?;
        let geom = self.geom()?;
        if !self.has_slot(slot) {
            return unsupported(format!(
                "resident measurement: slot {slot} has no resident tables from stage A"
            ));
        }
        let jobs = input.msa.len();
        if jobs == 0 {
            return Ok(Vec::new());
        }
        let n = geom.n();
        if input.msb.len() != jobs
            || input.rsa_site.len() != jobs
            || input.rsb_site.len() != jobs
            || input.ele_idx.len() != n
        {
            return Err(StageError::Failed(
                "resident measurement: inconsistent Green job arrays".into(),
            ));
        }
        let block = 256usize;
        let shared = (2 * n + 5 * block) * std::mem::size_of::<f64>();
        let to_i32 = |values: &[i64]| -> Vec<i32> { values.iter().map(|&v| v as i32).collect() };
        let msa = to_i32(input.msa);
        let msb = to_i32(input.msb);
        let rsa = to_i32(input.rsa_site);
        let rsb = to_i32(input.rsb_site);
        upload_i32(&mut self.green_msa, &self.stream, &msa)?;
        upload_i32(&mut self.green_msb, &self.stream, &msb)?;
        upload_i32(&mut self.green_rsa, &self.stream, &rsa)?;
        upload_i32(&mut self.green_rsb, &self.stream, &rsb)?;
        ensure_f64(&mut self.green_out, &self.stream, jobs)?;
        let mut stage = self.stage.take().expect("stage buffers");
        let ele_i32 = to_i32(input.ele_idx);
        if let Err(error) = self
            .stream
            .memcpy_htod(&ele_i32, &mut stage.ele.slice_mut(slot * n..(slot + 1) * n))
        {
            self.stage = Some(stage);
            return Err(cu(error));
        }
        let (jobs_i, n_i, ne_i, ns_i, nq_i) = (
            jobs as i32,
            n as i32,
            geom.n_elec as i32,
            geom.n_site as i32,
            geom.n_qp as i32,
        );
        let ele0 = (slot * n) as i32;
        let plane0 = (slot * geom.n_qp) as i32;
        let weights = self.qp_weight.as_ref().expect("qp weights");
        let slater = self.slater.as_ref().expect("slater uploaded");
        let msa_dev = self.green_msa.as_ref().expect("green msa");
        let msb_dev = self.green_msb.as_ref().expect("green msb");
        let rsa_dev = self.green_rsa.as_ref().expect("green rsa");
        let rsb_dev = self.green_rsb.as_ref().expect("green rsb");
        let mut out = self.green_out.take().expect("green out");
        let result = (|| -> Result<(), StageError> {
            let mut b = self.stream.launch_builder(&self.funcs.green_two);
            b.arg(slater)
                .arg(&stage.out)
                .arg(&stage.pf)
                .arg(&stage.ele)
                .arg(msa_dev)
                .arg(msb_dev)
                .arg(rsa_dev)
                .arg(rsb_dev)
                .arg(weights)
                .arg(&mut out)
                .arg(&ele0)
                .arg(&plane0)
                .arg(&jobs_i)
                .arg(&n_i)
                .arg(&ne_i)
                .arg(&ns_i)
                .arg(&nq_i);
            // SAFETY: kernel ABI (const double* x2, const int*, const double*, int, double*,
            // int x7); one block per job, dynamic shared = (2n + 5*256) doubles.
            unsafe {
                b.launch(LaunchConfig {
                    grid_dim: (jobs as u32, 1, 1),
                    block_dim: (block as u32, 1, 1),
                    shared_mem_bytes: shared as u32,
                })
            }
            .map_err(cu)?;
            self.stream.synchronize().map_err(cu)
        })();
        self.stage = Some(stage);
        if let Err(error) = result {
            self.green_out = Some(out);
            return Err(error);
        }
        let mut values = Vec::new();
        self.download_f64(&out, jobs, &mut values)?;
        self.green_out = Some(out);
        Ok(values)
    }
}

/// [`PfaffianStages`] stage A with the device-resident protocol (the host-plane
/// `pfaffian_inverse_batch` still runs through the batched CUDA engine of issue #423).
pub struct ResidentStageA {
    inner: BatchedStages<'static>,
    device: Arc<Mutex<ResidentDevice>>,
}

impl ResidentStageA {
    /// Stage A adapter sharing `device` with the resident measurement stage.
    pub fn new(inner: BatchedStages<'static>, device: Arc<Mutex<ResidentDevice>>) -> Self {
        Self { inner, device }
    }
}

impl PfaffianStages for ResidentStageA {
    fn label(&self) -> String {
        format!("{} + resident stage A", self.inner.label())
    }

    fn provider(&self) -> String {
        format!(
            "device-resident stage A (assemble + batched Pfaffian on CUDA, tables resident); \
             host-plane fallback: {}",
            self.inner.provider()
        )
    }

    fn resident_stage_a(&self) -> bool {
        true
    }

    fn stage_a_resident(
        &mut self,
        input: &ResidentStageAInput<'_>,
    ) -> Result<ResidentStageAOutput, StageError> {
        self.device
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .stage_a_run(input)
    }

    fn pfaffian_inverse_batch(
        &mut self,
        x: &[f64],
        n: usize,
        planes: usize,
    ) -> Result<PfInvBatch, StageError> {
        self.inner.pfaffian_inverse_batch(x, n, planes)
    }
}

/// [`MeasurementStages`] stage B: the device-resident local energy and Slater derivative when
/// stage A left the sample's tables resident, and the eager tenferro device path (issue #518
/// milestone 1) otherwise.
pub struct ResidentMeasurementStage {
    inner: TenferroMeasurement,
    device: Arc<Mutex<ResidentDevice>>,
}

impl ResidentMeasurementStage {
    /// Stage B adapter sharing `device` with the resident stage A.
    pub fn new(inner: TenferroMeasurement, device: Arc<Mutex<ResidentDevice>>) -> Self {
        Self { inner, device }
    }
}

impl MeasurementStages for ResidentMeasurementStage {
    fn label(&self) -> String {
        format!("{} + resident stage B", self.inner.label())
    }

    fn provider(&self) -> String {
        "device-resident measurement (assemble/Pfaffian/local energy/Slater derivative on \
         CUDA, tables resident); eager fallback: tenferro CUDA measurement"
            .to_string()
    }

    fn resident_measurement(&self) -> bool {
        true
    }

    fn resident_green(&self) -> bool {
        true
    }

    fn resident_prepare(
        &mut self,
        data: &ExpertModeData,
        state: &VmcOptimizationState,
    ) -> Result<(), StageError> {
        let mut device = self.device.lock().unwrap_or_else(|e| e.into_inner());
        mvmc_core::measurement_backend::resident_prepare(&mut *device, data, state)
    }

    fn resident_stats(&self) -> Option<ResidentStats> {
        self.device
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .stats()
    }

    fn measure_sample(
        &mut self,
        input: &mut MeasurementInput<'_>,
        slater_diff_len: usize,
        timing: &mut dyn MeasurementTiming,
    ) -> Result<MeasurementOutput, StageError> {
        let slot = input.batch_slot;
        let mut device = self.device.lock().unwrap_or_else(|e| e.into_inner());
        if device.has_slot(slot) {
            if timing.enabled() {
                let mut timer = CTimer::<true>::new();
                timer.diagnostics = timing.diagnostics();
                let out = mvmc_core::measurement_backend::resident_measure_sample(
                    &mut *device,
                    input,
                    slater_diff_len,
                    &mut timer,
                )?;
                timing.merge_elapsed(&timer.elapsed_ns);
                Ok(out)
            } else {
                let mut timer = CTimer::<false>::new();
                timer.diagnostics = timing.diagnostics();
                mvmc_core::measurement_backend::resident_measure_sample(
                    &mut *device,
                    input,
                    slater_diff_len,
                    &mut timer,
                )
            }
        } else {
            // No resident tables for this slot (stage A was not the resident backend): the
            // eager tenferro path of issue #518 milestone 1 handles the sample.
            drop(device);
            self.inner.measure_sample(input, slater_diff_len, timing)
        }
    }

    fn measure_green(
        &mut self,
        input: &mut MeasurementInput<'_>,
        one_len: usize,
        direct_len: usize,
        timing: &mut dyn MeasurementTiming,
    ) -> Result<Option<mvmc_core::stage_backend::GreenOutput>, StageError> {
        let mut device = self.device.lock().unwrap_or_else(|e| e.into_inner());
        if !device.has_slot(input.batch_slot) {
            return Ok(None);
        }
        debug_assert_eq!(one_len, input.data.green_one_terms.len());
        debug_assert_eq!(direct_len, input.data.green_two_terms.len());
        let _ = timing;
        mvmc_core::measurement_backend::resident_green_values(&mut *device, input).map(Some)
    }
}

/// Build the shared device state and the two stage objects for the stage backend of `ordinal`.
pub fn resident_stages(
    ordinal: usize,
    pfaffian: BatchedStages<'static>,
    measurement: TenferroMeasurement,
) -> Result<(ResidentStageA, ResidentMeasurementStage), String> {
    let device = Arc::new(Mutex::new(ResidentDevice::new(ordinal)?));
    Ok((
        ResidentStageA::new(pfaffian, device.clone()),
        ResidentMeasurementStage::new(measurement, device),
    ))
}
