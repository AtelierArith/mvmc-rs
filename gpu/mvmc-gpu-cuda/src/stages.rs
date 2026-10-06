//! `AcceleratedStages` over a tenferro `EagerRuntime` (CPU or CUDA) for the validation
//! harness of issue #424.
//!
//! `S` and `g` run through `dot_general`; the Pfaffian and inverse are not provided by
//! tenferro 0.7.1 and report `StageError::Unsupported` (no CPU fallback). Upload and
//! download are part of every call (`transfers_included = true` in the metadata).

use std::sync::Arc;

use mvmc_core::accel_validation::{centered, AcceleratedStages, PfInv, SrSg, StageError};
use tenferro_ad::{EagerRuntime, EagerTensor};
use tenferro_tensor::{DotGeneralConfig, Tensor, TensorRead};

/// Eager-runtime stages; the runtime is created once and reused.
pub struct EagerStages {
    ctx: Arc<EagerRuntime>,
    label: String,
    provider: String,
}

fn failed(e: impl std::fmt::Display) -> StageError {
    StageError::Failed(e.to_string())
}

impl EagerStages {
    /// tenferro CPU eager runtime.
    pub fn cpu() -> Result<Self, String> {
        Ok(Self {
            ctx: crate::bench::cpu_runtime()?,
            label: "tenferro-eager-cpu".to_string(),
            provider: "tenferro-ad EagerRuntime + CpuBackend (cpu-faer)".to_string(),
        })
    }

    /// tenferro CUDA eager runtime on `ordinal`.
    pub fn cuda(ordinal: usize) -> Result<Self, String> {
        Ok(Self {
            ctx: crate::bench::cuda_runtime(ordinal)?,
            label: format!("tenferro-cuda:{ordinal}"),
            provider: "tenferro-ad EagerRuntime + tenferro-gpu CudaBackend (cuBLAS)".to_string(),
        })
    }

    fn upload(&self, host: Tensor) -> Result<EagerTensor, StageError> {
        let dev = self
            .ctx
            .with_execution_session(|s| s.upload_host_tensor(TensorRead::from_tensor(&host)))
            .map_err(failed)?
            .map_err(failed)?;
        EagerTensor::from_tensor_in(dev, self.ctx.clone()).map_err(failed)
    }

    fn download(&self, t: &EagerTensor) -> Result<Vec<f64>, StageError> {
        let dev = t.to_tensor().map_err(failed)?;
        let host = self
            .ctx
            .with_execution_session(|s| s.download_to_host(TensorRead::from_tensor(&dev)))
            .map_err(failed)?
            .map_err(failed)?;
        Ok(host.as_slice::<f64>().map_err(failed)?.to_vec())
    }
}

impl AcceleratedStages for EagerStages {
    fn label(&self) -> String {
        self.label.clone()
    }
    fn provider(&self) -> String {
        self.provider.clone()
    }

    fn pfaffian_inverse(&mut self, _x: &[f64], _n: usize) -> Result<PfInv, StageError> {
        Err(StageError::Unsupported(
            "tenferro 0.7.1 has no skew-symmetric factorization or Pfaffian".to_string(),
        ))
    }

    fn sr_s_g(
        &mut self,
        o: &[f64],
        ns: usize,
        np: usize,
        e: &[f64],
        w: &[f64],
    ) -> Result<SrSg, StageError> {
        let (d, de) = centered(o, ns, np, e, w);
        let mut dw = d.clone();
        for p in 0..np {
            for k in 0..ns {
                dw[k + p * ns] *= w[k];
            }
        }
        let dt = self.upload(Tensor::from_vec_col_major(vec![ns, np], d).map_err(failed)?)?;
        let dwt = self.upload(Tensor::from_vec_col_major(vec![ns, np], dw).map_err(failed)?)?;
        let det = self.upload(Tensor::from_vec_col_major(vec![ns], de).map_err(failed)?)?;
        let cfg = DotGeneralConfig {
            lhs_contracting_dims: vec![0],
            rhs_contracting_dims: vec![0],
            lhs_batch_dims: vec![],
            rhs_batch_dims: vec![],
        };
        let s = dwt.dot_general(&dt, cfg.clone()).map_err(failed)?;
        let g = dwt.dot_general(&det, cfg).map_err(failed)?;
        self.ctx.synchronize().map_err(failed)?;
        Ok(SrSg {
            s: self.download(&s)?,
            g: self.download(&g)?,
        })
    }
}
