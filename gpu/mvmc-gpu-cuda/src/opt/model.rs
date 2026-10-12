//! Setup-only host model. No walker optimizer state is allocated here.
use mvmc_core::{ExpertModeData, InitialDef, SingleProcessReducer};
use mvmc_expert_parsers::utils::{
    parameter_init::init_parameter, read_input_parameters::read_input_parameters,
};
use std::path::Path;

use super::OptConfig;

pub(super) struct Model {
    pub data: ExpertModeData,
    pub seed: u64,
    pub ns: usize,
    pub ne: usize,
    pub nq: usize,
    pub ng: usize,
    pub nj: usize,
    pub np: usize,
    pub parameters: Vec<f64>,
    pub flags: Vec<i32>,
    pub shift: bool,
    pub gutz: Vec<i32>,
    pub jast: Vec<i32>,
    pub weights: Vec<f64>,
    pub pa: Vec<i32>,
    pub pb: Vec<i32>,
    pub sa: Vec<i32>,
    pub sb: Vec<i32>,
    pub cs: Vec<f64>,
    pub cc: Vec<f64>,
    pub ss: Vec<f64>,
    pub columns: Vec<i32>,
    /// C Slater row, column, QP for every sparse parameter coefficient.
    pub coordinates: Vec<i32>,
    pub coefficients: Vec<f64>,
    /// Julia GPUOperator layout: (dest1,src1,spin1,dest2,src2,spin2,pairs).
    pub operators: Vec<i32>,
    pub energies: Vec<f64>,
}

impl Model {
    pub fn prepare(path: &Path, config: &OptConfig) -> Result<Self, String> {
        let mut data =
            mvmc_expert_parsers::parse_expert_mode_files(path).map_err(|e| e.to_string())?;
        mvmc_core::validation::validate_para_opt(&data)?;
        // Match Julia's GPU setup: normalize translations before model construction.
        data.modpara.nmp_trans = data
            .modpara
            .nmp_trans
            .checked_abs()
            .ok_or("NMPTrans out of range")?
            .max(1);
        if data.modpara.nelec == 0 {
            data.modpara.nelec = (data.modpara.nlocspin + data.modpara.ncond) / 2;
        }
        let m = &data.modpara;
        if data.i_flg_orbital_general != 0
            || m.two_sz != 0
            || m.nex_update_path != 0
            || m.nsrcg != 0
            || m.nsplit_size != 1
            || m.nlocspin != 0
            || data.locspin_terms.iter().any(|s| s.spin_value != 0)
            || data.count_rbm_parameters() != 0
            || !data.opt_trans.is_empty()
            || !data.doublon_holon_2site_indices.is_empty()
            || !data.doublon_holon_4site_indices.is_empty()
            || mvmc_core::get_all_complex_flag(&data)?
        {
            return Err("resident GPU Opt requires real normal orbitals, hopping-only sampling, no local spins/RBM/DH/OptTrans, NSRCG=0 and serial execution (#560)".into());
        }
        let seed = mvmc_core::resolve_rnd_seed(m.rnd_seed, config.seed, 0, &SingleProcessReducer)?;
        data.modpara.rnd_seed = seed;
        let seed32 =
            u32::try_from(seed).map_err(|_| "parameter initialization seed must fit UInt32")?;
        let mut rng = sfmt19937::Sfmt19937Rng::new(seed32);
        init_parameter(&mut data, &mut rng).map_err(str::to_string)?;
        let initial = match &config.initial_def {
            InitialDef::Auto => {
                let p = path
                    .parent()
                    .unwrap_or_else(|| Path::new("."))
                    .join("initial.def");
                p.is_file().then_some(p)
            }
            InitialDef::None => None,
            InitialDef::Path(p) => Some(p.clone()),
        };
        if let Some(initial) = initial {
            if !mvmc_core::read_initial_def(&mut data, initial).map_err(|e| e.to_string())? {
                return Err("initial parameter file could not be loaded".into());
            }
        }
        read_input_parameters(&mut data, path)?;
        mvmc_core::sync::sync_modified_parameter_local(&mut data, true);
        mvmc_core::qp::init_qp_weight(&mut data);
        Self::from_data(data, seed as u64)
    }

    pub fn from_data(mut data: ExpertModeData, seed: u64) -> Result<Self, String> {
        let (ns, ne) = (data.modpara.nsite as usize, data.modpara.nelec as usize);
        if ne == 0 || ne > ns || 2 * ne > crate::pfaffian::MAX_N {
            return Err("unsupported electron geometry".into());
        }
        let packed = mvmc_core::pack_parameters(&data).map_err(|e| e.to_string())?;
        if packed.iter().any(|p| p.im != 0.0 || !p.re.is_finite()) {
            return Err("GPU Opt parameters must be finite and real".into());
        }
        let parameters: Vec<_> = packed.iter().map(|p| p.re).collect();
        let np = parameters.len();
        if np == 0 || np > i32::MAX as usize {
            return Err("invalid parameter count".into());
        }
        let layout = data.projection_layout();
        let (ng, nj) = (layout.n_gutzwiller, layout.n_jastrow);
        if layout.n_proj != ng + nj {
            return Err("resident GPU projection supports Gutzwiller/Jastrow (#560)".into());
        }
        let flags: Vec<i32> = (0..np)
            .map(|p| i32::from(data.optimization_flags.get(2 * p).copied().unwrap_or(1) != 0))
            .collect();
        let shift = ng > 0 && nj > 0 && flags[..ng + nj].iter().all(|&f| f != 0);
        data.ensure_orbital_idx_matrix();
        let nsp = data.modpara.nsp_gauss_leg.max(1) as usize;
        let nmp = data.modpara.nmp_trans.unsigned_abs().max(1) as usize;
        let nopt = data.n_qp_opt_trans.max(1) as usize;
        let nq = nsp * nmp * nopt;
        let weights = data.qp_weights.as_ref().ok_or("missing QP weights")?;
        if weights.qp_full_weight.len() != nq {
            return Err("QP weight geometry mismatch".into());
        }
        let real = |x: num_complex::Complex64| {
            if x.im == 0.0 && x.re.is_finite() {
                Ok(x.re)
            } else {
                Err("complex or nonfinite GPU model coefficient".to_string())
            }
        };
        let qw = weights
            .qp_full_weight
            .iter()
            .copied()
            .map(real)
            .collect::<Result<Vec<_>, _>>()?;
        let cs = (0..nq)
            .map(|q| real(weights.spgl_cos_sin[q % nsp]))
            .collect::<Result<Vec<_>, _>>()?;
        let cc = (0..nq)
            .map(|q| real(weights.spgl_cos_cos[q % nsp]))
            .collect::<Result<Vec<_>, _>>()?;
        let ss = (0..nq)
            .map(|q| real(weights.spgl_sin_sin[q % nsp]))
            .collect::<Result<Vec<_>, _>>()?;
        let oi = data
            .orbital_idx_matrix
            .as_ref()
            .ok_or("orbital indices missing")?;
        let os = data
            .orbital_sgn_matrix
            .as_ref()
            .ok_or("orbital signs missing")?;
        let mut pa = vec![];
        let mut pb = vec![];
        let mut sa = vec![];
        let mut sb = vec![];
        for q in 0..nq {
            let opt = q / (nsp * nmp);
            let mp = (q % (nsp * nmp)) / nsp;
            for ri in 0..ns {
                for rj in 0..ns {
                    let (i, si) = mvmc_core::qp::translated_site(&data, ri, opt, mp, true);
                    let (j, sj) = mvmc_core::qp::translated_site(&data, rj, opt, mp, true);
                    if i >= ns || j >= ns || oi[i][j] < 0 || oi[j][i] < 0 {
                        return Err("invalid orbital/translation mapping".into());
                    }
                    let a = ng + nj + oi[i][j] as usize;
                    let b = ng + nj + oi[j][i] as usize;
                    if a >= np || b >= np {
                        return Err("orbital parameter outside declared layout".into());
                    }
                    pa.push(a as i32);
                    pb.push(b as i32);
                    sa.push((os[i][j] * si * sj) as i32);
                    sb.push((os[j][i] * si * sj) as i32);
                }
            }
        }
        // Append in C's flat row order, exactly the Julia sparse-basis order.
        // Duplicate a/b parameter indices are combined before evaluation.
        let mut sparse: Vec<Vec<(i32, i32, i32, f64)>> = vec![vec![]; np];
        let unit = |a: f64, b: f64, q: usize, rs: usize, cs_: usize| match (rs, cs_) {
            (0, 0) => -(a - b) * cs[q],
            (0, 1) => a * cc[q] + b * ss[q],
            (1, 0) => -a * ss[q] - b * cc[q],
            _ => (a - b) * cs[q],
        };
        for q in 0..nq {
            for rs in 0..2 {
                for ri in 0..ns {
                    for colspin in 0..2 {
                        for rj in 0..ns {
                            let k = (q * ns + ri) * ns + rj;
                            let (a, b) = (pa[k] as usize, pb[k] as usize);
                            let (row, col) = ((ri + rs * ns) as i32, (rj + colspin * ns) as i32);
                            let x = unit(
                                sa[k] as f64,
                                if a == b { sb[k] as f64 } else { 0. },
                                q,
                                rs,
                                colspin,
                            );
                            if x != 0. {
                                sparse[a].push((row, col, q as i32, x));
                            }
                            if a != b {
                                let x = unit(0., sb[k] as f64, q, rs, colspin);
                                if x != 0. {
                                    sparse[b].push((row, col, q as i32, x));
                                }
                            }
                        }
                    }
                }
            }
        }
        let mut columns = vec![0i32];
        let mut coordinates = vec![];
        let mut coefficients = vec![];
        for entries in sparse {
            for (r, c, q, x) in entries {
                coordinates.extend([r, c, q]);
                coefficients.push(x);
            }
            columns.push(
                i32::try_from(coefficients.len()).map_err(|_| "too many sparse coefficients")?,
            );
        }
        let gutz = if ng == 0 {
            vec![-1; ns]
        } else {
            data.gutzwiller_idx
                .iter()
                .map(|&i| i32::try_from(i).map_err(|_| "Gutzwiller index out of range"))
                .collect::<Result<Vec<_>, _>>()?
        };
        let jast = if nj == 0 {
            vec![-1; ns * ns]
        } else {
            data.jastrow_idx
                .iter()
                .flatten()
                .map(|&i| i32::try_from(i).map_err(|_| "Jastrow index out of range"))
                .collect::<Result<Vec<_>, _>>()?
        };
        if gutz.len() != ns || jast.len() != ns * ns {
            return Err("projection index dimensions mismatch".into());
        }
        if gutz.iter().any(|&p| p < -1 || p >= ng as i32)
            || jast.iter().any(|&p| p < -1 || p >= nj as i32)
        {
            return Err("projection parameter index out of bounds".into());
        }
        let mut operators = vec![];
        let mut energies = vec![];
        let mut add = |op: [i64; 7], value: f64| -> Result<(), String> {
            for i in [0, 1, 3, 4] {
                if op[i] < 0 || op[i] >= ns as i64 {
                    return Err("Hamiltonian site out of bounds".into());
                }
            }
            if !(0..=1).contains(&op[2]) || !(0..=1).contains(&op[5]) || !value.is_finite() {
                return Err("invalid Hamiltonian coefficient/spin".into());
            }
            operators.extend(op.map(|x| x as i32));
            energies.push(value);
            Ok(())
        };
        for t in &data.coulomb_intra_terms {
            add([t.site, t.site, 0, t.site, t.site, 1, 2], t.value)?;
        }
        for t in &data.coulomb_inter_terms {
            for s in 0..2 {
                for u in 0..2 {
                    add([t.site1, t.site1, s, t.site2, t.site2, u, 2], t.value)?;
                }
            }
        }
        for t in &data.hund_terms {
            for s in 0..2 {
                add([t.site1, t.site1, s, t.site2, t.site2, s, 2], -t.value)?;
            }
        }
        for t in &data.transfer_terms {
            if t.spin1 != t.spin2 {
                return Err("GPU Transfer requires spin conservation".into());
            }
            add(
                [t.site1, t.site2, t.spin1.as_code() as i64, 0, 0, 0, 1],
                -real(t.value)?,
            )?;
        }
        for t in &data.pair_hop_terms {
            add([t.site1, t.site2, 0, t.site1, t.site2, 1, 2], t.value)?;
        }
        for t in &data.exchange_terms {
            for s in 0..2 {
                add([t.site1, t.site2, s, t.site2, t.site1, 1 - s, 2], t.value)?;
            }
        }
        for t in &data.inter_all_terms {
            if t.spin0 != t.spin1 || t.spin2 != t.spin3 {
                return Err("GPU InterAll requires spin-conserving pairs".into());
            }
            add(
                [t.site0, t.site1, t.spin0, t.site2, t.site3, t.spin2, 2],
                real(t.value)?,
            )?;
        }
        Ok(Self {
            data,
            seed,
            ns,
            ne,
            nq,
            ng,
            nj,
            np,
            parameters,
            flags,
            shift,
            gutz,
            jast,
            weights: qw,
            pa,
            pb,
            sa,
            sb,
            cs,
            cc,
            ss,
            columns,
            coordinates,
            coefficients,
            operators,
            energies,
        })
    }
}
