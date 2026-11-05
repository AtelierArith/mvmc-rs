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
    /// DH2 / DH4 table counts (C NDoublonHolon2siteIdx / NDoublonHolon4siteIdx).
    pub n2: usize,
    pub n4: usize,
    /// All projection parameters: `ng + nj + 6*n2 + 10*n4`.
    pub nproj: usize,
    pub np: usize,
    pub parameters: Vec<f64>,
    pub flags: Vec<i32>,
    /// C SetFlagShift: FlagShiftGJ, FlagShiftDH2, FlagShiftDH4.
    pub shift: bool,
    pub shift_dh2: bool,
    pub shift_dh4: bool,
    /// DH2 neighbors `[n2][nsite][2]`, then DH4 neighbors `[n4][nsite][4]`.
    pub dh: Vec<i32>,
    /// RBM coefficients (C `NRBM`), stored between projection and Slater.
    pub nrbm: usize,
    /// Hidden neurons: charge, then spin, then general (C `Nneuron`).
    pub nneuron: usize,
    /// Device RBM index tables (`resident_rbm.cuh`); empty without RBM.
    pub rbm: Vec<i32>,
    /// OptTrans sectors (C `NOptTrans`), the last parameter block (#674).
    pub nopt: usize,
    /// Fixed QP weights (C `QPFixWeight`, `NSPGaussLeg*NMPTrans`); empty
    /// without OptTrans, whose full weights never change.
    pub fix: Vec<f64>,
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
    /// Julia `get_loc_spn_array`: 1 for a local-spin site (`spin_value == 1`).
    pub localspins: Vec<i32>,
    /// Julia GPU exchange probability: 0 / 0.5 / 1 for NExUpdatePath 0 / 1 / 2.
    pub exchange: f64,
    /// C `AllComplexFlag != 0` (#678): `parameters`, `weights` and `energies`
    /// hold interleaved (re, im) pairs and `flags` the `2*np` real/imaginary
    /// component flags (C `OptFlag`). Projection factors use the real parts.
    pub complex: bool,
}

impl Model {
    /// FP64 values per device scalar: 2 for complex mode (interleaved re, im).
    pub fn lanes(&self) -> usize {
        1 + usize::from(self.complex)
    }

    /// Real projection parameters, the sampler's table (C `creal(Proj)`).
    pub fn projection(&self) -> Vec<f64> {
        let lanes = self.lanes();
        (0..self.nproj)
            .map(|p| self.parameters[lanes * p])
            .collect()
    }

    /// SR dimension: `np` real parameters, or C's `2*NPara` in complex mode.
    pub fn sr_dimension(&self) -> usize {
        self.lanes() * self.np
    }

    pub fn prepare(path: &Path, config: &OptConfig) -> Result<Self, String> {
        Self::prepare_input(path, config, false, None)
    }

    pub fn prepare_physcal(
        path: &Path,
        config: &OptConfig,
        opt_para: Option<&Path>,
    ) -> Result<Self, String> {
        Self::prepare_input(path, config, true, opt_para)
    }

    fn prepare_input(
        path: &Path,
        config: &OptConfig,
        physcal: bool,
        opt_para: Option<&Path>,
    ) -> Result<Self, String> {
        let mut data =
            mvmc_expert_parsers::parse_expert_mode_files(path).map_err(|e| e.to_string())?;
        if physcal {
            mvmc_core::validation::validate_phys_cal(&data)?;
            if data.modpara.lanczos_mode != 0 {
                return Err("resident GPU PhysCal requires NLanczosMode=0".into());
            }
        } else {
            mvmc_core::validation::validate_para_opt(&data)?;
        }
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
        // Julia GPU accepts NExUpdatePath 0..2 with local spins
        // (`cuda_state.jl:47`); path 3 needs general orbitals. DH2/DH4 follow
        // the C counters (#665) and RBM the C complex-normal factors (#667);
        // Julia GPU supports neither. NSRCG is accepted: PhysCal ignores it as
        // Julia GPU does (`cuda_runners.jl:27`); Opt runs C SR-CG (#670).
        // OptTrans follows C `qp.c`/`vmccal.c` (#674); Julia GPU rejects it.
        // Complex normal orbitals follow C's complex normal path (#678).
        if data.i_flg_orbital_general != 0
            || m.two_sz != 0
            || !(0..=2).contains(&m.nex_update_path)
            || m.nsplit_size != 1
        {
            return Err(
                "resident GPU requires normal orbitals, NExUpdatePath<=2 and serial execution"
                    .into(),
            );
        }
        if !physcal && m.nsrcg != 0 && mvmc_core::get_all_complex_flag(&data)? {
            return Err("resident GPU complex Opt requires direct SR (NSRCG=0)".into());
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
        if let Some(opt_para) = opt_para {
            let loaded = mvmc_core::read_opt_para_file(&mut data, opt_para)?;
            if loaded
                != mvmc_core::pack_parameters(&data)
                    .map_err(|e| e.to_string())?
                    .len()
            {
                return Err("incomplete optimized parameter file".into());
            }
        }
        // C PhysCal loads fixed parameters before applying explicit In*.def overlays.
        // Opt has no fixed parameter file, so its existing initialization order is unchanged.
        read_input_parameters(&mut data, path)?;
        mvmc_core::sync::sync_modified_parameter_local(&mut data, !physcal);
        mvmc_core::qp::init_qp_weight(&mut data);
        if physcal {
            Self::from_data_with_derivatives(data, seed as u64, false)
        } else {
            Self::from_data(data, seed as u64)
        }
    }

    pub fn from_data(data: ExpertModeData, seed: u64) -> Result<Self, String> {
        Self::from_data_with_derivatives(data, seed, true)
    }

    fn from_data_with_derivatives(
        mut data: ExpertModeData,
        seed: u64,
        derivatives: bool,
    ) -> Result<Self, String> {
        let (ns, ne) = (data.modpara.nsite as usize, data.modpara.nelec as usize);
        if ne == 0 || ne > ns || 2 * ne > crate::pfaffian::MAX_N {
            return Err("unsupported electron geometry".into());
        }
        let exchange = match data.modpara.nex_update_path {
            0 => 0.0,
            1 => 0.5,
            2 => 1.0,
            _ => return Err("resident GPU requires NExUpdatePath in 0:2".into()),
        };
        let mut loc = vec![0i64; ns];
        mvmc_core::sampling::init_loc_spn(&mut loc, &data);
        let localspins: Vec<i32> = loc.iter().map(|&x| x as i32).collect();
        let nlocal = localspins.iter().filter(|&&x| x == 1).count();
        // Julia initial_gpu_state (`cuda_initial_sampling.jl:87`).
        if nlocal > 2 * ne || 2 * ne > 2 * ns - nlocal {
            return Err("infeasible local-spin filling".into());
        }
        let complex = mvmc_core::get_all_complex_flag(&data)?;
        let packed = mvmc_core::pack_parameters(&data).map_err(|e| e.to_string())?;
        if complex {
            if packed
                .iter()
                .any(|p| !p.re.is_finite() || !p.im.is_finite())
            {
                return Err("resident GPU parameters must be finite".into());
            }
        } else if packed.iter().any(|p| p.im != 0.0 || !p.re.is_finite()) {
            return Err("resident GPU parameters must be finite and real".into());
        }
        let parameters: Vec<_> = if complex {
            packed.iter().flat_map(|p| [p.re, p.im]).collect()
        } else {
            packed.iter().map(|p| p.re).collect()
        };
        let np = packed.len();
        if np == 0 || np > i32::MAX as usize {
            return Err("invalid parameter count".into());
        }
        let layout = data.projection_layout();
        let (ng, nj, n2, n4) = (
            layout.n_gutzwiller,
            layout.n_jastrow,
            layout.n_dh2,
            layout.n_dh4,
        );
        let nproj = ng + nj + 6 * n2 + 10 * n4;
        let nrbm = data.count_rbm_parameters();
        if layout.n_proj != nproj
            || layout.dh2_offset != ng + nj
            || layout.dh4_offset != ng + nj + 6 * n2
            || nproj + nrbm > np
        {
            return Err("resident GPU projection supports Gutzwiller/Jastrow/DH2/DH4".into());
        }
        // C setmemory.c: OptTrans = Para + NProj + NRBM + NSlater, one weight
        // per sector of NQPFull = NQPFix * NQPOptTrans (readdef.c:984-1017).
        let nopt = data.count_opt_trans_parameters();
        let slater_end = np - nopt.min(np);
        if nopt > 0
            && (data.n_qp_opt_trans.max(1) as usize != nopt
                || nproj + nrbm + data.slater_params.len() + nopt != np)
        {
            return Err("resident GPU OptTrans requires one weight per NQPOptTrans sector".into());
        }
        // C OptFlag holds real and imaginary component flags; real mode uses
        // the real ones, complex mode all 2*np (stcopt.c).
        let flag = |k: usize| i32::from(data.optimization_flags.get(k).copied().unwrap_or(1) != 0);
        let real_flags: Vec<i32> = (0..np).map(|p| flag(2 * p)).collect();
        let flags: Vec<i32> = if complex {
            (0..2 * np).map(flag).collect()
        } else {
            real_flags.clone()
        };
        let shift = ng > 0 && nj > 0 && real_flags[..ng + nj].iter().all(|&f| f != 0);
        // C SetFlagShift (parameter.c:255-313), as the C-order sync applies it:
        // DH shifts need every Gutzwiller and every DH component optimized.
        let all = |range: std::ops::Range<usize>| real_flags[range].iter().all(|&f| f != 0);
        let gutz_free = ng > 0 && all(0..ng);
        let shift_dh2 = gutz_free && n2 > 0 && all(ng + nj..ng + nj + 6 * n2);
        let shift_dh4 = gutz_free && n4 > 0 && all(ng + nj + 6 * n2..nproj);
        let mut dh = Vec::with_capacity((2 * n2 + 4 * n4) * ns);
        let site = |r: i64| {
            i32::try_from(r)
                .ok()
                .filter(|&r| r >= 0 && (r as usize) < ns)
                .ok_or_else(|| "DH neighbor site out of bounds".to_string())
        };
        let rows2 = data
            .doublon_holon_2site_indices
            .get(..n2)
            .ok_or("DH2 tables missing")?;
        let rows4 = data
            .doublon_holon_4site_indices
            .get(..n4)
            .ok_or("DH4 tables missing")?;
        for table in rows2 {
            if table.neighbors.len() != ns {
                return Err("DH2 table must list every center site".into());
            }
            for &r in table.neighbors.iter().flatten() {
                dh.push(site(r)?);
            }
        }
        for table in rows4 {
            if table.neighbors.len() != ns {
                return Err("DH4 table must list every center site".into());
            }
            for &r in table.neighbors.iter().flatten() {
                dh.push(site(r)?);
            }
        }
        let (rbm, nneuron) = rbm_tables(&data)?;
        // Complex mode supports Gutzwiller/Jastrow projections and normal
        // orbitals (#678). These families have no complex resident kernels yet.
        if complex && (nrbm > 0 || nopt > 0 || n2 + n4 > 0 || nlocal > 0) {
            return Err(
                "resident GPU complex mode does not support RBM, OptTrans, DH2/DH4 or local spins"
                    .into(),
            );
        }
        data.ensure_orbital_idx_matrix();
        let nsp = data.modpara.nsp_gauss_leg.max(1) as usize;
        let nmp = data.modpara.nmp_trans.unsigned_abs().max(1) as usize;
        let nsector = data.n_qp_opt_trans.max(1) as usize;
        let nq = nsp * nmp * nsector;
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
        // Complex mode takes C's complex QPFullWeight (CalculateIP_fcmp); the
        // real path keeps rejecting complex weights.
        let finite = |x: num_complex::Complex64| x.re.is_finite() && x.im.is_finite();
        let qw = if complex {
            if !weights.qp_full_weight.iter().copied().all(finite) {
                return Err("nonfinite GPU QP weight".into());
            }
            weights
                .qp_full_weight
                .iter()
                .flat_map(|x| [x.re, x.im])
                .collect()
        } else {
            weights
                .qp_full_weight
                .iter()
                .copied()
                .map(real)
                .collect::<Result<Vec<_>, _>>()?
        };
        // C UpdateQPWeight rebuilds the full weights from these after every
        // parameter update (qp.c:130-148).
        let fix = if nopt > 0 {
            if weights.qp_fix_weight.len() * nopt != nq {
                return Err("QP weight geometry mismatch".into());
            }
            weights
                .qp_fix_weight
                .iter()
                .copied()
                .map(real)
                .collect::<Result<Vec<_>, _>>()?
        } else {
            vec![]
        };
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
                    // C setmemory.c: Slater = Para + NProj + NRBM.
                    let a = nproj + nrbm + oi[i][j] as usize;
                    let b = nproj + nrbm + oi[j][i] as usize;
                    if a >= slater_end || b >= slater_end {
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
        for q in 0..if derivatives { nq } else { 0 } {
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
        // Complex mode keeps complex Transfer/InterAll coefficients (C
        // calham.c); the real path requires real ones.
        let coefficient = |x: num_complex::Complex64| {
            if complex {
                finite(x)
                    .then_some(x)
                    .ok_or_else(|| "nonfinite GPU model coefficient".to_string())
            } else {
                real(x).map(|re| num_complex::Complex64::new(re, 0.0))
            }
        };
        let mut add = |op: [i64; 7], value: num_complex::Complex64| -> Result<(), String> {
            for i in [0, 1, 3, 4] {
                if op[i] < 0 || op[i] >= ns as i64 {
                    return Err("Hamiltonian site out of bounds".into());
                }
            }
            if !(0..=1).contains(&op[2]) || !(0..=1).contains(&op[5]) || !finite(value) {
                return Err("invalid Hamiltonian coefficient/spin".into());
            }
            operators.extend(op.map(|x| x as i32));
            energies.push(value.re);
            if complex {
                energies.push(value.im);
            }
            Ok(())
        };
        let r = |x: f64| num_complex::Complex64::new(x, 0.0);
        for t in &data.coulomb_intra_terms {
            add([t.site, t.site, 0, t.site, t.site, 1, 2], r(t.value))?;
        }
        for t in &data.coulomb_inter_terms {
            for s in 0..2 {
                for u in 0..2 {
                    add([t.site1, t.site1, s, t.site2, t.site2, u, 2], r(t.value))?;
                }
            }
        }
        for t in &data.hund_terms {
            for s in 0..2 {
                add([t.site1, t.site1, s, t.site2, t.site2, s, 2], r(-t.value))?;
            }
        }
        for t in &data.transfer_terms {
            if t.spin1 != t.spin2 {
                return Err("GPU Transfer requires spin conservation".into());
            }
            add(
                [t.site1, t.site2, t.spin1.as_code() as i64, 0, 0, 0, 1],
                -coefficient(t.value)?,
            )?;
        }
        for t in &data.pair_hop_terms {
            add([t.site1, t.site2, 0, t.site1, t.site2, 1, 2], r(t.value))?;
        }
        for t in &data.exchange_terms {
            for s in 0..2 {
                add(
                    [t.site1, t.site2, s, t.site2, t.site1, 1 - s, 2],
                    r(t.value),
                )?;
            }
        }
        for t in &data.inter_all_terms {
            if t.spin0 != t.spin1 || t.spin2 != t.spin3 {
                return Err("GPU InterAll requires spin-conserving pairs".into());
            }
            add(
                [t.site0, t.site1, t.spin0, t.site2, t.site3, t.spin2, 2],
                coefficient(t.value)?,
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
            n2,
            n4,
            nproj,
            np,
            parameters,
            flags,
            shift,
            shift_dh2,
            shift_dh4,
            dh,
            nrbm,
            nneuron,
            rbm,
            nopt,
            fix,
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
            localspins,
            exchange,
            complex,
        })
    }
}

/// Device RBM tables in the layout of `resident_rbm.cuh`, and the neuron count.
///
/// Indices are relative to the RBM block (C `RBM = Para + NProj`), in C
/// section order: physical layers (charge, spin, general), hidden layers, then
/// physical-hidden couplings. Every present section must map each of its
/// cells exactly once, as the C readers require (`readdef.c` GetInfoRBM_*).
/// An absent section leaves its cells at -1, which C skips
/// (`if(NChargeRBM_PhysHiddenIdx>0)`, `rbm.c` MakeRBMCnt/UpdateRBMCnt).
fn rbm_tables(data: &ExpertModeData) -> Result<(Vec<i32>, usize), String> {
    if !data.has_rbm_terms() {
        return if data.count_rbm_parameters() == 0 {
            Ok((vec![], 0))
        } else {
            Err("resident GPU RBM requires complete section mappings".into())
        };
    }
    let m = &data.modpara;
    if m.nneuron != 0 {
        return Err("resident GPU RBM requires Nneuron=0 (family neuron counts only)".into());
    }
    let ns = m.nsite as usize;
    let neurons = [m.nneuron_charge, m.nneuron_spin, m.nneuron_general]
        .map(|n| usize::try_from(n).unwrap_or(usize::MAX));
    if neurons.iter().any(|&n| n > 4096) {
        return Err("invalid RBM neuron count".into());
    }
    let h_total: usize = neurons.iter().sum();
    let w = data.rbm_section_sizes();
    let mut offset = [0usize; 9];
    for k in 1..9 {
        offset[k] = offset[k - 1] + w[k - 1];
    }
    let first = [0, neurons[0], neurons[0] + neurons[1]];
    let mut tab = vec![-1i32; 4 * ns + 2 * h_total + 2 * ns * h_total];
    let fail = |what: &str| format!("resident GPU RBM requires a complete {what} mapping");
    // One assignment per cell; `section` gives the relative index offset.
    let mut put = |cell: usize, section: usize, idx: i64, what: &str| -> Result<(), String> {
        if idx < 0 || idx as usize >= w[section] || tab[cell] != -1 {
            return Err(fail(what));
        }
        tab[cell] = (offset[section] + idx as usize) as i32;
        Ok(())
    };
    let site = |r: i64, n: usize| (r >= 0 && (r as usize) < n).then_some(r as usize);
    let spin = |s: i64| (0..=1).contains(&s).then_some(s as usize);
    for t in &data.charge_rbm_phys_layer_terms {
        let r = site(t.site, ns).ok_or_else(|| fail("ChargeRBM_PhysLayer"))?;
        put(r, 0, t.idx, "ChargeRBM_PhysLayer")?;
    }
    for t in &data.spin_rbm_phys_layer_terms {
        let r = site(t.site, ns).ok_or_else(|| fail("SpinRBM_PhysLayer"))?;
        put(ns + r, 1, t.idx, "SpinRBM_PhysLayer")?;
    }
    for t in &data.general_rbm_phys_layer_terms {
        let r = site(t.site, ns).ok_or_else(|| fail("GeneralRBM_PhysLayer"))?;
        let s = spin(t.spin).ok_or_else(|| fail("GeneralRBM_PhysLayer"))?;
        put(2 * ns + r + s * ns, 2, t.idx, "GeneralRBM_PhysLayer")?;
    }
    let bias = 4 * ns + h_total;
    for (family, terms) in [
        (
            0,
            data.charge_rbm_hidden_layer_terms
                .iter()
                .map(|t| (t.site, t.idx))
                .collect::<Vec<_>>(),
        ),
        (
            1,
            data.spin_rbm_hidden_layer_terms
                .iter()
                .map(|t| (t.site, t.idx))
                .collect(),
        ),
        (
            2,
            data.general_rbm_hidden_layer_terms
                .iter()
                .map(|t| (t.site, t.idx))
                .collect(),
        ),
    ] {
        for (h, idx) in terms {
            let h = site(h, neurons[family]).ok_or_else(|| fail("RBM_HiddenLayer"))?;
            put(bias + first[family] + h, 3 + family, idx, "RBM_HiddenLayer")?;
        }
    }
    let coupling = |h: usize, rs: usize| 4 * ns + 2 * h_total + h * 2 * ns + rs;
    for t in &data.charge_rbm_phys_hidden_terms {
        let r = site(t.site1, ns).ok_or_else(|| fail("ChargeRBM_PhysHidden"))?;
        let h = site(t.site2, neurons[0]).ok_or_else(|| fail("ChargeRBM_PhysHidden"))?;
        put(coupling(h, r), 6, t.idx, "ChargeRBM_PhysHidden")?;
    }
    for t in &data.spin_rbm_phys_hidden_terms {
        let r = site(t.site1, ns).ok_or_else(|| fail("SpinRBM_PhysHidden"))?;
        let h = site(t.site2, neurons[1]).ok_or_else(|| fail("SpinRBM_PhysHidden"))?;
        put(coupling(first[1] + h, r), 7, t.idx, "SpinRBM_PhysHidden")?;
    }
    for t in &data.general_rbm_phys_hidden_terms {
        let r = site(t.site1, ns).ok_or_else(|| fail("GeneralRBM_PhysHidden"))?;
        let s = spin(t.spin).ok_or_else(|| fail("GeneralRBM_PhysHidden"))?;
        let h = site(t.site2, neurons[2]).ok_or_else(|| fail("GeneralRBM_PhysHidden"))?;
        put(
            coupling(first[2] + h, r + s * ns),
            8,
            t.idx,
            "GeneralRBM_PhysHidden",
        )?;
    }
    // Completeness of every present section (C rejects partial mappings).
    let complete =
        |cells: Vec<usize>, section: usize| w[section] == 0 || cells.iter().all(|&c| tab[c] >= 0);
    let phys_hidden = |family: usize, width: usize| -> Vec<usize> {
        (0..neurons[family])
            .flat_map(|h| (0..width).map(move |rs| coupling(first[family] + h, rs)))
            .collect()
    };
    let hidden = |family: usize| -> Vec<usize> {
        (0..neurons[family])
            .map(|h| bias + first[family] + h)
            .collect()
    };
    if !complete((0..ns).collect(), 0)
        || !complete((ns..2 * ns).collect(), 1)
        || !complete((2 * ns..4 * ns).collect(), 2)
        || !complete(hidden(0), 3)
        || !complete(hidden(1), 4)
        || !complete(hidden(2), 5)
        || !complete(phys_hidden(0, ns), 6)
        || !complete(phys_hidden(1, ns), 7)
        || !complete(phys_hidden(2, 2 * ns), 8)
    {
        return Err("resident GPU RBM requires complete section mappings".into());
    }
    for (family, &count) in neurons.iter().enumerate() {
        for h in 0..count {
            tab[4 * ns + first[family] + h] = family as i32;
        }
    }
    Ok((tab, h_total))
}
