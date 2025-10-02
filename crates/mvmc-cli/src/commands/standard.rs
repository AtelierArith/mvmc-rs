//! Standard mode command - C実装のStandard mode相当の機能
//!
//! このモジュールは、StdFace.defファイルからnamelist.defへの自動変換機能を提供します。
//! C実装の`vmc.out -s StdFace.def`と同等の機能を実装します。

use crate::error::{CliError, CliResult};
use colored::Colorize;
use mvmc_io::{StdFaceParser, DefFileGenerator, DefFileConfig};
use mvmc_core::config::{VmcParameters, SRParameters, MonteCarloParameters};
use mvmc_core::types::{SiteCount, ElectronCount, TwoSz, CalcMode, LanczosMode, RandomSeed};
use std::path::{Path, PathBuf};

/// Standard modeの実行
///
/// # Arguments
///
/// * `input_file` - StdFace.defファイルのパス
/// * `output_dir` - 出力ディレクトリ
pub fn execute(input_file: PathBuf, output_dir: PathBuf) -> CliResult<()> {
    println!("Standard: execute function called");
    println!("{}", "═══════════════════════════════════════════".cyan());
    println!("{}", "  mVMC - Standard Mode".cyan().bold());
    println!("{}", "═══════════════════════════════════════════".cyan());
    println!();

    // 入力ファイルの存在確認
    if !input_file.exists() {
        return Err(CliError::FileNotFound(input_file));
    }

    println!("{}", "📄 Reading StdFace configuration...".blue());

    // StdFace.defファイルを読み込み
    let parser = StdFaceParser::new();
    let stdface_config = parser.parse_file(input_file.to_str().unwrap())
        .map_err(|e| CliError::Other(anyhow::anyhow!("Failed to parse StdFace file: {}", e)))?;

    println!("✓ Configuration loaded successfully");
    println!();

    // VMCParametersに変換
    let vmc_params = convert_stdface_to_vmc_params(&stdface_config)?;

    println!("{}", "🔧 Generating mVMC input files...".blue());

    // 出力ディレクトリを作成
    std::fs::create_dir_all(&output_dir)
        .map_err(|e| CliError::Other(anyhow::anyhow!("Failed to create output directory: {}", e)))?;

    // DefFileConfigを作成
    let def_config = DefFileConfig {
        nsite: vmc_params.nsite.get(),
        ne: vmc_params.ne.get(),
        two_sz: vmc_params.two_sz.get(),
        model: stdface_config.model.model_type.clone(),
        lattice: stdface_config.lattice.lattice_type.clone(),
        j: stdface_config.model.parameters.get("j").copied(),
        t: stdface_config.model.parameters.get("t").copied(),
        u: stdface_config.model.parameters.get("u").copied(),
        mu: stdface_config.model.parameters.get("mu").copied(),
        nsr_opt_itr_step: vmc_params.sr_params.iteration_steps,
        nsr_opt_itr_smp: Some(100),
        dsr_opt_red_cut: Some(vmc_params.sr_params.reduction_cutoff),
        dsr_opt_step_dt: Some(vmc_params.sr_params.step_size),
        dsr_opt_sta_del: Some(vmc_params.sr_params.stability_delta),
        nvmc_warm_up: Some(vmc_params.mc_params.warmup_steps),
        nvmc_interval: Some(vmc_params.mc_params.sampling_interval),
        nvmc_sample: Some(vmc_params.mc_params.num_samples),
        rnd_seed: Some(vmc_params.random_seed.get()),
        c_data_file_head: "zvo".to_string(),
        c_para_file_head: "zqp".to_string(),
        n_data_idx_start: 1,
        n_data_qty_smp: 1,
    };

    // すべての.defファイルを生成
    let generator = DefFileGenerator::new(def_config);
    generator.generate_all(&output_dir)
        .map_err(|e| CliError::Other(anyhow::anyhow!("Failed to generate definition files: {}", e)))?;

    println!("✓ All input files generated successfully");
    println!();

    // VMC計算を実行
    println!("{}", "🔬 Running VMC calculation...".yellow());
    run_vmc_calculation(&vmc_params, &stdface_config, &output_dir)?;
    println!("✓ VMC calculation completed successfully");
    println!();
    println!("{}", "═══════════════════════════════════════════".cyan());
    println!("{}", "✓ Standard mode completed successfully".green().bold());
    println!("{}", "═══════════════════════════════════════════".cyan());

    Ok(())
}

/// StdFace設定をVmcParametersに変換
fn convert_stdface_to_vmc_params(stdface_config: &mvmc_io::StdFaceConfig) -> CliResult<VmcParameters> {
    // 基本的なパラメータを抽出
    let nsite = stdface_config.lattice.dimensions.iter().product::<usize>();
    let ne = stdface_config.calculation.n_particles.unwrap_or(nsite);
    let two_sz = stdface_config.calculation.total_sz.unwrap_or(0);

    // VmcParametersを作成
    let params = VmcParameters {
        nsite: SiteCount::new(nsite),
        ne: ElectronCount::new(ne),
        two_sz: TwoSz::new(two_sz),
        calc_mode: CalcMode::Expectation,
        lanczos_mode: LanczosMode::None,
        random_seed: RandomSeed::new(stdface_config.calculation.random_seed.unwrap_or(12345)),
        sr_params: SRParameters {
            iteration_steps: stdface_config.optimization.sr_steps.unwrap_or(100),
            iteration_sample: 100,
            fixed_sample_steps: 1,
            reduction_cutoff: stdface_config.optimization.sr_reduction_cutoff.unwrap_or(1e-6),
            stability_delta: stdface_config.optimization.sr_stabilization_delta.unwrap_or(1e-6),
            step_size: stdface_config.optimization.sr_step_delta.unwrap_or(0.1),
            cg_max_iterations: stdface_config.optimization.sr_cg_max_iter.unwrap_or(1000),
            cg_tolerance: stdface_config.optimization.sr_cg_tol.unwrap_or(1e-10),
            use_cg: stdface_config.optimization.sr_cg.unwrap_or(0) != 0,
        },
        mc_params: MonteCarloParameters {
            warmup_steps: stdface_config.monte_carlo.vmc_warmup_steps.unwrap_or(1000),
            sampling_interval: stdface_config.monte_carlo.vmc_sampling_interval.unwrap_or(10),
            num_samples: stdface_config.monte_carlo.vmc_samples.unwrap_or(1000),
            exchange_update: stdface_config.monte_carlo.ex_update_path.unwrap_or(0) != 0,
            block_update_size: stdface_config.monte_carlo.block_update_size.unwrap_or(1),
            exchange_ratio: stdface_config.monte_carlo.ex_update_ratio.unwrap_or(0.3),
        },
    };

    Ok(params)
}

/// VMC計算を実行してzvo_out_001.datを生成
fn run_vmc_calculation(vmc_params: &VmcParameters, stdface_config: &mvmc_io::StdFaceConfig, output_dir: &Path) -> CliResult<()> {
    use std::fs::File;
    use std::io::Write;
    use mvmc_core::vmc::VmcEngine;
    use mvmc_physics::hamiltonian::{HeisenbergHamiltonian, HubbardHamiltonian, Hamiltonian};
    use mvmc_physics::lattice::{ChainLattice, SquareLattice, Lattice};
    use mvmc_physics::wavefunction::CombinedWavefunction;
    use mvmc_core::types::CalcMode;

    // 最適化モードに設定
    let mut opt_params = vmc_params.clone();
    opt_params.calc_mode = CalcMode::Optimization;

    // Lattice selection based on StdFace
    let nsite = opt_params.nsite.get();
    let dims = &stdface_config.lattice.dimensions;
    let periodic = parse_periodic_from_stdface(&stdface_config);

    // Build Hamiltonian and wavefunction based on model type
    let model = stdface_config.model.model_type.to_lowercase();

    // Heisenberg (Spin) model
    let (mut wavefunction, hamiltonian_boxed): (CombinedWavefunction, Box<dyn Hamiltonian>) = if model == "spin" {
        // Ensure ne = 0 for spin model
        if opt_params.ne.get() != 0 {
            eprintln!("warning: Spin model detected but ne != 0; overriding ne -> 0 for spin-only calculation");
        }

        let j_exchange = *stdface_config.model.parameters.get("J").unwrap_or(&1.0);
        // Build lattice
        let h = match dims.len() {
            1 => {
                let lat = ChainLattice::new(dims[0], periodic)
                    .map_err(|e| CliError::Other(anyhow::anyhow!("Failed to create Chain lattice: {}", e)))?;
                HeisenbergHamiltonian::new(lat, j_exchange, 0.0)
            }
            2 => {
                let lat = SquareLattice::new(dims[0], dims[1], periodic)
                    .map_err(|e| CliError::Other(anyhow::anyhow!("Failed to create Square lattice: {}", e)))?;
                HeisenbergHamiltonian::new(lat, j_exchange, 0.0)
            }
            _ => {
                let lat = ChainLattice::new(nsite, true)
                    .map_err(|e| CliError::Other(anyhow::anyhow!("Failed to create lattice: {}", e)))?;
                HeisenbergHamiltonian::new(lat, j_exchange, 0.0)
            }
        }
            .map_err(|e| CliError::Other(anyhow::anyhow!("Failed to create Heisenberg Hamiltonian: {}", e)))?;

        // For spin model, construct combined wavefunction with ne=0 (Slater part becomes constant 1)
        let mut wf = CombinedWavefunction::new(nsite, 0)
            .map_err(|e| CliError::Other(anyhow::anyhow!("Failed to create wavefunction: {}", e)))?;

        // Add a simple spin-Jastrow projector to avoid constant wavefunction in spin model
        // Pairs: nearest-neighbor bonds from lattice
        let mut pairs: Vec<(usize, usize)> = Vec::new();
        for i in 0..nsite {
            for &j in h.lattice().neighbors(i).iter() {
                if i < j { pairs.push((i, j)); }
            }
        }
        // Small negative alpha favors antiferromagnetic correlations
        let alpha = -0.05_f64;
        wf.add_projector(Box::new(mvmc_physics::wavefunction::projection::SpinJastrowProjector::new(nsite, pairs, alpha)));

        (wf, Box::new(h))
    } else {
        // Fermionic (Hubbard) model: use parameters t, U, mu when available
        let t = *stdface_config.model.parameters.get("t").unwrap_or(&1.0);
        let u = *stdface_config.model.parameters.get("U").unwrap_or(&0.0);
        let mu = *stdface_config.model.parameters.get("mu").unwrap_or(&0.0);

        let h = match dims.len() {
            1 => {
                let lat = ChainLattice::new(dims[0], periodic)
                    .map_err(|e| CliError::Other(anyhow::anyhow!("Failed to create Chain lattice: {}", e)))?;
                HubbardHamiltonian::new(lat, t, u, mu)
            }
            2 => {
                let lat = SquareLattice::new(dims[0], dims[1], periodic)
                    .map_err(|e| CliError::Other(anyhow::anyhow!("Failed to create Square lattice: {}", e)))?;
                HubbardHamiltonian::new(lat, t, u, mu)
            }
            _ => {
                let lat = ChainLattice::new(nsite, true)
                    .map_err(|e| CliError::Other(anyhow::anyhow!("Failed to create lattice: {}", e)))?;
                HubbardHamiltonian::new(lat, t, u, mu)
            }
        }
            .map_err(|e| CliError::Other(anyhow::anyhow!("Failed to create Hubbard Hamiltonian: {}", e)))?;

        // Use CombinedWavefunction::new to initialize a Slater-based wavefunction
        let wf = CombinedWavefunction::new(nsite, opt_params.ne.get())
            .map_err(|e| CliError::Other(anyhow::anyhow!("Failed to create wavefunction: {}", e)))?;

        (wf, Box::new(h))
    };

    // VMCエンジンを作成
    let mut vmc_engine = VmcEngine::new(
        opt_params.clone(),
        wavefunction,
        hamiltonian_boxed,
    ).map_err(|e| CliError::Other(anyhow::anyhow!("Failed to create VMC engine: {}", e)))?;

    // 出力管理器（C準拠のファイル名）
    let out_mgr = mvmc_io::output::OutputManager::new(&output_dir, "zvo", 1);
    // Clean previous outputs to avoid mixing formats from earlier runs
    out_mgr.clean().map_err(|e| CliError::Other(anyhow::anyhow!("Failed to clean output dir: {}", e)))?;
    let srinfo_path = out_mgr.output_dir.join(format!("{}_SRinfo.dat", out_mgr.file_head));

    // VMC最適化を実行
    let num_iterations = opt_params.sr_params.iteration_steps;

    println!("  Running VMC optimization with:");
    println!("    - Combined wavefunction (Slater/Pfaffian + projections)");
    println!("    - {} Hamiltonian", if model == "spin" { "Heisenberg" } else { "Hubbard" });
    println!("    - Monte Carlo sampling");
    println!("    - Stochastic Reconfiguration optimization");
    println!("    - Parameter updates between iterations");
    println!();

    // 逐次イテレーションを実行し、各ステップで出力
    for iter in 0..num_iterations {
        println!("Standard: About to call run_single_iteration for iteration {}", iter);
        let result = vmc_engine
            .run_single_iteration()
            .map_err(|e| CliError::Other(anyhow::anyhow!("VMC iteration {} failed: {}", iter, e)))?;
        println!("Standard: run_single_iteration completed for iteration {}", iter);

        // zvo_out: Re(E), Im(E), Re(E^2), Re(Var), Re(Sz), Re(Sz^2)
        let energy_data = mvmc_io::output::EnergyData::new(
            num_complex::Complex64::new(result.energy.re, result.energy.im),
            num_complex::Complex64::new(result.energy_squared.re, result.energy_squared.im),
            num_complex::Complex64::new(result.sz_total, 0.0),
            num_complex::Complex64::new(result.sz_squared, 0.0),
        );
        // append one line to zvo_out_001.dat
        {
            use std::io::Write as _;
            let out_path = out_mgr.energy_output_path();
            std::fs::create_dir_all(out_path.parent().unwrap())
                .map_err(|e| CliError::Other(anyhow::anyhow!("Failed to ensure output dir: {}", e)))?;
            let mut f = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&out_path)
                .map_err(|e| CliError::Other(anyhow::anyhow!("Failed to open zvo_out: {}", e)))?;
            writeln!(
                f,
                "{:.18e} {:.18e}  {:.18e} {:.18e} {:.18e} {:.18e}",
                energy_data.energy.re,
                energy_data.energy.im,
                energy_data.energy_squared.re,
                energy_data.variance.re,
                energy_data.sz_total.re,
                energy_data.sz_squared.re,
            ).map_err(|e| CliError::Other(anyhow::anyhow!("Failed to append zvo_out: {}", e)))?;
        }

        // zvo_var: [E_re E_im 0.0 E2_re E2_im 0.0] then parameters [Re Im 0.0]*
        let params_vec = vmc_engine.wavefunction().export_parameters();
        let var_data = mvmc_io::output::VariationalData::new(energy_data, params_vec);
        out_mgr.append_variational(&var_data)
            .map_err(|e| CliError::Other(anyhow::anyhow!("Failed to append zvo_var: {}", e)))?;

        // zvo_SRinfo.dat: header once, then one line per step
        if !srinfo_path.exists() || std::fs::metadata(&srinfo_path).map(|m| m.len()).unwrap_or(0) == 0 {
            use std::io::Write as _;
            let mut f = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&srinfo_path)
                .map_err(|e| CliError::Other(anyhow::anyhow!("Failed to open zvo_SRinfo.dat: {}", e)))?;
            writeln!(f, "#Npara Msize optCut diagCut sDiagMax  sDiagMin    absRmax      imax")
                .map_err(|e| CliError::Other(anyhow::anyhow!("Failed to write zvo_SRinfo header: {}", e)))?;
        }
        if let Some(info) = &result.sr_info {
            use std::io::Write as _;
            let mut f = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&srinfo_path)
                .map_err(|e| CliError::Other(anyhow::anyhow!("Failed to open zvo_SRinfo.dat: {}", e)))?;
            writeln!(
                f,
                "{:5} {:5} {:5} {:5}  {:.5e}  {:.5e} {:.5e} {:6}",
                info.npara,
                info.msize,
                info.opt_cut,
                info.diag_cut,
                info.sdiag_max,
                info.sdiag_min,
                info.abs_rmax,
                info.imax,
            ).map_err(|e| CliError::Other(anyhow::anyhow!("Failed to append zvo_SRinfo.dat: {}", e)))?;
        }

        if iter % (num_iterations.max(1) / 20 + 1) == 0 || iter + 1 == num_iterations {
            println!("  Iteration {}/{}: Energy = {:.6}", iter + 1, num_iterations, result.energy.re);
        }
    }

    Ok(())
}

/// Try to interpret periodic boundary from StdFace.
/// Defaults to true (mVMC standard). Supports keys in additional:
/// - periodic = 0/1
/// - boundary = Open/Periodic (case-insensitive)
fn parse_periodic_from_stdface(cfg: &mvmc_io::StdFaceConfig) -> bool {
    // default periodic
    let mut periodic = true;
    // normalize keys
    for (k, v) in &cfg.additional {
        let key = k.to_lowercase();
        let val = v.to_lowercase();
        if key == "periodic" {
            if let Ok(n) = val.parse::<i32>() { return n != 0; }
            if val.contains("true") { return true; }
            if val.contains("false") { return false; }
        }
        if key == "boundary" || key == "boundarycondition" {
            if val.starts_with("open") { return false; }
            if val.starts_with("periodic") { return true; }
        }
    }
    periodic
}
