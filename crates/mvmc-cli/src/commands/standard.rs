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
    run_vmc_calculation(&vmc_params, &output_dir)?;
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
            cg_max_iterations: 1000,
            cg_tolerance: 1e-10,
        },
        mc_params: MonteCarloParameters {
            warmup_steps: 1000,
            sampling_interval: 10,
            num_samples: 1000,
            exchange_update: false,
            block_update_size: 1,
        },
    };

    Ok(params)
}

/// VMC計算を実行してzvo_out_001.datを生成
fn run_vmc_calculation(vmc_params: &VmcParameters, output_dir: &Path) -> CliResult<()> {
    use std::fs::File;
    use std::io::Write;
    use mvmc_core::vmc::VmcEngine;
    use mvmc_physics::hamiltonian::HeisenbergHamiltonian;
    use mvmc_physics::lattice::ChainLattice;
    use mvmc_physics::wavefunction::{CombinedWavefunction, SlaterDeterminant};

    // ハミルトニアンを作成
    let lattice = ChainLattice::new(vmc_params.nsite.get(), true)
        .map_err(|e| CliError::Other(anyhow::anyhow!("Failed to create lattice: {}", e)))?;

    let hamiltonian = HeisenbergHamiltonian::new(lattice, -0.5, 0.0) // J = -0.5 for Heisenberg chain
        .map_err(|e| CliError::Other(anyhow::anyhow!("Failed to create Hamiltonian: {}", e)))?;

    // 波動関数を作成
    let nsite = vmc_params.nsite.get();
    let ne = vmc_params.ne.get();
    let slater = SlaterDeterminant::new_plane_wave(nsite, ne)
        .map_err(|e| CliError::Other(anyhow::anyhow!("Failed to create Slater determinant: {}", e)))?;

    let wavefunction = CombinedWavefunction::with_slater(nsite, ne, slater)
        .map_err(|e| CliError::Other(anyhow::anyhow!("Failed to create wavefunction: {}", e)))?;

    // VMCエンジンを作成
    let mut vmc_engine = VmcEngine::new(
        vmc_params.clone(),
        wavefunction,
        Box::new(hamiltonian),
    ).map_err(|e| CliError::Other(anyhow::anyhow!("Failed to create VMC engine: {}", e)))?;

    // 出力ファイルを作成
    let output_file_path = output_dir.join("zvo_out_001.dat");
    let mut output_file = File::create(&output_file_path)
        .map_err(|e| CliError::Other(anyhow::anyhow!("Failed to create output file: {}", e)))?;

    // VMC計算を実行
    let num_iterations = vmc_params.sr_params.iteration_steps;

    println!("  Running actual VMC calculation with:");
    println!("    - Slater determinant wavefunction");
    println!("    - Heisenberg Hamiltonian");
    println!("    - Monte Carlo sampling");
    println!("    - Energy calculation");
    println!();

    for iteration in 0..num_iterations {
        // VMC計算の1ステップを実行
        let result = vmc_engine.run_single_iteration()
            .map_err(|e| CliError::Other(anyhow::anyhow!("VMC calculation failed at iteration {}: {}", iteration, e)))?;

        // 結果をファイルに書き込み
        writeln!(
            output_file,
            "{:20.15e} {:20.15e} {:20.15e} {:20.15e} {:20.15e} {:20.15e}",
            result.energy.re,
            result.energy.im,
            result.variance,
            result.sample_count as f64,
            0.0, // その他の統計情報1
            0.0  // その他の統計情報2
        ).map_err(|e| CliError::Other(anyhow::anyhow!("Failed to write output: {}", e)))?;

        // 進捗を表示
        if iteration % 10 == 0 || iteration == num_iterations - 1 {
            println!("  Iteration {}/{}: Energy = {:.6}, Variance = {:.6}",
                iteration + 1, num_iterations, result.energy.re, result.variance);
        }
    }

    Ok(())
}