//! The fused SR steps of `SrStages` (issue #447) on the host backends: the C-order defaults
//! equal the stage-by-stage composition (they are the oracle of the resident device step), the
//! tenferro CPU backend agrees within the documented bounds, and the CG default is exactly the
//! production `SampledSrOperator` solve.

use mvmc_core::sr_backend::{
    COrderSr, CgStepInput, DirectStepInput, RealView, SrAssembleInput, SrStages,
};
use mvmc_core::sr_cg::SampledSrOperator;
use mvmc_core::stage_backend::StageBackend;

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    }
}

/// Store with the real layout: row 0 constant, rows 1.. sampled, scaled by `1/sqrt(samples)`.
fn store(n: usize, samples: usize, seed: u64) -> Vec<f64> {
    let mut rng = Lcg(seed);
    let scale = 1.0 / (samples as f64).sqrt();
    let mut s = vec![0.0; n * samples];
    for k in 0..samples {
        s[k * n] = scale;
        for p in 1..n {
            s[p + k * n] = scale * (rng.next() + rng.next() + 0.2 * (p % 5) as f64);
        }
    }
    s
}

#[test]
fn c_order_direct_step_is_the_stage_composition_and_tenferro_agrees() {
    let (n, samples) = (24, 90);
    let store = store(n, samples, 5);
    let mut rng = Lcg(9);
    let ho: Vec<f64> = (0..n).map(|_| rng.next()).collect();
    let map: Vec<usize> = (0..n - 1).filter(|p| p % 5 != 4).collect();
    let input = DirectStepInput {
        store: &store,
        n,
        samples,
        ho: &ho,
        map: &map,
        offset: 1,
        sta_del: 0.01,
        step_dt: 0.003,
    };
    let mut c = COrderSr::default();
    let x = c.direct_step(&input).expect("direct step");

    // manual composition through the individual stages
    let nm = map.len();
    let mut gram = vec![0.0; n * n];
    c.gram_real(&store, n, samples, &mut gram).unwrap();
    let mut s = vec![0.0; nm * nm];
    let mut g = vec![0.0; nm];
    c.assemble_s_g(
        &SrAssembleInput {
            oo: RealView::Real(&gram),
            ho: RealView::Real(&ho),
            map: &map,
            ld: n,
            offset: 1,
            sta_del: 0.01,
            step_dt: 0.003,
        },
        &mut s,
        &mut g,
    )
    .unwrap();
    c.cholesky_solve(&mut s, &mut g, nm).unwrap();
    assert!(x.iter().zip(&g).all(|(a, b)| a.to_bits() == b.to_bits()));

    // tenferro CPU: same fused step through its own stages, within solve-level bounds
    let mut t = StageBackend::tenferro_cpu().expect("tenferro cpu");
    let xt = t.sr().direct_step(&input).expect("tenferro direct step");
    let num: f64 = x
        .iter()
        .zip(&xt)
        .map(|(a, b)| (a - b) * (a - b))
        .sum::<f64>()
        .sqrt();
    let den: f64 = x.iter().map(|a| a * a).sum::<f64>().sqrt();
    assert!(num / den < 1e-9, "tenferro vs C order: {}", num / den);
}

#[test]
fn indefinite_direct_step_fails_with_a_typed_error() {
    let (n, samples) = (12, 40);
    let store = store(n, samples, 3);
    let ho = vec![0.1; n];
    let map: Vec<usize> = (0..n - 1).collect();
    let input = DirectStepInput {
        store: &store,
        n,
        samples,
        ho: &ho,
        map: &map,
        offset: 1,
        sta_del: -2.0,
        step_dt: 0.003,
    };
    assert!(COrderSr::default().direct_step(&input).is_err());
}

#[test]
fn c_order_cg_step_is_the_production_operator_solve() {
    let (n, samples) = (30, 120);
    let st = store(n, samples, 7);
    let comp = n - 1;
    let mut operand = vec![0.0; comp * samples];
    let mut mean = vec![0.0; comp];
    let mut second = vec![0.0; comp];
    for k in 0..samples {
        for p in 0..comp {
            let o = st[p + 1 + k * n];
            operand[p + k * comp] = o;
            mean[p] += o * st[k * n];
            second[p] += o * o;
        }
    }
    let diagonal: Vec<f64> = second.iter().zip(&mean).map(|(s, m)| s - m * m).collect();
    let mut rng = Lcg(2);
    let gradient: Vec<f64> = (0..comp).map(|_| rng.next()).collect();
    let input = CgStepInput {
        real: &operand,
        imag: None,
        components: comp,
        samples,
        gradient: &gradient,
        mean: &mean,
        diagonal: &diagonal,
        inv_weight: 1.0,
        shift: 0.3,
        tolerance: 1e-10,
        max_iterations: comp,
    };
    let step = COrderSr::default().cg_step(&input).expect("cg step");

    let mut op = SampledSrOperator::new(comp, samples, false);
    op.mean.copy_from_slice(&mean);
    op.diagonal.copy_from_slice(&diagonal);
    op.real_samples.copy_from_slice(&operand);
    let reference = op.solve(&gradient, 1.0, 0.3, 1e-10, comp);
    assert_eq!(step.iterations, reference.iterations);
    assert!(step
        .solution
        .iter()
        .zip(&reference.solution)
        .all(|(a, b)| a.to_bits() == b.to_bits()));
}
