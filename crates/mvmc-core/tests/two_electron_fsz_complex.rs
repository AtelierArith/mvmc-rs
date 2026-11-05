//! #679: complex FSZ `UpdateMAllTwo_fsz` (rank-two update of an accepted
//! exchange) against an independent Pfaffian and inverse of the post-move
//! matrix.
use mvmc_core::sampling::updates::update_m_all_two_fsz_complex_flat;
use mvmc_core::SlaterElmFlat;
use num_complex::Complex64;

type C = Complex64;

/// Expansion along the first row (exact up to roundoff for n <= 8).
fn pf(m: &[Vec<C>]) -> C {
    let n = m.len();
    if n == 0 {
        return C::new(1.0, 0.0);
    }
    let mut acc = C::new(0.0, 0.0);
    for j in 1..n {
        let rest: Vec<usize> = (1..n).filter(|&k| k != j).collect();
        let sub: Vec<Vec<C>> = rest
            .iter()
            .map(|&r| rest.iter().map(|&c| m[r][c]).collect())
            .collect();
        let sign = if (j - 1) % 2 == 0 { 1.0 } else { -1.0 };
        acc += m[0][j] * pf(&sub) * sign;
    }
    acc
}

/// Gauss-Jordan inverse with partial pivoting.
#[allow(clippy::needless_range_loop)]
fn inverse(m: &[Vec<C>]) -> Vec<Vec<C>> {
    let n = m.len();
    let mut a: Vec<Vec<C>> = (0..n)
        .map(|i| {
            let mut r = m[i].clone();
            r.extend((0..n).map(|j| C::new(f64::from(u8::from(i == j)), 0.0)));
            r
        })
        .collect();
    for c in 0..n {
        let p = (c..n)
            .max_by(|&x, &y| a[x][c].norm().total_cmp(&a[y][c].norm()))
            .unwrap();
        a.swap(c, p);
        let d = a[c][c];
        for v in a[c].iter_mut() {
            *v /= d;
        }
        for r in 0..n {
            if r != c {
                let f = a[r][c];
                for k in 0..2 * n {
                    let t = a[c][k];
                    a[r][k] -= f * t;
                }
            }
        }
    }
    a.into_iter().map(|r| r[n..].to_vec()).collect()
}

fn max_abs(m: &[Vec<C>]) -> f64 {
    m.iter().flatten().map(|z| z.norm()).fold(0.0, f64::max)
}

fn lcg(s: &mut u64) -> f64 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*s >> 11) as f64) / ((1u64 << 53) as f64) - 0.5
}

struct Fsz {
    n_site: usize,
    n_elec: usize,
    slt: SlaterElmFlat<C>,
    ele_idx: Vec<i64>,
    ele_spn: Vec<i64>,
    ele_cfg: Vec<i64>,
}

impl Fsz {
    fn new(n_site: usize, n_elec: usize, seed: u64) -> Self {
        let n2 = 2 * n_site;
        let mut s = seed;
        let mut slt = SlaterElmFlat::<C>::zeros(1, n_site);
        for i in 0..n2 {
            for j in (i + 1)..n2 {
                let v = C::new(lcg(&mut s), lcg(&mut s));
                slt.set(0, i, j, v);
                slt.set(0, j, i, -v);
            }
        }
        // Electrons alternate spins over the first sites, as a Neel start.
        let n_size = 2 * n_elec;
        let mut ele_idx = vec![0; n_size];
        let mut ele_spn = vec![0; n_size];
        let mut ele_cfg = vec![-1; n2];
        for m in 0..n_size {
            let (r, spin) = (m / 2 + (m % 2) * (n_site / 2), m % 2);
            ele_idx[m] = r as i64;
            ele_spn[m] = spin as i64;
            ele_cfg[r + spin * n_site] = m as i64;
        }
        Self {
            n_site,
            n_elec,
            slt,
            ele_idx,
            ele_spn,
            ele_cfg,
        }
    }

    fn matrix(&self) -> Vec<Vec<C>> {
        let n_size = 2 * self.n_elec;
        let rs = |m: usize| (self.ele_idx[m] + self.ele_spn[m] * self.n_site as i64) as usize;
        (0..n_size)
            .map(|i| (0..n_size).map(|j| self.slt.get(0, rs(i), rs(j))).collect())
            .collect()
    }

    /// The vmcmake_fsz.c exchange: `mi` (spin s) hops ri -> rj, then the
    /// spin-t electron `mj` at rj hops rj -> ri. Returns (mi, s, mj, t, ri, rj).
    fn exchange(&mut self, seed: &mut u64) -> Option<(usize, u8, usize, u8, usize, usize)> {
        let n_size = 2 * self.n_elec;
        for _ in 0..1000 {
            let mi = ((lcg(seed) + 0.5) * n_size as f64) as usize % n_size;
            let s = self.ele_spn[mi] as usize;
            let t = 1 - s;
            let ri = self.ele_idx[mi] as usize;
            let rj = ((lcg(seed) + 0.5) * self.n_site as f64) as usize % self.n_site;
            let n = self.n_site;
            // rj must hold a t electron and no s electron; ri no t electron.
            if self.ele_cfg[rj + t * n] < 0
                || self.ele_cfg[rj + s * n] >= 0
                || self.ele_cfg[ri + t * n] >= 0
            {
                continue;
            }
            let mj = self.ele_cfg[rj + t * n] as usize;
            self.ele_cfg[ri + s * n] = -1;
            self.ele_cfg[rj + t * n] = -1;
            self.ele_idx[mi] = rj as i64;
            self.ele_idx[mj] = ri as i64;
            self.ele_cfg[rj + s * n] = mi as i64;
            self.ele_cfg[ri + t * n] = mj as i64;
            return Some((mi, s as u8, mj, t as u8, ri, rj));
        }
        None
    }
}

/// Twenty consecutive accepted exchanges per case, as in a sampling interval
/// before the `nAccept > Nsite` refresh. After each update the Pfaffian and
/// every inverse entry are compared with a fresh first-row Pfaffian expansion
/// and a Gauss-Jordan inverse of the post-move matrix.
///
/// Bound: the rank-two update's error after k steps is a sum of k
/// O(n_size^2 eps) inverse perturbations amplified by the condition number,
/// so each entry is checked against `16 * (k+1) * n_size * eps * kappa *
/// max|InvM|` (and the Pfaffian relatively, with the same factor), with
/// `kappa = max|M| * max|InvM| * n_size` the max-norm condition estimate of
/// the current matrix. The largest observed error is 0.09 of this bound.
#[test]
fn complex_fsz_rank_two_update_matches_fresh_pfaffian_and_inverse() {
    let mut worst: f64 = 0.0;
    let mut steps = 0;
    for (n_site, n_elec, seed) in [(4, 2, 11), (6, 3, 7), (8, 4, 3), (8, 3, 19)] {
        let mut case = Fsz::new(n_site, n_elec, seed);
        let n_size = 2 * n_elec;
        let stride = n_size * n_size + 1;
        let m0 = case.matrix();
        let inv0 = inverse(&m0);
        let mut inv = vec![C::new(0.0, 0.0); stride];
        for i in 0..n_size {
            for j in 0..n_size {
                inv[i * n_size + j] = inv0[i][j];
            }
        }
        let mut pfm = vec![pf(&m0)];
        let mut moves = seed.wrapping_mul(977);
        for k in 0..20 {
            let Some((mi, s, mj, t, ri, rj)) = case.exchange(&mut moves) else {
                panic!("no exchange available for case {n_site}/{n_elec}");
            };
            update_m_all_two_fsz_complex_flat(
                mi,
                s,
                mj,
                t,
                ri,
                rj,
                &case.ele_idx,
                &case.ele_spn,
                &case.slt,
                &mut inv,
                stride,
                &mut pfm,
                0,
                1,
                n_site,
                n_elec,
            );
            let m = case.matrix();
            let expected_inv = inverse(&m);
            let expected_pf = pf(&m);
            let inv_scale = max_abs(&expected_inv);
            let kappa = max_abs(&m) * inv_scale * n_size as f64;
            let factor = 16.0 * (k + 1) as f64 * n_size as f64 * f64::EPSILON * kappa;
            let pf_err = (pfm[0] - expected_pf).norm() / expected_pf.norm();
            assert!(
                pf_err <= factor,
                "{n_site}/{n_elec} step {k}: Pfaffian relative error {pf_err:e} > {factor:e}"
            );
            for i in 0..n_size {
                for j in 0..n_size {
                    let err = (inv[i * n_size + j] - expected_inv[i][j]).norm();
                    assert!(
                        err <= factor * inv_scale,
                        "{n_site}/{n_elec} step {k}: InvM[{i}][{j}] error {err:e} > {:e}",
                        factor * inv_scale
                    );
                    worst = worst.max(err / (factor * inv_scale));
                }
            }
            worst = worst.max(pf_err / factor);
            steps += 1;
        }
    }
    assert_eq!(steps, 80);
    eprintln!("worst error / bound = {worst:e}");
}

/// The port uses the mVMC 1.4.0 `rsbOld = rbOld + t*Nsite` (upstream #100);
/// 1.3.0 used `raOld`. The old `(a,b)` element cancels, so any `rb_old`
/// argument, including the 1.3.0 value `ra_old`, gives the same update to
/// roundoff (6x6, O(1) entries, one update: 1e-11 against observed ~1e-14).
#[test]
fn complex_fsz_rank_two_update_is_independent_of_the_old_ab_element() {
    let (n_site, n_elec) = (6, 3);
    let n_size = 2 * n_elec;
    let stride = n_size * n_size + 1;
    let mut case = Fsz::new(n_site, n_elec, 23);
    let m0 = case.matrix();
    let inv0 = inverse(&m0);
    let pf0 = pf(&m0);
    let mut seed = 5;
    let (mi, s, mj, t, ri, rj) = case.exchange(&mut seed).unwrap();
    let m = case.matrix();
    let expected_inv = inverse(&m);
    let expected_pf = pf(&m);
    for rb_old in 0..n_site {
        let mut inv = vec![C::new(0.0, 0.0); stride];
        for i in 0..n_size {
            for j in 0..n_size {
                inv[i * n_size + j] = inv0[i][j];
            }
        }
        let mut pfm = vec![pf0];
        update_m_all_two_fsz_complex_flat(
            mi,
            s,
            mj,
            t,
            ri,
            rb_old,
            &case.ele_idx,
            &case.ele_spn,
            &case.slt,
            &mut inv,
            stride,
            &mut pfm,
            0,
            1,
            n_site,
            n_elec,
        );
        assert!(
            (pfm[0] - expected_pf).norm() < 1e-11,
            "rb_old={rb_old} (rj={rj})"
        );
        for i in 0..n_size {
            for j in 0..n_size {
                assert!(
                    (inv[i * n_size + j] - expected_inv[i][j]).norm() < 1e-11,
                    "rb_old={rb_old} InvM[{i}][{j}]"
                );
            }
        }
    }
}

/// QP planes outside `qp_start..qp_end` are left untouched and each plane in
/// range is updated independently with its own Slater table.
#[test]
fn complex_fsz_rank_two_update_touches_only_the_requested_qp_planes() {
    let (n_site, n_elec) = (4, 2);
    let n_size = 2 * n_elec;
    let stride = n_size * n_size + 1;
    let mut base = Fsz::new(n_site, n_elec, 5);
    let other = Fsz::new(n_site, n_elec, 6);
    // Three QP planes: plane 1 uses the second Slater table.
    let mut slt = SlaterElmFlat::<C>::zeros(3, n_site);
    for qp in 0..3 {
        let source = if qp == 1 { &other.slt } else { &base.slt };
        for i in 0..2 * n_site {
            for j in 0..2 * n_site {
                slt.set(qp, i, j, source.get(0, i, j));
            }
        }
    }
    let old = base.clone_config();
    let mut seed = 99;
    let (mi, s, mj, t, ri, rj) = base.exchange(&mut seed).unwrap();
    let mut inv = vec![C::new(0.0, 0.0); 3 * stride];
    let mut pfm = vec![C::new(0.0, 0.0); 3];
    let build = |table: &SlaterElmFlat<C>, idx: &[i64], spn: &[i64]| -> Vec<Vec<C>> {
        let rs = |m: usize| (idx[m] + spn[m] * n_site as i64) as usize;
        (0..n_size)
            .map(|i| (0..n_size).map(|j| table.get(0, rs(i), rs(j))).collect())
            .collect()
    };
    for qp in 0..3 {
        let table = if qp == 1 { &other.slt } else { &base.slt };
        let m = build(table, &old.0, &old.1);
        let m_inv = inverse(&m);
        for i in 0..n_size {
            for j in 0..n_size {
                inv[qp * stride + i * n_size + j] = m_inv[i][j];
            }
        }
        pfm[qp] = pf(&m);
    }
    let untouched = (inv[..stride].to_vec(), pfm[0]);
    update_m_all_two_fsz_complex_flat(
        mi,
        s,
        mj,
        t,
        ri,
        rj,
        &base.ele_idx,
        &base.ele_spn,
        &slt,
        &mut inv,
        stride,
        &mut pfm,
        1,
        3,
        n_site,
        n_elec,
    );
    assert_eq!(inv[..stride], untouched.0[..]);
    assert_eq!(pfm[0], untouched.1);
    for (qp, &updated) in pfm.iter().enumerate().skip(1) {
        let table = if qp == 1 { &other.slt } else { &base.slt };
        let m = build(table, &base.ele_idx, &base.ele_spn);
        let expected = pf(&m);
        // 4x4, O(1) entries, one update: 1e-12 is far above the observed 1e-15.
        assert!(
            (updated - expected).norm() <= 1e-12 * expected.norm().max(1.0),
            "qp {qp}"
        );
    }
}

impl Fsz {
    fn clone_config(&self) -> (Vec<i64>, Vec<i64>) {
        (self.ele_idx.clone(), self.ele_spn.clone())
    }
}

/// 20-step repeatability of complex FSZ optimization with the rank-two
/// exchange update (AGENTS.md long-run policy). The Heisenberg FSZ chain
/// samples with `NExUpdatePath 2`, so accepted exchanges go through
/// `update_m_all_two_fsz_complex_flat`. The same inputs and seed must
/// reproduce every output byte.
#[test]
fn complex_fsz_twenty_step_optimization_is_repeatable() {
    let input = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../extern/Julia-mVMC/examples/inputs/heisenberg_chain_fsz/namelist.def");
    let base = std::env::temp_dir().join(format!("mvmc-679-repeat-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&base);
    let run = |name: &str| {
        let dir = base.join(name);
        let config = mvmc_core::RunConfig {
            nsmp: Some(20),
            seed: Some(679),
            output_dir: Some(dir.clone()),
            ..mvmc_core::RunConfig::new(20, "fsz")
        };
        let result = mvmc_core::run_para_opt_from_namelist(&input, config).unwrap();
        assert_eq!(result.status, 0);
        (dir, result.final_energy_per_site)
    };
    let (first, energy) = run("first");
    let (second, repeated) = run("second");
    assert!(energy.is_finite());
    assert_eq!(energy.to_bits(), repeated.to_bits());
    for name in ["zvo_out.dat", "zqp_opt.dat", "zvo_var.dat"] {
        assert_eq!(
            std::fs::read(first.join(name)).unwrap(),
            std::fs::read(second.join(name)).unwrap(),
            "{name}"
        );
    }
    let _ = std::fs::remove_dir_all(&base);
}
