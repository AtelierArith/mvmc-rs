use mvmc_core::sampling::updates::update_m_all_two_complex_flat;
use mvmc_core::SlaterElmFlat;
use num_complex::Complex64;

type C = Complex64;

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

#[allow(clippy::needless_range_loop)]
fn inverse(m: &[Vec<C>]) -> Vec<Vec<C>> {
    let n = m.len();
    let mut a: Vec<Vec<C>> = (0..n)
        .map(|i| {
            let mut r = m[i].clone();
            r.extend((0..n).map(|j| {
                if i == j {
                    C::new(1.0, 0.0)
                } else {
                    C::new(0.0, 0.0)
                }
            }));
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

fn lcg(s: &mut u64) -> f64 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*s >> 11) as f64) / ((1u64 << 53) as f64) - 0.5
}

/// One two-electron move: electron `ma` (spin 0) hops to `ra`, electron `mb` (spin 1) hops to `rb`.
struct Case {
    n_site: usize,
    n_elec: usize,
    old_idx: Vec<i64>,
    ma: usize,
    mb: usize,
    ra: usize,
    rb: usize,
}

/// Returns (Pf error, InvM error, updated Pf, updated InvM). Errors are against an independent
/// recomputation of the Pfaffian and the inverse of the new matrix.
fn run(case: &Case, ra_old_arg: usize, rb_old_arg: usize) -> (f64, f64, C, Vec<C>) {
    let Case {
        n_site,
        n_elec,
        ma,
        mb,
        ..
    } = *case;
    let n_size = 2 * n_elec;
    let n2 = 2 * n_site;
    let mut seed = 12345u64;
    let mut slt = SlaterElmFlat::<C>::zeros(1, n_site);
    for i in 0..n2 {
        for j in (i + 1)..n2 {
            let v = C::new(lcg(&mut seed), lcg(&mut seed));
            slt.set(0, i, j, v);
            slt.set(0, j, i, -v);
        }
    }
    let mut new_idx = case.old_idx.clone();
    new_idx[ma] = case.ra as i64;
    new_idx[mb + n_elec] = case.rb as i64;
    let build = |idx: &[i64]| -> Vec<Vec<C>> {
        (0..n_size)
            .map(|i| {
                (0..n_size)
                    .map(|j| {
                        slt.get(
                            0,
                            idx[i] as usize + (i / n_elec) * n_site,
                            idx[j] as usize + (j / n_elec) * n_site,
                        )
                    })
                    .collect()
            })
            .collect()
    };
    let m_old = build(&case.old_idx);
    let m_new = build(&new_idx);
    let inv_old = inverse(&m_old);
    let inv_new = inverse(&m_new);
    let stride = n_size * n_size + 1;
    let mut inv: Vec<C> = vec![C::new(0.0, 0.0); stride];
    for i in 0..n_size {
        for j in 0..n_size {
            inv[i * n_size + j] = inv_old[i][j];
        }
    }
    let mut pfm = vec![pf(&m_old)];
    update_m_all_two_complex_flat(
        ma, 0, mb, 1, ra_old_arg, rb_old_arg, &new_idx, &slt, &mut inv, stride, &mut pfm, 0, 1,
        n_site, n_elec,
    );
    let pf_err = (pfm[0] - pf(&m_new)).norm();
    let mut inv_err: f64 = 0.0;
    for i in 0..n_size {
        for j in 0..n_size {
            inv_err = inv_err.max((inv[i * n_size + j] - inv_new[i][j]).norm());
        }
    }
    (pf_err, inv_err, pfm[0], inv)
}

/// C defines `rsbOld = raOld + t*Nsite` in `pfupdate_two_fcmp.c:227` (and the real/fsz/fsz_real
/// variants) where `rbOld + t*Nsite` was evidently meant, and the Rust real paths reproduce it.
/// The complex path uses `rb_old`. The old (a,b) matrix element only enters through
/// `vec_s[msb]`, and the updated Pfaffian and inverse do not depend on it beyond roundoff:
/// every (raOld, rbOld) argument pair must reproduce the independently recomputed Pfaffian and
/// inverse of the new matrix. Tolerance 1e-11: entries are O(1), matrices are 4x4/6x6, and the
/// observed errors are about 1e-14 (inverse) and 1e-15 (Pfaffian).
#[test]
fn complex_two_electron_update_is_independent_of_old_ab_element() {
    let cases = [
        // exchange: the two electrons swap sites (the vmcmake / lslocgrn call pattern)
        Case {
            n_site: 4,
            n_elec: 2,
            old_idx: vec![0, 1, 2, 3],
            ma: 0,
            mb: 0,
            ra: 2,
            rb: 0,
        },
        // two independent hops to empty sites
        Case {
            n_site: 5,
            n_elec: 2,
            old_idx: vec![0, 1, 2, 3],
            ma: 0,
            mb: 1,
            ra: 4,
            rb: 1,
        },
        Case {
            n_site: 6,
            n_elec: 3,
            old_idx: vec![0, 1, 2, 3, 4, 5],
            ma: 2,
            mb: 0,
            ra: 3,
            rb: 2,
        },
    ];
    for (ci, case) in cases.iter().enumerate() {
        let mut reference: Option<(C, Vec<C>)> = None;
        for ra_old in 0..case.n_site {
            for rb_old in 0..case.n_site {
                let (pf_err, inv_err, pf_v, inv_v) = run(case, ra_old, rb_old);
                assert!(
                    pf_err < 1e-11,
                    "case {ci} ({ra_old},{rb_old}) pf err {pf_err:e}"
                );
                assert!(
                    inv_err < 1e-11,
                    "case {ci} ({ra_old},{rb_old}) inv err {inv_err:e}"
                );
                match &reference {
                    None => reference = Some((pf_v, inv_v)),
                    Some((pf_r, inv_r)) => {
                        assert!((pf_v - pf_r).norm() < 1e-11);
                        for (a, b) in inv_v.iter().zip(inv_r.iter()) {
                            assert!((a - b).norm() < 1e-11);
                        }
                    }
                }
            }
        }
    }
}
