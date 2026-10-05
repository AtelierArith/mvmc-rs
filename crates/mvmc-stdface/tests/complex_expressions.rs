//! The compound `double complex` expressions of the C StdFace evaluated on a grid that includes
//! `-0.0`, compared (as IEEE bit patterns) with the C compiler's results.
//!
//! The grid, the expressions and the FNV-1a family digests come from
//! `c_toolbox/stdface/complex_expr.c`; `tests/fixtures/stdface/complex_expr.digests` stores the
//! digest of every family produced by that probe (gcc -O3; -O0 and -O2 agree).
//! Set `STDFACE_LINES=1` to print the raw lines (diffable against `STDFACE_LINES=1 ./probe`).

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use mvmc_stdface::ccomplex::C64;
use mvmc_stdface::cexpr;

#[derive(Default)]
struct Digests {
    order: Vec<String>,
    map: BTreeMap<String, (u64, u64)>,
    print_lines: bool,
}

impl Digests {
    fn emit(&mut self, name: &str, inputs: &[f64], v: C64) {
        let mut line = name.to_string();
        for x in inputs {
            line.push_str(&format!(" {:016x}", x.to_bits()));
        }
        line.push_str(&format!(
            " -> {:016x} {:016x}\n",
            v.re.to_bits(),
            v.im.to_bits()
        ));
        if self.print_lines {
            print!("{line}");
            return;
        }
        if !self.map.contains_key(name) {
            self.order.push(name.to_string());
        }
        let entry = self
            .map
            .entry(name.to_string())
            .or_insert((1469598103934665603, 0));
        for b in line.bytes() {
            entry.0 ^= b as u64;
            entry.0 = entry.0.wrapping_mul(1099511628211);
        }
        entry.1 += 1;
    }
}

fn run_grid(d: &mut Digests) {
    let g = [-2.0, -0.0, 0.0, 0.5, 3.0];
    let spins = [(0.5, 0.5), (0.5, -0.5), (1.0, 1.0), (1.0, 0.0), (1.0, -1.0)];
    for &a in &g {
        d.emit("hl_im_neg", &[a], cexpr::hubbard_local_im_neg(a));
        d.emit("hl_im_pos", &[a], cexpr::hubbard_local_im_pos(a));
    }
    for &a in &g {
        for &b in &g {
            for &(s, sz) in &spins {
                let inp = [a, b, s, sz];
                let m = cexpr::mag_field_minus(a, b, s, sz);
                let p = cexpr::mag_field_plus(a, b, s, sz);
                d.emit("mf_minus", &inp, m);
                d.emit("mf_plus", &inp, p);
                d.emit("conj_mf_minus", &inp, m.conj());
            }
        }
    }
    for &a in &g {
        for &b in &g {
            for &(s, sz) in &spins {
                d.emit("gj_zz", &[a, b, s], cexpr::general_j_zz(a, b, sz));
            }
        }
    }
    for &a in &g {
        for &b in &g {
            for &c in &g {
                for &e in &g {
                    let mut j = [[0.0; 3]; 3];
                    j[0][0] = a;
                    j[1][1] = b;
                    j[0][1] = c;
                    j[1][0] = e;
                    for &(si, siz) in &spins {
                        for &(sj, sjz) in &spins {
                            let inp = [a, b, c, e, si, siz, sj, sjz];
                            let pm = cexpr::general_j_pm(&j, si, siz, sj, sjz);
                            d.emit("gj_pm", &inp, pm);
                            d.emit("conj_gj_pm", &inp, pm.conj());
                            let pp = cexpr::general_j_pp(&j, si, siz, sj, sjz);
                            d.emit("gj_pp", &inp, pp);
                            d.emit("conj_gj_pp", &inp, pp.conj());
                        }
                    }
                }
            }
        }
    }
    for &a in &g {
        for &b in &g {
            for &(si, siz) in &spins {
                for &(sj, sjz) in &spins {
                    let mut j = [[0.0; 3]; 3];
                    j[0][2] = a;
                    j[1][2] = b;
                    j[2][0] = a;
                    j[2][1] = b;
                    let inp = [a, b, si, siz, sj, sjz];
                    let pz = cexpr::general_j_pz(&j, si, siz, sjz);
                    d.emit("gj_pz", &inp, pz);
                    d.emit("conj_gj_pz", &inp, pz.conj());
                    let zp = cexpr::general_j_zp(&j, sj, siz, sjz);
                    d.emit("gj_zp", &inp, zp);
                    d.emit("conj_gj_zp", &inp, zp.conj());
                }
            }
        }
    }
}

#[test]
fn complex_expressions_match_the_c_compiler_bitwise() {
    let mut d = Digests {
        print_lines: std::env::var_os("STDFACE_LINES").is_some(),
        ..Default::default()
    };
    run_grid(&mut d);
    if d.print_lines {
        return;
    }
    let fixture = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/stdface/complex_expr.digests");
    let expected = fs::read_to_string(fixture).unwrap();
    let mut got = String::new();
    for name in &d.order {
        let (hash, count) = d.map[name];
        got.push_str(&format!("{name} {count} {hash:016x}\n"));
    }
    assert_eq!(got, expected);
}
