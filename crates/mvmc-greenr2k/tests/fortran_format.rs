//! The Fortran output formats must match gfortran exactly (text, not numerics).
//!
//! `tests/fixtures/greenr2k/format_probe.txt` was produced by gfortran 13.3.0 from
//! `c_toolbox/greenr2k/format_probe.f90` for the bit patterns in
//! `c_toolbox/greenr2k/probe_values.txt` (see the fixture PROVENANCE).

use mvmc_greenr2k::fortran_fmt::{fmt_e, fmt_f, fmt_i, fmt_i_min, list_real, RealKind};

const PROBE: &str = include_str!("../../../tests/fixtures/greenr2k/format_probe.txt");

#[test]
fn edit_descriptors_and_list_directed_reals_match_gfortran() {
    let mut checked = 0;
    for line in PROBE.lines() {
        let (head, rest) = line.split_once(" [").expect("record");
        let (hex, tag) = head.split_once(' ').expect("tag");
        let expected = rest.strip_suffix(']').expect("closing bracket").trim_end();
        let x = f64::from_bits(u64::from_str_radix(hex, 16).expect("hex bits"));
        let actual = match tag.trim() {
            "r8" => format!(" {}", list_real(RealKind::Double, x)),
            "r4" => format!(" {}", list_real(RealKind::Single, x)),
            "e15" => fmt_e(15, 5, x),
            "f15" => fmt_f(15, 10, x),
            "f10" => fmt_f(10, 5, x),
            "f72" => fmt_f(7, 2, x),
            other => panic!("unknown tag {other}"),
        };
        assert_eq!(actual.trim_end(), expected, "{hex} {tag}");
        checked += 1;
    }
    assert!(
        checked > 3000,
        "probe fixture unexpectedly small: {checked}"
    );
}

#[test]
fn integer_edit_descriptors() {
    assert_eq!(fmt_i(3, 7), "  7");
    assert_eq!(fmt_i(3, 1000), "***");
    assert_eq!(fmt_i(4, -12), " -12");
    assert_eq!(fmt_i(0, 12345), "12345");
    assert_eq!(fmt_i_min(3, 3, 7), "007");
    assert_eq!(fmt_i_min(3, 3, 1000), "***");
    assert_eq!(fmt_i(11, 6), "          6");
}
