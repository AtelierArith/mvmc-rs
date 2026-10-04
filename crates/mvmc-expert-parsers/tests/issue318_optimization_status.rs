//! Independent A145 reporting literals; no reference runtime or RNG.
use mvmc_expert_parsers::{
    write_optimization_status, DoublonHolon2SiteIndex, DoublonHolon4SiteIndex, ExpertModeData,
    GutzwillerTerm, JastrowTerm, OrbitalTerm,
};
use num_complex::Complex64;
use std::io::{self, Write};

fn data() -> ExpertModeData {
    let mut data = ExpertModeData::new();
    data.modpara.n_orbital_idx = 0;
    data
}
fn orbital(idx: i64, site1: i64, site2: i64, sign: i64) -> OrbitalTerm {
    OrbitalTerm {
        idx,
        site1,
        site2,
        sign,
        is_complex: false,
    }
}
fn render(data: &ExpertModeData) -> String {
    let before = format!("{data:?}");
    let mut bytes = Vec::new();
    write_optimization_status(data, &mut bytes).unwrap();
    assert_eq!(format!("{data:?}"), before);
    String::from_utf8(bytes).unwrap()
}
const TITLE: &str = "Optimization status (real component; optimized iff flag == 1)\n";
const EMPTY_PREFIX: &str = "Gutzwiller parameters=0\nGutzwiller representative_rows=0 without_metadata=0\nJastrow parameters=0\nJastrow representative_rows=0 without_metadata=0\n";

#[test]
fn shared_indices_and_representatives_have_independent_literal_output() {
    let mut d = data();
    d.n_gutzwiller_idx = 2;
    d.n_jastrow_idx = 1;
    d.modpara.n_orbital_idx = 3;
    d.gutzwiller_terms = [2, 0]
        .map(|site| GutzwillerTerm {
            site,
            value: Complex64::new(0.0, 0.0),
            is_complex: false,
        })
        .to_vec();
    d.jastrow_terms = vec![JastrowTerm {
        site1: 1,
        site2: 2,
        value: Complex64::new(0.0, 0.0),
        is_complex: false,
    }];
    d.optimization_flags = vec![1, 99, 0, 99, 2, 99, 1, 99, -1, 99, 1, 99];
    d.orbital_terms = vec![
        orbital(2, 0, 1, 1),
        orbital(2, 1, 0, -1),
        orbital(0, 0, 0, 1),
    ];
    assert_eq!(
        render(&d),
        concat!(
        "Optimization status (real component; optimized iff flag == 1)\n",
        "Gutzwiller parameters=2\n  parameter=0 optimized\n  parameter=1 fixed\n",
        "Gutzwiller representative_rows=2 without_metadata=0\n",
        "  row=0 parameter=0 site=2 optimized\n  row=1 parameter=1 site=0 fixed\n",
        "Jastrow parameters=1\n  parameter=0 fixed\n",
        "Jastrow representative_rows=1 without_metadata=0\n  row=0 parameter=0 sites=1,2 fixed\n",
        "Slater parameters=3 optimized=2 fixed=1\n",
        "  parameter=0 optimized\n  parameter=1 fixed\n  parameter=2 optimized\n",
        "Slater mapping_rows=3\n",
        "  row=0 parameter=2 sites=0,1 sign=1 optimized\n",
        "  row=1 parameter=2 sites=1,0 sign=-1 optimized\n",
        "  row=2 parameter=0 sites=0,0 sign=1 optimized\n"
    )
    );
}

#[test]
fn empty_missing_flags_and_imaginary_only_flags_are_fixed() {
    let d = data();
    assert_eq!(
        render(&d),
        format!(
            "{TITLE}{EMPTY_PREFIX}Slater parameters=0 optimized=0 fixed=0\nSlater mapping_rows=0\n"
        )
    );
    let mut d = data();
    d.modpara.n_orbital_idx = 2;
    d.optimization_flags = vec![0, 1];
    assert_eq!(render(&d), format!("{TITLE}{EMPTY_PREFIX}Slater parameters=2 optimized=0 fixed=2\n  parameter=0 fixed\n  parameter=1 fixed\nSlater mapping_rows=0\n"));
}

#[test]
fn reserved_metadata_and_fallback_widths_are_explicit() {
    let mut d = data();
    d.n_gutzwiller_idx = 3;
    d.n_jastrow_idx = 2;
    d.gutzwiller_terms = vec![GutzwillerTerm {
        site: 7,
        value: Complex64::new(0.0, 0.0),
        is_complex: false,
    }];
    d.jastrow_terms = vec![JastrowTerm {
        site1: 4,
        site2: 5,
        value: Complex64::new(0.0, 0.0),
        is_complex: false,
    }];
    assert_eq!(render(&d), format!("{TITLE}Gutzwiller parameters=3\n  parameter=0 fixed\n  parameter=1 fixed\n  parameter=2 fixed\nGutzwiller representative_rows=1 without_metadata=2\n  row=0 parameter=0 site=7 fixed\nJastrow parameters=2\n  parameter=0 fixed\n  parameter=1 fixed\nJastrow representative_rows=1 without_metadata=1\n  row=0 parameter=0 sites=4,5 fixed\nSlater parameters=0 optimized=0 fixed=0\nSlater mapping_rows=0\n"));
    d.n_gutzwiller_idx = -1;
    d.n_jastrow_idx = 0;
    let out = render(&d);
    assert!(out.contains("Gutzwiller parameters=1\n"));
    assert!(out.contains("Jastrow parameters=1\n"));
}

#[test]
fn declared_projection_and_nine_rbm_prefix_use_literal_components136_and140() {
    let mut d = data();
    d.n_gutzwiller_idx = 3;
    d.n_jastrow_idx = 4;
    d.doublon_holon_2site_indices = vec![DoublonHolon2SiteIndex { neighbors: vec![] }];
    d.doublon_holon_4site_indices = vec![DoublonHolon4SiteIndex { neighbors: vec![] }];
    d.rbm_section_widths = [1, 2, 3, 4, 5, 6, 7, 8, 9];
    d.modpara.n_orbital_idx = 3;
    d.optimization_flags = vec![0; 142];
    d.optimization_flags[136] = 1;
    d.optimization_flags[140] = 1;
    d.orbital_terms = vec![orbital(2, 0, 1, 1)];
    assert!(render(&d).ends_with("Slater parameters=3 optimized=2 fixed=1\n  parameter=0 optimized\n  parameter=1 fixed\n  parameter=2 optimized\nSlater mapping_rows=1\n  row=0 parameter=2 sites=0,1 sign=1 optimized\n"));
}

#[test]
fn dense_and_mapping_truncation_are_independent() {
    let mut d = data();
    d.modpara.n_orbital_idx = 12;
    d.orbital_terms = vec![orbital(11, 0, 1, 1), orbital(11, 1, 0, -1)];
    assert!(render(&d).ends_with(concat!(
        "Slater parameters=12 optimized=0 fixed=12\n",
        "  parameter=0 fixed\n  parameter=1 fixed\n  parameter=2 fixed\n  parameter=3 fixed\n  parameter=4 fixed\n  ...\n",
        "  parameter=7 fixed\n  parameter=8 fixed\n  parameter=9 fixed\n  parameter=10 fixed\n  parameter=11 fixed\n",
        "Slater mapping_rows=2\n  row=0 parameter=11 sites=0,1 sign=1 fixed\n  row=1 parameter=11 sites=1,0 sign=-1 fixed\n"
    )));
    d.modpara.n_orbital_idx = 1;
    d.orbital_terms = vec![orbital(0, 0, 1, 1); 12];
    assert!(render(&d).ends_with(concat!(
        "Slater parameters=1 optimized=0 fixed=1\n  parameter=0 fixed\nSlater mapping_rows=12\n",
        "  row=0 parameter=0 sites=0,1 sign=1 fixed\n  row=1 parameter=0 sites=0,1 sign=1 fixed\n",
        "  row=2 parameter=0 sites=0,1 sign=1 fixed\n  row=3 parameter=0 sites=0,1 sign=1 fixed\n  row=4 parameter=0 sites=0,1 sign=1 fixed\n  ...\n",
        "  row=7 parameter=0 sites=0,1 sign=1 fixed\n  row=8 parameter=0 sites=0,1 sign=1 fixed\n",
        "  row=9 parameter=0 sites=0,1 sign=1 fixed\n  row=10 parameter=0 sites=0,1 sign=1 fixed\n  row=11 parameter=0 sites=0,1 sign=1 fixed\n"
    )));
}

fn rejects_without_write(d: &ExpertModeData) {
    let before = format!("{d:?}");
    let mut out = Vec::new();
    assert_eq!(
        write_optimization_status(d, &mut out).unwrap_err().kind(),
        io::ErrorKind::InvalidInput
    );
    assert!(out.is_empty());
    assert_eq!(format!("{d:?}"), before);
}

#[test]
fn invalid_layout_and_overflow_reject_before_write_without_mutation() {
    for idx in [-1, 1] {
        let mut d = data();
        d.modpara.n_orbital_idx = 1;
        d.orbital_terms = vec![orbital(idx, 0, 1, 1)];
        rejects_without_write(&d);
    }
    let mut d = data();
    d.modpara.n_orbital_idx = -1;
    rejects_without_write(&d);
    let mut d = data();
    d.n_gutzwiller_idx = 1;
    d.gutzwiller_terms = vec![
        GutzwillerTerm {
            site: 0,
            value: Complex64::new(0.0, 0.0),
            is_complex: false
        };
        2
    ];
    rejects_without_write(&d);
    let mut d = data();
    d.n_jastrow_idx = 1;
    d.jastrow_terms = vec![
        JastrowTerm {
            site1: 0,
            site2: 1,
            value: Complex64::new(0.0, 0.0),
            is_complex: false
        };
        2
    ];
    rejects_without_write(&d);
    let mut d = data();
    d.n_gutzwiller_idx = i64::MAX;
    d.n_jastrow_idx = i64::MAX;
    rejects_without_write(&d);
    let mut d = data();
    d.rbm_section_widths = [usize::MAX, 1, 0, 0, 0, 0, 0, 0, 0];
    rejects_without_write(&d);
    let mut d = data();
    d.orbital_terms = vec![orbital(0, 0, 1, 1)];
    rejects_without_write(&d);
}

struct LimitedWriter {
    bytes: Vec<u8>,
    remaining: usize,
}
impl Write for LimitedWriter {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if self.remaining == 0 {
            return Err(io::Error::new(
                io::ErrorKind::BrokenPipe,
                "literal test failure",
            ));
        }
        let n = bytes.len().min(self.remaining);
        self.bytes.extend_from_slice(&bytes[..n]);
        self.remaining -= n;
        Ok(n)
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}
#[test]
fn writer_failure_propagates_with_partial_output_but_no_data_mutation() {
    let d = data();
    let before = format!("{d:?}");
    let mut writer = LimitedWriter {
        bytes: vec![],
        remaining: 12,
    };
    let err = write_optimization_status(&d, &mut writer).unwrap_err();
    assert_eq!(err.kind(), io::ErrorKind::BrokenPipe);
    assert_eq!(err.to_string(), "literal test failure");
    assert_eq!(writer.bytes, b"Optimization");
    assert_eq!(format!("{d:?}"), before);
}
struct InterruptOnce {
    interrupted: bool,
    bytes: Vec<u8>,
}
impl Write for InterruptOnce {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if !self.interrupted {
            self.interrupted = true;
            return Err(io::Error::from(io::ErrorKind::Interrupted));
        }
        self.bytes.extend_from_slice(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}
#[test]
fn interrupted_writer_is_retried_by_standard_write_all() {
    let d = data();
    let mut writer = InterruptOnce {
        interrupted: false,
        bytes: vec![],
    };
    write_optimization_status(&d, &mut writer).unwrap();
    assert!(writer.interrupted);
    assert_eq!(
        String::from_utf8(writer.bytes).unwrap(),
        format!(
            "{TITLE}{EMPTY_PREFIX}Slater parameters=0 optimized=0 fixed=0\nSlater mapping_rows=0\n"
        )
    );
}
