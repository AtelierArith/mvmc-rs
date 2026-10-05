//! Pure-Rust port of the mVMC post-processing tool `tool/greenr2k.F90`.
//!
//! The tool reads a NameList file and a geometry file, the one- and two-body
//! correlation functions written by mVMC (`zvo_cisajs_???.dat`,
//! `zvo_cisajscktalt_???.dat`) or HPhi, Fourier transforms them onto a k path
//! and a k grid and writes `output/zvo_corr*.dat`, `kpath.gp` and FermiSurfer
//! `.frmsf` files. The control flow, operation order, console messages and the
//! Fortran output formats follow `greenr2k.F90` (see `docs/manual`).
//!
//! All file names are resolved relative to the working directory passed to
//! [`run`], like the Fortran program resolves them relative to its cwd.

// The numerical loops deliberately mirror the explicit index loops of the Fortran
// source (AGENTS.md: keep numerical kernels explicit; follow the C/Fortran operation
// order), so iterator rewrites are not wanted here.
#![allow(clippy::needless_range_loop)]

pub mod fortran_fmt;
pub mod listread;

use std::fmt;
use std::fs;
use std::io::Write;
use std::path::Path;

use fortran_fmt::{fmt_e, fmt_f, fmt_i, fmt_i_min, ListRecord, RealKind};
use listread::{parse_int, parse_real, Records};

/// Errors; the binary exits with status 2 (the gfortran runtime-error status).
#[derive(Debug)]
pub enum Error {
    Io(String),
    Parse(String),
    /// Fortran `STOP "Missing indices for the Green function."` after the list of
    /// missing indices was printed to stdout.
    MissingIndices,
    Other(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Io(s) | Error::Parse(s) | Error::Other(s) => write!(f, "{s}"),
            Error::MissingIndices => write!(f, "Missing indices for the Green function."),
        }
    }
}

impl std::error::Error for Error {}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e.to_string())
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct C {
    re: f64,
    im: f64,
}

impl C {
    const ZERO: C = C { re: 0.0, im: 0.0 };
    fn new(re: f64, im: f64) -> C {
        C { re, im }
    }
    fn add(self, o: C) -> C {
        C::new(self.re + o.re, self.im + o.im)
    }
    fn sub(self, o: C) -> C {
        C::new(self.re - o.re, self.im - o.im)
    }
    fn mul(self, o: C) -> C {
        C::new(
            self.re * o.re - self.im * o.im,
            self.re * o.im + self.im * o.re,
        )
    }
    fn scale(self, r: f64) -> C {
        C::new(self.re * r, self.im * r)
    }
    fn div_real(self, r: f64) -> C {
        C::new(self.re / r, self.im / r)
    }
    fn neg(self) -> C {
        C::new(-self.re, -self.im)
    }
}

/// Console output (`WRITE(*,...)`).
struct Console<'a> {
    out: &'a mut dyn Write,
}

impl Console<'_> {
    /// `WRITE(*,*)` with an empty list.
    fn blank(&mut self) -> Result<(), Error> {
        writeln!(self.out)?;
        Ok(())
    }
    fn line(&mut self, text: &str) -> Result<(), Error> {
        writeln!(self.out, "{text}")?;
        Ok(())
    }
    fn list(&mut self, build: impl FnOnce(&mut ListRecord)) -> Result<(), Error> {
        let mut record = ListRecord::new();
        build(&mut record);
        writeln!(self.out, "{}", record.finish())?;
        Ok(())
    }
}

fn lower(key: &str) -> String {
    key.chars()
        .map(|c| {
            if c.is_ascii_uppercase() {
                c.to_ascii_lowercase()
            } else {
                c
            }
        })
        .collect()
}

#[derive(Default)]
struct Params {
    file_one: String,
    file_two: String,
    filehead: String,
    nsite: i64,
    numave: i64,
    interval: i64,
    nwfc: i64,
    calctype: i64,
    filetail: Vec<String>,
}

fn resolve(cwd: &Path, name: &str) -> std::path::PathBuf {
    cwd.join(name)
}

fn read_filename(cwd: &Path, namelist_arg: &str, con: &mut Console) -> Result<Params, Error> {
    let mut p = Params::default();
    let mut modpara = String::new();
    let mut calcmod = String::new();
    let mut lanczos_max: i64 = 0;
    let mut idx_start: i64 = 0;
    con.blank()?;
    con.list(|r| {
        r.str("#####  Read HPhi/mVMC Input Files  #####");
    })?;
    con.blank()?;
    //
    let mut rec = Records::open(&resolve(cwd, namelist_arg))?;
    while let Some(key) = rec.peek_key() {
        match lower(&key).as_str() {
            "onebodyg" => p.file_one = rec.read(2)?.remove(1),
            "twobodyg" => p.file_two = rec.read(2)?.remove(1),
            "modpara" => modpara = rec.read(2)?.remove(1),
            "calcmod" => calcmod = rec.read(2)?.remove(1),
            _ => {
                rec.read(1)?;
            }
        }
    }
    con.list(|r| {
        r.str("  Read from ").str(namelist_arg);
    })?;
    con.list(|r| {
        r.str("    OneBodyG file : ").str(&p.file_one);
    })?;
    con.list(|r| {
        r.str("    TwoBodyG file : ").str(&p.file_two);
    })?;
    con.list(|r| {
        r.str("    ModPara file : ").str(&modpara);
    })?;
    con.list(|r| {
        r.str("    CalcMod file : ").str(&calcmod);
    })?;
    //
    let mut rec = Records::open(&resolve(cwd, &modpara))?;
    while let Some(key) = rec.peek_key() {
        let name = rec.name().to_string();
        match lower(&key).as_str() {
            "nsite" => p.nsite = parse_int(&rec.read(2)?[1], &name)?,
            "cdatafilehead" => p.filehead = rec.read(2)?.remove(1),
            "numave" => p.numave = parse_int(&rec.read(2)?[1], &name)?,
            "lanczos_max" => lanczos_max = parse_int(&rec.read(2)?[1], &name)?,
            "expecinterval" => p.interval = parse_int(&rec.read(2)?[1], &name)?,
            "exct" => p.nwfc = parse_int(&rec.read(2)?[1], &name)?,
            "ndataidxstart" => idx_start = parse_int(&rec.read(2)?[1], &name)?,
            "ndataqtysmp" => p.numave = parse_int(&rec.read(2)?[1], &name)?,
            _ => {
                rec.read(1)?;
            }
        }
    }
    con.list(|r| {
        r.str("  Read from ").str(&modpara);
    })?;
    con.list(|r| {
        r.str("    FileHead : ").str(&p.filehead);
    })?;
    con.list(|r| {
        r.str("    Number of site : ").int(p.nsite);
    })?;
    p.filehead = format!("output/{}", p.filehead);
    //
    if calcmod.is_empty() {
        p.calctype = 4;
    } else {
        let mut rec = Records::open(&resolve(cwd, &calcmod))?;
        while let Some(key) = rec.peek_key() {
            let name = rec.name().to_string();
            if lower(&key) == "calctype" {
                p.calctype = parse_int(&rec.read(2)?[1], &name)?;
            } else {
                rec.read(1)?;
            }
        }
        con.list(|r| {
            r.str("  Read from ").str(&calcmod);
        })?;
        con.list(|r| {
            r.str("    CalcType : ").int(p.calctype);
        })?;
    }
    //
    match p.calctype {
        0 => {
            p.nwfc = 1;
            p.filetail = vec![".dat".to_string()];
            con.list(|r| {
                r.str("    Method : Lanczos");
            })?;
        }
        1 => {
            if p.interval == 0 {
                return Err(Error::Other("ExpecInterval is zero for TPQ".to_string()));
            }
            p.nwfc = p.numave * (1 + (lanczos_max - 1) / p.interval);
            for istep in 0..lanczos_max {
                for irun in 0..p.numave {
                    if istep % p.interval == 0 {
                        p.filetail.push(format!("_set{irun}step{istep}.dat"));
                    }
                }
            }
            con.list(|r| {
                r.str("    Method : TPQ");
            })?;
            con.list(|r| {
                r.str("    Number of Run : ").int(p.numave);
            })?;
            con.list(|r| {
                r.str("    Maximum Iteration : ").int(lanczos_max);
            })?;
            con.list(|r| {
                r.str("    Expectation Interval : ").int(p.interval);
            })?;
        }
        2 => {
            // READ(fi, '("  MAX DIMENSION idim_max=1", i16)') nwfc: the constant string
            // in an input format is a gfortran runtime error; other compilers
            // skip as many characters as the string holds. Follow the latter.
            let path = resolve(cwd, "output/CHECK_Memory.dat");
            let text = fs::read_to_string(&path)
                .map_err(|e| Error::Io(format!("cannot open '{}': {e}", path.display())))?;
            let line = text.lines().next().unwrap_or("");
            let field: String = line.chars().skip(26).take(16).collect();
            let field = field.replace(' ', "");
            p.nwfc = if field.is_empty() {
                0
            } else {
                parse_int(&field, "output/CHECK_Memory.dat")?
            };
            for iwfc in 0..p.nwfc {
                p.filetail.push(format!("_eigen{iwfc}.dat"));
            }
            con.list(|r| {
                r.str("    Method : Full Diagonalization");
            })?;
        }
        3 => {
            for iwfc in 0..p.nwfc {
                p.filetail.push(format!("_eigen{iwfc}.dat"));
            }
            con.list(|r| {
                r.str("    Method : LOBCG");
            })?;
        }
        _ => {
            p.nwfc = p.numave;
            for iwfc in 1..=p.nwfc {
                p.filetail
                    .push(format!("_{}.dat", fmt_i_min(3, 3, iwfc - 1 + idx_start)));
            }
            con.list(|r| {
                r.str("    Method : mVMC");
            })?;
            con.list(|r| {
                r.str("    Start Index : ").int(idx_start);
            })?;
        }
    }
    con.list(|r| {
        r.str("    Number of States : ").int(p.nwfc);
    })?;
    Ok(p)
}

struct Geometry {
    recipr: [[f64; 3]; 3],
    nr: usize,
    norb: usize,
    rindx: Vec<usize>,
    orb: Vec<usize>,
    nreq: Vec<usize>,
    irv: Vec<Vec<[i64; 3]>>,
    phase: Vec<Vec<f64>>,
    nnode: usize,
    nk_line: usize,
    knode: Vec<[f64; 3]>,
    kname: Vec<String>,
    nkg: [i64; 3],
}

fn invert3(a: [[f64; 3]; 3]) -> Result<[[f64; 3]; 3], Error> {
    // LU factorisation with partial pivoting (dgetrf) and inverse (dgetri).
    let mut lu = a;
    let mut piv = [0usize; 3];
    for k in 0..3 {
        let mut p = k;
        for i in k + 1..3 {
            if lu[i][k].abs() > lu[p][k].abs() {
                p = i;
            }
        }
        piv[k] = p;
        if lu[p][k] == 0.0 {
            return Err(Error::Other(
                "singular lattice vectors in geometry file".to_string(),
            ));
        }
        lu.swap(k, p);
        for i in k + 1..3 {
            lu[i][k] /= lu[k][k];
            for j in k + 1..3 {
                lu[i][j] -= lu[i][k] * lu[k][j];
            }
        }
    }
    let mut inv = [[0.0f64; 3]; 3];
    for col in 0..3 {
        let mut b = [0.0f64; 3];
        b[col] = 1.0;
        // Apply the row interchanges in order.
        for k in 0..3 {
            b.swap(k, piv[k]);
        }
        for i in 0..3 {
            for j in 0..i {
                b[i] -= lu[i][j] * b[j];
            }
        }
        for i in (0..3).rev() {
            for j in i + 1..3 {
                b[i] -= lu[i][j] * b[j];
            }
            b[i] /= lu[i][i];
        }
        for i in 0..3 {
            inv[i][col] = b[i];
        }
    }
    Ok(inv)
}

fn read_geometry(
    cwd: &Path,
    geometry_arg: &str,
    nsite: i64,
    con: &mut Console,
) -> Result<Geometry, Error> {
    con.blank()?;
    con.list(|r| {
        r.str("#####  Read Geometry Input File  #####");
    })?;
    con.blank()?;
    con.list(|r| {
        r.str("  Read from ").str(geometry_arg);
    })?;
    let mut rec = Records::open(&resolve(cwd, geometry_arg))?;
    let name = rec.name().to_string();
    let reals = |rec: &mut Records, n: usize| -> Result<Vec<f64>, Error> {
        rec.read(n)?.iter().map(|t| parse_real(t, &name)).collect()
    };
    let ints = |rec: &mut Records, n: usize| -> Result<Vec<i64>, Error> {
        rec.read(n)?.iter().map(|t| parse_int(t, &name)).collect()
    };
    // direct[c][v]: component c of lattice vector v (Fortran direct(1:3, v)).
    let mut direct = [[0.0f64; 3]; 3];
    for v in 0..3 {
        let row = reals(&mut rec, 3)?;
        for c in 0..3 {
            direct[c][v] = row[c];
        }
    }
    con.list(|r| {
        r.str("    Direct LATTICE VECTOR :");
    })?;
    for v in 0..3 {
        con.line(&format!(
            "    {}{}{}",
            fmt_f(15, 10, direct[0][v]),
            fmt_f(15, 10, direct[1][v]),
            fmt_f(15, 10, direct[2][v])
        ))?;
    }
    let phase_deg = reals(&mut rec, 3)?;
    con.list(|r| {
        r.str("    Boundary phase[degree] : ");
    })?;
    con.line(&format!(
        "    {}{}{}",
        fmt_f(15, 10, phase_deg[0]),
        fmt_f(15, 10, phase_deg[1]),
        fmt_f(15, 10, phase_deg[2])
    ))?;
    let mut phase0 = [0.0f64; 3];
    for c in 0..3 {
        phase0[c] = phase_deg[c] * std::f64::consts::PI / 180.0;
    }
    // box[c][v]: component c of supercell vector v (Fortran box(1:3, v)).
    let mut boxm = [[0i64; 3]; 3];
    for v in 0..3 {
        let row = ints(&mut rec, 3)?;
        for c in 0..3 {
            boxm[c][v] = row[c];
        }
    }
    con.list(|r| {
        r.str("    Supercell Index :");
    })?;
    for v in 0..3 {
        con.line(&format!(
            "{}{}{}",
            fmt_i(8, boxm[0][v]),
            fmt_i(8, boxm[1][v]),
            fmt_i(8, boxm[2][v])
        ))?;
    }
    // R-vector and orbital index.
    let nsite_u =
        usize::try_from(nsite).map_err(|_| Error::Other("negative number of sites".to_string()))?;
    let mut irv1: Vec<[i64; 3]> = Vec::new();
    let mut orb = vec![0usize; nsite_u];
    let mut rindx = vec![0usize; nsite_u];
    for isite in 0..nsite_u {
        let v = ints(&mut rec, 4)?;
        let irv0 = [v[0], v[1], v[2]];
        match irv1.iter().position(|r| *r == irv0) {
            Some(ir) => rindx[isite] = ir,
            None => {
                irv1.push(irv0);
                rindx[isite] = irv1.len() - 1;
            }
        }
        orb[isite] = usize::try_from(v[3] + 1)
            .map_err(|_| Error::Other("negative orbital index in geometry file".to_string()))?;
    }
    let nr = irv1.len();
    let norb = orb.iter().copied().max().unwrap_or(0);
    con.list(|r| {
        r.str("    Number of orbitals :").int(norb as i64);
    })?;
    // k-point.
    let nn = ints(&mut rec, 2)?;
    let (nnode, nk_line) = (nn[0], nn[1]);
    if nnode < 1 || nk_line < 0 {
        return Err(Error::Other(
            "invalid k-node counts in geometry file".to_string(),
        ));
    }
    let (nnode, nk_line) = (nnode as usize, nk_line as usize);
    con.list(|r| {
        r.str("    Number of k-node, and k-points along lines :")
            .int(nnode as i64)
            .int(nk_line as i64);
    })?;
    con.list(|r| {
        r.str("      k-node :");
    })?;
    let mut knode = Vec::new();
    let mut kname = Vec::new();
    for _ in 0..nnode {
        let items = rec.read(4)?;
        let k = [
            parse_real(&items[1], &name)?,
            parse_real(&items[2], &name)?,
            parse_real(&items[3], &name)?,
        ];
        con.line(&format!(
            "      {}{}{}{}",
            items[0].trim(),
            fmt_f(10, 5, k[0]),
            fmt_f(10, 5, k[1]),
            fmt_f(10, 5, k[2])
        ))?;
        kname.push(items[0].clone());
        knode.push(k);
    }
    let g = ints(&mut rec, 3)?;
    let nkg = [g[0], g[1], g[2]];
    con.line(&format!(
        "k-grid for momentum distribution :{}{}{}",
        fmt_i(3, nkg[0]),
        fmt_i(3, nkg[1]),
        fmt_i(3, nkg[2])
    ))?;
    // Reciprocal lattice vector: inverse of transpose(direct).
    let mut transposed = [[0.0f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            transposed[i][j] = direct[j][i];
        }
    }
    let recipr = invert3(transposed)?;
    con.list(|r| {
        r.str("    Reciplocal lattice vector :");
    })?;
    for j in 0..3 {
        con.line(&format!(
            "    {}{}{}",
            fmt_f(15, 10, recipr[0][j]),
            fmt_f(15, 10, recipr[1][j]),
            fmt_f(15, 10, recipr[2][j])
        ))?;
    }
    // Move each R-vector to the nearest images (periodic boundary conditions).
    con.list(|r| {
        r.str("    Number of R-vector :").int(nr as i64);
    })?;
    let mut nreq = vec![0usize; nr];
    let mut irv: Vec<Vec<[i64; 3]>> = vec![Vec::new(); nr];
    let mut phase: Vec<Vec<f64>> = vec![Vec::new(); nr];
    for ir in 0..nr {
        let mut lenrv0 = 1.0e10f64;
        for i1 in -2i64..=2 {
            for i2 in -2i64..=2 {
                for i3 in -2i64..=2 {
                    let ii = [i1, i2, i3];
                    let mut irv0 = [0i64; 3];
                    for c in 0..3 {
                        irv0[c] = irv1[ir][c]
                            + (boxm[c][0] * ii[0] + boxm[c][1] * ii[1] + boxm[c][2] * ii[2]);
                    }
                    let mut rrv = [0.0f64; 3];
                    for c in 0..3 {
                        rrv[c] = direct[c][0] * irv0[0] as f64
                            + direct[c][1] * irv0[1] as f64
                            + direct[c][2] * irv0[2] as f64;
                    }
                    let lenrv = (rrv[0] * rrv[0] + rrv[1] * rrv[1] + rrv[2] * rrv[2]).sqrt();
                    if lenrv < lenrv0 - 1.0e-6 {
                        lenrv0 = lenrv;
                        nreq[ir] = 1;
                        irv[ir].clear();
                        phase[ir].clear();
                    } else if (lenrv - lenrv0).abs() < 1.0e-6 {
                        nreq[ir] += 1;
                    } else {
                        continue;
                    }
                    irv[ir].push(irv0);
                    phase[ir].push(
                        i1 as f64 * phase0[0] + i2 as f64 * phase0[1] + i3 as f64 * phase0[2],
                    );
                }
            }
        }
        let mut text = String::new();
        for ireq in 0..nreq[ir] {
            let v = irv[ir][ireq];
            text.push_str(&format!(
                "{}{}{}{}, ",
                fmt_i(5, v[0]),
                fmt_i(5, v[1]),
                fmt_i(5, v[2]),
                fmt_f(7, 2, phase[ir][ireq])
            ));
        }
        con.line(&text)?;
    }
    Ok(Geometry {
        recipr,
        nr,
        norb,
        rindx,
        orb,
        nreq,
        irv,
        phase,
        nnode,
        nk_line,
        knode,
        kname,
        nkg,
    })
}

fn set_kpoints(g: &Geometry, con: &mut Console) -> Result<(Vec<[f64; 3]>, usize), Error> {
    let nkgp = g.nkg[0] * g.nkg[1] * g.nkg[2];
    let nk = (g.nk_line * (g.nnode - 1) + 1) as i64 + nkgp;
    con.list(|r| {
        r.str("     Number of k : ").int(nk);
    })?;
    let mut kvec = Vec::new();
    kvec.push(g.knode[0]);
    for inode in 0..g.nnode - 1 {
        for ik in 1..=g.nk_line {
            let xx = ik as f64 / g.nk_line as f64;
            let mut k = [0.0f64; 3];
            for c in 0..3 {
                k[c] = (1.0 - xx) * g.knode[inode][c] + xx * g.knode[inode + 1][c];
            }
            kvec.push(k);
        }
    }
    let ikk = kvec.len();
    for i1 in 1..=g.nkg[0] {
        for i2 in 1..=g.nkg[1] {
            for i3 in 1..=g.nkg[2] {
                kvec.push([
                    (i1 - 1) as f64 / g.nkg[0] as f64,
                    (i2 - 1) as f64 / g.nkg[1] as f64,
                    (i3 - 1) as f64 / g.nkg[2] as f64,
                ]);
            }
        }
    }
    Ok((kvec, ikk))
}

/// `indx(ir, kind, orbL, orbR)` flattened; kind in 0..8. Values are 1-based
/// positions in the correlation files, 0 meaning "not present".
struct Indx {
    nr: usize,
    norb: usize,
    data: Vec<usize>,
}

impl Indx {
    fn at(&self, ir: usize, kind: usize, l: usize, r: usize) -> usize {
        self.data[ir + self.nr * (kind + 8 * (l + self.norb * r))]
    }
    fn set(&mut self, ir: usize, kind: usize, l: usize, r: usize, v: usize) {
        self.data[ir + self.nr * (kind + 8 * (l + self.norb * r))] = v;
    }
    fn count(&self, kinds: std::ops::Range<usize>) -> usize {
        let mut n = 0;
        for r in 0..self.norb {
            for l in 0..self.norb {
                for k in kinds.clone() {
                    for ir in 0..self.nr {
                        if self.at(ir, k, l, r) != 0 {
                            n += 1;
                        }
                    }
                }
            }
        }
        n
    }
}

fn read_corrindx(
    cwd: &Path,
    p: &Params,
    g: &Geometry,
    con: &mut Console,
) -> Result<(Indx, usize, usize), Error> {
    con.blank()?;
    con.list(|r| {
        r.str("#####  Read Correlation Index File  #####");
    })?;
    con.blank()?;
    let nr = g.nr;
    let mut indx = Indx {
        nr,
        norb: g.norb,
        data: vec![0; nr * 8 * g.norb * g.norb],
    };
    let site = |v: i64, what: &str| -> Result<usize, Error> {
        let i = v + 1;
        if i >= 1 && (i as usize) <= g.rindx.len() {
            Ok(i as usize - 1)
        } else {
            Err(Error::Other(format!(
                "site index {v} out of range in {what}"
            )))
        }
    };
    let not_r0 = |s: usize| g.irv[g.rindx[s]][0] != [0, 0, 0];
    //
    let mut rec = Records::open(&resolve(cwd, &p.file_one))?;
    con.list(|r| {
        r.str("  Read from ").str(&p.file_one);
    })?;
    let name = rec.name().to_string();
    rec.read(1)?;
    let ncor1 = parse_int(&rec.read(2)?[1], &name)?;
    for _ in 0..3 {
        rec.read(1)?;
    }
    con.list(|r| {
        r.str("    Number of Correlation Function (One body) : ")
            .int(ncor1);
    })?;
    for icor in 1..=ncor1 {
        let mut t = [0i64; 4];
        for (i, tok) in rec.read(4)?.iter().enumerate() {
            t[i] = parse_int(tok, &name)?;
        }
        let s1 = site(t[0], &name)?;
        if not_r0(s1) {
            con.line(&format!(
                "       REMARK : The left operator at # {icor} is not at R=0."
            ))?;
            continue;
        }
        let s3 = site(t[2], &name)?;
        if t[1] == 0 && t[3] == 0 {
            indx.set(g.rindx[s3], 0, g.orb[s1] - 1, g.orb[s3] - 1, icor as usize);
        } else if t[1] == 1 && t[3] == 1 {
            indx.set(g.rindx[s3], 1, g.orb[s1] - 1, g.orb[s3] - 1, icor as usize);
        }
    }
    con.list(|r| {
        r.str("    Number of Up-Up Index : ")
            .int(indx.count(0..1) as i64);
    })?;
    con.list(|r| {
        r.str("    Number of Down-Down Index : ")
            .int(indx.count(1..2) as i64);
    })?;
    //
    let mut rec = Records::open(&resolve(cwd, &p.file_two))?;
    con.list(|r| {
        r.str("  Read from ").str(&p.file_two);
    })?;
    let name = rec.name().to_string();
    rec.read(1)?;
    let ncor2 = parse_int(&rec.read(2)?[1], &name)?;
    for _ in 0..3 {
        rec.read(1)?;
    }
    con.list(|r| {
        r.str("    Number of Correlation Function (Two body) : ")
            .int(ncor2);
    })?;
    for icor in 1..=ncor2 {
        let mut t = [0i64; 8];
        for (i, tok) in rec.read(8)?.iter().enumerate() {
            t[i] = parse_int(tok, &name)?;
        }
        let ic = icor as usize;
        let s1 = site(t[0], &name)?;
        let s5 = site(t[4], &name)?;
        let remark = |con: &mut Console| -> Result<(), Error> {
            con.line(&format!(
                "       REMARK : The left operator at # {icor} is not at R=0."
            ))
        };
        if t[0] == t[2] && t[4] == t[6] {
            if not_r0(s1) {
                remark(con)?;
                continue;
            }
            let (l, r) = (g.orb[s1] - 1, g.orb[s5] - 1);
            let ir = g.rindx[s5];
            if t[1] == 0 && t[3] == 0 {
                if t[5] == 0 && t[7] == 0 {
                    indx.set(ir, 2, l, r, ic);
                } else if t[5] == 1 && t[7] == 1 {
                    indx.set(ir, 3, l, r, ic);
                }
            } else if t[1] == 1 && t[3] == 1 {
                if t[5] == 0 && t[7] == 0 {
                    indx.set(ir, 4, l, r, ic);
                } else if t[5] == 1 && t[7] == 1 {
                    indx.set(ir, 5, l, r, ic);
                }
            } else if p.calctype != 4 {
                if (t[1] == 0 && t[3] == 1) && (t[5] == 1 && t[7] == 0) {
                    indx.set(ir, 6, l, r, ic);
                } else if (t[1] == 1 && t[3] == 0) && (t[5] == 0 && t[7] == 1) {
                    indx.set(ir, 7, l, r, ic);
                }
            }
        }
        if p.calctype == 4 && (t[0] == t[6] && t[2] == t[4]) {
            if not_r0(s1) {
                remark(con)?;
                continue;
            }
            let (l, r) = (g.orb[s1] - 1, g.orb[s5] - 1);
            let ir = g.rindx[s5];
            if (t[1] == 0 && t[3] == 0) && (t[5] == 1 && t[7] == 1) {
                indx.set(ir, 6, l, r, ic);
            } else if (t[1] == 1 && t[3] == 1) && (t[5] == 0 && t[7] == 0) {
                indx.set(ir, 7, l, r, ic);
            }
        }
    }
    //
    if indx.count(2..8) != nr * 6 * g.norb * g.norb {
        con.list(|r| {
            r.str("ERROR! The following correlation function is missed:");
        })?;
        con.list(|r| {
            r.str("R,   kind,   orb1,   orb2");
        })?;
        let labels = [
            " Up-Up-Up-Up         ",
            " Up-Up-Down-Down     ",
            " Down-Down-Up-Up     ",
            " Down-Down-Down-Down ",
            " Up-Down-Down-Up     ",
            " Down-Up-Up-Down     ",
        ];
        for icor in 2..8 {
            for ir in 0..nr {
                for iorb in 0..g.norb {
                    for jorb in 0..g.norb {
                        if indx.at(ir, icor, iorb, jorb) == 0 {
                            con.line(&format!(
                                "{}{}{}  {}",
                                ir + 1,
                                labels[icor - 2],
                                iorb + 1,
                                jorb + 1
                            ))?;
                        }
                    }
                }
            }
        }
        return Err(Error::MissingIndices);
    }
    for (label, kind) in [
        ("    Number of UpUpUpUp         Index : ", 2),
        ("    Number of UpUpDownDown     Index : ", 3),
        ("    Number of DownDownUpUp     Index : ", 4),
        ("    Number of DownDownDownDown Index : ", 5),
        ("    Number of Plus-Minus       Index : ", 6),
        ("    Number of Minus-Plus       Index : ", 7),
    ] {
        con.list(|r| {
            r.str(label).int(indx.count(kind..kind + 1) as i64);
        })?;
    }
    Ok((indx, ncor1 as usize, ncor2 as usize))
}

struct Cor {
    nr: usize,
    norb: usize,
    data: Vec<C>,
}

impl Cor {
    fn idx(&self, ir: usize, k: usize, jo: usize, io: usize, w: usize) -> usize {
        ir + self.nr * (k + 6 * (jo + self.norb * (io + self.norb * w)))
    }
}

fn read_corrfile(
    cwd: &Path,
    p: &Params,
    g: &Geometry,
    indx: &Indx,
    ncor1: usize,
    ncor2: usize,
) -> Result<Cor, Error> {
    let nr = g.nr;
    let norb = g.norb;
    let nwfc = p.filetail.len();
    let mut cor = Cor {
        nr,
        norb,
        data: vec![C::ZERO; nr * 6 * norb * norb * nwfc],
    };
    let mut cor0 = vec![C::ZERO; ncor1.max(ncor2) + 1];
    let ir0 = (0..nr)
        .find(|&ir| g.irv[ir][0] == [0, 0, 0])
        .ok_or_else(|| Error::Other("no R=0 vector among the sites".to_string()))?;
    for iwfc in 0..nwfc {
        // One-body correlation function.
        let filename = format!("{}_cisajs{}", p.filehead, p.filetail[iwfc]);
        let mut rec = Records::open(&resolve(cwd, &filename))?;
        let name = rec.name().to_string();
        for icor in 1..=ncor1 {
            let t = rec.read(6)?;
            for tok in &t[..4] {
                parse_int(tok, &name)?;
            }
            cor0[icor] = C::new(parse_real(&t[4], &name)?, parse_real(&t[5], &name)?);
        }
        for iorb in 0..norb {
            for jorb in 0..norb {
                for ir in 0..nr {
                    for k in 0..2 {
                        let i = cor.idx(ir, k, jorb, iorb, iwfc);
                        cor.data[i] = cor0[indx.at(ir, k, jorb, iorb)];
                    }
                }
            }
        }
        // Two-body correlation function.
        let filename = format!("{}_cisajscktalt{}", p.filehead, p.filetail[iwfc]);
        let mut rec = Records::open(&resolve(cwd, &filename))?;
        let name = rec.name().to_string();
        for icor in 1..=ncor2 {
            let t = rec.read(10)?;
            for tok in &t[..8] {
                parse_int(tok, &name)?;
            }
            cor0[icor] = C::new(parse_real(&t[8], &name)?, parse_real(&t[9], &name)?);
        }
        for iorb in 0..norb {
            for jorb in 0..norb {
                for ir in 0..nr {
                    let c3 = cor0[indx.at(ir, 2, jorb, iorb)]
                        .add(cor0[indx.at(ir, 3, jorb, iorb)])
                        .add(cor0[indx.at(ir, 4, jorb, iorb)])
                        .add(cor0[indx.at(ir, 5, jorb, iorb)]);
                    let sum_i = cor.data[cor.idx(ir0, 0, iorb, iorb, iwfc)]
                        .add(cor.data[cor.idx(ir0, 1, iorb, iorb, iwfc)]);
                    let sum_j = cor.data[cor.idx(ir0, 0, jorb, jorb, iwfc)]
                        .add(cor.data[cor.idx(ir0, 1, jorb, jorb, iwfc)]);
                    let i3 = cor.idx(ir, 2, jorb, iorb, iwfc);
                    cor.data[i3] = c3.sub(sum_i.mul(sum_j));
                    let c4 = cor0[indx.at(ir, 2, jorb, iorb)]
                        .sub(cor0[indx.at(ir, 3, jorb, iorb)])
                        .sub(cor0[indx.at(ir, 4, jorb, iorb)])
                        .add(cor0[indx.at(ir, 5, jorb, iorb)]);
                    let i4 = cor.idx(ir, 3, jorb, iorb, iwfc);
                    cor.data[i4] = c4.scale(0.25);
                    for k in 0..2 {
                        let i = cor.idx(ir, 4 + k, jorb, iorb, iwfc);
                        cor.data[i] = cor0[indx.at(ir, 6 + k, jorb, iorb)];
                    }
                }
            }
        }
        // For mVMC:
        //   Ciu+ Cid Cjd+ Cju = delta_{ij} Ciu+ Ciu - Ciu+ Cju Cjd+ Cid
        //   Cid+ Ciu Cju+ Cjd = delta_{ij} Cid+ Cid - Cid+ Cjd Cju+ Ciu
        if p.calctype == 4 {
            for iorb in 0..norb {
                for jorb in 0..norb {
                    for ir in 0..nr {
                        for k in 4..6 {
                            let i = cor.idx(ir, k, jorb, iorb, iwfc);
                            cor.data[i] = cor.data[i].neg();
                        }
                    }
                }
            }
            for iorb in 0..norb {
                for k in 0..2 {
                    let src = cor.data[cor.idx(0, k, iorb, iorb, iwfc)];
                    let i = cor.idx(0, 4 + k, iorb, iorb, iwfc);
                    cor.data[i] = cor.data[i].add(src);
                }
            }
        }
        // S.S = Sz Sz + 0.5 * (S+S- + S-S+)
        for iorb in 0..norb {
            for jorb in 0..norb {
                for ir in 0..nr {
                    let c4 = cor.data[cor.idx(ir, 3, jorb, iorb, iwfc)];
                    let c5 = cor.data[cor.idx(ir, 4, jorb, iorb, iwfc)];
                    let c6 = cor.data[cor.idx(ir, 5, jorb, iorb, iwfc)];
                    let i = cor.idx(ir, 5, jorb, iorb, iwfc);
                    cor.data[i] = c4.add(c5.add(c6).scale(0.5));
                }
            }
        }
    }
    Ok(cor)
}

/// Fourier transformation: `cor_k(ik, 6, norb, norb, nwfc)` flattened with `ik` fastest.
fn fourier_cor(g: &Geometry, kvec: &[[f64; 3]], nwfc: usize, cor: &Cor) -> Vec<C> {
    let nk = kvec.len();
    let nr = g.nr;
    let tpi = 2.0 * std::f64::consts::PI;
    // fmat[ik + nk*ir] = (1/nreq) sum_ireq exp(-i k.R + i phase)
    let mut fmat = vec![C::ZERO; nk * nr];
    for ik in 0..nk {
        for ir in 0..nr {
            let mut f = C::ZERO;
            for ireq in 0..g.nreq[ir] {
                let r = g.irv[ir][ireq];
                let dot = kvec[ik][0] * r[0] as f64
                    + kvec[ik][1] * r[1] as f64
                    + kvec[ik][2] * r[2] as f64;
                let theta = -tpi * dot + g.phase[ir][ireq];
                f = f.add(C::new(theta.cos(), theta.sin()));
            }
            fmat[ik + nk * ir] = f.div_real(g.nreq[ir] as f64);
        }
    }
    let ncol = 6 * g.norb * g.norb * nwfc;
    let mut cor_k = vec![C::ZERO; nk * ncol];
    for col in 0..ncol {
        for ik in 0..nk {
            let mut acc = C::ZERO;
            for ir in 0..nr {
                acc = acc.add(fmat[ik + nk * ir].mul(cor.data[ir + nr * col]));
            }
            cor_k[ik + nk * col] = acc;
        }
    }
    // cor_k(:,3:6) /= nr; the one-body parts (1:2) are not divided.
    for col in 0..ncol {
        if col % 6 >= 2 {
            for ik in 0..nk {
                let i = ik + nk * col;
                cor_k[i] = cor_k[i].div_real(nr as f64);
            }
        }
    }
    cor_k
}

fn write_file(path: &Path, text: &str) -> Result<(), Error> {
    fs::write(path, text).map_err(|e| Error::Io(format!("cannot write '{}': {e}", path.display())))
}

fn output_cor(
    cwd: &Path,
    p: &Params,
    g: &Geometry,
    kvec: &[[f64; 3]],
    ikk: usize,
    cor_k: &[C],
    con: &mut Console,
) -> Result<(), Error> {
    let nk = kvec.len();
    let norb = g.norb;
    let nwfc = p.filetail.len();
    let nnode = g.nnode;
    // x-position for plotting the band.
    let mut xk = vec![0.0f64; ikk];
    let mut xk_label = vec![0.0f64; nnode];
    let mut pos = 0usize;
    for inode in 0..nnode - 1 {
        let mut dk = [0.0f64; 3];
        for c in 0..3 {
            dk[c] = g.knode[inode + 1][c] - g.knode[inode][c];
        }
        let mut dk_cart = [0.0f64; 3];
        for i in 0..3 {
            dk_cart[i] = g.recipr[i][0] * dk[0] + g.recipr[i][1] * dk[1] + g.recipr[i][2] * dk[2];
        }
        let klength = (dk_cart[0] * dk_cart[0] + dk_cart[1] * dk_cart[1] + dk_cart[2] * dk_cart[2])
            .sqrt()
            / g.nk_line as f64;
        xk_label[inode] = xk[pos];
        for _ in 0..g.nk_line {
            xk[pos + 1] = xk[pos] + klength;
            pos += 1;
        }
    }
    xk_label[nnode - 1] = xk[pos];
    let at = |ik: usize, k: usize, jo: usize, io: usize, w: usize| -> C {
        cor_k[ik + nk * (k + 6 * (jo + norb * (io + norb * w)))]
    };
    //
    con.blank()?;
    con.list(|r| {
        r.str("#####  Output Files  #####");
    })?;
    con.blank()?;
    con.list(|r| {
        r.str("  Correlation in k-space : ")
            .str(&format!("{}_corr", p.filehead))
            .str("*.dat");
    })?;
    if p.calctype == 1 || p.calctype == 4 {
        // TPQ/mVMC: average and standard error over `numave` samples.
        if p.numave <= 0 {
            return Err(Error::Other(
                "number of samples must be positive".to_string(),
            ));
        }
        let numave = p.numave as usize;
        for istep in 1..=nwfc / numave {
            let iwfc1 = numave * (istep - 1);
            let iwfc2 = numave * istep;
            let n = ikk * 6 * norb * norb;
            let mut ave = vec![C::ZERO; n];
            let mut err = vec![C::ZERO; n];
            let flat = |ik: usize, k: usize, jo: usize, io: usize| -> usize {
                ik + ikk * (k + 6 * (jo + norb * io))
            };
            for io in 0..norb {
                for jo in 0..norb {
                    for k in 0..6 {
                        for ik in 0..ikk {
                            let mut s = C::ZERO;
                            for w in iwfc1..iwfc2 {
                                s = s.add(at(ik, k, jo, io, w));
                            }
                            ave[flat(ik, k, jo, io)] = s.div_real(numave as f64);
                        }
                    }
                }
            }
            for w in iwfc1..iwfc2 {
                for io in 0..norb {
                    for jo in 0..norb {
                        for k in 0..6 {
                            for ik in 0..ikk {
                                let d = at(ik, k, jo, io, w).sub(ave[flat(ik, k, jo, io)]);
                                let i = flat(ik, k, jo, io);
                                err[i] = err[i].add(C::new(d.re * d.re, d.im * d.im));
                            }
                        }
                    }
                }
            }
            if numave == 1 {
                err.iter_mut().for_each(|e| *e = C::ZERO);
            } else {
                let denom = ((numave * (numave - 1)) as f64).sqrt();
                for e in err.iter_mut() {
                    *e = C::new(e.re.sqrt(), e.im.sqrt()).div_real(denom);
                }
            }
            let filename = if p.calctype == 1 {
                format!(
                    "{}_corr_step{}.dat",
                    p.filehead,
                    p.interval * (istep as i64 - 1)
                )
            } else {
                format!("{}_corr.dat", p.filehead)
            };
            let mut text = String::from(" # k-length[1]\n");
            let mut ii = 1i64;
            for iorb in 1..=norb {
                for jorb in 1..=norb {
                    text.push_str(&format!(
                        "# Orbital{} to Orbital{}\n",
                        fmt_i(3, iorb as i64),
                        fmt_i(3, jorb as i64)
                    ));
                    text.push_str(&format!(
                        "#  UpUp[{},{},{},{}] (Re. Im. Err.) DownDown[{},{},{},{}]\n",
                        fmt_i(4, ii + 1),
                        fmt_i(4, ii + 2),
                        fmt_i(4, ii + 13),
                        fmt_i(4, ii + 14),
                        fmt_i(4, ii + 3),
                        fmt_i(4, ii + 4),
                        fmt_i(4, ii + 15),
                        fmt_i(4, ii + 16)
                    ));
                    text.push_str(&format!(
                        "#  Density[{},{},{},{}] SzSz[{},{},{},{}] S+S-[{},{},{},{}] S.S[{},{},{},{}]\n",
                        fmt_i(4, ii + 5),
                        fmt_i(4, ii + 6),
                        fmt_i(4, ii + 17),
                        fmt_i(4, ii + 18),
                        fmt_i(4, ii + 7),
                        fmt_i(4, ii + 8),
                        fmt_i(4, ii + 19),
                        fmt_i(4, ii + 20),
                        fmt_i(4, ii + 9),
                        fmt_i(4, ii + 10),
                        fmt_i(4, ii + 21),
                        fmt_i(4, ii + 22),
                        fmt_i(4, ii + 11),
                        fmt_i(4, ii + 12),
                        fmt_i(4, ii + 23),
                        fmt_i(4, ii + 24)
                    ));
                    ii += 24;
                }
            }
            for ik in 0..ikk {
                text.push_str(&fmt_e(15, 5, xk[ik]));
                for iorb in 0..norb {
                    for jorb in 0..norb {
                        for k in 0..6 {
                            let v = ave[flat(ik, k, jorb, iorb)];
                            text.push_str(&fmt_e(15, 5, v.re));
                            text.push_str(&fmt_e(15, 5, v.im));
                        }
                        for k in 0..6 {
                            let v = err[flat(ik, k, jorb, iorb)];
                            text.push_str(&fmt_e(15, 5, v.re));
                            text.push_str(&fmt_e(15, 5, v.im));
                        }
                    }
                }
                text.push('\n');
            }
            write_file(&resolve(cwd, &filename), &text)?;
        }
    } else {
        // HPhi.
        for iwfc in 0..nwfc {
            let filename = format!("{}_corr{}", p.filehead, p.filetail[iwfc]);
            let mut text = String::from(" # k-length[1]\n");
            let mut ii = 1i64;
            for iorb in 1..=norb {
                for jorb in 1..=norb {
                    text.push_str(&format!(
                        "# Orbital{} to Orbital{}\n",
                        fmt_i(3, iorb as i64),
                        fmt_i(3, jorb as i64)
                    ));
                    text.push_str(&format!(
                        "#  UpUp[{},{}] (Re. Im.) DownDown[{},{}]\n",
                        fmt_i(4, ii + 1),
                        fmt_i(4, ii + 2),
                        fmt_i(4, ii + 3),
                        fmt_i(4, ii + 4)
                    ));
                    text.push_str(&format!(
                        "#  Density[{},{}] SzSz[{},{}] S+S-[{},{}] S.S[{},{}]\n",
                        fmt_i(4, ii + 5),
                        fmt_i(4, ii + 6),
                        fmt_i(4, ii + 7),
                        fmt_i(4, ii + 8),
                        fmt_i(4, ii + 9),
                        fmt_i(4, ii + 10),
                        fmt_i(4, ii + 11),
                        fmt_i(4, ii + 12)
                    ));
                    ii += 12;
                }
            }
            for ik in 0..ikk {
                // WRITE(fo,'(1000e15.5)') xk(ik), cor_k(ik, 1:6, 1:norb, 1:norb, iwfc)
                let mut items: Vec<f64> = vec![xk[ik]];
                for io in 0..norb {
                    for jo in 0..norb {
                        for k in 0..6 {
                            let v = at(ik, k, jo, io, iwfc);
                            items.push(v.re);
                            items.push(v.im);
                        }
                    }
                }
                for chunk in items.chunks(1000) {
                    for v in chunk {
                        text.push_str(&fmt_e(15, 5, *v));
                    }
                    text.push('\n');
                }
            }
            write_file(&resolve(cwd, &filename), &text)?;
        }
    }
    // kpath.gp
    let mut gp = String::from("set xtics (");
    for inode in 0..nnode - 1 {
        gp.push_str(&format!(
            "'{}'  {}, ",
            g.kname[inode].trim(),
            fmt_f(10, 5, xk_label[inode])
        ));
    }
    gp.push_str(&format!(
        "'{}' {})\n",
        g.kname[nnode - 1].trim(),
        fmt_f(10, 5, xk_label[nnode - 1])
    ));
    gp.push_str("set ylabel 'Correlation function'\n");
    gp.push_str("set grid xtics lt 1 lc 0\n");
    write_file(&resolve(cwd, "kpath.gp"), &gp)?;
    // FermiSurfer file.
    for iwfc in 0..nwfc {
        let filename = format!("{}_corr{}.frmsf", p.filehead, p.filetail[iwfc]);
        let mut text = String::new();
        let mut rec = ListRecord::new();
        for v in g.nkg {
            rec.int(v);
        }
        text.push_str(&rec.finish());
        text.push('\n');
        let single = |vals: &[f64]| -> String {
            let mut rec = ListRecord::new();
            for v in vals {
                rec.real(RealKind::Single, *v);
            }
            format!("{}\n", rec.finish())
        };
        let mut rec = ListRecord::new();
        rec.int(1);
        text.push_str(&rec.finish());
        text.push('\n');
        let mut rec = ListRecord::new();
        rec.int(norb as i64);
        text.push_str(&rec.finish());
        text.push('\n');
        for j in 0..3 {
            text.push_str(&single(&[g.recipr[0][j], g.recipr[1][j], g.recipr[2][j]]));
        }
        for iorb in 0..norb {
            for ik in ikk..nk {
                let v = at(ik, 0, iorb, iorb, iwfc).re + at(ik, 1, iorb, iorb, iwfc).re;
                let mut rec = ListRecord::new();
                rec.real(RealKind::Double, v);
                text.push_str(&rec.finish());
                text.push('\n');
            }
        }
        for iorb in 1..=norb {
            for _ in ikk..nk {
                text.push_str(&single(&[iorb as f64]));
            }
        }
        write_file(&resolve(cwd, &filename), &text)?;
    }
    Ok(())
}

/// Run `greenr2k <namelist> <geometry>` in working directory `cwd`; console output
/// goes to `stdout`.
pub fn run(
    namelist: &str,
    geometry: &str,
    cwd: &Path,
    stdout: &mut dyn Write,
) -> Result<(), Error> {
    let mut con = Console { out: stdout };
    let params = read_filename(cwd, namelist, &mut con)?;
    let geo = read_geometry(cwd, geometry, params.nsite, &mut con)?;
    let (kvec, ikk) = set_kpoints(&geo, &mut con)?;
    let (indx, ncor1, ncor2) = read_corrindx(cwd, &params, &geo, &mut con)?;
    let cor = read_corrfile(cwd, &params, &geo, &indx, ncor1, ncor2)?;
    let cor_k = fourier_cor(&geo, &kvec, params.filetail.len(), &cor);
    output_cor(cwd, &params, &geo, &kvec, ikk, &cor_k, &mut con)?;
    con.blank()?;
    con.list(|r| {
        r.str("#####  Done  #####");
    })?;
    con.blank()?;
    Ok(())
}
