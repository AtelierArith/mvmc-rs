//! C run-log files: `{head}_time_{NDataIdxStart:03}.dat` and `{head}_SRinfo.dat`.
//!
//! C authority (`extern/mVMC-1.3.0/src/mVMC/`): `InitFile` (`initfile.c:35-66`)
//! opens both files with `"w"` on rank 0 (SRinfo only for `NVMCCalMode==0`),
//! `OutputTime` (`vmcclock.c:34-46`) writes the time rows, `FlushFile`
//! (`initfile.c:192-203`) flushes every `NFileFlushInterval` steps and
//! `CloseFile` closes them. C keeps stdio-buffered `FILE*` handles open for the
//! whole run; [`RunFiles`] holds buffered writers the same way, so rows reach
//! disk only at [`RunFiles::flush_file`] or when the files are closed (dropped).
//!
//! `NFileFlushInterval` is set only by the C command-line option `-F`
//! (`vmcmain.c:123-148`, default 1); `modpara.def` has no such keyword.

use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::Path;

use mvmc_expert_parsers::ExpertModeData;

/// Whether the run is parameter optimization or physical-quantity calculation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RunKind {
    /// `NVMCCalMode == 0`: `_time_` and `_SRinfo` files.
    ParaOpt,
    /// `NVMCCalMode == 1`: `_time_` file only.
    PhysCal,
}

/// One `_SRinfo.dat` row; field order follows the C `fprintf` argument order.
#[derive(Debug, Clone, PartialEq)]
pub struct SrInfoRow {
    /// `NPara` (the global parameter count, not `OFFSET*NPara`).
    pub n_para: i64,
    /// `nSmat`.
    pub n_smat: i64,
    /// `optNum`: components fixed by `OptFlag`.
    pub opt_num: i64,
    /// `cutNum`: components removed by the diagonal cut.
    pub cut_num: i64,
    /// `sDiagMax` (`eigenMax` in diagonalization mode).
    pub s_diag_max: f64,
    /// `sDiagMin` (`eigenMin` in diagonalization mode).
    pub s_diag_min: f64,
    /// Signed element of the solution with the largest magnitude.
    pub r_max: f64,
    /// `smatToParaIdx[simax]`.
    pub i_max: i64,
    /// CG rows append `", %d"` with the solver `info`; direct rows do not.
    pub cg_info: Option<i64>,
}

/// Header of `_SRinfo.dat` (`initfile.c:46-52`); `SRFlag==1` is diagonalization mode.
pub fn sr_info_header(diagonalization: bool) -> &'static str {
    if diagonalization {
        "#Npara Msize optCut diagCut sEigenMax  sEigenMin    absRmax       imax\n"
    } else {
        "#Npara Msize optCut diagCut sDiagMax  sDiagMin    absRmax       imax\n"
    }
}

/// C `printf("% .5e", x)`.
pub fn format_c_exp5(x: f64) -> String {
    if x.is_nan() {
        return if x.is_sign_negative() { "-nan" } else { " nan" }.to_string();
    }
    if x.is_infinite() {
        return if x < 0.0 { "-inf" } else { " inf" }.to_string();
    }
    let raw = format!("{x:.5e}");
    let (mantissa, exponent) = raw.split_once('e').expect("exponent marker");
    let exponent: i32 = exponent.parse().expect("numeric exponent");
    let sign = if mantissa.starts_with('-') { "" } else { " " };
    format!("{sign}{mantissa}e{exponent:+03}")
}

/// Format one SRinfo row, including the trailing newline.
///
/// C: `"%5d %5d %5d %5d % .5e % .5e % .5e %5d\n"` (direct) and
/// `"%5d %5d %5d %5d % .5e % .5e % .5e %5d, %d\n"` (CG).
pub fn format_sr_info_row(row: &SrInfoRow) -> String {
    let mut line = format!(
        "{:5} {:5} {:5} {:5} {} {} {} {:5}",
        row.n_para,
        row.n_smat,
        row.opt_num,
        row.cut_num,
        format_c_exp5(row.s_diag_max),
        format_c_exp5(row.s_diag_min),
        format_c_exp5(row.r_max),
        row.i_max
    );
    if let Some(info) = row.cg_info {
        line.push_str(&format!(", {info}"));
    }
    line.push('\n');
    line
}

/// Format one `OutputTime` row with the supplied `ctime(3)` string (which
/// carries its own trailing newline).
///
/// `counter` is C's `Counter[]`; slots 0..6 are hop/exchange/LSF attempts and
/// acceptances. Step 0 is the column-title line, as in C.
pub fn format_time_line(step: i64, counter: &[i64], ctime: &str) -> String {
    if step == 0 {
        return format!("{step:05}  acc_hop acc_ex  acc_lsf n_hop    n_ex      n_lsf   : {ctime}");
    }
    let ratio = |all: i64, accepted: i64| {
        if all == 0 {
            0.0
        } else {
            accepted as f64 / all as f64
        }
    };
    let p_hop = ratio(counter[0], counter[1]);
    let p_ex = ratio(counter[2], counter[3]);
    let p_lsf = ratio(counter[4], counter[5]);
    // C passes int counters to %-8d; truncate like the C int conversion.
    format!(
        "{step:05}  {p_hop:.5} {p_ex:.5} {p_lsf:.5} {:<8} {:<8}  {:<8}: {ctime}",
        counter[0] as i32, counter[2] as i32, counter[4] as i32
    )
}

/// `ctime(3)` for the current local time, e.g. `"Thu Oct  5 12:34:56 2026\n"`.
#[cfg(unix)]
pub fn c_ctime_now() -> String {
    use std::os::raw::{c_char, c_long};
    extern "C" {
        fn time(t: *mut c_long) -> c_long;
        fn ctime_r(t: *const c_long, buf: *mut c_char) -> *mut c_char;
    }
    let mut buffer = [0 as c_char; 64];
    // SAFETY: `ctime_r` writes at most 26 bytes (NUL-terminated) into `buffer`.
    unsafe {
        let now = time(std::ptr::null_mut());
        if ctime_r(&now, buffer.as_mut_ptr()).is_null() {
            return String::from("\n");
        }
        std::ffi::CStr::from_ptr(buffer.as_ptr())
            .to_string_lossy()
            .into_owned()
    }
}

/// UTC fallback with the same `ctime` layout on non-Unix targets.
#[cfg(not(unix))]
pub fn c_ctime_now() -> String {
    let seconds = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs() as i64);
    let days = seconds.div_euclid(86_400);
    let rem = seconds.rem_euclid(86_400);
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);
    let weekday = (days + 4).rem_euclid(7) as usize;
    const DAYS: [&str; 7] = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
    const MONTHS: [&str; 12] = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ];
    format!(
        "{} {} {:2} {:02}:{:02}:{:02} {}\n",
        DAYS[weekday],
        MONTHS[(month - 1) as usize],
        day,
        rem / 3600,
        rem % 3600 / 60,
        rem % 60,
        year
    )
}

/// Open run-log files of one rank-0 process (C `FileTime` and `FileSRinfo`).
pub struct RunFiles {
    time: BufWriter<File>,
    sr_info: Option<BufWriter<File>>,
    flush_interval: i64,
    flush_count: u64,
}

impl RunFiles {
    /// C `InitFile`: create (truncate) the files under `dir`.
    ///
    /// `flush_interval` is `NFileFlushInterval` (`None` is C's default 1).
    pub fn init(
        data: &ExpertModeData,
        dir: &Path,
        kind: RunKind,
        flush_interval: Option<i64>,
    ) -> io::Result<Self> {
        let flush_interval = flush_interval.unwrap_or(1);
        if flush_interval < 1 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "FileFlushInterval should be natural number",
            ));
        }
        std::fs::create_dir_all(dir)?;
        let head = if data.modpara.c_data_file_head.is_empty() {
            "zvo"
        } else {
            data.modpara.c_data_file_head.as_str()
        };
        // C formats the signed NDataIdxStart with %03d (for example "-01").
        let time = File::create(dir.join(format!(
            "{head}_time_{:03}.dat",
            data.modpara.n_data_idx_start
        )))?;
        let sr_info = if kind == RunKind::ParaOpt {
            let mut file = BufWriter::new(File::create(dir.join(format!("{head}_SRinfo.dat")))?);
            file.write_all(sr_info_header(data.modpara.sr_flag).as_bytes())?;
            Some(file)
        } else {
            None
        };
        Ok(Self {
            time: BufWriter::new(time),
            sr_info,
            flush_interval,
            flush_count: 0,
        })
    }

    /// C `OutputTime(step)`.
    pub fn output_time(&mut self, step: i64, counter: &[i64]) -> io::Result<()> {
        self.time
            .write_all(format_time_line(step, counter, &c_ctime_now()).as_bytes())
    }

    /// Append one SRinfo row (no-op outside parameter optimization).
    pub fn write_sr_info(&mut self, row: &SrInfoRow) -> io::Result<()> {
        match self.sr_info.as_mut() {
            Some(file) => file.write_all(format_sr_info_row(row).as_bytes()),
            None => Ok(()),
        }
    }

    /// C `FlushFile(step)`: flush when `step % NFileFlushInterval == 0`.
    pub fn flush_file(&mut self, step: i64) -> io::Result<()> {
        if step % self.flush_interval != 0 {
            return Ok(());
        }
        self.flush_count += 1;
        self.time.flush()?;
        if let Some(file) = self.sr_info.as_mut() {
            file.flush()?;
        }
        Ok(())
    }

    /// Number of flushes performed by [`Self::flush_file`] (for lifecycle tests).
    pub fn flush_count(&self) -> u64 {
        self.flush_count
    }

    /// C `CloseFile`: flush and close both files.
    pub fn close(mut self) -> io::Result<()> {
        self.time.flush()?;
        if let Some(file) = self.sr_info.as_mut() {
            file.flush()?;
        }
        Ok(())
    }
}
