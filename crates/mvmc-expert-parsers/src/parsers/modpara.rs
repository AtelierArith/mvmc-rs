//! `modpara.def` parser.
//!
//! Pure-Rust port of C mVMC 1.3.0 `SetDefaultValuesModPara` and
//! `GetInfoFromModPara` (`readdef.c`), including the NBlockSize_RBMRatio
//! adjustment that `ReadDefFileNInt` applies after the read. Julia's
//! `modpara_parser.jl` is the architectural reference only; its aliases,
//! `name = value` syntax, `useDiagScale`/`RescaleSmat` and silent handling of
//! unknown keywords are not C behavior and are not accepted.
//!
//! C contract reproduced here:
//! * Every key starts at the C default (`ModParaParameters::default`), so a
//!   key absent from the file keeps that default.
//! * Lines are read with `fgets` into 256-byte buffers. Lines 1, 3, 4, 5 and 8
//!   are skipped, line 2 is scanned and ignored, line 6 supplies
//!   `CDataFileHead` and line 7 `CParaFileHead` (second word, positional).
//! * Later lines are skipped when they begin with `\n` or `-`; otherwise
//!   `sscanf("%s %lf")` is applied. As in C, a failed `%s` or `%lf` leaves the
//!   previous keyword/value in place (so a stale value can be re-applied).
//! * Keywords match ASCII case-insensitively and in full (`CheckWords`); an
//!   unknown keyword is an error naming it. `2Sz -1` is rejected.
//! * Values are `%lf` doubles converted with a C `(int)` cast, which truncates
//!   toward zero (`1.0e2` is 100, `2.9` is 2). Out-of-range or NaN casts are
//!   undefined in C; Rust uses the saturating `as i32`.
//! * C replaces `RndSeed < 0` by `time(NULL)` here. Rust keeps the raw negative
//!   value and resolves it once, in `mvmc_core::resolve_rnd_seed`.
//! * The `output/` prefix C puts on both file heads is the Rust output
//!   directory (default `<namelist dir>/output`, overridden by `--out-dir`), so
//!   the heads are stored without it; see
//!   `ModParaParameters::c_data_file_path_head`.
//! * `NFileFlushInterval` is not a C modpara keyword and is rejected (flush
//!   semantics are tracked separately in #346).

use std::io;
use std::path::Path;

use crate::types::ModParaParameters;
use crate::utils::c_numeric::Scan;
use crate::utils::file::read_def_file;

/// `D_FileNameMax`: size of the C `fgets` line buffers.
const C_LINE_BUFFER: usize = 256;

/// Parse a `modpara.def` file from disk.
pub fn parse_modpara_def<P: AsRef<Path>>(path: P) -> io::Result<ModParaParameters> {
    let content = read_def_file(path)?;
    parse_modpara_content(&content)
}

fn invalid(message: String) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

/// C `fgets(buf, 256, fp)`: at most 255 bytes, including a terminating newline.
fn fgets_lines(content: &[u8]) -> Vec<&[u8]> {
    let mut lines = Vec::new();
    let mut start = 0;
    while start < content.len() {
        let limit = (start + C_LINE_BUFFER - 1).min(content.len());
        let end = content[start..limit]
            .iter()
            .position(|&b| b == b'\n')
            .map_or(limit, |i| start + i + 1);
        lines.push(&content[start..end]);
        start = end;
    }
    lines
}

/// C `CheckWords`: full-string equality after ASCII lowercasing.
fn check_words(word: &[u8], keyword: &str) -> bool {
    word.eq_ignore_ascii_case(keyword.as_bytes())
}

/// Parse a `modpara.def` payload from memory following C `GetInfoFromModPara`.
pub fn parse_modpara_content(content: &str) -> io::Result<ModParaParameters> {
    let mut p = ModParaParameters::default();
    let incomplete = || invalid("ModPara file is incomplete (fewer than 8 header lines)".into());
    let mut it = fgets_lines(content.as_bytes()).into_iter();
    // Lines 1-5: 1, 3, 4, 5 skipped; 2 scanned for "%s %d" and ignored.
    for _ in 0..5 {
        it.next().ok_or_else(incomplete)?;
    }
    // Line 6: "%s %s" -> CDataFileHead; line 7: CParaFileHead.
    let second_word = |line: &[u8]| -> String {
        let mut scan = Scan::new(line);
        scan.token();
        scan.token()
            .map(|w| String::from_utf8_lossy(w).into_owned())
            .unwrap_or_default()
    };
    p.c_data_file_head = second_word(it.next().ok_or_else(incomplete)?);
    p.c_para_file_head = second_word(it.next().ok_or_else(incomplete)?);
    // Line 8 is skipped but remains in C's `ctmp` keyword buffer.
    let mut keyword: Vec<u8> = it.next().ok_or_else(incomplete)?.to_vec();
    // C's `dtmp` is uninitialized until the first successful `%lf`.
    let mut value = 0.0_f64;

    for line in it {
        if line.first() == Some(&b'\n') || line.first() == Some(&b'-') {
            continue;
        }
        let mut scan = Scan::new(line);
        if let Some(word) = scan.token() {
            keyword = word.to_vec();
            if let Some(number) = scan.float() {
                value = number;
            }
        }
        // `(int) dtmp`: truncation toward zero (saturating, see module docs).
        let int = value as i32;
        let int64 = i64::from(int);
        let k = keyword.as_slice();
        if check_words(k, "NVMCCalMode") {
            p.vmc_calc_mode = int64;
        } else if check_words(k, "NLanczosMode") {
            p.lanczos_mode = int64;
        } else if check_words(k, "NDataIdxStart") {
            p.n_data_idx_start = int64;
        } else if check_words(k, "NDataQtySmp") {
            p.n_data_qty_smp = int64;
        } else if check_words(k, "Nsite") {
            p.nsite = int64;
        } else if check_words(k, "Ne") || check_words(k, "Nelectron") {
            p.nelec = int64;
        } else if check_words(k, "Ncond") {
            p.ncond = int64;
        } else if check_words(k, "2Sz") {
            p.two_sz = int64;
            if int == -1 {
                return Err(invalid(
                    "2Sz must be even number (2Sz = -1 is rejected)".into(),
                ));
            }
        } else if check_words(k, "NSPGaussLeg") {
            p.nsp_gauss_leg = int64;
        } else if check_words(k, "NSPStot") {
            p.nsp_stot = int64;
        } else if check_words(k, "NMPTrans") {
            p.nmp_trans = int64;
        } else if check_words(k, "NSROptItrStep") {
            p.nsr_opt_itr_step = int64;
        } else if check_words(k, "NSROptItrSmp") {
            p.nsr_opt_itr_smp = int64;
        } else if check_words(k, "DSROptRedCut") {
            p.dsr_opt_red_cut = value;
        } else if check_words(k, "DSROptStaDel") {
            p.dsr_opt_sta_del = value;
        } else if check_words(k, "DSROptStepDt") {
            p.dsr_opt_step_dt = value;
        } else if check_words(k, "NSROptCGMaxIter") {
            p.nsr_opt_cg_max_iter = int64;
        } else if check_words(k, "DSROptCGTol") {
            p.dsr_opt_cg_tol = value;
        } else if check_words(k, "NVMCWarmUp") {
            p.nvmc_warmup = int64;
        } else if check_words(k, "NVMCInterval") {
            p.nvmc_interval = int64;
        } else if check_words(k, "NVMCSample") {
            p.nvmc_sample = int64;
        } else if check_words(k, "NExUpdatePath") {
            p.nex_update_path = int64;
        } else if check_words(k, "RndSeed") {
            p.rnd_seed = int64;
        } else if check_words(k, "NSplitSize") {
            p.nsplit_size = int64;
        } else if check_words(k, "NStore") {
            p.nstore_o = int64;
        } else if check_words(k, "NSRCG") {
            p.nsrcg = int64;
        } else if check_words(k, "Nneuron") {
            p.nneuron = int64;
        } else if check_words(k, "NneuronCharge") {
            p.nneuron_charge = int64;
        } else if check_words(k, "NneuronSpin") {
            p.nneuron_spin = int64;
        } else if check_words(k, "NneuronGeneral") {
            p.nneuron_general = int64;
        } else if check_words(k, "NBlockSize_RBMRatio") {
            p.nblock_size_rbm_ratio = int64;
        } else {
            return Err(invalid(format!(
                "keyword \" {} \" is incorrect in modpara.def",
                String::from_utf8_lossy(k).trim_end_matches('\n')
            )));
        }
    }

    // readdef.c:375-380: the RBM block size must be a multiple of 8. C `int`
    // arithmetic: only positive remainders are adjusted; the quotient rounds
    // down but never below one block.
    let block = p.nblock_size_rbm_ratio as i32;
    if block % 8 > 0 {
        p.nblock_size_rbm_ratio = i64::from((block / 8).max(1) * 8);
    }
    Ok(p)
}
