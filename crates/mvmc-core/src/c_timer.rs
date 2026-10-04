//! Inclusive C-compatible section timers, ported from Julia's c_timer.jl.
//! Const generics erase clock reads and slot accesses in disabled kernels.
use std::{
    fs::File,
    io::{self, Write},
    path::{Path, PathBuf},
    time::Instant,
};

/// C's NTimer, including diagnostic IDs through 966.
pub const CTIMER_N: usize = 1000;

/// Per-run timer. A given ID has one start slot and must not be reentered.
/// Shared-memory workers must use separate timers and merge elapsed times.
#[derive(Clone, Debug)]
pub struct CTimer<const ENABLED: bool> {
    /// Accumulated nanoseconds indexed by upstream timer ID.
    pub elapsed_ns: [u64; CTIMER_N],
    /// Most recent start timestamp, in nanoseconds, indexed by timer ID.
    pub start_ns: [u64; CTIMER_N],
    /// Enabled diagnostic families for this run.
    pub diagnostics: TimerEnv,
    epoch: Option<Instant>,
}
impl<const ENABLED: bool> Default for CTimer<ENABLED> {
    fn default() -> Self {
        Self::new()
    }
}
impl<const ENABLED: bool> CTimer<ENABLED> {
    /// Query the type-level switch, matching Julia's enabled/disabled timer dispatch.
    pub const fn enabled(&self) -> bool {
        ENABLED
    }

    /// Construct zeroed timer storage and a monotonic clock origin.
    pub fn new() -> Self {
        Self {
            elapsed_ns: [0; CTIMER_N],
            start_ns: [0; CTIMER_N],
            epoch: if ENABLED { Some(Instant::now()) } else { None },
            diagnostics: TimerEnv::default(),
        }
    }
    /// Start this ID; disabled timers do not read the clock.
    #[inline(always)]
    pub fn start(&mut self, id: usize) {
        if ENABLED {
            self.start_at(
                id,
                self.epoch
                    .as_ref()
                    .expect("enabled timer has an epoch")
                    .elapsed()
                    .as_nanos() as u64,
            );
        }
    }
    /// Accumulate elapsed time for this ID; parent sections keep running.
    #[inline(always)]
    pub fn stop(&mut self, id: usize) {
        if ENABLED {
            self.stop_at(
                id,
                self.epoch
                    .as_ref()
                    .expect("enabled timer has an epoch")
                    .elapsed()
                    .as_nanos() as u64,
            );
        }
    }
    /// Start a diagnostic section only when its family is enabled.
    /// Passing a resolved environment predicate selects the parent timer or a no-op,
    /// corresponding to Julia's `ctimer_if_env` without a disabled singleton borrow.
    #[inline(always)]
    pub fn start_diag(&mut self, id: usize, enabled: bool) {
        if ENABLED && enabled {
            self.start(id);
        }
    }
    /// Stop a diagnostic section only when its family is enabled.
    #[inline(always)]
    pub fn stop_diag(&mut self, id: usize, enabled: bool) {
        if ENABLED && enabled {
            self.stop(id);
        }
    }
    /// Start with an explicit timestamp for deterministic tests or a shared clock.
    #[inline(always)]
    pub fn start_at(&mut self, id: usize, now: u64) {
        if ENABLED {
            self.start_ns[id] = now;
        }
    }
    /// Stop with an explicit timestamp, preserving Julia's UInt64 arithmetic.
    #[inline(always)]
    pub fn stop_at(&mut self, id: usize, now: u64) {
        if ENABLED {
            self.elapsed_ns[id] =
                self.elapsed_ns[id].wrapping_add(now.wrapping_sub(self.start_ns[id]));
        }
    }
    /// Reset both arrays between runs.
    pub fn reset(&mut self) {
        self.elapsed_ns.fill(0);
        self.start_ns.fill(0);
    }
    /// Convert accumulated nanoseconds to seconds only when queried or written.
    pub fn seconds(&self, id: usize) -> f64 {
        self.elapsed_ns[id] as f64 / 1.0e9
    }
    /// Add worker-local elapsed counters, preserving parent's running slots.
    pub fn merge_elapsed(&mut self, worker: &Self) {
        if ENABLED {
            for (parent, worker) in self.elapsed_ns.iter_mut().zip(&worker.elapsed_ns) {
                *parent = parent.wrapping_add(*worker);
            }
        }
    }
    /// Write the upstream optimization report. The directory must already exist.
    /// Pass `"zvo"` for the original Julia default prefix.
    pub fn write_para_opt(&self, dir: &Path, prefix: &str) -> io::Result<PathBuf> {
        self.write_lines(dir, prefix, "CalcTimer", PARA_OPT_LINES)
    }
    /// Write the upstream diagnostic report, including uninstrumented zero slots.
    /// Pass `"zvo"` for the original Julia default prefix.
    pub fn write_diag(&self, dir: &Path, prefix: &str) -> io::Result<PathBuf> {
        self.write_lines(dir, prefix, "CalcTimerDiag", DIAG_LINES)
    }
    /// Write the upstream PhysCal report (`NVMCCalMode=1`).
    /// Pass `"zvo"` for the original Julia default prefix.
    pub fn write_phys_cal(&self, dir: &Path, prefix: &str) -> io::Result<PathBuf> {
        self.write_lines(dir, prefix, "CalcTimer", PHYS_CAL_LINES)
    }
    fn write_lines(
        &self,
        dir: &Path,
        prefix: &str,
        suffix: &str,
        lines: &[(&str, usize)],
    ) -> io::Result<PathBuf> {
        let path = dir.join(format!("{prefix}_{suffix}.dat"));
        let mut file = File::create(&path)?;
        for &(label, id) in lines {
            writeln!(file, "{label}{:12.5}", self.seconds(id))?;
        }
        Ok(path)
    }
}

/// Environment controls. Every present value other than literal "0" enables a flag.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct TimerEnv {
    /// Primary MVMC_C_TIMER switch.
    pub primary: bool,
    /// Deprecated MVMC_TIMER alias.
    pub legacy: bool,
    /// MVMC_CALHAM1_DIAG.
    pub calham1: bool,
    /// MVMC_SLATER_DIAG.
    pub slater: bool,
    /// MVMC_MAINCAL_DIAG.
    pub maincal: bool,
    /// MVMC_WEIGHTAVG_DIAG.
    pub weightavg: bool,
}
impl TimerEnv {
    /// Read controls from the process environment.
    pub fn from_env() -> Self {
        Self::from_lookup(|key| std::env::var(key).ok())
    }
    /// Read through an injected lookup without mutating process-wide environment.
    pub fn from_lookup(mut get: impl FnMut(&str) -> Option<String>) -> Self {
        let mut enabled = |key| get(key).is_some_and(|value| value != "0");
        Self {
            primary: enabled("MVMC_C_TIMER"),
            legacy: enabled("MVMC_TIMER"),
            calham1: enabled("MVMC_CALHAM1_DIAG"),
            slater: enabled("MVMC_SLATER_DIAG"),
            maincal: enabled("MVMC_MAINCAL_DIAG"),
            weightavg: enabled("MVMC_WEIGHTAVG_DIAG"),
        }
    }
    /// A diagnostic flag enables the parent timer even without the main switch.
    pub fn enabled(&self) -> bool {
        self.primary || self.legacy || self.any_diag()
    }
    /// Whether a diagnostic report is requested.
    pub fn any_diag(&self) -> bool {
        self.calham1 || self.slater || self.maincal || self.weightavg
    }
    /// Julia warns only when the deprecated alias is used without the main switch.
    pub fn legacy_warning(&self) -> bool {
        self.legacy && !self.primary
    }
}

const PARA_OPT_LINES: &[(&str, usize)] = &[
    ("All                         [0] ", 0),
    ("Initialization              [1] ", 1),
    ("  read options             [10] ", 10),
    ("  ReadDefFile              [11] ", 11),
    ("  SetMemory                [12] ", 12),
    ("  InitParameter            [13] ", 13),
    ("VMCParaOpt                  [2] ", 2),
    ("  VMCMakeSample             [3] ", 3),
    ("    makeInitialSample      [30] ", 30),
    ("    make candidate         [31] ", 31),
    ("    hopping update         [32] ", 32),
    ("      UpdateProjCnt        [60] ", 60),
    ("      CalculateNewPfM2     [61] ", 61),
    ("      CalculateLogIP       [62] ", 62),
    ("      UpdateMAll           [63] ", 63),
    ("    exchange update        [33] ", 33),
    ("      UpdateProjCnt        [65] ", 65),
    ("      CalculateNewPfMTwo2  [66] ", 66),
    ("      CalculateLogIP       [67] ", 67),
    ("      UpdateMAllTwo        [68] ", 68),
    ("    lspinflip update       [36] ", 36),
    ("      UpdateProjCnt       [600] ", 600),
    ("      CalculateNewPfMTwo2 [601] ", 601),
    ("      CalculateLogIP      [602] ", 602),
    ("      UpdateMAllTwo       [603] ", 603),
    ("    recal PfM and InvM     [34] ", 34),
    ("    save electron config   [35] ", 35),
    ("  VMCMainCal                [4] ", 4),
    ("    CalculateMAll          [40] ", 40),
    ("    LocEnergyCal           [41] ", 41),
    ("      CalHamiltonian0      [70] ", 70),
    ("      CalHamiltonian1      [71] ", 71),
    ("      CalHamiltonian2      [72] ", 72),
    ("    ReturnSlaterElmDiff    [42] ", 42),
    ("    calculate OO and HO    [43] ", 43),
    ("    multiply store OO      [45] ", 45),
    ("  StochasticOpt             [5] ", 5),
    ("    preprocess             [50] ", 50),
    ("    stcOptMain             [51] ", 51),
    ("      initBLACS            [55] ", 55),
    ("      calculate S and g    [56] ", 56),
    ("      DPOSV                [57] ", 57),
    ("      gatherParaChange     [58] ", 58),
    ("    postprocess            [52] ", 52),
    ("  UpdateSlaterElm          [20] ", 20),
    ("  WeightAverage            [21] ", 21),
    ("  outputData               [22] ", 22),
    ("  SyncModifiedParameter    [23] ", 23),
    ("  cal                      [24] ", 24),
    ("  SR                       [25] ", 25),
    ("  MAll                     [69] ", 69),
];

const DIAG_LINES: &[(&str, usize)] = &[
    ("CalH1 GreenFunc1Real       [920] ", 920),
    ("  UpdateProjCnt            [921] ", 921),
    ("  ProjRatio                [922] ", 922),
    ("  CalculateNewPfM2_real    [923] ", 923),
    ("  CalculateIP_real         [924] ", 924),
    ("  CalH1 fast prep          [925] ", 925),
    ("  CalH1 fast term loop     [926] ", 926),
    ("  GreenFunc1 setup/check   [927] ", 927),
    ("  GreenFunc1 restore       [928] ", 928),
    ("  CalH1 threaded combine   [929] ", 929),
    ("SlaterElmDiff_fcmp         [930] ", 930),
    ("  Slater scratch reset     [931] ", 931),
    ("  Slater transOrb build    [932] ", 932),
    ("  Slater buffer accumulate [933] ", 933),
    ("  Slater srOptO store      [934] ", 934),
    ("  CalH1 direct proj ratio  [935] ", 935),
    ("  CalH1 PfM2/IP fused      [936] ", 936),
    ("VMCMainCal unmeasured      [940] ", 940),
    ("  accumulator init/reset   [941] ", 941),
    ("  sample copy/check        [942] ", 942),
    ("  post CalculateMAll       [943] ", 943),
    ("  CalculateIP              [944] ", 944),
    ("  weight/check             [945] ", 945),
    ("  energy accumulate/check  [946] ", 946),
    ("  Green measurement        [947] ", 947),
    ("  SR setup/proj/RBM        [948] ", 948),
    ("  optTrans diff            [949] ", 949),
    ("  clear/merge/final check  [950] ", 950),
    ("WeightAverage diagnostic   [960] ", 960),
    ("  WE allreduce             [961] ", 961),
    ("  WE normalize             [962] ", 962),
    ("  SR OO allreduce          [963] ", 963),
    ("  SR HO allreduce          [964] ", 964),
    ("  SR normalize             [965] ", 965),
    ("  ReduceCounter            [966] ", 966),
];

/// C's `OutputTimerPhysCal` line set (`vmcclock.c:140-191`). Distinct from the
/// ParaOpt set: ids 50-53 mean GreenFunc1/GreenFunc2/addPhysCA/addPhysCACA in
/// this mode, and StochasticOpt/ReturnSlaterElmDiff ids are absent.
const PHYS_CAL_LINES: &[(&str, usize)] = &[
    ("All                         [0] ", 0),
    ("Initialization              [1] ", 1),
    ("  read options             [10] ", 10),
    ("  ReadDefFile              [11] ", 11),
    ("  SetMemory                [12] ", 12),
    ("  InitParameter            [13] ", 13),
    ("VMCPhysCal                  [2] ", 2),
    ("  VMCMakeSample             [3] ", 3),
    ("    makeInitialSample      [30] ", 30),
    ("    make candidate         [31] ", 31),
    ("    hopping update         [32] ", 32),
    ("      UpdateProjCnt        [60] ", 60),
    ("      CalculateNewPfM2     [61] ", 61),
    ("      CalculateLogIP       [62] ", 62),
    ("      UpdateMAll           [63] ", 63),
    ("    exchange update        [33] ", 33),
    ("      UpdateProjCnt        [65] ", 65),
    ("      CalculateNewPfMTwo2  [66] ", 66),
    ("      CalculateLogIP       [67] ", 67),
    ("      UpdateMAllTwo        [68] ", 68),
    ("    lspinflip update       [36] ", 36),
    ("      UpdateProjCnt       [600] ", 600),
    ("      CalculateNewPfMTwo2 [601] ", 601),
    ("      CalculateLogIP      [602] ", 602),
    ("      UpdateMAllTwo       [603] ", 603),
    ("    recal PfM and InvM     [34] ", 34),
    ("    save electron config   [35] ", 35),
    ("  VMCMainCal                [4] ", 4),
    ("    CalculateMAll          [40] ", 40),
    ("    LocEnergyCal           [41] ", 41),
    ("      CalHamiltonian0      [70] ", 70),
    ("      CalHamiltonian1      [71] ", 71),
    ("      CalHamiltonian2      [72] ", 72),
    ("    CalculateGreenFunc     [42] ", 42),
    ("      GreenFunc1           [50] ", 50),
    ("      GreenFunc2           [51] ", 51),
    ("      addPhysCA            [52] ", 52),
    ("      addPhysCACA          [53] ", 53),
    ("    Lanczos1               [43] ", 43),
    ("    Lanczos2               [44] ", 44),
    ("  UpdateSlaterElm          [20] ", 20),
    ("  WeightAverage            [21] ", 21),
    ("  outputData               [22] ", 22),
];
