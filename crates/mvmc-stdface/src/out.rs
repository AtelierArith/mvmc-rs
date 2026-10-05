//! Output context: the C program's `stdout` stream and its working directory.
//!
//! C writes progress messages to `stdout` and the Expert files to the current directory. Here
//! messages accumulate in [`Out::log`] (byte-for-byte what C prints) and files are written into
//! [`Out::dir`]. `StdFace_exit(code)` becomes [`StdFaceError::Exit`].

use std::fmt;
use std::path::{Path, PathBuf};

/// Why StdFace stopped.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StdFaceError {
    /// `StdFace_exit(code)`; the message that explains it is already in [`Out::log`].
    Exit(i32),
    /// A generated file could not be written.
    Io(String),
}

impl fmt::Display for StdFaceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StdFaceError::Exit(code) => write!(f, "StdFace exited with code {code}"),
            StdFaceError::Io(message) => write!(f, "{message}"),
        }
    }
}

impl std::error::Error for StdFaceError {}

/// Result type of every StdFace routine.
pub type Res<T> = Result<T, StdFaceError>;

/// `StdFace_exit(-1)` after the diagnostic has been printed.
pub fn exit<T>(code: i32) -> Res<T> {
    Err(StdFaceError::Exit(code))
}

/// The `stdout` log plus the directory generated files are written to.
#[derive(Debug)]
pub struct Out {
    /// Everything C would have printed to `stdout`.
    pub log: String,
    /// Everything C would have printed to `stderr`.
    pub err: String,
    /// Directory receiving the generated files (C: the current directory).
    pub dir: PathBuf,
    /// Directory the lattice data files (Wannier90 `zvo_*.dat`) are read from (C: the current
    /// directory).
    pub data_dir: PathBuf,
}

impl Out {
    /// New context writing into `dir`.
    pub fn new(dir: impl AsRef<Path>) -> Self {
        Self {
            log: String::new(),
            err: String::new(),
            dir: dir.as_ref().to_path_buf(),
            data_dir: PathBuf::from("."),
        }
    }

    /// Read data files from `data_dir` instead of the current directory.
    pub fn with_data_dir(mut self, data_dir: impl AsRef<Path>) -> Self {
        self.data_dir = data_dir.as_ref().to_path_buf();
        self
    }

    /// Append to the `stderr` log.
    pub fn eprint(&mut self, text: &str) {
        self.err.push_str(text);
    }

    /// Append to the `stdout` log.
    pub fn print(&mut self, text: &str) {
        self.log.push_str(text);
    }

    /// Create `name` in the output directory with `content` (`fopen(name, "w")` + writes).
    pub fn write_file(&mut self, name: &str, content: &str) -> Res<()> {
        let path = self.dir.join(name);
        std::fs::write(&path, content)
            .map_err(|e| StdFaceError::Io(format!("cannot write {}: {e}", path.display())))
    }
}

/// `fprintf(stdout, ...)` into an [`Out`].
#[macro_export]
macro_rules! outf {
    ($o:expr, $($arg:tt)*) => {
        $o.print(&format!($($arg)*))
    };
}
