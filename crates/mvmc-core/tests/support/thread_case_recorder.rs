//! Proposed test-only observer. Not installed in a production crate or gate.
//! The caller wraps ORIGINAL assertions, never substitutes a boolean claim.
use std::fs::{File, OpenOptions};
use std::io::{self, Write};
use std::path::Path;

pub struct Recorder {
    file: File,
    sequence: usize,
    run: String,
}

fn atom(value: &str) -> io::Result<()> {
    if value.is_empty()
        || !value
            .bytes()
            .all(|c| c.is_ascii_alphanumeric() || b"_-/.".contains(&c))
        || value.split('/').any(|part| matches!(part, "" | "." | ".."))
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "invalid record atom",
        ));
    }
    Ok(())
}

impl Recorder {
    /// Exclusive file creation. The harness must supply an independently bound run ID.
    pub fn create(path: &Path, run: &str) -> io::Result<Self> {
        atom(run)?;
        let file = OpenOptions::new().write(true).create_new(true).open(path)?;
        Ok(Self {
            file,
            sequence: 0,
            run: run.to_owned(),
        })
    }

    pub(super) fn event(
        &mut self,
        phase: &str,
        case: &str,
        boundary: &str,
        worker: usize,
        repeat: usize,
    ) -> io::Result<()> {
        atom(case)?;
        atom(boundary)?;
        if ![1, 2, 4].contains(&worker) || repeat == 0 {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, "worker/repeat"));
        }
        writeln!(
            self.file,
            "{}|{}|{}|{}|{}|{}|{}",
            self.sequence, self.run, phase, case, boundary, worker, repeat
        )?;
        self.file.flush()?;
        self.sequence += 1;
        Ok(())
    }

    /// When disabled, calls assertions once without creating a file or copying data.
    /// On panic only START remains. A write failure propagates to the test caller.
    /// COMPLETE means this closure returned; its meaning requires reviewed call-site source.
    pub fn assertions<T>(
        recorder: Option<&mut Self>,
        case: &str,
        boundary: &str,
        worker: usize,
        repeat: usize,
        assertions: impl FnOnce() -> T,
    ) -> io::Result<T> {
        if let Some(recorder) = recorder {
            recorder.event("START", case, boundary, worker, repeat)?;
            let value = assertions();
            recorder.event("COMPLETE", case, boundary, worker, repeat)?;
            Ok(value)
        } else {
            Ok(assertions())
        }
    }
}

// Deliberately no generic test-PASS emitter, no checks:true API, no RNG access,
// and no Drop completion. All 83 reviewed groups require explicit call-site wiring.

#[cfg(test)]
mod tests {
    use super::*;
    use std::panic::{catch_unwind, AssertUnwindSafe};
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn path() -> std::path::PathBuf {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        std::env::temp_dir().join(format!(
            "thread-recorder-proposal-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ))
    }

    #[test]
    fn disabled_calls_original_once() {
        let mut calls = 0;
        let answer = Recorder::assertions(None, "unused", "unused", 1, 1, || {
            calls += 1;
            7
        })
        .unwrap();
        assert_eq!((answer, calls), (7, 1));
    }

    #[test]
    fn completion_follows_assertions_and_duplicate_file_rejected() {
        let path = path();
        let mut recorder = Recorder::create(&path, "run").unwrap();
        assert!(Recorder::create(&path, "run").is_err());
        Recorder::assertions(Some(&mut recorder), "case", "boundary", 2, 1, || {
            assert_eq!(2 + 2, 4);
        })
        .unwrap();
        drop(recorder);
        assert_eq!(
            std::fs::read_to_string(&path).unwrap(),
            "0|run|START|case|boundary|2|1\n1|run|COMPLETE|case|boundary|2|1\n"
        );
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn assertion_panic_never_completes() {
        let path = path();
        let mut recorder = Recorder::create(&path, "run").unwrap();
        let failure = catch_unwind(AssertUnwindSafe(|| {
            Recorder::assertions(Some(&mut recorder), "case", "boundary", 1, 1, || {
                panic!("original assertion failure")
            })
            .unwrap();
        }));
        assert!(failure.is_err());
        drop(recorder);
        assert_eq!(
            std::fs::read_to_string(&path).unwrap(),
            "0|run|START|case|boundary|1|1\n"
        );
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn existing_directory_and_invalid_atom_rejected() {
        assert!(Recorder::create(&std::env::temp_dir(), "run").is_err());
        assert!(Recorder::create(&path(), "bad\nrun").is_err());
    }

    #[test]
    fn traversal_and_delimiter_case_atoms_rejected() {
        for value in [
            "../case",
            "case/../other",
            "/case",
            "case//other",
            "case|other",
            "case\nother",
        ] {
            assert!(atom(value).is_err(), "{value:?}");
        }
        assert!(atom("long20/real-fsz/qp32/store1/cg0").is_ok());
    }
}
