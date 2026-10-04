//! Test-only ownership bridge for the inherited acquisition descriptor.
//! Node stdio[3]='pipe' may be a Unix socket: reopening /proc/self/fd/3
//! is not supported. Duplicate the existing descriptor instead.
#![allow(unsafe_code)]

use std::fs::File;
use std::io;
#[cfg(unix)]
use std::os::fd::BorrowedFd;

#[cfg(unix)]
pub(super) fn duplicate(fd: BorrowedFd<'_>) -> io::Result<File> {
    fd.try_clone_to_owned().map(File::from)
}

pub(super) fn inherited_fd3() -> io::Result<File> {
    // SAFETY: the reviewed CI harness spawns this exact selected test with
    // stdio[3]='pipe'. FD3 is inherited open and owned by the process, and no
    // code in this single-test acquisition closes/replaces it. The BorrowedFd
    // exists only during duplication; we never take ownership of original FD3.
    // The returned File owns ONLY the independently duplicated descriptor.
    #[cfg(unix)]
    {
        let inherited = unsafe { BorrowedFd::borrow_raw(3) };
        duplicate(inherited)
    }
    #[cfg(not(unix))]
    {
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "Unix inherited acquisition FD3 required",
        ))
    }
}

#[cfg(all(test, unix))]
mod controls {
    use super::*;
    use std::io::{Read, Write};
    use std::os::fd::{AsFd, AsRawFd};
    use std::os::unix::net::UnixStream;
    use std::time::Duration;

    #[test]
    fn socket_duplicate_drop_preserves_original_and_receiver_eof() {
        let (mut original, receiver) = UnixStream::pair().unwrap();
        receiver
            .set_read_timeout(Some(Duration::from_secs(2)))
            .unwrap();
        let unrelated = File::options().write(true).open("/dev/null").unwrap();
        let mut unrelated_copy = duplicate(unrelated.as_fd()).unwrap();
        let mut output = duplicate(original.as_fd()).unwrap();
        assert_ne!(output.as_raw_fd(), original.as_raw_fd());
        output.write_all(b"one").unwrap();
        drop(output);
        original.write_all(b"two").unwrap();
        unrelated_copy.write_all(b"still-owned").unwrap();
        drop(original);
        let mut data = Vec::new();
        receiver.take(64).read_to_end(&mut data).unwrap();
        assert_eq!(data, b"onetwo");
        // Both independent handles remain usable after stream writers close.
        unrelated_copy.write_all(b"no-double-close").unwrap();
        assert!(unrelated.metadata().is_ok());
    }

    #[test]
    fn pipe_duplicate_drop_preserves_original_and_receiver_eof() {
        let (receiver, mut original) = std::io::pipe().unwrap();
        let mut output = duplicate(original.as_fd()).unwrap();
        output.write_all(b"pipe").unwrap();
        drop(output);
        original.write_all(b"original").unwrap();
        drop(original);
        let mut data = Vec::new();
        receiver.take(64).read_to_end(&mut data).unwrap();
        assert_eq!(data, b"pipeoriginal");
    }

    #[test]
    fn dropping_original_does_not_invalidate_owned_duplicate() {
        let (original, receiver) = UnixStream::pair().unwrap();
        receiver
            .set_read_timeout(Some(Duration::from_secs(2)))
            .unwrap();
        let mut output = duplicate(original.as_fd()).unwrap();
        drop(original);
        output.write_all(b"independent-owner").unwrap();
        drop(output);
        let mut data = Vec::new();
        receiver.take(64).read_to_end(&mut data).unwrap();
        assert_eq!(data, b"independent-owner");
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn original_proc_socket_reopen_is_enxio_but_dup_writes() {
        let (original, receiver) = UnixStream::pair().unwrap();
        receiver
            .set_read_timeout(Some(Duration::from_secs(2)))
            .unwrap();
        let path = format!("/proc/self/fd/{}", original.as_raw_fd());
        let error = File::options().write(true).open(path).unwrap_err();
        assert_eq!(error.raw_os_error(), Some(6));
        let mut output = duplicate(original.as_fd()).unwrap();
        output.write_all(b"dup-not-reopen").unwrap();
        drop(output);
        drop(original);
        let mut data = Vec::new();
        receiver.take(64).read_to_end(&mut data).unwrap();
        assert_eq!(data, b"dup-not-reopen");
    }
}
