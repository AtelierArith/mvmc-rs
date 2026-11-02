//! Linux calling-thread affinity. New worker threads inherit the caller's mask.
use std::{collections::BTreeSet, marker::PhantomData, rc::Rc};

fn current_raw() -> Result<libc::cpu_set_t, String> {
    // cpu_set_t consists of integer mask storage; zero is valid initialized storage.
    let mut mask = unsafe { std::mem::zeroed::<libc::cpu_set_t>() };
    // The writable buffer is a correctly sized cpu_set_t; pid 0 selects this thread.
    let status = unsafe { libc::sched_getaffinity(0, std::mem::size_of_val(&mask), &mut mask) };
    if status != 0 {
        return Err(std::io::Error::last_os_error().to_string());
    }
    Ok(mask)
}

pub(crate) fn current() -> Result<BTreeSet<usize>, String> {
    let mask = current_raw()?;
    Ok((0..libc::CPU_SETSIZE as usize)
        .filter(|&cpu| unsafe { libc::CPU_ISSET(cpu, &mask) })
        .collect())
}

fn set(mask: &libc::cpu_set_t) -> Result<(), String> {
    // The read-only buffer has the exact mask size; only the calling thread is changed.
    let status = unsafe { libc::sched_setaffinity(0, std::mem::size_of_val(mask), mask) };
    if status != 0 {
        return Err(std::io::Error::last_os_error().to_string());
    }
    Ok(())
}

pub(crate) struct ThreadAffinity {
    original: Option<libc::cpu_set_t>,
    // Restoring from a different OS thread would change the wrong affinity.
    _same_thread: PhantomData<Rc<()>>,
}

impl ThreadAffinity {
    pub(crate) fn restrict(cpus: &BTreeSet<usize>) -> Result<Self, String> {
        let allowed = current()?;
        if cpus.is_empty() || !cpus.is_subset(&allowed) {
            return Err(
                "requested CPU affinity must be a nonempty subset of the current mask".into(),
            );
        }
        let original = current_raw()?;
        // A zero cpu_set_t is valid empty mask storage, populated before the syscall.
        let mut mask = unsafe { std::mem::zeroed::<libc::cpu_set_t>() };
        for &cpu in cpus {
            // Every id came from current(), so it fits CPU_SETSIZE.
            unsafe {
                libc::CPU_SET(cpu, &mut mask);
            }
        }
        set(&mask)?;
        Ok(Self {
            original: Some(original),
            _same_thread: PhantomData,
        })
    }

    pub(crate) fn restore(&mut self) -> Result<(), String> {
        if let Some(mask) = &self.original {
            set(mask)?;
        }
        self.original = None;
        Ok(())
    }
}

impl Drop for ThreadAffinity {
    fn drop(&mut self) {
        let _ = self.restore();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn restores_calling_thread_and_rejects_empty_mask() {
        let original = current().unwrap();
        assert!(ThreadAffinity::restrict(&BTreeSet::new()).is_err());
        let first = BTreeSet::from([*original.first().unwrap()]);
        {
            let mut guard = ThreadAffinity::restrict(&first).unwrap();
            assert_eq!(current().unwrap(), first);
            guard.restore().unwrap();
            assert_eq!(current().unwrap(), original);
        }
        assert_eq!(current().unwrap(), original);
    }

    #[test]
    fn restores_nested_masks_and_unwinding_without_crossing_threads() {
        let original = current().unwrap();
        let first = BTreeSet::from([*original.first().unwrap()]);
        let result = std::panic::catch_unwind(|| {
            let _outer = ThreadAffinity::restrict(&first).unwrap();
            {
                let _inner = ThreadAffinity::restrict(&first).unwrap();
                assert_eq!(current().unwrap(), first);
            }
            assert_eq!(current().unwrap(), first);
            panic!("exercise affinity guard unwind");
        });
        assert!(result.is_err());
        assert_eq!(current().unwrap(), original);
    }
}
