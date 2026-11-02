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

thread_local! {
    static POOL_MASKS: std::cell::RefCell<Vec<(u64, ThreadAffinity)>> = const {
        std::cell::RefCell::new(Vec::new())
    };
}

/// Guards stay on their owning OS workers; only the restoration token crosses threads.
pub(crate) struct PoolAffinity {
    pool: &'static rayon::ThreadPool,
    token: u64,
    active: bool,
}

impl PoolAffinity {
    pub(crate) fn restrict_existing(cpus: &BTreeSet<usize>) -> Result<Option<Self>, String> {
        let Some(pool) = crate::threading::existing_inner_pool() else {
            return Ok(None);
        };
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let token = NEXT
            .try_update(
                std::sync::atomic::Ordering::Relaxed,
                std::sync::atomic::Ordering::Relaxed,
                |n| n.checked_add(1),
            )
            .map_err(|_| "CPU affinity token space exhausted")?;
        let mut owner = Self {
            pool,
            token,
            active: true,
        };
        let results = pool.broadcast(|_| {
            let guard = ThreadAffinity::restrict(cpus)?;
            POOL_MASKS.with(|masks| masks.borrow_mut().push((token, guard)));
            Ok::<(), String>(())
        });
        if let Some(error) = results.into_iter().find_map(Result::err) {
            let rollback = owner.restore();
            return Err(format!(
                "inner worker affinity failed: {error}; rollback={rollback:?}"
            ));
        }
        Ok(Some(owner))
    }

    pub(crate) fn restore(&mut self) -> Result<(), String> {
        if !self.active {
            return Ok(());
        }
        let token = self.token;
        let results = self.pool.broadcast(|_| {
            POOL_MASKS.with(|masks| {
                let mut masks = masks.borrow_mut();
                let Some(index) = masks.iter().position(|(id, _)| *id == token) else {
                    return Ok(()); // This worker already restored, or failed to apply.
                };
                if index + 1 != masks.len() {
                    return Err("inner worker affinity guards must restore in reverse order".into());
                }
                masks[index].1.restore()?;
                masks.pop();
                Ok::<(), String>(())
            })
        });
        if let Some(error) = results.into_iter().find_map(Result::err) {
            return Err(error);
        }
        self.active = false;
        Ok(())
    }
}

impl Drop for PoolAffinity {
    fn drop(&mut self) {
        let _ = self.restore();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn restores_existing_pool_after_nested_guards_and_unwind() {
        if std::env::var("MVMC_AFFINITY_POOL_CHILD").as_deref() != Ok("1") {
            let mut child = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "cpu_affinity::tests::restores_existing_pool_after_nested_guards_and_unwind",
                ])
                .env("MVMC_AFFINITY_POOL_CHILD", "1")
                .env("MVMC_RS_INNER_THREADS", "2")
                .spawn()
                .unwrap();
            let started = std::time::Instant::now();
            loop {
                if let Some(status) = child.try_wait().unwrap() {
                    assert!(status.success());
                    return;
                }
                if started.elapsed() > std::time::Duration::from_secs(30) {
                    child.kill().unwrap();
                    child.wait().unwrap();
                    panic!("affinity pool test timed out");
                }
                std::thread::sleep(std::time::Duration::from_millis(10));
            }
        }
        crate::threading::install(|| {
            let pool = crate::threading::existing_inner_pool().unwrap();
            let before = pool.broadcast(|_| current().unwrap());
            assert_eq!(before.len(), 2);
            let mask = BTreeSet::from([*before[0].first().unwrap()]);
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let _outer = PoolAffinity::restrict_existing(&mask).unwrap().unwrap();
                {
                    let _inner = PoolAffinity::restrict_existing(&mask).unwrap().unwrap();
                    assert!(pool
                        .broadcast(|_| current().unwrap())
                        .iter()
                        .all(|v| v == &mask));
                }
                assert!(pool
                    .broadcast(|_| current().unwrap())
                    .iter()
                    .all(|v| v == &mask));
                panic!("exercise existing worker restoration");
            }));
            assert!(result.is_err());
            assert_eq!(pool.broadcast(|_| current().unwrap()), before);
            if before[0].len() >= 2 {
                let other = BTreeSet::from([*before[0].iter().nth(1).unwrap()]);
                // Make the actual affinity reject a subset on one worker only. A successful
                // peer must roll back rather than retain the partially applied mask.
                pool.broadcast(|context| {
                    if context.index() == 0 {
                        let guard = ThreadAffinity::restrict(&mask).unwrap();
                        POOL_MASKS.with(|v| v.borrow_mut().push((u64::MAX, guard)));
                    }
                });
                let mut constrained = PoolAffinity {
                    pool,
                    token: u64::MAX,
                    active: true,
                };
                let split = pool.broadcast(|_| current().unwrap());
                assert!(PoolAffinity::restrict_existing(&other).is_err());
                assert_eq!(pool.broadcast(|_| current().unwrap()), split);
                constrained.restore().unwrap();
                assert_eq!(pool.broadcast(|_| current().unwrap()), before);
            }
        });
    }

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
