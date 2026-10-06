//! pfapack per-plane reference backend.

use rayon::prelude::*;

use crate::{BatchOutput, PfScalar, PlaneStatus};

/// Per-plane scratch reused across planes of one task.
pub struct PlaneWorkspace<T> {
    pub(crate) pivots: Vec<pfapack::PivotIndex1Based>,
    pub(crate) vt: Vec<T>,
    pub(crate) m: Vec<T>,
}

impl<T: PfScalar> PlaneWorkspace<T> {
    pub fn new(n: usize) -> Self {
        Self {
            pivots: vec![pfapack::PivotIndex1Based(0); n],
            vt: vec![T::ZERO; n - 1],
            m: vec![T::ZERO; n * n],
        }
    }
}

pub(crate) fn run<T: PfScalar>(
    planes: &[T],
    n: usize,
    count: usize,
    parallel: bool,
) -> BatchOutput<T> {
    let nn = n * n;
    let mut inv = planes.to_vec();
    let mut pf = vec![T::ZERO; count];
    let mut status = vec![PlaneStatus::Ok; count];
    if parallel {
        inv.par_chunks_mut(nn)
            .zip(pf.par_iter_mut())
            .zip(status.par_iter_mut())
            .for_each_init(
                || PlaneWorkspace::<T>::new(n),
                |ws, ((plane, pf), status)| {
                    let (value, st) = T::pfapack_plane(plane, n, ws);
                    *pf = value;
                    *status = st;
                },
            );
    } else {
        let mut ws = PlaneWorkspace::<T>::new(n);
        for ((plane, pf), status) in inv.chunks_mut(nn).zip(pf.iter_mut()).zip(status.iter_mut()) {
            let (value, st) = T::pfapack_plane(plane, n, &mut ws);
            *pf = value;
            *status = st;
        }
    }
    (pf, inv, status)
}
