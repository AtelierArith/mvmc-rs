//! Literal Julia MPI87 operands, with C operator/formula authority.
//! Provenance: docs/reference/c-to-julia/verification/issue-179-literal-operator.md.
#![cfg(feature = "mpi")]

use mvmc_core::sr_cg::{install_cg_observer, CgObserver, CgProductPhase, SampledSrOperator};
use mvmc_core::{mpi::MpiContext, Reducer};
use std::{cell::RefCell, rc::Rc};

type Product = (CgProductPhase, Vec<f64>, Vec<f64>);

#[derive(Default)]
struct Products(RefCell<Vec<Product>>);

impl CgObserver for Products {
    fn product(&self, phase: CgProductPhase, search: &[f64], product: &[f64]) {
        self.0
            .borrow_mut()
            .push((phase, search.to_vec(), product.to_vec()));
    }
}

fn close(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (&a, &b) in actual.iter().zip(expected) {
        // Two components, two samples, two ranks; max raw product magnitude 97.
        // Absolute 1e-12 covers this short BLAS/reduction/correction path only,
        // not a solver forward error or sampling-trajectory allowance.
        assert!(a.is_finite() && (a - b).abs() <= 1e-12, "{a} != {b}");
    }
}

#[test]
#[ignore = "requires an isolated mpiexec -n 2 process"]
fn issue179_literal_two_rank_cg_products() {
    let world = MpiContext::initialize().unwrap();
    assert_eq!(
        world.world_size(),
        2,
        "this literal fixture requires two ranks"
    );
    let rank = world.rank();
    for complex in [false, true] {
        let mut operator = SampledSrOperator::new(2, 2, complex);
        // Column-major: Julia matrices [1 2; 3 4] and [5 6; 7 8].
        operator.real_samples.copy_from_slice(if rank == 0 {
            &[1.0, 3.0, 2.0, 4.0]
        } else {
            &[5.0, 7.0, 6.0, 8.0]
        });
        if complex {
            operator.imag_samples.copy_from_slice(if rank == 0 {
                &[0.5, 1.5, -1.0, 0.25]
            } else {
                &[-0.5, 0.75, 1.25, -1.5]
            });
        }
        operator.mean.copy_from_slice(&[0.25, -0.5]);
        operator.diagonal.copy_from_slice(&[2.0, 3.0]);
        let mut search = if rank == 0 {
            [0.5, -1.0]
        } else {
            [999.0, 999.0]
        };
        let mut product = [f64::NAN; 2];
        let observed = Rc::new(Products::default());
        let guard = install_cg_observer(observed.clone()).unwrap();
        operator
            .apply_with_reducer(&mut product, &mut search, 0.25, 0.1, &world)
            .unwrap();
        drop(guard);
        assert_eq!(
            search,
            [0.5, -1.0],
            "root input must replace non-root poison"
        );
        let expected = if complex {
            [-14.4859375, -24.2375]
        } else {
            [-15.30625, -22.7375]
        };
        close(&product, &expected);
        let records = observed.0.borrow();
        assert_eq!(records.len(), 4);
        for (record, phase) in records.iter().zip([
            CgProductPhase::RootSearch,
            CgProductPhase::Local,
            CgProductPhase::Global,
            CgProductPhase::Corrected,
        ]) {
            assert_eq!(record.0, phase);
            assert_eq!(record.1, [0.5, -1.0]);
        }
        assert!(records[0].2.is_empty());
        let mut local = if rank == 0 {
            [-8.5, -19.5]
        } else {
            [-52.5, -71.5]
        };
        if complex {
            let imaginary = if rank == 0 {
                [0.125, -2.0625]
            } else {
                [3.15625, -3.9375]
            };
            for i in 0..2 {
                local[i] += imaginary[i];
            }
        }
        close(&records[1].2, &local);
        close(
            &records[2].2,
            if complex {
                &[-57.71875, -97.0]
            } else {
                &[-61.0, -91.0]
            },
        );
        close(&records[3].2, &expected);
        println!(
            "LITERAL_CG rank={rank} world=2 complex={complex} x={search:?} product={product:?}"
        );
    }
}
