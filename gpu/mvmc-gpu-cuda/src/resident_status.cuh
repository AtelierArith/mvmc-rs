// Sticky resident status word (#573). The first failing stage wins via
// atomicCAS(error, 0, code); later kernels observe a nonzero word and return.
// Keep these values identical to `ResidentFailure::from_status` in
// src/interval/resident.rs (checked by a host unit test).
#ifndef MVMC_RESIDENT_STATUS
#define MVMC_RESIDENT_STATUS
enum {
  RESIDENT_REBUILD = 1,      // singular/nonfinite QP plane or zero/nonfinite overlap
  RESIDENT_ENERGY = 2,       // nonfinite local energy or observable
  RESIDENT_SR = 3,           // SR Cholesky failure
  RESIDENT_PARAMETERS = 4,   // nonfinite parameter update
  RESIDENT_SCHEDULE = 5,     // a walker did not finish its scheduled hops
  RESIDENT_DERIVATIVES = 6,  // nonfinite log-derivative
  RESIDENT_CG = 7,           // nonfinite SR-CG increment (C StcOpt info=1)
};
#endif
