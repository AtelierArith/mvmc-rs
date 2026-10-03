inline double xdot(const int n, const double * const p, const double * const q) {
  int i;
  double z=0;
  #pragma loop noalias
  for(i=0;i<n;i++) {
    z += p[i]*q[i];
  }
  return z;
}
