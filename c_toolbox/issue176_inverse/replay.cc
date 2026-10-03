// Original diagnostic driver; invokes unchanged upstream MPL-2.0 routines.
// Optional developer check only: never built or invoked by Cargo.
#include <complex>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <cmath>
extern "C" void utu2inv_d(int,double*,int,int*,double*,double*,int);
extern "C" void utu2inv_z(int,std::complex<double>*,int,int*,std::complex<double>*,std::complex<double>*,int);
static void show(const char *label,double x) {
  uint64_t bits; std::memcpy(&bits,&x,sizeof bits);
  std::printf("%s %.17g %016llx\n",label,x,(unsigned long long)bits);
}
int main() {
  // setup.txt header 2 1 1 0: indices [0,1], spins [1,1], QP1.
  // Slater S[2,3]=a. Column-major assembled A=[0,-a,a,0].
  // Already-factorized n=2 upper skew matrix, pivots [1,2].
  const double a=0.5535714285714286;
  double real[4]={0,-a,a,0}, vt[1], work[4]; int piv[2]={1,2};
  std::complex<double> z[4]={{0,0},{-a,0},{a,0},{0,0}}, zvt[1], zw[4];
  show("input-upper",a);
  utu2inv_d(2,real,2,piv,vt,work,2);
  utu2inv_z(2,z,2,piv,zvt,zw,2);
  show("real-vT",vt[0]); show("complex-vT",zvt[0].real());
  show("real-post-sign-row1-col0",-real[1]);
  show("complex-post-sign-row1-col0",-z[1].real());
  show("complex-imag-row1-col0",-z[1].imag());
  show("real-inverse-residual",a*real[1]-1);
  show("complex-inverse-residual",a*z[1].real()-1);
  // Analytic 2x2 inverse: no cross-platform computed-bit acceptance gate.
  return !std::isfinite(real[1]) || !std::isfinite(z[1].real()) ||
         std::abs(a*real[1]-1)>8e-16 ||
         std::abs(a*z[1].real()-1)>8e-16 || z[1].imag()!=0;
}
