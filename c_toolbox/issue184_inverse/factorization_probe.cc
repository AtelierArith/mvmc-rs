// Optional standalone original-C kernel acquisition; no Cargo consumer.
// invert.tcc retains its upstream MPL-2.0 notice. Fortran sources are
// compiled unchanged from mVMC-1.3.0; see factorization-README.md.
#include <iomanip>
#include <iostream>
#include <vector>
#include "invert.tcc"
// Use a separate C++ identifier for the actual LP64 Fortran ABI. Upstream
// blalink_fort.h declares this symbol with BLIS dim_t (host long), whereas
// these unchanged default-integer Fortran objects take 32-bit integers.
extern "C" void issue184_zsktrf(const char*, const char*, const int*,
    std::complex<double>*, const int*, int*, std::complex<double>*, const int*, int*) asm("zsktrf_");
int main() {
    int n;
    if (!(std::cin >> n) || n != 6) return 2;
    using Z = std::complex<double>;
    std::vector<Z> a(n*n), work(n*n), m(n*n,Z(17)), vt(n-1,Z(19));
    std::vector<int> piv(n);
    for (auto &z : a) {
        double re,im;
        if (!(std::cin >> re >> im)) return 2;
        z=Z(re,im);
    }
    std::string extra;
    if (std::cin >> extra) return 2;
    const int lwork=n*n; int info=-999;
    issue184_zsktrf("U","N",&n,a.data(),&n,piv.data(),work.data(),&lwork,&info);
    std::cout << std::setprecision(17) << info << '\n';
    if (info != 0) return 3;
    for (auto z : a) std::cout << z.real() << ' ' << z.imag() << '\n';
    for (auto p : piv) std::cout << p << '\n';
    const auto before=piv;
    utu2inv<Z>(n,a.data(),n,piv.data(),vt.data(),m.data(),n);
    if (piv!=before) return 4;
    for (auto *v : {&a,&m,&vt}) for(auto z : *v)
        std::cout << z.real() << ' ' << z.imag() << '\n';
    for (auto p : piv) std::cout << p << '\n';
}
