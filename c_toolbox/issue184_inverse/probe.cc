// Developer-only adapter. Numerical bodies are included unchanged from
// mVMC-1.3.0 (MPL-2.0); see README.md. Never invoked by Cargo.
#include <iomanip>
#include <iostream>
#include <vector>
#include "invert.tcc"

template <typename T> void run(int n) {
    std::vector<T> a(n*n), m(n*n, T(17)), vt(n-1, T(19));
    std::vector<int> piv(n);
    for (auto &x : a) {
        double re, im;
        if (!(std::cin >> re >> im)) throw std::runtime_error("short A");
        if constexpr (std::is_same_v<T, double>) {
            if (im != 0) throw std::runtime_error("imaginary real input");
            x = re;
        } else x = T(re, im);
    }
    for (auto &p : piv) if (!(std::cin >> p)) throw std::runtime_error("short pivot");
    utu2inv<T>(n, a.data(), n, piv.data(), vt.data(), m.data(), n);
    std::cout << std::setprecision(17);
    for (auto *v : {&a, &m, &vt}) for (auto x : *v)
        std::cout << std::real(x) << ' ' << std::imag(x) << '\n';
    for (auto p : piv) std::cout << p << '\n';
}
int main() {
    char kind; int n;
    if (!(std::cin >> kind >> n) || (n != 4 && n != 6)) return 2;
    if (kind == 'r') run<double>(n);
    else if (kind == 'c') run<std::complex<double>>(n);
    else return 2;
}
