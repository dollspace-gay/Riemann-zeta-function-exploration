// Finite Vasyunin formula for the real Nyman--Beurling Gram matrix.
// Long-double evaluation is numerical, not interval arithmetic.
// Formula: Ehm, arXiv:2405.06349, (17) and (30).
#include <cmath>
#include <cstddef>
#include <numeric>
#include <vector>

extern "C" int nb_vasyunin_gram(int size, double* output) noexcept {
    if (size < 1 || size > 8192 || output == nullptr) return 1;
    try {
        const std::size_t stride = static_cast<std::size_t>(size) + 1;
        std::vector<long double> table(stride * stride, 0.0L);
        const long double pi = acosl(-1.0L);
        const long double gamma = 0.577215664901532860606512090082402431L;
        const long double constant = logl(2 * pi) - gamma;
        #pragma omp parallel for schedule(dynamic, 1)
        for (int denominator = 2; denominator <= size; ++denominator) {
            const int half = (denominator - 1) / 2;
            std::vector<long double> cotangents(half + 1);
            for (int k = 1; k <= half; ++k)
                cotangents[k] = 1 / tanl(pi * k / denominator);
            for (int numerator = 1; numerator <= denominator / 2; ++numerator) {
                if (std::gcd(numerator, denominator) != 1) continue;
                long double sum = 0, compensation = 0;
                int residue = 0;
                for (int k = 1; k <= half; ++k) {
                    residue += numerator;
                    if (residue >= denominator) residue -= denominator;
                    // Pair k with denominator-k, avoiding cotangent near pi.
                    const long double term = (2.0L * residue / denominator - 1) * cotangents[k];
                    const long double corrected = term - compensation;
                    const long double next = sum + corrected;
                    compensation = (next - sum) - corrected;
                    sum = next;
                }
                table[denominator * stride + numerator] = sum;
                table[denominator * stride + denominator - numerator] = -sum;
            }
        }
        #pragma omp parallel for schedule(static)
        for (int i = 1; i <= size; ++i) {
            for (int j = i; j <= size; ++j) {
                const int divisor = std::gcd(i, j);
                const int m = i / divisor, n = j / divisor;
                long double value;
                if (m == n) {
                    value = constant / divisor;
                } else {
                    const long double vmn = table[n * stride + m];
                    const long double vnm = table[m * stride + n % m];
                    value = ((m + n) * constant - (m - n) * logl(static_cast<long double>(m) / n)
                             - pi * (vmn + vnm)) / (2.0L * m * n * divisor);
                }
                output[static_cast<std::size_t>(i - 1) * size + j - 1] = static_cast<double>(value);
                output[static_cast<std::size_t>(j - 1) * size + i - 1] = static_cast<double>(value);
            }
        }
        return 0;
    } catch (...) {
        return 2;
    }
}
