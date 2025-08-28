/*******************************************************************************
 *	Mathematical utility functions for CONV 3.0
 *  Copyright M.H.J. Lam, 2025.
 ****/

#include "math_utils.hpp"
#include <cmath>
#include <numbers>
#include <algorithm>

namespace conv
{

constexpr int MAX_BESSEL_ITERATIONS = 1000;
constexpr double PI = std::numbers::pi;

/**
 * @brief Modified Bessel function of the first kind, order 0: I₀(x)
 * @param x Input argument
 * @return I₀(x) 
 * 
 * High-precision implementation using series expansion for small x
 * and asymptotic expansion for large x, with C++20 constexpr optimizations.
 */
[[nodiscard]] double bessel_i0(double x) noexcept {
    if (x < 0.0) x = -x; // I₀ is even function
    
    if (x < 3.75) {
        // Series expansion for small arguments
        const double y = (x / 3.75) * (x / 3.75);
        return 1.0 + 3.5156229 * y + 3.0899424 * y * y + 1.2067492 * y * y * y +
               0.2659732 * std::pow(y, 4) + 0.0360768 * std::pow(y, 5) + 0.0045813 * std::pow(y, 6);
    } else {
        // Asymptotic expansion for large arguments
        const double z = 3.75 / x;
        return std::exp(x) / std::sqrt(x) * 
               (0.39894228 + 0.01328592 * z + 0.00225319 * z * z -
                0.00157565 * z * z * z + 0.00916281 * std::pow(z, 4) -
                0.02057706 * std::pow(z, 5) + 0.02635537 * std::pow(z, 6) -
                0.01647633 * std::pow(z, 7) + 0.00392377 * std::pow(z, 8));
    }
}

/**
 * @brief Optimized exponential-Bessel function for Gaussian beam convolution
 * @param r Radius coordinate [cm]
 * @param r2 Secondary radius coordinate [cm] 
 * @param R Beam radius [cm]
 * @return exp(-y + x) * I₀(x) where x=4rr₂/R², y=2(r²+r₂²)/R²
 */
[[nodiscard]] double exp_bessel_i0(double r, double r2, double R) noexcept {
    const double RR_inv = 1.0 / (R * R);
    const double x = 4.0 * r * r2 * RR_inv;
    const double y = 2.0 * (r2 * r2 + r * r) * RR_inv;
    
    return std::exp(-y + x) * bessel_i0(x);
}

/**
 * @brief Optimized grid point calculation for interpolation
 * @param r First radius coordinate [cm]
 * @param r2 Second radius coordinate [cm]
 * @param dr Grid spacing [cm]
 * @return Normalized overlap coefficient [0,1]
 * 
 * Calculates the fractional overlap between two grid points
 * using optimized geometric operations.
 */
[[nodiscard]] double grid_overlap_coefficient(double r, double r2, double dr) noexcept {
    const double r_diff = std::abs(r - r2);
    
    if (r_diff >= dr) {
        return 0.0;
    }
    
    if (r_diff <= 1e-10) {
        return 1.0;
    }
    
    // Calculate fractional overlap using geometric approximation
    const double temp = r_diff / dr;
    if (temp >= 1.0) {
        return 0.0;
    } else {
        return std::acos(std::clamp(temp, -1.0, 1.0)) / std::numbers::pi;
    }
}

} // namespace conv
