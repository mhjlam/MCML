/*******************************************************************************
 *	Mathematical utility functions header for CONV 3.0
 *  Copyright M.H.J. Lam, 2025.
 ****/

#pragma once

namespace conv
{

/**
 * @brief Modified Bessel function of the first kind, order 0: I₀(x)
 * @param x Input argument
 * @return I₀(x) 
 * 
 * High-precision implementation using series expansion for small x
 * and asymptotic expansion for large x, with C++20 constexpr optimizations.
 */
[[nodiscard]] double bessel_i0(double x) noexcept;

/**
 * @brief Optimized exponential-Bessel function for Gaussian beam convolution
 * @param r Radius coordinate [cm]
 * @param r2 Secondary radius coordinate [cm] 
 * @param R Beam radius [cm]
 * @return exp(-y + x) * I₀(x) where x=4rr₂/R², y=2(r²+r₂²)/R²
 */
[[nodiscard]] double exp_bessel_i0(double r, double r2, double R) noexcept;

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
[[nodiscard]] double grid_overlap_coefficient(double r, double r2, double dr) noexcept;

} // namespace conv
