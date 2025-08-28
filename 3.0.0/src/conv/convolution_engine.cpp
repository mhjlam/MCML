/*******************************************************************************
 *	Convolution engine implementation for CONV 3.0
 *  Copyright M.H.J. Lam, 2025.
 ****/

#include "convolution_engine.hpp"
#include "math_utils.hpp"
#include "mcml_reader.hpp"
#include "beam_profile.hpp"
#include "binary_tree.hpp"
#include <algorithm>
#include <stdexcept>
#include <chrono>
#include <cmath>
#include <numbers>
#include <functional>

namespace conv
{

// Mathematical constants
constexpr double GAUSSIAN_LIMIT_FACTOR = 4.0;
constexpr double BESSEL_PRECISION = 1e-15;

// Helper functions in detail namespace
namespace detail {

/**
 * @brief Optimal radius calculation for cylindrical coordinates
 * @param ir Radial index
 * @param dr Radial spacing
 * @return Optimal radius value
 */
[[nodiscard]] double optimal_radius(size_t ir, double dr) noexcept {
    return (static_cast<double>(ir) + 0.5) * dr;
}

/**
 * @brief Flat beam area overlap factor
 * @param rc Center radius
 * @param r Sample radius  
 * @param R Beam radius
 * @return Area overlap coefficient [0,1]
 */
[[nodiscard]] double flat_beam_area(double rc, double r, double R) noexcept {
    if (r > R) return 0.0;
    
    const double r_diff = std::abs(rc - r);
    if (r_diff >= R) return 0.0;
    
    // Simplified area overlap calculation
    return 1.0 - (r_diff / R);
}

/**
 * @brief Exponentially scaled modified Bessel function
 * @param x Argument
 * @return exp(-x) * I_0(x) for high precision at large x
 */
[[nodiscard]] double exp_bessel_i0(double x) noexcept {
    if (x <= 0.0) return 1.0;
    
    // For small x, use direct calculation
    if (x < 3.0) {
        const double bessel = bessel_i0(x);
        return std::exp(-x) * bessel;
    }
    
    // For larger x, use asymptotic approximation to avoid overflow
    const double term = 1.0 / std::sqrt(2.0 * std::numbers::pi * x);
    
    // Leading correction terms in the asymptotic expansion
    const double x_inv = 1.0 / x;
    const double x_inv2 = x_inv * x_inv;
    const double correction = 1.0 + x_inv / 8.0 - 9.0 * x_inv2 / 128.0;
    
    return term * correction;
}

/**
 * @brief Linear interpolation for impulse response
 * @param grid_points Grid data points
 * @param r Target radius
 * @param dr Grid spacing  
 * @return Interpolated value
 */
[[nodiscard]] double interpolate_impulse_response(const std::vector<double>& grid_points, 
                                                double r, double dr) noexcept {
    if (grid_points.empty()) return 0.0;
    
    const double r_index = r / dr;
    const size_t i0 = static_cast<size_t>(r_index);
    
    if (i0 >= grid_points.size() - 1) {
        return grid_points.back();
    }
    
    const double frac = r_index - static_cast<double>(i0);
    return grid_points[i0] * (1.0 - frac) + grid_points[i0 + 1] * frac;
}

} // namespace detail

void ConvolutionEngine::set_parameters(const ConvolutionParams& params) {
    if (!params.is_valid()) {
        throw std::invalid_argument("Invalid convolution parameters");
    }
    m_params = params;
}

void ConvolutionEngine::clear_cache() {
    m_bessel_cache.clear();
    m_exp_bessel_cache.clear();
}

size_t ConvolutionEngine::cache_memory_usage() const noexcept {
    return m_bessel_cache.memory_usage() + m_exp_bessel_cache.memory_usage();
}

ConvolutionResult ConvolutionEngine::convolve_reflectance(const McmlInputData& input_data,
                                                         const McmlOutputData& output_data,
                                                         const BeamProfile& beam,
                                                         const GridParameters& grid_params) {
    // Convert McmlInputData to GridParameters for compatibility
    GridParameters input_grid;
    input_grid.nr = input_data.nr;
    input_grid.dr = input_data.dr;
    
    return perform_convolution(output_data.rd_ra, beam, input_grid, grid_params);
}

ConvolutionResult ConvolutionEngine::convolve_transmittance(const McmlInputData& input_data,
                                                           const McmlOutputData& output_data,
                                                           const BeamProfile& beam,
                                                           const GridParameters& grid_params) {
    // Convert McmlInputData to GridParameters for compatibility
    GridParameters input_grid;
    input_grid.nr = input_data.nr;
    input_grid.dr = input_data.dr;
    
    return perform_convolution(output_data.td_ra, beam, input_grid, grid_params);
}

ConvolutionResult ConvolutionEngine::convolve_absorption(const McmlInputData& input_data,
                                                        const McmlOutputData& output_data,
                                                        const BeamProfile& beam,
                                                        const GridParameters& grid_params) {
    // Convert McmlInputData to GridParameters for compatibility  
    GridParameters input_grid;
    input_grid.nr = input_data.nr;
    input_grid.dr = input_data.dr;
    
    return perform_convolution(output_data.a_rz, beam, input_grid, grid_params);
}

ConvolutionResult ConvolutionEngine::perform_convolution(const Matrix2D<double>& impulse_response,
                                                        const BeamProfile& beam,
                                                        const GridParameters& input_grid,
                                                        const GridParameters& output_grid) {
    const auto start_time = std::chrono::high_resolution_clock::now();
    
    // Prepare result structure
    ConvolutionResult result;
    result.convolved_data = Matrix2D<double>(output_grid.nr, impulse_response.cols());
    
    const BeamParameters& beam_params = beam.parameters();
    const double P = beam_params.total_power;
    const double R = beam_params.radius;
    
    // Main convolution loop
    for (size_t iz = 0; iz < impulse_response.cols(); ++iz) {
        for (size_t ir = 0; ir < output_grid.nr; ++ir) {
            const double rc = detail::optimal_radius(ir, output_grid.dr);
            
            double convolved_value = 0.0;
            
            // Select convolution method based on beam profile
            switch (beam_params.type) {
                case BeamType::Flat:
                    convolved_value = compute_flat_convolution(rc, iz, impulse_response, 
                                                             input_grid, P, R);
                    break;
                    
                case BeamType::Gaussian:
                    convolved_value = compute_gaussian_convolution(rc, iz, impulse_response,
                                                                 input_grid, P, R);
                    break;
                    
                case BeamType::Arbitrary:
                    convolved_value = compute_arbitrary_convolution(rc, iz, impulse_response,
                                                                  input_grid, beam_params);
                    break;
                    
                default:
                    convolved_value = impulse_response(ir, iz); // No convolution
                    break;
            }
            
            result.convolved_data(ir, iz) = convolved_value;
        }
    }
    
    // Record computation time
    const auto end_time = std::chrono::high_resolution_clock::now();
    const auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    result.computation_time = static_cast<double>(duration.count()) / 1e6;
    
    result.converged = true;
    result.iterations_used = output_grid.nr * impulse_response.cols();
    
    return result;
}

double ConvolutionEngine::compute_flat_convolution(double rc, size_t ix,
                                                  const Matrix2D<double>& impulse_response,
                                                  const GridParameters& input_grid,
                                                  double P, double R) {
    double sum = 0.0;
    
    // Integrate over beam area with cylindrical coordinates
    for (size_t ir = 0; ir < input_grid.nr; ++ir) {
        const double r = detail::optimal_radius(ir, input_grid.dr);
        
        // Only include points within beam radius
        if (r <= R) {
            const double area_factor = detail::flat_beam_area(rc, r, R);
            const double impulse_val = interpolate_impulse_response(r, ix, impulse_response, input_grid);
            
            // Add contribution with proper area weighting
            sum += area_factor * impulse_val * r * input_grid.dr;
        }
    }
    
    // Normalize by beam area: π R²
    const double beam_area = std::numbers::pi * R * R;
    return (4.0 * P / (R * R)) * sum / beam_area;
}

double ConvolutionEngine::compute_gaussian_convolution(double rc, size_t ix,
                                                      const Matrix2D<double>& impulse_response,
                                                      const GridParameters& input_grid,
                                                      double P, double R) {
    double sum = 0.0;
    
    // Integration limit for Gaussian beam (4σ ≈ 2R)
    const double r_max = GAUSSIAN_LIMIT_FACTOR * R;
    
    // Gaussian beam profile: I(r) = (2P/πR²) * exp(-2r²/R²)
    const double gauss_coeff = 2.0 * P / (std::numbers::pi * R * R);
    
    for (size_t ir = 0; ir < input_grid.nr; ++ir) {
        const double r = detail::optimal_radius(ir, input_grid.dr);
        
        if (r <= r_max) {
            // Use exp_bessel function for high precision
            const double bessel_arg = 4.0 * rc * r / (R * R);
            const double exp_arg = 2.0 * (rc * rc + r * r) / (R * R);
            const double bessel_factor = detail::exp_bessel_i0(bessel_arg);
            const double gauss_weight = std::exp(-exp_arg) * bessel_factor;
            
            const double impulse_val = interpolate_impulse_response(r, ix, impulse_response, input_grid);
            
            // Add contribution with Gaussian weighting
            sum += gauss_weight * impulse_val * r * input_grid.dr;
        }
    }
    
    return gauss_coeff * sum;
}

double ConvolutionEngine::compute_arbitrary_convolution(double /* rc */, size_t ix,
                                                       const Matrix2D<double>& impulse_response,
                                                       const GridParameters& input_grid,
                                                       const BeamParameters& beam_params) {
    // Use the power densities from the arbitrary beam profile
    if (beam_params.radii.size() != beam_params.power_densities.size() || 
        beam_params.radii.empty()) {
        return 0.0; // Invalid beam profile
    }
    
    double sum = 0.0;
    
    // Integration over the beam profile
    for (size_t ir = 0; ir < input_grid.nr; ++ir) {
        const double r = detail::optimal_radius(ir, input_grid.dr);
        
        // Interpolate beam profile at radius r
        double beam_intensity = 0.0;
        for (size_t i = 0; i < beam_params.radii.size() - 1; ++i) {
            if (r >= beam_params.radii[i] && r <= beam_params.radii[i + 1]) {
                const double frac = (r - beam_params.radii[i]) / 
                                  (beam_params.radii[i + 1] - beam_params.radii[i]);
                beam_intensity = beam_params.power_densities[i] * (1.0 - frac) +
                               beam_params.power_densities[i + 1] * frac;
                break;
            }
        }
        
        if (beam_intensity > 0.0) {
            const double impulse_val = interpolate_impulse_response(r, ix, impulse_response, input_grid);
            sum += beam_intensity * impulse_val * r * input_grid.dr;
        }
    }
    
    return sum;
}

double ConvolutionEngine::interpolate_impulse_response(double r, size_t ix,
                                                      const Matrix2D<double>& impulse_response,
                                                      const GridParameters& input_grid) {
    if (impulse_response.empty() || ix >= impulse_response.cols()) {
        return 0.0;
    }
    
    const double r_index = r / input_grid.dr;
    const size_t ir_base = static_cast<size_t>(r_index);
    
    if (ir_base >= impulse_response.rows() - 1) {
        return impulse_response(impulse_response.rows() - 1, ix);
    }
    
    // Linear interpolation
    const double frac = r_index - static_cast<double>(ir_base);
    const double val1 = impulse_response(ir_base, ix);
    const double val2 = impulse_response(ir_base + 1, ix);
    
    return val1 * (1.0 - frac) + val2 * frac;
}

double ConvolutionEngine::compute_convolution_point(double rc, size_t ix,
                                                   const Matrix2D<double>& impulse_response,
                                                   const BeamProfile& beam,
                                                   const GridParameters& input_grid) {
    const BeamParameters& beam_params = beam.parameters();
    const double P = beam_params.total_power;
    const double R = beam_params.radius;
    
    switch (beam_params.type) {
        case BeamType::Flat:
            return compute_flat_convolution(rc, ix, impulse_response, input_grid, P, R);
            
        case BeamType::Gaussian:
            return compute_gaussian_convolution(rc, ix, impulse_response, input_grid, P, R);
            
        case BeamType::Arbitrary:
            return compute_arbitrary_convolution(rc, ix, impulse_response, input_grid, beam_params);
            
        default:
            // No convolution - return original data with interpolation
            return interpolate_impulse_response(rc, ix, impulse_response, input_grid);
    }
}

// Caching Bessel functions implementation
double ConvolutionEngine::bessel_i0(double x) {
    if (m_params.use_caching) {
        auto cached = m_bessel_cache.find(x);
        if (cached.has_value()) {
            return cached.value();
        }
        
        const double result = conv::bessel_i0(x);
        m_bessel_cache.insert(x, result);
        return result;
    }
    return conv::bessel_i0(x);
}

double ConvolutionEngine::exp_bessel_i0(double x, double y) {
    if (m_params.use_caching) {
        const double key = x * 1e6 + y; // Simple composite key
        auto cached = m_exp_bessel_cache.find(key);
        if (cached.has_value()) {
            return cached.value();
        }
        
        const double result = std::exp(x) * conv::bessel_i0(y);
        m_exp_bessel_cache.insert(key, result);
        return result;
    }
    return std::exp(x) * conv::bessel_i0(y);
}

// Integration methods implementation
double ConvolutionEngine::adaptive_simpson(const std::function<double(double)>& integrand, 
                                          double a, double b, double tolerance) {
    const size_t initial_intervals = 8;
    double prev_result = simpson_rule(integrand, a, b, initial_intervals);
    
    for (size_t n = initial_intervals * 2; n <= m_params.max_iterations; n *= 2) {
        const double new_result = simpson_rule(integrand, a, b, n);
        const double error = std::abs(new_result - prev_result);
        const double relative_error = error / std::max(std::abs(new_result), 1e-15);
        
        if (relative_error < tolerance) {
            return new_result;
        }
        
        prev_result = new_result;
    }
    
    // Return best estimate if not converged
    return prev_result;
}

double ConvolutionEngine::simpson_rule(const std::function<double(double)>& integrand, 
                                      double a, double b, size_t n) {
    if (n % 2 != 0) n++; // Ensure even number of intervals
    
    const double h = (b - a) / n;
    double sum = integrand(a) + integrand(b);
    
    // Add even-indexed terms (coefficient 2)
    for (size_t i = 2; i < n; i += 2) {
        sum += 2.0 * integrand(a + i * h);
    }
    
    // Add odd-indexed terms (coefficient 4)  
    for (size_t i = 1; i < n; i += 2) {
        sum += 4.0 * integrand(a + i * h);
    }
    
    return sum * h / 3.0;
}

double ConvolutionEngine::interpolate_2d(const Matrix2D<double>& matrix, 
                                        double r_idx, size_t x_idx) const {
    if (matrix.empty() || x_idx >= matrix.cols()) {
        return 0.0;
    }
    
    const size_t ir_base = static_cast<size_t>(r_idx);
    
    if (ir_base >= matrix.rows() - 1) {
        return matrix(matrix.rows() - 1, x_idx);
    }
    
    // Linear interpolation in r direction
    const double frac = r_idx - static_cast<double>(ir_base);
    const double val1 = matrix(ir_base, x_idx);
    const double val2 = matrix(ir_base + 1, x_idx);
    
    return val1 * (1.0 - frac) + val2 * frac;
}

bool ConvolutionEngine::validate_inputs(const McmlInputData& input_data,
                                       const McmlOutputData& output_data,
                                       const GridParameters& grid_params) const noexcept {
    // Check input data validity
    if (!input_data.is_valid() || !output_data.is_valid()) {
        return false;
    }
    
    // Check grid parameters
    if (!grid_params.is_valid() || grid_params.nr == 0 || grid_params.dr <= 0.0) {
        return false;
    }
    
    // Check compatibility between input and output data
    if (input_data.nr == 0 || input_data.dr <= 0.0) {
        return false;
    }
    
    // Check that matrices have appropriate sizes
    if (output_data.rd_ra.empty() && output_data.td_ra.empty() && output_data.a_rz.empty()) {
        return false; // No data to convolve
    }
    
    return true;
}

} // namespace conv
