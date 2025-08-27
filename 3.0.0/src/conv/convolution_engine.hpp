/*******************************************************************************
 *	Convolution engine for finite-size beam processing in CONV 3.0
 *  Copyright M.H.J. Lam, 2025.
 ****/

#pragma once

#include "beam_profile.hpp"
#include "binary_tree.hpp"
#include "conv.hpp"
#include "matrix.hpp"

#include <functional>

namespace conv
{

// Forward declarations
struct McmlInputData;
struct McmlOutputData;

/**
 * @brief Configuration for convolution computation
 */
struct ConvolutionParams {
	double epsilon = DEFAULT_EPSILON;     // Relative error tolerance
	size_t max_iterations = 10000;        // Maximum integration steps
	double integration_tolerance = 1e-12; // Integration convergence tolerance
	bool use_caching = true;              // Enable result caching
	bool adaptive_integration = true;     // Use adaptive step size

	[[nodiscard]] bool is_valid() const noexcept;
};

/**
 * @brief Result of convolution computation
 */
struct ConvolutionResult {
	Matrix2D<double> convolved_data; // Result matrix
	double estimated_error = 0.0;    // Estimated relative error
	size_t iterations_used = 0;      // Number of integration steps
	double computation_time = 0.0;   // Time taken [seconds]
	bool converged = false;          // Convergence status

	[[nodiscard]] bool is_valid() const noexcept { return convolved_data.size() > 0; }
};

/**
 * @brief Engine for performing convolution of MCML data with finite beam profiles
 */
class ConvolutionEngine
{
public:
	ConvolutionEngine() = default;
	~ConvolutionEngine() = default;

	// Non-copyable but movable
	ConvolutionEngine(const ConvolutionEngine&) = delete;
	ConvolutionEngine& operator=(const ConvolutionEngine&) = delete;
	ConvolutionEngine(ConvolutionEngine&&) = default;
	ConvolutionEngine& operator=(ConvolutionEngine&&) = default;

	/**
	 * @brief Set convolution parameters
	 * @param params Convolution configuration
	 */
	void set_parameters(const ConvolutionParams& params);

	/**
	 * @brief Perform convolution for reflectance data
	 * @param input_data Original MCML input parameters
	 * @param output_data Original MCML results
	 * @param beam Beam profile for convolution
	 * @param grid_params Output grid parameters
	 * @return Convolution result
	 * @throws ConvolutionException if computation fails
	 */
	[[nodiscard]] ConvolutionResult convolve_reflectance(const McmlInputData& input_data,
														 const McmlOutputData& output_data, const BeamProfile& beam,
														 const GridParameters& grid_params);

	/**
	 * @brief Perform convolution for transmittance data
	 * @param input_data Original MCML input parameters
	 * @param output_data Original MCML results
	 * @param beam Beam profile for convolution
	 * @param grid_params Output grid parameters
	 * @return Convolution result
	 * @throws ConvolutionException if computation fails
	 */
	[[nodiscard]] ConvolutionResult convolve_transmittance(const McmlInputData& input_data,
														   const McmlOutputData& output_data, const BeamProfile& beam,
														   const GridParameters& grid_params);

	/**
	 * @brief Perform convolution for absorption data
	 * @param input_data Original MCML input parameters
	 * @param output_data Original MCML results
	 * @param beam Beam profile for convolution
	 * @param grid_params Output grid parameters
	 * @return Convolution result
	 * @throws ConvolutionException if computation fails
	 */
	[[nodiscard]] ConvolutionResult convolve_absorption(const McmlInputData& input_data,
														const McmlOutputData& output_data, const BeamProfile& beam,
														const GridParameters& grid_params);

	/**
	 * @brief Get current parameters
	 * @return Current convolution parameters
	 */
	[[nodiscard]] const ConvolutionParams& parameters() const noexcept { return m_params; }

	/**
	 * @brief Clear computation caches
	 */
	void clear_cache();

	/**
	 * @brief Get cache statistics
	 * @return Cache memory usage in bytes
	 */
	[[nodiscard]] size_t cache_memory_usage() const noexcept;

private:
	/**
	 * @brief Generic convolution implementation
	 * @param impulse_response 2D impulse response matrix
	 * @param beam Beam profile
	 * @param input_grid Input grid parameters
	 * @param output_grid Output grid parameters
	 * @return Convolution result
	 */
	[[nodiscard]] ConvolutionResult perform_convolution(const Matrix2D<double>& impulse_response,
														const BeamProfile& beam, const GridParameters& input_grid,
														const GridParameters& output_grid);

	/**
	 * @brief Compute convolution integral at a single point
	 * @param rc Output radius coordinate
	 * @param ix Output x-coordinate index
	 * @param impulse_response Input impulse response
	 * @param beam Beam profile
	 * @param input_grid Input grid parameters
	 * @return Convolved value at (rc, ix)
	 */
	[[nodiscard]] double compute_convolution_point(double rc, size_t ix, const Matrix2D<double>& impulse_response,
												   const BeamProfile& beam, const GridParameters& input_grid);

	/**
	 * @brief Compute flat beam convolution at point
	 * @param rc Output radius coordinate
	 * @param ix Output x-coordinate index
	 * @param impulse_response Input impulse response matrix
	 * @param input_grid Input grid parameters
	 * @param P Beam power
	 * @param R Beam radius
	 * @return Convolved value
	 */
	[[nodiscard]] double compute_flat_convolution(double rc, size_t ix,
												  const Matrix2D<double>& impulse_response,
												  const GridParameters& input_grid,
												  double P, double R);

	/**
	 * @brief Compute Gaussian beam convolution at point
	 * @param rc Output radius coordinate
	 * @param ix Output x-coordinate index
	 * @param impulse_response Input impulse response matrix
	 * @param input_grid Input grid parameters
	 * @param P Beam power
	 * @param R Beam radius
	 * @return Convolved value
	 */
	[[nodiscard]] double compute_gaussian_convolution(double rc, size_t ix,
													  const Matrix2D<double>& impulse_response,
													  const GridParameters& input_grid,
													  double P, double R);

	/**
	 * @brief Compute arbitrary beam convolution at point
	 * @param rc Output radius coordinate
	 * @param ix Output x-coordinate index
	 * @param impulse_response Input impulse response matrix
	 * @param input_grid Input grid parameters
	 * @param beam_params Beam parameters
	 * @return Convolved value
	 */
	[[nodiscard]] double compute_arbitrary_convolution(double rc, size_t ix,
													   const Matrix2D<double>& impulse_response,
													   const GridParameters& input_grid,
													   const BeamParameters& beam_params);

	/**
	 * @brief Interpolate impulse response data
	 * @param r Radius coordinate
	 * @param ix X-coordinate index
	 * @param impulse_response Input matrix
	 * @param input_grid Input grid parameters
	 * @return Interpolated value
	 */
	[[nodiscard]] double interpolate_impulse_response(double r, size_t ix,
													  const Matrix2D<double>& impulse_response,
													  const GridParameters& input_grid);

	/**
	 * @brief Adaptive integration using Simpson's rule
	 * @param integrand Function to integrate
	 * @param a Lower limit
	 * @param b Upper limit
	 * @param tolerance Relative tolerance
	 * @return Integration result
	 */
	[[nodiscard]] double adaptive_simpson(const std::function<double(double)>& integrand, double a, double b,
										  double tolerance);

	/**
	 * @brief Simpson's rule implementation
	 * @param integrand Function to integrate
	 * @param a Lower limit
	 * @param b Upper limit
	 * @param n Number of intervals (must be even)
	 * @return Integration result
	 */
	[[nodiscard]] double simpson_rule(const std::function<double(double)>& integrand, double a, double b, size_t n);

	/**
	 * @brief Compute Modified Bessel function I0(x) with caching
	 * @param x Argument
	 * @return I0(x)
	 */
	[[nodiscard]] double bessel_i0(double x);

	/**
	 * @brief Compute exp(x) * I0(y) with caching for numerical stability
	 * @param x Exponential argument
	 * @param y Bessel function argument
	 * @return exp(x) * I0(y)
	 */
	[[nodiscard]] double exp_bessel_i0(double x, double y);

	/**
	 * @brief Interpolate 2D impulse response at fractional indices
	 * @param matrix Input matrix
	 * @param r_idx Fractional r index
	 * @param x_idx Integer x index
	 * @return Interpolated value
	 */
	[[nodiscard]] double interpolate_2d(const Matrix2D<double>& matrix, double r_idx, size_t x_idx) const;

	/**
	 * @brief Validate input data for convolution
	 * @param input_data MCML input parameters
	 * @param output_data MCML output results
	 * @param grid_params Grid parameters
	 * @return true if valid
	 */
	[[nodiscard]] bool validate_inputs(const McmlInputData& input_data, const McmlOutputData& output_data,
									   const GridParameters& grid_params) const noexcept;

	ConvolutionParams m_params;
	FloatCache m_bessel_cache {1e-12};     // Cache for Bessel function values
	FloatCache m_exp_bessel_cache {1e-12}; // Cache for exp*Bessel values
};

} // namespace conv
