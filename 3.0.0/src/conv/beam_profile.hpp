/*******************************************************************************
 *	Beam profile management for CONV 3.0
 *  Copyright M.H.J. Lam, 2025.
 ****/

#pragma once

#include "conv.hpp"
#include "matrix.hpp"

#include <fstream>
#include <numbers>

namespace conv
{

/**
 * @brief Manages photon beam profiles for convolution
 */
class BeamProfile
{
public:
	explicit BeamProfile(const BeamParameters& params = {});
	~BeamProfile() = default;

	// Non-copyable but movable
	BeamProfile(const BeamProfile&) = delete;
	BeamProfile& operator=(const BeamProfile&) = delete;
	BeamProfile(BeamProfile&&) = default;
	BeamProfile& operator=(BeamProfile&&) = default;

	/**
	 * @brief Set beam parameters
	 * @param params Beam configuration
	 * @throws ConvException if parameters are invalid
	 */
	void set_parameters(const BeamParameters& params);

	/**
	 * @brief Get current beam parameters
	 * @return Current beam configuration
	 */
	[[nodiscard]] const BeamParameters& parameters() const noexcept { return m_params; }

	/**
	 * @brief Calculate beam profile value at given radius
	 * @param radius Distance from beam center [cm]
	 * @return Normalized intensity at radius
	 */
	[[nodiscard]] double profile_value(double radius) const;

	/**
	 * @brief Load arbitrary beam profile from file
	 * @param filename Path to profile file
	 * @throws FileException if file cannot be read
	 */
	void load_arbitrary_profile(const std::string& filename);

	/**
	 * @brief Validate current beam parameters
	 * @return true if parameters are valid
	 */
	[[nodiscard]] bool is_valid() const noexcept;

	/**
	 * @brief Calculate effective beam radius (radius containing given fraction of power)
	 * @param power_fraction Fraction of total power (0.0 to 1.0)
	 * @return Effective radius [cm]
	 */
	[[nodiscard]] double effective_radius(double power_fraction = 0.95) const;

	/**
	 * @brief Get maximum radius for which profile is defined
	 * @return Maximum radius [cm]
	 */
	[[nodiscard]] double max_radius() const noexcept;

private:
	/**
	 * @brief Interpolate arbitrary beam profile at given radius
	 * @param radius Distance from beam center [cm]
	 * @return Interpolated power density
	 */
	[[nodiscard]] double interpolate_arbitrary(double radius) const;

	/**
	 * @brief Normalize arbitrary beam profile
	 */
	void normalize_arbitrary_profile();

	/**
	 * @brief Validate arbitrary beam profile data
	 * @return true if data is valid
	 */
	[[nodiscard]] bool validate_arbitrary_data() const noexcept;

	BeamParameters m_params;
	bool m_profile_loaded = false;
};

// Inline implementation for performance-critical methods

inline double BeamProfile::profile_value(double radius) const {
	if (radius < 0.0) {
		return 0.0;
	}

	switch (m_params.type) {
		case BeamType::Original:
			// Delta function approximation - very narrow Gaussian
			return (radius < 1e-10) ? 1.0 : 0.0;

		case BeamType::Flat:
			if (m_params.radius <= 0.0)
				return 0.0;
			return (radius <= m_params.radius) ? 1.0 / (std::numbers::pi * m_params.radius * m_params.radius) : 0.0;

		case BeamType::Gaussian:
			if (m_params.radius <= 0.0)
				return 0.0;
			{
				const double sigma = m_params.radius / std::sqrt(2.0); // 1/e² radius to standard deviation
				const double factor = 1.0 / (2.0 * std::numbers::pi * sigma * sigma);
				return factor * std::exp(-(radius * radius) / (2.0 * sigma * sigma));
			}

		case BeamType::Arbitrary:
			if (!m_profile_loaded)
				return 0.0;
			return interpolate_arbitrary(radius);

		default: return 0.0;
	}
}

inline double BeamProfile::max_radius() const noexcept {
	switch (m_params.type) {
		case BeamType::Original: return 0.0;
		case BeamType::Flat: return m_params.radius;
		case BeamType::Gaussian: return m_params.radius * 3.0; // Practical cutoff at 3 sigma
		case BeamType::Arbitrary: return m_params.radii.empty() ? 0.0 : m_params.radii.back();
		default: return 0.0;
	}
}

} // namespace conv
