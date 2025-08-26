/**
 * @file tracer.hpp
 * @brief Photon tracing and scattering implementation for Monte Carlo simulation
 * @author M.H.J. Lam
 * @date 2025
 */

#pragma once

#include "mcml.hpp"

#include <memory>
#include <vector>

template<typename T>
using vec1 = std::vector<T>;
template<typename T>
using vec2 = std::vector<std::vector<T>>;
template<typename T>
using vec3 = std::vector<std::vector<std::vector<T>>>;

class Random;
struct RunParams;

/**
 * @brief Photon tracer class for Monte Carlo light transport simulation
 * 
 * Handles photon launching, propagation, scattering, and boundary interactions
 * in multi-layered turbid media.
 */
class Tracer
{
public:
	/**
	 * @brief Construct tracer with default radiance
	 * @param params Simulation parameters reference
	 * @param random Shared pointer to random number generator
	 */
	Tracer(RunParams& params, std::shared_ptr<Random>& random);
	
	/**
	 * @brief Construct tracer with custom radiance
	 * @param params Simulation parameters reference
	 * @param random Shared pointer to random number generator
	 * @param radiance Radiance object for output recording
	 */
	Tracer(RunParams& params, std::shared_ptr<Random>& random, Radiance radiance);
	
	~Tracer() = default;

	/**
	 * @brief Choose a new direction for photon propagation by sampling scattering angles
	 * 
	 * Samples:
	 * 1. The polar (deflection) angle θ, measuring from downwards z-axis
	 * 2. The azimuthal angle Ψ, measuring rotation around the z-axis in the xy-plane
	 * 
	 * θ range:    0 - π  (0 to 180 degrees)    [sin(θ) range:  0 to 1]
	 * Ψ range:    0 - 2π (0 to 360 degrees)    [cos(Ψ) range: -1 to 1]
	 * 
	 * Since cos²(θ) + sin²(θ) = 1, and if sin(θ) is known; cos(θ) = sqrt(1-sin(θ)²)
	 * 
	 * In the Henyey-Greenstein phase function, g is the asymmetry parameter and typically
	 * takes values in the range [-1, 1]:
	 * - g = 0: isotropic scattering (uniform in all directions).
	 * - g > 0: forward scattering (continue in the same direction).
	 * - g < 0: backward scattering (scatter in the opposite direction).
	 * 
	 * @param photon Reference to photon being scattered
	 * @param g Henyey-Greenstein asymmetry parameter
	 */
	void spin(Photon& photon, double g);

	/**
	 * @brief Initialize a photon packet for simulation
	 * 
	 * If an isotropic source is launched inside a glass layer, check whether the
	 * photon will be total-internally reflected. If so, kill the photon to avoid
	 * infinite travelling inside the glass layer.
	 * 
	 * @return Initialized photon packet
	 */
	Photon launch();

	/**
	 * @brief Move the photon away in the current layer
	 */
	void hop(Photon& photon, double dist, double n);

	/**
	 * @brief Drop photon weight inside the tissue due to absorption
	 * 
	 * The photon is assumed alive.
	 * The weight drop is dw = w * μa / (μa + μs).
	 * The dropped weight is assigned to the absorption array elements.
	 * 
	 * @param photon Reference to photon losing weight
	 */
	void drop(Photon& photon);

	/**
	 * @brief Determine photon survival via roulette when photon weight becomes too small
	 * @param photon Reference to photon undergoing roulette
	 */
	void roulette(Photon& photon);

	/**
	 * @brief Compute the Fresnel reflectance
	 * 
	 * Make sure that the cosine of the incident angle ai is positive,
	 * and the case when the angle is greater than the critical angle is ruled out.
	 * 
	 * Avoid trigonometric function operations as much as possible, because they are computationally intensive.
	 * 
	 * @param eta_i Incident refractive index
	 * @param eta_t Transmit refractive index
	 * @param cos_ai Cosine of the incident angle ai
	 * @param cos_at Reference to cosine of the transmission angle at
	 * @return Fresnel reflectance value
	 */
	double fresnel(double eta_i, double eta_t, double cos_ai, double& cos_at);

	/**
	 * @brief Decide whether the photon will be transmitted or reflected on the upper boundary
	 * 
	 * Handles photon interactions when uz < 0 (moving upward) at the current layer boundary.
	 * 
	 * If current_layer is the first layer, the photon packet will be partially transmitted and
	 * partially reflected if PARTIAL_REFLECTION active, or the photon packet will be either
	 * transmitted or reflected determined statistically if PARTIAL_REFLECTION is inactive.
	 * 
	 * Record the transmitted photon weight as reflection.
	 * 
	 * If the current_layer is not the first layer and the photon packet is transmitted,
	 * move the photon to the previous layer.
	 * 
	 * @param photon Reference to photon crossing upper boundary
	 */
	void cross_up(Photon& photon);

	/**
	 * @brief Decide whether the photon will be transmitted or reflected on the bottom boundary
	 * 
	 * Handles photon interactions when uz > 0 (moving downward) at the current layer boundary.
	 * 
	 * If the photon is transmitted, move the photon to current_layer + 1.
	 * If current_layer is the last layer, record the weight as transmittance.
	 * 
	 * @param photon Reference to photon crossing lower boundary
	 */
	void cross_down(Photon& photon);

	/**
	 * @brief Set a step size if the previous step has finished
	 * 
	 * If the step size fits in the current layer, move the photon, drop weight,
	 * and choose a new photon direction for propagation.
	 * 
	 * If the step size is long enough for the photon to hit an interface, this step
	 * is divided into three steps:
	 * 
	 * 1. Move the photon to the boundary free of absorption or scattering.
	 * 2. Update the step size to the unfinished step size.
	 * 3. Decide whether the photon is reflected or transmitted.
	 * 
	 * @param photon Reference to photon being processed
	 */
	void hop_drop_spin(Photon& photon);

	/**
	 * @brief Trace a photon through the medium
	 * 
	 * Trace a photon, then compute the absorption, transmittance, and reflection
	 * constants, including their standard errors.
	 * 
	 * @param photon Reference to photon being traced
	 */
	void trace(Photon& photon);

	/**
	 * @brief Record photon weight exiting the first layer to the reflectance array
	 * 
	 * Record photon weight exiting the first layer (uz < 0) to the reflectance
	 * array and update its weight.
	 * 
	 * @param photon Reference to photon exiting upward
	 * @param reflectance Reflectance value to record
	 */
	void record_reflectance(Photon& photon, double reflectance);

	/**
	 * @brief Record photon weight exiting the last layer to the transmittance array
	 * 
	 * Record the photon weight exiting the last layer (uz > 0),
	 * no matter whether the layer is glass or not, to the transmittance array.
	 * 
	 * @param photon Reference to photon exiting downward
	 * @param reflectance Reflectance value (actually transmittance)
	 */
	void record_transmittance(Photon& photon, double reflectance);

	/**
	 * @brief Return radiance object
	 * @return Reference to internal radiance object for simulation results
	 */
	operator Radiance&() { return m_radiance; }

private:
	RunParams& m_params;
	std::shared_ptr<Random> m_random;
	Radiance m_radiance;
};
