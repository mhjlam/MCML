////////////////////////////////////////////////////////////////////////////////
// CUDAMC - GPU Monte Carlo for Time-Resolved Photon Transport
//
// GPU Device Functions for CUDAMC Photon Transport
//
// This file contains CUDA device functions that run on the GPU to simulate
// photon transport through homogeneous turbid media with time resolution.
// Functions handle photon initialization, scattering, boundary interactions,
// and detection for time-domain spectroscopy applications.
//
// Authors: Erik Alerstam, Tomas Svensson, Stefan Andersson-Engels
// Modernized: 2025
////////////////////////////////////////////////////////////////////////////////

#include "cudamc.h"

// Define PI constant for scattering calculations
#define PI 3.14159265f

////////////////////////////////////////////////////////////////////////////////
// MAIN MONTE CARLO SIMULATION KERNEL

/**
 * Main GPU kernel for time-resolved Monte Carlo photon transport
 *
 * Each GPU thread simulates multiple photon trajectories through homogeneous
 * tissue with time-resolved detection. Photons undergo scattering, absorption,
 * and boundary interactions while tracking time-of-flight for spectroscopy
 * applications.
 *
 * KERNEL EXECUTION MODEL:
 * -----------------------
 * - One thread per GPU core (typically 512-1024 threads per block)
 * - Each thread simulates ~500,000 photon trajectories
 * - Coalesced memory access for optimal GPU memory bandwidth
 * - Atomic operations for thread-safe histogram accumulation
 *
 * @param rng_states      Per-thread RNG state array
 * @param rng_constant    Per-thread RNG constant array
 * @param rng_multipliers Per-thread RNG multiplier array
 * @param num_device      Device array for storing photon counts per thread
 * @param time_histogram  Time-resolved detection histogram (shared)
 */
__global__ void mc(uint32_t* rng_states_32, uint32_t* rng_constant_32, uint32_t* rng_multipliers,
				   uint32_t* num_device, uint32_t* time_histogram) {
	// Thread identification and memory indexing
	const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

	// Local photon state variables
	float3 photon_position;     // Current photon position [cm]
	float3 photon_direction;    // Current photon direction (unit vector)
	float time_of_flight;       // Total photon time-of-flight [ps]

	// Convert 32-bit RNG arrays to 64-bit for internal use
	uint64_t rng_state = ((uint64_t)rng_constant_32[thread_id] << 32) | rng_states_32[thread_id];
	uint32_t rng_multiplier = rng_multipliers[thread_id];

	// Tissue optical properties (loaded from constant memory)
	float mus_max = 100.0f;    // Maximum scattering coefficient [cm⁻¹]
	float anisotropy_g = 0.9f; // Henyey-Greenberg anisotropy parameter
	float light_speed = 0.0299792458f; // Speed of light in tissue [cm/ps]
	float critical_cos = 0.292f;       // Critical angle cosine for TIR
	float refractive_ratio = 1.4f;     // Refractive index ratio (tissue/air)

	// Detection and simulation counters
	uint32_t num_detected_photons = 0;

	// Launch initial photon state
	launch_photon(&photon_position, &photon_direction, &time_of_flight);

	// Main photon transport loop (500,000 photons per thread)
	for (uint32_t photon_index = 0; photon_index < 500000; photon_index++) {
		// Sample step size from exponential distribution
		float step_size = __fdividef(-__logf(rand_mwc_oc(&rng_state, &rng_multiplier)), mus_max);

		// Move photon and check for boundary crossing
		photon_position.x += photon_direction.x * step_size;
		photon_position.y += photon_direction.y * step_size;
		photon_position.z += photon_direction.z * step_size;

		// Handle boundary interaction if photon exits tissue (z < 0)
		uint32_t boundary_flag = 0;
		if (photon_position.z < 0.0f) {
			boundary_flag = reflect(&photon_direction, &photon_position, &time_of_flight,
									&light_speed, &critical_cos, &refractive_ratio,
									&rng_state, &rng_multiplier, time_histogram);
		}

		// Update time-of-flight
		time_of_flight += __fdividef(step_size, light_speed);

		// Sample new scattering direction
		spin(&photon_direction, &anisotropy_g, &rng_state, &rng_multiplier);

		// Check termination conditions
		if (time_of_flight >= 2000.0f || boundary_flag >= 1) {
			if (boundary_flag == 1) {
				num_detected_photons++;
			}
			// Reset photon for next trajectory
			launch_photon(&photon_position, &photon_direction, &time_of_flight);
		}
	}

	// Store final detection count and convert RNG state back to 32-bit format
	num_device[thread_id] = num_detected_photons;
	rng_states_32[thread_id] = (uint32_t)(rng_state & 0xFFFFFFFFULL);
}

////////////////////////////////////////////////////////////////////////////////
// RANDOM NUMBER GENERATION (GPU DEVICE FUNCTIONS)

/**
 * Multiply-With-Carry RNG: Closed-Open interval [0, 1) - GPU version
 *
 * @param rng_state      RNG state pointer (modified)
 * @param rng_multiplier RNG multiplier pointer
 * @return               Random float in [0, 1)
 */
__device__ float rand_mwc_co(uint64_t* rng_state, uint32_t* rng_multiplier) {
	*rng_state = (*rng_state & 0xffffffffull) * (*rng_multiplier) + (*rng_state >> 32);
	return ((float)((uint32_t)(*rng_state & 0xffffffffull)) / (UINT_MAX));
}

/**
 * Multiply-With-Carry RNG: Open-Closed interval (0, 1] - GPU version
 *
 * @param rng_state      RNG state pointer (modified)
 * @param rng_multiplier RNG multiplier pointer
 * @return               Random float in (0, 1]
 */
__device__ float rand_mwc_oc(uint64_t* rng_state, uint32_t* rng_multiplier) {
	*rng_state = (*rng_state & 0xffffffffull) * (*rng_multiplier) + (*rng_state >> 32);
	return (1.0f - (float)((uint32_t)(*rng_state & 0xffffffffull)) / (UINT_MAX));
}

////////////////////////////////////////////////////////////////////////////////
// PHOTON INITIALIZATION (GPU DEVICE FUNCTIONS)

/**
 * Launch new photon at origin - GPU version
 *
 * @param photon_position  Photon position (modified)
 * @param photon_direction Photon direction (modified)
 * @param time_of_flight   Time of flight (modified)
 */
__device__ void launch_photon(float3* photon_position, float3* photon_direction, float* time_of_flight) {
	// Initialize at tissue surface
	photon_position->x = 0.0f;
	photon_position->y = 0.0f;
	photon_position->z = 0.0f;

	// Direction: straight into tissue (+z)
	photon_direction->x = 0.0f;
	photon_direction->y = 0.0f;
	photon_direction->z = 1.0f;

	// Initialize time
	*time_of_flight = 0.0f;
}

////////////////////////////////////////////////////////////////////////////////
// PHOTON SCATTERING (GPU DEVICE FUNCTIONS)

/**
 * Sample scattering direction using Henyey-Greenberg phase function - GPU version
 *
 * @param photon_direction Current direction (modified)
 * @param anisotropy_g     Anisotropy parameter pointer
 * @param rng_state        RNG state pointer
 * @param rng_multiplier   RNG multiplier pointer
 */
__device__ void spin(float3* photon_direction, float* anisotropy_g, uint64_t* rng_state, uint32_t* rng_multiplier) {
	float cost, sint; // cos(θ) and sin(θ) for polar angle
	float cosp, sinp; // cos(φ) and sin(φ) for azimuthal angle
	float temp, tempdir = photon_direction->x;

	// Sample polar angle from Henyey-Greenberg distribution
	if ((*anisotropy_g) == 0.0f) {
		// Isotropic scattering: uniform cos(θ) in [-1, 1]
		cost = 2.0f * rand_mwc_co(rng_state, rng_multiplier) - 1.0f;
	}
	else {
		// Anisotropic scattering: Henyey-Greenberg phase function
		temp = __fdividef((1.0f - (*anisotropy_g) * (*anisotropy_g)), 
						  (1.0f - (*anisotropy_g) + 2.0f * (*anisotropy_g) * rand_mwc_co(rng_state, rng_multiplier)));
		cost = __fdividef((1.0f + (*anisotropy_g) * (*anisotropy_g) - temp * temp), (2.0f * (*anisotropy_g)));
	}
	sint = sqrtf(1.0f - cost * cost);

	// Sample azimuthal angle uniformly in [0, 2π)
	__sincosf(2.0f * PI * rand_mwc_co(rng_state, rng_multiplier), &sinp, &cosp); // Fast GPU sincos

	// Transform to lab coordinate system
	temp = sqrtf(1.0f - photon_direction->z * photon_direction->z);

	if (temp == 0.0f) {
		// Normal incidence case
		photon_direction->x = sint * cosp;
		photon_direction->y = sint * sinp;
		photon_direction->z = copysignf(cost, photon_direction->z * cost);
	}
	else {
		// General incidence: apply rotation matrix
		photon_direction->x = __fdividef(sint * (photon_direction->x * photon_direction->z * cosp - photon_direction->y * sinp), temp) + photon_direction->x * cost;
		photon_direction->y = __fdividef(sint * (photon_direction->y * photon_direction->z * cosp + tempdir * sinp), temp) + photon_direction->y * cost;
		photon_direction->z = -sint * cosp * temp + photon_direction->z * cost;
	}

	// Renormalize direction vector (GPU-optimized)
	temp = rsqrtf(photon_direction->x * photon_direction->x + photon_direction->y * photon_direction->y + photon_direction->z * photon_direction->z); // Fast inverse sqrt
	photon_direction->x *= temp;
	photon_direction->y *= temp;
	photon_direction->z *= temp;
}

////////////////////////////////////////////////////////////////////////////////
// BOUNDARY REFLECTION (GPU DEVICE FUNCTIONS)

/**
 * Handle boundary interaction at tissue-air interface - GPU version
 *
 * @param photon_direction    Current direction (modified if reflected)
 * @param photon_position     Current position (modified)
 * @param time_of_flight      Time of flight (modified)
 * @param light_speed         Speed of light in medium
 * @param critical_cos        Critical angle cosine
 * @param refractive_ratio    Refractive index ratio
 * @param rng_state           RNG state pointer
 * @param rng_multiplier      RNG multiplier pointer
 * @param time_histogram      Time histogram array
 * @return                    0=reflected, 1=detected, 2=transmitted
 */
__device__ uint32_t reflect(float3* photon_direction, float3* photon_position, float* time_of_flight, float* light_speed, 
							float* critical_cos, float* refractive_ratio, uint64_t* rng_state,
							uint32_t* rng_multiplier, uint32_t* time_histogram) {
	// Detection parameters
	float fibre_separation = 1.0f; // Source-detector separation [cm]
	float fibre_diameter = 0.05f;  // Detector diameter [cm]

	// Calculate Fresnel reflectance
	float r;
	if (-photon_direction->z <= *critical_cos) {
		// Total internal reflection
		r = 1.0f;
	}
	else {
		if (-photon_direction->z == 1.0f) {
			// Normal incidence: simplified Fresnel
			r = __fdividef((1.0f - *refractive_ratio), (1.0f + *refractive_ratio));
			r *= r;
		}
		else {
			// Oblique incidence: full Fresnel calculation
			float sin_angle_i = sqrtf(1.0f - photon_direction->z * photon_direction->z);
			float sin_angle_t = *refractive_ratio * sin_angle_i;
			float cos_angle_t = sqrtf(1.0f - sin_angle_t * sin_angle_t);

			float cos_sum_angle = (-photon_direction->z * cos_angle_t) - sin_angle_i * sin_angle_t;
			float cos_dif_angle = (-photon_direction->z * cos_angle_t) + sin_angle_i * sin_angle_t;
			float sin_sum_angle = sin_angle_i * cos_angle_t + (-photon_direction->z * sin_angle_t);
			float sin_dif_angle = sin_angle_i * cos_angle_t - (-photon_direction->z * sin_angle_t);

			// Average of s and p polarization reflectance
			r = 0.5f * sin_dif_angle * sin_dif_angle
				* __fdividef((cos_dif_angle * cos_dif_angle + cos_sum_angle * cos_sum_angle),
							 (sin_sum_angle * sin_sum_angle * cos_dif_angle * cos_dif_angle));
		}
	}

	// Monte Carlo reflection/transmission decision
	if (r < 1.0f) {
		if (rand_mwc_co(rng_state, rng_multiplier) <= r) {
			// Photon reflects
			r = 1.0f;
		}
		else {
			// Photon transmits: calculate exit position and check detection
			r = __fdividef(-photon_position->z, photon_direction->z); // Time to reach boundary
			photon_position->x += photon_direction->x * r;
			photon_position->y += photon_direction->y * r;
			*time_of_flight += __fdividef(r, *light_speed);         // Update total time of flight

			// Calculate radial distance from source at exit
			r = sqrtf(photon_position->x * photon_position->x + photon_position->y * photon_position->y);

			// Check for fiber detection
			if (fabsf(r - fibre_separation) <= fibre_diameter) {
				// Photon detected: add to time-resolved histogram
				uint32_t bin = __float2uint_rz(__fdividef((*time_of_flight), DT));
				if (bin < TEMP_SIZE) {               // Bounds check
					atomicAdd(time_histogram + bin, 1); // Thread-safe increment
				}
				return 1;                            // Detected
			}
			else {
				return 2;                            // Transmitted but not detected
			}
		}
	}

	// Handle reflection back into medium
	if (r == 1.0f) {
		photon_position->z *= -1; // Mirror z-position across boundary
		photon_direction->z *= -1; // Mirror z-direction component
	}

	return 0;         // Reflected
}
