/*
 *  CPU Gold Standard Implementation for CUDAMC
 *
 *  This file provides the CPU reference implementation that mirrors the GPU Monte Carlo
 *  photon transport simulation. Results should match GPU output for validation purposes.
 *
 *  Physical Model:
 *  - Semi-infinite turbid medium with refractive index mismatch
 *  - Scattering follows Henyey-Greenstein phase function
 *  - Absorption and scattering events sampled stochastically
 *  - Time-resolved detection at boundary with fiber geometry
 *
 *	This file is part of CUDAMC.
 *	Licensed under GNU General Public License v3 or later.
 */

#include "cudamc.h"

#include <limits.h>
#include <math.h>
#include <stdint.h>

// UTILITY MACROS
#define SIGN(x) ((x) >= 0 ? 1 : -1)

// FUNCTION DECLARATIONS
float gs_rand_mwc_coh(uint64_t*, uint32_t*);
float gs_rand_mwc_och(uint64_t*, uint32_t*);
void gs_launch_photon(float3*, float3*, float*);
void gs_spin(float3*, float*, uint64_t*, uint32_t*);
uint32_t gs_reflect(float3*, float3*, float*, float*, float*, float*, uint64_t*, uint32_t*, uint32_t*);

////////////////////////////////////////////////////////////////////////////////
// MAIN MONTE CARLO SIMULATION (CPU GOLD STANDARD)

/**
 * CPU Monte Carlo photon transport simulation
 *
 * Runs NUM_THREADS_CPU independent photon histories, each with NUM_STEPS_CPU steps.
 * Simulates photon propagation in semi-infinite turbid medium with time-resolved
 * detection at the boundary using fiber geometry.
 *
 * @param xd    RNG state lower 32 bits for each thread
 * @param cd    RNG carry values for each thread
 * @param ad    RNG multipliers for each thread
 * @param numh  Output: number of detected photons per thread
 * @param histh Output: time-resolved histogram bins
 */
void gs_mc(uint32_t* xd, uint32_t* cd, uint32_t* ad, uint32_t* numh, uint32_t* histh) {
	// Thread loop variables
	uint32_t thread;          // Current thread index
	uint32_t ii;              // Step counter within each photon history
	uint64_t x;               // Combined 64-bit RNG state
	uint32_t a;               // RNG multiplier for current thread
	uint32_t num_det_photons; // Detected photons counter for current thread
	uint32_t flag;            // Boundary interaction result flag

	// Photon state variables
	float3 pos; // Photon position (x, y, z) in cm
	float3 dir; // Photon direction unit vector
	float t;    // Time of flight in picoseconds
	float s;    // Step length for current move in cm

	// Medium optical properties
	float mus_max = 90.0f;    // Maximum scattering coefficient [1/cm]
	float v = 0.0214f;        // Speed of light in medium [cm/ps] = c₀/(n=1.4)
	float cos_crit = 0.6999f; // Critical angle cosine = √(1-(n₁/n₂)²) for TIR
	float g = 0.9f;           // Anisotropy parameter for Henyey-Greenstein scattering
	float n = 1.4f;           // Refractive index of medium

	// Process each thread (photon trajectory) independently
	for (thread = 0; thread < NUM_THREADS_CPU; thread++) {
		// Initialize RNG state for this thread
		x = cd[thread];             // Get carry value
		x = (x << 32) + xd[thread]; // Combine with lower 32 bits
		a = ad[thread];             // Get multiplier for this thread

		// Initialize counters for this thread
		num_det_photons = 0;
		flag = 0;

		// Launch initial photon (position at origin, direction +z)
		gs_launch_photon(&pos, &dir, &t);

		// Main photon step loop
		for (ii = 0; ii < NUM_STEPS_CPU; ii++) {
			// Sample step length using exponential distribution
			// s = -ln(ξ)/μₛ where ξ ∈ (0,1] to avoid ln(0)
			s = -logf(gs_rand_mwc_och(&x, &a)) / mus_max;

			// Check if photon will cross boundary (z = 0) during this step
			if ((pos.z + dir.z * s) <= 0) {
				// Handle boundary interaction (reflection/transmission)
				flag = gs_reflect(&dir, &pos, &t, &v, &cos_crit, &n, &x, &a, histh);
			}

			// Move photon to new position
			pos.x += s * dir.x;
			pos.y += s * dir.y;
			pos.z += s * dir.z;
			t += s / v; // Update time of flight

			// Sample new scattering direction
			gs_spin(&dir, &g, &x, &a);

			// Check termination conditions
			if (t >= T_MAX || flag >= 1) {
				num_det_photons++;                // Count this photon
				flag = 0;                         // Reset boundary flag
				gs_launch_photon(&pos, &dir, &t); // Launch new photon
			}
		}

		// Store results for this thread
		numh[thread] = num_det_photons;
	}
}

////////////////////////////////////////////////////////////////////////////////
// RANDOM NUMBER GENERATORS

/**
 * Multiply-With-Carry RNG: Closed-Open interval [0, 1)
 *
 * Returns random values from 0 (inclusive) to 1 (exclusive).
 * Used for angle sampling and most Monte Carlo decisions where exact 0 is acceptable.
 *
 * Algorithm: x = (x_low * a) + x_high, return x_low / UINT_MAX
 *
 * @param x  Pointer to 64-bit RNG state (modified)
 * @param a  Pointer to RNG multiplier
 * @return   Random float in [0, 1)
 */
float gs_rand_mwc_coh(uint64_t* x, uint32_t* a) {
	// Split 64-bit state: multiply low 32 bits by multiplier, add high 32 bits as carry
	*x = (*x & 0xffffffffull) * (*a) + (*x >> 32);

	// Return normalized low 32 bits
	return ((float)((uint32_t)(*x & 0xffffffffull)) / (UINT_MAX));
}

/**
 * Multiply-With-Carry RNG: Open-Closed interval (0, 1]
 *
 * Returns random values from 0 (exclusive) to 1 (inclusive).
 * Used for sampling exponential distributions where ln(0) would be undefined.
 *
 * @param x  Pointer to 64-bit RNG state (modified)
 * @param a  Pointer to RNG multiplier
 * @return   Random float in (0, 1]
 */
float gs_rand_mwc_och(uint64_t* x, uint32_t* a) {
	*x = (*x & 0xffffffffull) * (*a) + (*x >> 32);

	// Subtract from 1.0 to flip interval from [0,1) to (0,1]
	return (1.0f - (float)((uint32_t)(*x & 0xffffffffull)) / (UINT_MAX));
}

////////////////////////////////////////////////////////////////////////////////
// PHOTON INITIALIZATION

/**
 * Launch new photon at origin
 *
 * Initializes photon state for tissue surface launch:
 * - Position at origin (0, 0, 0)
 * - Direction along +z axis (into tissue)
 * - Time reset to zero
 *
 * @param pos  Photon position vector (modified)
 * @param dir  Photon direction vector (modified)
 * @param t    Time of flight pointer (modified)
 */
void gs_launch_photon(float3* pos, float3* dir, float* t) {
	// Initialize at tissue surface
	pos->x = 0.0f;
	pos->y = 0.0f;
	pos->z = 0.0f;

	// Direction: straight down into tissue (+z direction)
	dir->x = 0.0f;
	dir->y = 0.0f;
	dir->z = 1.0f;

	// Reset time counter
	*t = 0.0f;
}

////////////////////////////////////////////////////////////////////////////////
// PHOTON SCATTERING

/**
 * Sample new scattering direction using Henyey-Greenstein phase function
 *
 * Updates photon direction after scattering event. Uses two-step process:
 * 1. Sample polar angle θ from Henyey-Greenstein distribution
 * 2. Sample azimuthal angle φ uniformly from [0, 2π)
 * 3. Rotate coordinate system to apply angles relative to current direction
 *
 * @param dir  Current photon direction unit vector (modified)
 * @param g    Anisotropy parameter: -1 (backscatter) to +1 (forward)
 * @param x    RNG state pointer
 * @param a    RNG multiplier pointer
 */
void gs_spin(float3* dir, float* g, uint64_t* x, uint32_t* a) {
	// Polar angle sampling variables
	float cost, sint;       // cos(θ) and sin(θ) for polar deflection angle
	float cosp, sinp;       // cos(φ) and sin(φ) for azimuthal angle
	float temp;
	float tempdir = dir->x; // Save original x-direction for coordinate transform

	// Sample polar angle θ from Henyey-Greenstein phase function
	if ((*g) == 0.0f) {
		// Isotropic scattering: cos(θ) uniform in [-1, 1]
		cost = 2.0f * gs_rand_mwc_coh(x, a) - 1.0f;
	}
	else {
		// Henyey-Greenstein scattering:
		// P(cos θ) ∝ (1-g²)/[1+g²-2g·cos θ]^(3/2)
		temp = (1.0f - (*g) * (*g)) / (1.0f - (*g) + 2.0f * (*g) * gs_rand_mwc_coh(x, a));
		cost = (1.0f + (*g) * (*g) - temp * temp) / (2.0f * (*g));
	}
	sint = sqrtf(1.0f - cost * cost); // sin(θ) = √(1 - cos²(θ))

	// Sample azimuthal angle φ uniformly in [0, 2π)
	temp = 2.0f * PI * gs_rand_mwc_coh(x, a);
	cosp = cosf(temp);
	sinp = sinf(temp);

	// Coordinate system transformation
	temp = sqrtf(1.0f - dir->z * dir->z); // √(1 - cos²(θ_incident))

	if (temp == 0.0f) {
		// Special case: normal incidence (dir = ±ẑ)
		// Direct assignment since no rotation needed
		dir->x = sint * cosp;
		dir->y = sint * sinp;
		dir->z = copysignf(cost, dir->z * cost);
	}
	else {
		// General case: rotate scattered direction relative to incident direction
		// Apply rotation matrix to transform from scattering frame to lab frame
		dir->x = sint * (dir->x * dir->z * cosp - dir->y * sinp) / temp + dir->x * cost;
		dir->y = sint * (dir->y * dir->z * cosp + tempdir * sinp) / temp + dir->y * cost;
		dir->z = -sint * cosp * temp + dir->z * cost;
	}

	// Renormalize direction vector to account for floating-point errors
	temp = 1.0f / sqrtf(dir->x * dir->x + dir->y * dir->y + dir->z * dir->z);
	dir->x *= temp;
	dir->y *= temp;
	dir->z *= temp;
}

////////////////////////////////////////////////////////////////////////////////
// BOUNDARY INTERACTION AND DETECTION

/**
 * Handle photon interaction at tissue-air boundary
 *
 * When photon reaches z=0 boundary, determines whether it reflects back into
 * tissue or transmits out. If transmitted, checks for detection by fiber optic.
 * Uses Fresnel reflectance calculation with total internal reflection.
 *
 * @param dir             Current photon direction (modified if reflected)
 * @param pos             Current photon position (modified)
 * @param t               Time of flight pointer (modified)
 * @param v               Speed of light in medium pointer
 * @param cos_crit        Critical angle cosine for TIR
 * @param n               Refractive index ratio pointer
 * @param x               RNG state pointer
 * @param a               RNG multiplier pointer
 * @param histh           Time-resolved histogram array (modified)
 * @return                0=reflected, 1=detected, 2=transmitted but not detected
 */
uint32_t gs_reflect(float3* dir, float3* pos, float* t, float* v, float* cos_crit, float* n, uint64_t* x, uint32_t* a,
					uint32_t* histh) {
	// Detection geometry parameters
	float fibre_separation = 1.0f; // Distance from source to detector center [cm]
	float fibre_diameter = 0.05f;  // Detector fiber diameter [cm]

	// Calculate Fresnel reflectance
	float r;      // Fresnel reflectance coefficient
	uint32_t adr; // Histogram bin address

	// Check for total internal reflection
	if (-dir->z <= *cos_crit) {
		// Angle exceeds critical angle: total internal reflection
		r = 1.0f;
	}
	else {
		// Calculate Fresnel reflectance for transmitted light
		if (-dir->z == 1.0f) {
			// Normal incidence: simplified Fresnel equation
			// R = [(n₁-n₂)/(n₁+n₂)]²
			r = (1 - *n) / (1 + *n);
			r *= r;
		}
		else {
			// Oblique incidence: full Fresnel calculation
			// Calculate angles using Snell's law: n₁sin θ₁ = n₂sin θ₂
			float sin_angle_i = sqrtf(1 - dir->z * dir->z);           // sin θᵢ in medium
			float sin_angle_t = *n * sin_angle_i;                     // sin θₜ in air
			float cos_angle_t = sqrtf(1 - sin_angle_t * sin_angle_t); // cos θₜ

			// Calculate Fresnel coefficients for s and p polarization
			float cos_sum_angle = (-dir->z * cos_angle_t) - sin_angle_i * sin_angle_t;
			float cos_dif_angle = (-dir->z * cos_angle_t) + sin_angle_i * sin_angle_t;
			float sin_sum_angle = sin_angle_i * cos_angle_t + (-dir->z * sin_angle_t);
			float sin_dif_angle = sin_angle_i * cos_angle_t - (-dir->z * sin_angle_t);

			// Average of s and p polarization: R = ½(Rs + Rp)
			r = 0.5f * sin_dif_angle * sin_dif_angle * (cos_dif_angle * cos_dif_angle + cos_sum_angle * cos_sum_angle)
				/ (sin_sum_angle * sin_sum_angle * cos_dif_angle * cos_dif_angle);
		}
	}

	// Determine reflection vs transmission using Monte Carlo

	if (r < 1.0f) {
		// Partial transmission possible: sample random decision
		if (gs_rand_mwc_coh(x, a) <= r) {
			// Photon reflects back into medium
			r = 1.0f;
		}
		else {
			// Photon transmits: calculate exit position and check detection
			// Calculate intersection time with z=0 plane
			r = -(pos->z / dir->z); // Travel time to boundary (dir->z < 0)

			// Update position to boundary intersection point
			pos->x += dir->x * r;
			pos->y += dir->y * r;
			*t += r / (*v); // Update total time of flight

			// Calculate radial distance from source at exit point
			r = sqrtf(pos->x * pos->x + pos->y * pos->y);

			// Check if photon hits detector fiber
			if (fabsf(r - fibre_separation) <= fibre_diameter) {
				// Photon detected: add to time-resolved histogram
				adr = (uint32_t)floorf((*t) / DT); // Convert time to bin index

				if (adr < TEMP_SIZE) {             // Bounds check
					histh[adr] = histh[adr] + 1;
				}
				return 1;                          // Return: detected
			}
			else {
				// Transmitted but missed detector
				return 2; // Return: transmitted but not detected
			}
		}
	}

	// Handle reflection back into medium
	if (r == 1.0f) {
		// Reflect photon: mirror position and direction across z=0 plane
		pos->z *= -1; // Mirror z-coordinate across boundary
		dir->z *= -1; // Mirror z-direction component
	}

	return 0;         // Return: reflected back into medium
}
