/*==============================================================================
 * CUDAMCML Transport Module - GPU-Accelerated Photon Transport in Layered Media
 *
 * This module implements the core Monte Carlo photon transport physics for
 * multi-layered biological tissues. Provides massively parallel GPU kernels
 * for simulating photon propagation, scattering, absorption, and boundary
 * interactions with high computational efficiency.
 *
 * PHYSICAL MODEL IMPLEMENTATION:
 * ------------------------------
 * - Monte Carlo photon transport with statistical weight tracking
 * - Multi-layered geometry with arbitrary optical properties per layer
 * - Henyey-Greenberg scattering phase function for anisotropic scattering
 * - Fresnel reflection/transmission at all refractive index boundaries
 * - Beer-Lambert absorption with layer-specific absorption coefficients
 * - Russian roulette photon termination for computational efficiency
 *
 * GPU COMPUTATIONAL ARCHITECTURE:
 * -------------------------------
 * - One photon per GPU thread for massive parallelization
 * - Coalesced memory access patterns for optimal bandwidth utilization
 * - Template specialization for performance optimization (absorption detection)
 * - Atomic operations for thread-safe detection accumulation
 * - Efficient random number generation with per-thread state management
 *
 * DETECTION CAPABILITIES:
 * -----------------------
 * - Reflectance: Rd(r,α) - spatially and angularly resolved top surface detection
 * - Absorption: A(r,z) - volumetric energy deposition throughout tissue layers
 * - Transmittance: Tt(r,α) - spatially and angularly resolved bottom surface detection
 * - Real-time photon counting and statistical accumulation
 *
 * PERFORMANCE OPTIMIZATIONS:
 * ---------------------------
 * - Template specialization for compile-time optimization paths
 * - Single-precision floating-point arithmetic for GPU efficiency
 * - Minimized divergent branching in hot computation paths
 * - Efficient use of GPU constant memory for frequently accessed parameters
 * - Optimized atomic operations for detection data accumulation
 *
 * LICENSE:
 * --------
 * This file is part of CUDAMCML.
 *
 * CUDAMCML is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * CUDAMCML is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with CUDAMCML.  If not, see <http://www.gnu.org/licenses/>.
 */

////////////////////////////////////////////////////////////////////////////////
// INCLUDES AND DEPENDENCIES

// Standard library includes
#include <cstdint> // Standard integer types for cross-platform compatibility

////////////////////////////////////////////////////////////////////////////////
// DEVICE FUNCTION FORWARD DECLARATIONS

/**
 * Forward declarations of device functions for proper compilation ordering.
 * These functions implement the core physics of photon transport and are
 * called extensively within the main transport kernels.
 */

// Core transport kernel (template for performance optimization)
template<int ignoreAdetection>
__global__ void mc_d(MemStruct DeviceMem);

// Random number generation functions (imported from RNG module)
__device__ float rand_mwc_oc(uint64_t* rng_state, uint32_t* rng_multiplier); // (0,1] interval
__device__ float rand_mwc_co(uint64_t* rng_state, uint32_t* rng_multiplier); // [0,1) interval

// Photon lifecycle management
__device__ void launch_photon(PhotonStruct* photon_state, uint64_t* rng_state, uint32_t* rng_multiplier);
__global__ void launch_photon_global(MemStruct device_memory);
__device__ uint32_t photon_survive(PhotonStruct* photon_state, uint64_t* rng_state, uint32_t* rng_multiplier);

// Physics implementations
__device__ void spin(PhotonStruct* photon_state, float anisotropy_g, uint64_t* rng_state, uint32_t* rng_multiplier);
__device__ uint32_t reflect(PhotonStruct* photon_state, int target_layer, uint64_t* rng_state, uint32_t* rng_multiplier);

// Atomic operations for thread-safe detection accumulation
__device__ void atomic_add(uint64_t* memory_address, uint32_t value_to_add);

////////////////////////////////////////////////////////////////////////////////
// MAIN MONTE CARLO TRANSPORT KERNEL

/**
 * Primary Monte Carlo photon transport kernel
 *
 * This is the core computational kernel that simulates photon transport through
 * multi-layered biological tissues. Each GPU thread simulates one photon at a time,
 * with thousands of threads running in parallel for maximum computational throughput.
 *
 * COMPUTATIONAL WORKFLOW:
 * -----------------------
 * 1. Thread initialization: Load photon state and RNG state from global memory
 * 2. Main transport loop: Propagate photon through tissue layers
 *    - Sample step length from exponential distribution
 *    - Check for layer boundary crossings
 *    - Handle Fresnel reflection/transmission at boundaries
 *    - Accumulate absorbed energy and detection events
 *    - Perform anisotropic scattering (Henyey-Greenberg)
 *    - Apply Russian roulette termination for low-weight photons
 * 3. Photon lifecycle management: Launch new photons as current ones terminate
 * 4. State preservation: Save photon and RNG states to global memory
 *
 * TEMPLATE SPECIALIZATION:
 * ------------------------
 * The template parameter 'ignoreAdetection' enables compile-time optimization:
 * - ignoreAdetection=0: Full simulation with absorption detection (default)
 * - ignoreAdetection=1: Skip absorption detection for performance optimization
 *
 * PERFORMANCE CHARACTERISTICS:
 * ----------------------------
 * - Memory bandwidth optimized through coalesced access patterns
 * - Divergent branching minimized for SIMT efficiency
 * - Register usage optimized for maximum occupancy
 * - Atomic operations used only for thread-safe accumulation
 *
 * @param DeviceMem Complete device memory structure with all simulation arrays
 */
template<int ignoreAdetection>
__global__ void mc_d(MemStruct device_memory) {
	// THREAD IDENTIFICATION AND MEMORY ACCESS SETUP

	// Calculate global thread ID and memory access indices
	const int block_id = blockIdx.x;   // Block index within grid
	const int thread_id = threadIdx.x; // Thread index within block
	const int global_thread_index = NUM_THREADS_PER_BLOCK * block_id + thread_id;

	// Load RNG state and multipliers for this thread (coalesced access)
	uint64_t rng_state = device_memory.x[global_thread_index];      // MWC RNG state
	uint32_t rng_multiplier = device_memory.a[global_thread_index]; // Safe prime multiplier

	// Physics and detection variables
	float step_length;                     // Photon step length [cm]
	uint32_t detection_index;              // Array index for detection accumulation
	uint32_t absorbed_weight = 0;          // Accumulated absorbed weight
	uint32_t detection_index_previous = 0; // Previous detection index for optimization

	// Load photon state from global memory
	PhotonStruct photon = device_memory.p[global_thread_index];

	// Layer transition tracking
	int target_layer;

	// THREAD ACTIVITY CHECK AND MAIN TRANSPORT LOOP INITIALIZATION

	// Initialize loop counter and check thread activity status
	uint32_t loop_iteration = 0;
	if (!device_memory.thread_active[global_thread_index]) {
		loop_iteration = NUM_STEPS_GPU; // Skip main loop if thread is inactive
	}

	// MAIN MONTE CARLO TRANSPORT LOOP

	for (; loop_iteration < NUM_STEPS_GPU; loop_iteration++) {
		//=======================================================================
		// STEP LENGTH SAMPLING
		//=======================================================================

		// Sample step length from exponential distribution: s = -ln(ξ)/μₜ
		if (layers_dc[photon.layer].mutr != FLT_MAX) {
			// Normal tissue: exponential sampling with transport mean free path
			step_length = -__logf(rand_mwc_oc(&rng_state, &rng_multiplier)) * layers_dc[photon.layer].mutr;
		}
		else {
			// Glass layer: effectively infinite mean free path
			step_length = 100.0f; // Large step through non-scattering medium
		}

		// LAYER BOUNDARY INTERSECTION ANALYSIS

		// Initialize target layer (may change due to boundary crossing)
		target_layer = photon.layer;

		// Check for upward boundary crossing (reflection/transmission to layer above)
		if (photon.z + step_length * photon.dz < layers_dc[photon.layer].z_min) {
			target_layer = photon.layer - 1;
			step_length = __fdividef(layers_dc[photon.layer].z_min - photon.z, photon.dz);
		}

		// Check for downward boundary crossing (reflection/transmission to layer below)
		if (photon.z + step_length * photon.dz > layers_dc[photon.layer].z_max) {
			target_layer = photon.layer + 1;
			step_length = __fdividef(layers_dc[photon.layer].z_max - photon.z, photon.dz);
		}

		// PHOTON PROPAGATION

		// Move photon to new position
		photon.x += photon.dx * step_length;
		photon.y += photon.dy * step_length;
		photon.z += photon.dz * step_length;

		// Ensure photon stays within layer boundaries (numerical precision safety)
		photon.z = fminf(photon.z, layers_dc[photon.layer].z_max);
		photon.z = fmaxf(photon.z, layers_dc[photon.layer].z_min);

		// BOUNDARY INTERACTION PROCESSING

		if (target_layer != photon.layer) {
			// Reset step length for boundary interaction
			step_length = 0.0f;

			// Process Fresnel reflection/transmission at layer boundary
			if (reflect(&photon, target_layer, &rng_state, &rng_multiplier) == 0U) {
				// Photon transmitted through boundary

				// Diffuse reflectance detection (exiting top surface)
				if (target_layer == 0) {
					// Calculate detection bin indices: [angle][radius]
					const int angle_bin = __float2int_rz(acosf(-photon.dz) * 2.0f * RPI * det_dc[0].na);
					const int radius_bin =
						min(__float2int_rz(__fdividef(sqrtf(photon.x * photon.x + photon.y * photon.y), det_dc[0].dr)),
							static_cast<int>(det_dc[0].nr) - 1);
					detection_index = angle_bin * det_dc[0].nr + radius_bin;

					// Accumulate reflectance detection with thread-safe atomic operation
					atomic_add(&device_memory.Rd_ra[detection_index], photon.weight);

					// Terminate this photon (set weight to zero)
					photon.weight = 0;
				}

				// Transmittance detection (exiting bottom surface)
				if (target_layer > *n_layers_dc) {
					// Calculate detection bin indices: [angle][radius]
					const int angle_bin = __float2int_rz(acosf(photon.dz) * 2.0f * RPI * det_dc[0].na);
					const int radius_bin =
						min(__float2int_rz(__fdividef(sqrtf(photon.x * photon.x + photon.y * photon.y), det_dc[0].dr)),
							static_cast<int>(det_dc[0].nr) - 1);
					detection_index = angle_bin * det_dc[0].nr + radius_bin;

					// Accumulate transmittance detection with thread-safe atomic operation
					atomic_add(&device_memory.Tt_ra[detection_index], photon.weight);

					// Terminate this photon (set weight to zero)
					photon.weight = 0;
				}
			}
		}

		// ABSORPTION AND SCATTERING PROCESSING

		if (step_length > 0.0f) {
			// ENERGY ABSORPTION CALCULATION

			// Calculate absorbed weight using Beer-Lambert law: ΔW = W × μₐ/μₜ × s
			const uint32_t weight_absorbed = __float2uint_rn(layers_dc[photon.layer].mua * layers_dc[photon.layer].mutr
															 * __uint2float_rn(photon.weight));
			photon.weight -= weight_absorbed;

			// ABSORPTION DETECTION ACCUMULATION (TEMPLATE-CONTROLLED)

			// Template specialization: compile-time optimization for absorption detection
			if (ignoreAdetection == 0) {
				// Calculate detection bin index: A(r,z) spatial grid
				const int depth_bin =
					min(__float2int_rz(__fdividef(photon.z, det_dc[0].dz)), static_cast<int>(det_dc[0].nz) - 1);
				const int radius_bin =
					min(__float2int_rz(__fdividef(sqrtf(photon.x * photon.x + photon.y * photon.y), det_dc[0].dr)),
						static_cast<int>(det_dc[0].nr) - 1);
				detection_index = depth_bin * det_dc[0].nr + radius_bin;

				// Optimize atomic operations by accumulating within same spatial bin
				if (detection_index == detection_index_previous) {
					absorbed_weight += weight_absorbed;
				}
				else {
					// Commit previous accumulated weight to global memory
					if (absorbed_weight > 0) {
						atomic_add(&device_memory.A_rz[detection_index_previous], absorbed_weight);
					}
					// Start new accumulation for current spatial bin
					detection_index_previous = detection_index;
					absorbed_weight = weight_absorbed;
				}
			}

			// ANISOTROPIC SCATTERING (HENYEY-GREENBERG)

			// Apply anisotropic scattering with layer-specific anisotropy parameter
			spin(&photon, layers_dc[photon.layer].g, &rng_state, &rng_multiplier);
		}

		// PHOTON SURVIVAL AND LIFECYCLE MANAGEMENT

		// Check photon survival using Russian roulette termination
		if (!photon_survive(&photon, &rng_state, &rng_multiplier)) {
			// Current photon terminated - check if new photon should be launched
			if (atomicAdd(device_memory.num_terminated_photons, 1U) < (*num_photons_dc - NUM_THREADS)) {
				// Launch new photon to maintain simulation throughput
				launch_photon(&photon, &rng_state, &rng_multiplier);
			}
			else {
				// No more photons needed - deactivate this thread
				device_memory.thread_active[global_thread_index] = 0U;
				loop_iteration = NUM_STEPS_GPU; // Exit main transport loop
			}
		}

		// FINAL ABSORPTION ACCUMULATION (TEMPLATE-CONTROLLED CLEANUP)

		// Handle any remaining accumulated absorption weight
		if (ignoreAdetection == 1 && absorbed_weight != 0) {
			atomic_add(&device_memory.A_rz[detection_index_previous], absorbed_weight);
		}
	}

	// THREAD SYNCHRONIZATION AND STATE PRESERVATION

	// Ensure all threads complete before state saving (may not be necessary)
	__syncthreads();

	// Save photon and RNG states to global memory for potential continuation
	device_memory.p[global_thread_index] = photon;
	device_memory.x[global_thread_index] = rng_state;
}

////////////////////////////////////////////////////////////////////////////////
// PHOTON INITIALIZATION AND LIFECYCLE MANAGEMENT

/**
 * Initialize new photon state
 *
 * Sets up a new photon with standard initialization parameters for pencil
 * beam incident on the top surface of the multi-layered medium. This function
 * is called when a photon is first launched or when an old photon terminates
 * and needs to be replaced.
 *
 * INITIALIZATION PARAMETERS:
 * --------------------------
 * - Position: (0,0,0) - pencil beam incident at origin
 * - Direction: (0,0,1) - normal incidence (downward)
 * - Layer: 1 - first tissue layer (layer 0 is air above)
 * - Weight: Adjusted for specular reflection at top surface
 *
 * PHYSICAL CONSIDERATIONS:
 * ------------------------
 * - Specular reflection already accounted for in initial weight
 * - Fresnel reflection at air-tissue boundary handled during boundary crossing
 * - Weight normalization ensures proper statistical sampling
 *
 * @param photon_state   Pointer to photon structure to initialize
 * @param rng_state      Pointer to RNG state (not used in current implementation)
 * @param rng_multiplier Pointer to RNG multiplier (not used in current implementation)
 */
__device__ void launch_photon(PhotonStruct* photon_state, uint64_t* rng_state, uint32_t* rng_multiplier) {
	// Set initial position: pencil beam at origin
	photon_state->x = 0.0f; // [cm] - lateral position x
	photon_state->y = 0.0f; // [cm] - lateral position y
	photon_state->z = 0.0f; // [cm] - depth position (top surface)

	// Set initial direction: normal incidence (downward)
	photon_state->dx = 0.0f; // x-direction cosine
	photon_state->dy = 0.0f; // y-direction cosine
	photon_state->dz = 1.0f; // z-direction cosine (downward)

	// Set initial layer: first tissue layer (not air above)
	photon_state->layer = 1;

	// Set initial weight: accounts for specular reflection at top surface
	photon_state->weight = *start_weight_dc;
}

/**
 * Global photon launch kernel
 *
 * GPU kernel for initializing photon states across all threads simultaneously.
 * This kernel is typically called once at the beginning of simulation to
 * set up the initial photon population across all GPU threads.
 *
 * KERNEL EXECUTION MODEL:
 * -----------------------
 * - One thread per photon initialization
 * - Coalesced memory access for optimal bandwidth
 * - Minimal computation - just initialization
 *
 * @param device_memory Complete device memory structure containing photon arrays
 */
__global__ void launch_photon_global(MemStruct device_memory) {
	// Calculate global thread index for memory access
	const int block_id = blockIdx.x;
	const int thread_id = threadIdx.x;
	const int global_thread_index = NUM_THREADS_PER_BLOCK * block_id + thread_id;

	// Local photon structure for initialization
	PhotonStruct photon;

	// Load RNG state for this thread (though not used in current launch_photon)
	uint64_t rng_state = device_memory.x[global_thread_index];
	uint32_t rng_multiplier = device_memory.a[global_thread_index];

	// Initialize photon with standard parameters
	launch_photon(&photon, &rng_state, &rng_multiplier);

	// Store initialized photon state to global memory
	device_memory.p[global_thread_index] = photon;
}

////////////////////////////////////////////////////////////////////////////////
// ANISOTROPIC SCATTERING IMPLEMENTATION

/**
 * Henyey-Greenberg anisotropic scattering
 *
 * Updates photon direction according to the Henyey-Greenberg phase function,
 * which models anisotropic scattering in biological tissues. The scattering
 * anisotropy parameter g controls the forward/backward scattering preference.
 *
 * PHASE FUNCTION MATHEMATICS:
 * ---------------------------
 * P(cos θ) = (1-g²) / (1 + g² - 2g cos θ)^(3/2)
 *
 * Where θ is the scattering angle and g is the anisotropy parameter:
 * - g = 0: Isotropic scattering (uniform in all directions)
 * - g > 0: Forward-peaked scattering (typical for biological tissues)
 * - g < 0: Back-scattered scattering (rare in biology)
 * - |g| → 1: Highly anisotropic scattering
 *
 * SCATTERING COORDINATE SYSTEM:
 * ------------------------------
 * - Polar angle θ: Sampled from Henyey-Greenberg distribution
 * - Azimuthal angle φ: Uniformly distributed [0, 2π)
 * - Coordinate transformation from local to global coordinate system
 * - Special handling for normal incidence (dz ≈ ±1)
 *
 * NUMERICAL OPTIMIZATION:
 * -----------------------
 * - GPU intrinsic functions for fast trigonometry
 * - Optimized coordinate transformations
 * - Renormalization to handle floating-point precision loss
 *
 * @param photon_state   Pointer to photon structure (direction modified in-place)
 * @param anisotropy_g   Anisotropy parameter [-1,1] for Henyey-Greenberg scattering
 * @param rng_state      Pointer to RNG state
 * @param rng_multiplier Pointer to RNG multiplier
 */
__device__ void spin(PhotonStruct* photon_state, float anisotropy_g, uint64_t* rng_state, uint32_t* rng_multiplier) {
	// Scattering angle parameters
	float cos_theta, sin_theta; // Cosine and sine of polar scattering angle θ
	float cos_phi, sin_phi;     // Cosine and sine of azimuthal angle φ
	float temp, temp_dx;        // Temporary variables for coordinate transformation

	// POLAR ANGLE SAMPLING (HENYEY-GREENBERG)

	if (anisotropy_g == 0.0f) {
		// Special case: Isotropic scattering (g = 0)
		cos_theta = 2.0f * rand_mwc_co(rng_state, rng_multiplier) - 1.0f; // Uniform [-1, 1]
	}
	else {
		// General case: Anisotropic Henyey-Greenberg scattering
		// Sample using inverse transform method
		temp = __fdividef((1.0f - anisotropy_g * anisotropy_g), 
						  (1.0f - anisotropy_g + 2.0f * anisotropy_g * rand_mwc_co(rng_state, rng_multiplier)));
		cos_theta = __fdividef((1.0f + anisotropy_g * anisotropy_g - temp * temp), (2.0f * anisotropy_g));
	}

	// Calculate sine from cosine (sin² + cos² = 1)
	sin_theta = sqrtf(1.0f - cos_theta * cos_theta);

	// AZIMUTHAL ANGLE SAMPLING

	// Sample azimuthal angle uniformly from [0, 2π)
	__sincosf(2.0f * PI * rand_mwc_co(rng_state, rng_multiplier), &sin_phi, &cos_phi);

	// COORDINATE SYSTEM TRANSFORMATION

	// Calculate perpendicular component magnitude for coordinate transformation
	temp = sqrtf(1.0f - photon_state->dz * photon_state->dz);

	if (temp == 0.0f) {
		// SPECIAL CASE: NORMAL INCIDENCE (dz ≈ ±1)

		// Simple transformation for normal incidence
		photon_state->dx = sin_theta * cos_phi;
		photon_state->dy = sin_theta * sin_phi;
		photon_state->dz = copysignf(cos_theta, photon_state->dz * cos_theta); // Preserve sign
	}
	else {
		// GENERAL CASE: OBLIQUE INCIDENCE

		// Store original dx for coordinate transformation
		temp_dx = photon_state->dx;

		// Apply full 3D coordinate transformation
		photon_state->dx = __fdividef(sin_theta * (photon_state->dx * photon_state->dz * cos_phi - photon_state->dy * sin_phi), temp) + photon_state->dx * cos_theta;
		photon_state->dy = __fdividef(sin_theta * (photon_state->dy * photon_state->dz * cos_phi + temp_dx * sin_phi), temp) + photon_state->dy * cos_theta;
		photon_state->dz = -sin_theta * cos_phi * temp + photon_state->dz * cos_theta;
	}

	// DIRECTION VECTOR RENORMALIZATION

	// Renormalize direction vector to account for floating-point precision loss
	// This is critical for maintaining unit vector properties over many scattering events
	temp = rsqrtf(photon_state->dx * photon_state->dx + photon_state->dy * photon_state->dy + photon_state->dz * photon_state->dz); // Fast inverse square root
	photon_state->dx *= temp;
	photon_state->dy *= temp;
	photon_state->dz *= temp;
}

////////////////////////////////////////////////////////////////////////////////
// FRESNEL REFLECTION AND TRANSMISSION

/**
 * Process Fresnel reflection/transmission at layer boundaries
 *
 * Calculates whether a photon is reflected or transmitted when crossing
 * a boundary between layers with different refractive indices. Uses the
 * complete Fresnel equations to account for both polarization states
 * and handles special cases like total internal reflection.
 *
 * FRESNEL REFLECTION PHYSICS:
 * ---------------------------
 * The reflection probability depends on:
 * - Incident angle θᵢ relative to surface normal
 * - Refractive indices of both layers (n₁, n₂)
 * - Polarization state (averaged over both s and p polarizations)
 *
 * SPECIAL CASES:
 * --------------
 * 1. Refractive index matching (n₁ = n₂): Automatic transmission
 * 2. Normal incidence (θᵢ = 0): Simple Fresnel formula
 * 3. Total internal reflection (n₁ > n₂, θᵢ > θc): Complete reflection
 * 4. General case: Full Fresnel calculation with optimized algorithm
 *
 * COMPUTATIONAL OPTIMIZATIONS:
 * -----------------------------
 * - Fast algorithm avoids expensive inverse trigonometry
 * - GPU-optimized mathematical operations
 * - Single random number sample for reflection/transmission decision
 * - Efficient direction vector updates
 *
 * @param p         Pointer to photon structure (layer and direction modified)
 * @param new_layer Target layer index if transmitted
 * @param x         Pointer to RNG state
 * @param a         Pointer to RNG multiplier
 * @return 1 if photon is reflected, 0 if transmitted
 */
__device__ uint32_t reflect(PhotonStruct* photon_state, int target_layer, uint64_t* rng_state, uint32_t* rng_multiplier) {
	// REFRACTIVE INDICES AND INITIAL SETUP

	// Extract refractive indices for current and target layers
	const float n1 = layers_dc[photon_state->layer].n;  // Current layer refractive index
	const float n2 = layers_dc[target_layer].n; // Target layer refractive index

	// Calculate incident angle cosine (angle with respect to surface normal)
	const float cos_incident = fabsf(photon_state->dz);

	// SPECIAL CASE: REFRACTIVE INDEX MATCHING

	if (n1 == n2) {
		// Perfect index matching: automatic transmission with no direction change
		photon_state->layer = target_layer;
		return 0U; // Transmitted
	}

	// SPECIAL CASE: TOTAL INTERNAL REFLECTION

	if (n1 > n2) {
		// Check for total internal reflection condition
		// Critical angle: sin(θc) = n₂/n₁
		// TIR occurs when: sin²(θᵢ) > (n₂/n₁)²
		const float sin_squared_incident = 1.0f - cos_incident * cos_incident;
		const float index_ratio_squared = (n2 / n1) * (n2 / n1);

		if (sin_squared_incident > index_ratio_squared) {
			// Total internal reflection: mirror z-direction only
			photon_state->dz *= -1.0f;
			return 1U; // Reflected
		}
	}

	// SPECIAL CASE: NORMAL INCIDENCE

	if (cos_incident == 1.0f) {
		// Normal incidence: simplified Fresnel formula
		const float r_normal = (n1 - n2) / (n1 + n2);
		const float reflectance = r_normal * r_normal;

		if (rand_mwc_co(rng_state, rng_multiplier) <= reflectance) {
			// Reflection: mirror z-direction only
			photon_state->dz *= -1.0f;
			return 1U; // Reflected
		}
		else {
			// Transmission: no direction change, only layer update
			photon_state->layer = target_layer;
			return 0U; // Transmitted
		}
	}

	// GENERAL CASE: FULL FRESNEL CALCULATION

	// Calculate transmission angle using Snell's law
	// sin²(θₜ) = (n₁/n₂)² × sin²(θᵢ)
	const float index_ratio = n1 / n2;
	const float sin_squared_transmitted = index_ratio * index_ratio * (1.0f - cos_incident * cos_incident);

	// Optimized Fresnel reflectance calculation
	// This algorithm avoids expensive trigonometric functions while maintaining accuracy
	float temp_factor = 2.0f
						* sqrtf((1.0f - cos_incident * cos_incident) * (1.0f - sin_squared_transmitted)
								* sin_squared_transmitted * cos_incident * cos_incident);

	// Intermediate calculation for Fresnel coefficients
	float fresnel_term =
		sin_squared_transmitted + (cos_incident * cos_incident) * (1.0f - 2.0f * sin_squared_transmitted);

	// Calculate final reflectance (averaged over both polarizations)
	float reflectance = fresnel_term
						* __fdividef((1.0f - fresnel_term - temp_factor),
									 ((1.0f - fresnel_term + temp_factor) * (fresnel_term + temp_factor)));

	// REFLECTION/TRANSMISSION DECISION

	if (rand_mwc_co(rng_state, rng_multiplier) <= reflectance) {
		// PHOTON REFLECTED
		photon_state->dz *= -1.0f; // Mirror z-direction component
		return 1U;      // Reflected
	}
	else {
		// PHOTON TRANSMITTED

		// Update direction vector for refracted ray
		const float cos_transmitted = sqrtf(1.0f - sin_squared_transmitted);

		// Scale lateral components by index ratio
		photon_state->dx *= index_ratio;
		photon_state->dy *= index_ratio;

		// Calculate transmitted z-component (preserve sign, use refracted magnitude)
		photon_state->dz = copysignf(cos_transmitted, photon_state->dz);

		// Update photon layer
		photon_state->layer = target_layer;

		return 0U; // Transmitted
	}
}

////////////////////////////////////////////////////////////////////////////////
// PHOTON SURVIVAL AND TERMINATION

/**
 * Russian roulette photon survival algorithm
 *
 * Determines whether a low-weight photon survives for continued simulation
 * or is terminated to maintain computational efficiency. Uses the Russian
 * roulette technique to maintain statistical accuracy while eliminating
 * photons that contribute negligibly to the final result.
 *
 * RUSSIAN ROULETTE ALGORITHM:
 * ---------------------------
 * - High-weight photons (W > WEIGHTI): Always survive
 * - Zero-weight photons: Always terminate (exited simulation domain)
 * - Low-weight photons: Survive with probability CHANCE
 * - Surviving photons: Weight boosted by factor 1/CHANCE
 *
 * STATISTICAL CORRECTNESS:
 * ------------------------
 * The expected contribution remains unchanged:
 * E[W_new] = CHANCE × (W/CHANCE) + (1-CHANCE) × 0 = W
 *
 * This maintains unbiased results while eliminating computationally
 * expensive tracking of very low-weight photons.
 *
 * @param p Pointer to photon structure (weight may be modified)
 * @param x Pointer to RNG state
 * @param a Pointer to RNG multiplier
 * @return 1 if photon survives, 0 if terminated
 */
__device__ uint32_t photon_survive(PhotonStruct* photon_state, uint64_t* rng_state, uint32_t* rng_multiplier) {
	// HIGH-WEIGHT PHOTON: AUTOMATIC SURVIVAL

	if (photon_state->weight > WEIGHTI) {
		return 1U;  // High-weight photon always survives
	}

	// ZERO-WEIGHT PHOTON: AUTOMATIC TERMINATION

	if (photon_state->weight == 0U) {
		return 0U; // Zero-weight photon (exited domain) always terminates
	}

	// LOW-WEIGHT PHOTON: RUSSIAN ROULETTE DECISION

	// Apply Russian roulette with survival probability CHANCE
	if (rand_mwc_co(rng_state, rng_multiplier) < CHANCE) {
		// Photon survives: boost weight to maintain statistical accuracy
		photon_state->weight = __float2uint_rn(__fdividef(static_cast<float>(photon_state->weight), CHANCE));
		return 1U; // Survived with boosted weight
	}

	// Photon terminated by Russian roulette
	return 0U;
}

////////////////////////////////////////////////////////////////////////////////
// ATOMIC OPERATIONS FOR THREAD-SAFE DETECTION

/**
 * 64-bit atomic addition for detection accumulation
 *
 * Performs thread-safe addition of 32-bit values to 64-bit detection arrays.
 * This function is essential for accumulating detection events from multiple
 * GPU threads without race conditions or data corruption.
 *
 * IMPLEMENTATION DETAILS:
 * -----------------------
 * - Uses 32-bit atomic operations on lower and upper halves separately
 * - Handles carry propagation between lower and upper 32-bit words
 * - Compatible with GPU Compute Capability 1.1+ requirements
 * - Ensures atomicity of complete 64-bit operation
 *
 * PERFORMANCE CONSIDERATIONS:
 * ---------------------------
 * - Optimized for common case (no carry propagation)
 * - Minimal overhead for most addition operations
 * - Coalesced memory access when possible
 * - Essential for massively parallel detection accumulation
 *
 * @param address Pointer to 64-bit value to be incremented
 * @param add     32-bit value to add atomically
 */
__device__ void atomic_add(uint64_t* memory_address, uint32_t value_to_add) {
	// Perform atomic addition on lower 32 bits
	uint32_t old_lower = atomicAdd(reinterpret_cast<uint32_t*>(memory_address), value_to_add);

	// Check for overflow (carry needed to upper 32 bits)
	// Overflow condition: (old_lower + value_to_add) < value_to_add, which indicates wraparound
	if (old_lower + value_to_add < value_to_add) {
		// Propagate carry to upper 32 bits
		atomicAdd(reinterpret_cast<uint32_t*>(memory_address) + 1, 1U);
	}
}
