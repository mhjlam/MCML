/*
 *  CUDAMC Header - Monte Carlo Photon Transport in Semi-Infinite Media
 *
 *  This header defines constants, data structures, and function prototypes for
 *  GPU-accelerated Monte Carlo simulation of photon migration in turbid tissues.
 *
 *  Simulation Features:
 *  - GPU/CPU parallel processing with identical algorithms
 *  - Time-resolved detection with configurable fiber geometry
 *  - Henyey-Greenstein scattering with anisotropy parameter
 *  - Fresnel reflection at refractive index boundaries
 *  - High-quality Multiply-With-Carry random number generation
 *
 *  Physical Model:
 *  - Semi-infinite turbid medium (z >= 0)
 *  - Point source at origin, detector at specified separation
 *  - Exponential step length distribution
 *  - Anisotropic scattering via Henyey-Greenstein phase function
 *
 *  This file is part of CUDAMC.
 *  Licensed under GNU General Public License v3 or later.
 */

#ifndef CUDA_MC_H
#define CUDA_MC_H

////////////////////////////////////////////////////////////////////////////////
// SYSTEM INCLUDES
////////////////////////////////////////////////////////////////////////////////
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdint.h>

#include <cstdlib>

////////////////////////////////////////////////////////////////////////////////
// PHYSICAL AND SIMULATION CONSTANTS

// Mathematical and Physical Constants
#define PI 3.14159265f // Mathematical constant π

// Temporal Parameters
#define T_MAX 2000.0f // [ps] Maximum photon time of flight before termination
#define DT    10.0f   // [ps] Time binning resolution for histogram

// GPU Configuration Parameters
#define NUM_THREADS_PER_BLOCK 512 // Threads per CUDA block (optimal for most modern GPUs)
#define NUM_BLOCKS            128 // Number of CUDA blocks
#define NUM_THREADS           (NUM_THREADS_PER_BLOCK * NUM_BLOCKS) // Total GPU threads

// Monte Carlo Step Counts
#define NUM_STEPS_GPU   500000 // Photon steps per GPU thread (high for accuracy)
#define NUM_STEPS_CPU   10000  // Photon steps per CPU thread (reduced for speed)
#define NUM_THREADS_CPU 1024   // Number of CPU threads for comparison

// Data Array Sizes
#define TEMP_SIZE 201 // Number of time histogram bins

////////////////////////////////////////////////////////////////////////////////
// CUDA ERROR HANDLING MACROS

/**
 * CUDA Error Checking Macro
 *
 * Wraps CUDA API calls with automatic error checking. If a CUDA error occurs,
 * prints detailed error information and terminates the program.
 *
 * Usage: CUDA_CHECK_ERROR(cudaMalloc(&ptr, size));
 */
#define cuda_check_error(call)                                                                                       \
	do {                                                                                                             \
		cudaError_t error = call;                                                                                    \
		if (error != cudaSuccess) {                                                                                  \
			fprintf(stderr, "CUDA Error at %s:%d - %s: %s\n", __FILE__, __LINE__, #call, cudaGetErrorString(error)); \
			exit(EXIT_FAILURE);                                                                                      \
		}                                                                                                            \
	}                                                                                                                \
	while (0)

// Backward compatibility alias
#define CUDA_CHECK_ERROR(call) cuda_check_error(call)

////////////////////////////////////////////////////////////////////////////////
// DATA STRUCTURES

/**
 * GPU Device Capability Information
 *
 * Stores hardware characteristics and computed optimal parameters for
 * adaptive GPU configuration based on actual device properties.
 */
struct DeviceCapability {
	// Hardware characteristics
	int major;                 // CUDA compute capability major version
	int minor;                 // CUDA compute capability minor version
	int multi_processor_count; // Number of streaming multiprocessors
	int max_threads_per_block; // Maximum threads per block
	int max_grid_size;         // Maximum grid dimension
	size_t total_global_mem;   // Total global memory in bytes

	// Computed optimal parameters for this device
	int optimal_block_size; // Recommended threads per block
	int optimal_grid_size;  // Recommended number of blocks
};

////////////////////////////////////////////////////////////////////////////////
// FUNCTION DECLARATIONS

// GPU Device Functions (executed on GPU)

/**
 * Main Monte Carlo kernel - executes on GPU
 *
 * Each thread simulates independent photon trajectories using identical
 * algorithms to the CPU version. Threads process different RNG seeds
 * to ensure statistical independence.
 *
 * @param x_device    RNG state lower 32 bits per thread
 * @param c_device    RNG carry values per thread
 * @param a_device    RNG multipliers per thread
 * @param num_device  Output: detected photons per thread
 * @param hist_device Output: time-resolved histogram bins (shared)
 */
__global__ void mc(uint32_t* x_device, uint32_t* c_device, uint32_t* a_device, uint32_t* num_device,
				   uint32_t* hist_device);

/**
 * Random number generators - GPU device versions
 *
 * Multiply-With-Carry generators optimized for GPU execution.
 * Two variants provide different interval coverage for numerical stability.
 */
__device__ float rand_mwc_oc(uint64_t* x, uint32_t* a); // Open-closed (0,1]
__device__ float rand_mwc_co(uint64_t* x, uint32_t* a); // Closed-open [0,1)

/**
 * Photon physics functions - GPU device versions
 *
 * Core photon transport algorithms: initialization, scattering, and
 * boundary interaction with Fresnel reflection/transmission.
 */
__device__ void launch_photon(float3* pos, float3* dir, float* t);
__device__ void spin(float3* dir, float* g, uint64_t* x, uint32_t* a);
__device__ uint32_t reflect(float3* dir, float3* pos, float* t, float* v, float* cos_crit, float* n, uint64_t* x,
							uint32_t* a, uint32_t* hist_device);

//------------------------------------------------------------------------------
// CPU Host Functions (executed on CPU - "Gold Standard" reference)
//------------------------------------------------------------------------------

/**
 * CPU reference implementations with "gs_" prefix
 *
 * These functions provide identical algorithms to GPU versions but run
 * on CPU for validation and performance comparison purposes.
 */
float gs_rand_mwc_och(uint64_t* x, uint32_t* a); // Open-closed RNG
float gs_rand_mwc_coh(uint64_t* x, uint32_t* a); // Closed-open RNG
void gs_launch_photon(float3* pos, float3* dir, float* t);
void gs_spin(float3* dir, float* g, uint64_t* x, uint32_t* a);
uint32_t gs_reflect(float3* dir, float3* pos, float* t, float* v, float* cos_crit, float* n, uint64_t* x, uint32_t* a,
					uint32_t* hist);

/**
 * Main CPU Monte Carlo simulation
 *
 * Runs NUM_THREADS_CPU independent photon histories for performance
 * comparison with GPU implementation.
 */
void gs_mc(uint32_t* xd, uint32_t* cd, uint32_t* ad, uint32_t* numh, uint32_t* histh);

// Application Control Functions (executed on CPU)

/**
 * Initialize random number generator arrays
 *
 * Sets up RNG states for all threads using safe prime multipliers
 * to ensure good statistical properties and thread independence.
 */
void initialize_rng();

/**
 * Main simulation coordinator
 *
 * Orchestrates GPU and CPU simulations, manages memory transfers,
 * and provides comprehensive performance analysis and reporting.
 *
 * @param x  RNG state arrays (lower 32 bits)
 * @param c  RNG carry arrays
 * @param a  RNG multiplier arrays
 */
void run_monte_carlo_simulation(uint32_t* x, uint32_t* c, uint32_t* a);

#endif // CUDA_MC_H
