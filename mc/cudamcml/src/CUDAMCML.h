/*==============================================================================
 * CUDAMCML - GPU-Accelerated Monte Carlo Multi-Layer Photon Transport
 *
 * CUDA-based Monte Carlo simulation of photon migration in layered turbid media.
 * This implementation extends basic Monte Carlo photon transport to complex
 * multi-layered geometries with arbitrary optical properties per layer.
 *
 * PHYSICAL MODEL:
 * ---------------
 * - Multi-layered turbid media with configurable optical properties per layer
 * - Arbitrary layer thicknesses and refractive index mismatches
 * - Fresnel reflection/transmission at all layer boundaries
 * - Henyey-Greenberg scattering phase function with layer-specific anisotropy
 * - Spatially-resolved detection grids for comprehensive flux analysis
 * - Weight-based photon termination using Russian roulette techniques
 *
 * DETECTION GEOMETRY:
 * -------------------
 * - Cylindrical coordinate system (r,z) for detection grids
 * - Reflectance detection at top surface: Rd(r,α)
 * - Absorption detection throughout volume: A(r,z)
 * - Transmittance detection at bottom surface: Tt(r,α)
 * - Angular resolution for directional flux analysis
 *
 * COMPUTATIONAL FEATURES:
 * -----------------------
 * - Dynamic GPU configuration based on hardware capabilities
 * - Memory-optimized data structures with CUDA alignment
 * - Constant memory utilization for frequently accessed parameters
 * - Exception-safe memory management with modern C++ practices
 * - Comprehensive error handling with detailed CUDA diagnostics
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

#ifndef CUDAMCML_H
#define CUDAMCML_H

// Standard library includes
#include <cstdlib>

////////////////////////////////////////////////////////////////////////////////
// GPU HARDWARE CONFIGURATION CONSTANTS

/**
 * Default GPU Thread Configuration
 *
 * Optimized for modern GPU architectures (Compute Capability 3.0+).
 * These values provide good occupancy across different GPU generations
 * while maintaining compatibility with memory alignment requirements.
 */
#define DEFAULT_THREADS_PER_BLOCK   512 // Optimal for most modern GPUs (multiple of 32)
#define GPU_OVERSUBSCRIPTION_FACTOR 3   // Run 3x multiprocessor count for better occupancy

////////////////////////////////////////////////////////////////////////////////
// PHYSICAL AND MATHEMATICAL CONSTANTS

/**
 * Mathematical constants for photon transport calculations
 *
 * High precision values ensure accurate trigonometric and geometric
 * computations throughout the Monte Carlo simulation.
 */
#define PI         3.141592654f // Pi - ratio of circumference to diameter
#define RPI        0.318309886f // 1/Pi - reciprocal for efficient division
#define MAX_LAYERS 100          // Maximum supported tissue layers
#define STR_LEN    200          // Maximum string length for file paths

/**
 * Monte Carlo simulation control parameters
 *
 * WEIGHTI: Integer representation of minimum photon weight (0.0001f)
 * CHANCE:  Survival probability in Russian roulette termination
 * NUM_STEPS_GPU: Maximum steps per kernel launch (memory management)
 */
#define WEIGHTI       429497u // 0xFFFFFFFFu*WEIGHT (0.0001f converted to integer)
#define CHANCE        0.1f    // Russian roulette survival probability
#define NUM_STEPS_GPU 500000  // Maximum transport steps per kernel launch

////////////////////////////////////////////////////////////////////////////////
// CUDA ERROR HANDLING AND DIAGNOSTICS

/**
 * Comprehensive CUDA Error Checking Macro
 *
 * Provides detailed error reporting with file location, function name,
 * and human-readable error descriptions. Essential for debugging
 * GPU memory operations and kernel launches.
 *
 * USAGE:
 *   CUDA_CHECK_ERROR(cudaMalloc(&ptr, size));
 *   CUDA_CHECK_ERROR(kernel<<<blocks, threads>>>(...));
 *   CUDA_CHECK_ERROR(cudaDeviceSynchronize());
 */
#define CUDA_CHECK_ERROR(call)                                                                                       \
	do {                                                                                                             \
		cudaError_t error = call;                                                                                    \
		if (error != cudaSuccess) {                                                                                  \
			fprintf(stderr, "CUDA Error at %s:%d - %s: %s\n", __FILE__, __LINE__, #call, cudaGetErrorString(error)); \
			exit(EXIT_FAILURE);                                                                                      \
		}                                                                                                            \
	}                                                                                                                \
	while (0)

#define CUDA_SAFE_CALL(call) CUDA_CHECK_ERROR(call)

////////////////////////////////////////////////////////////////////////////////
// ADAPTIVE GPU THREAD CONFIGURATION

/**
 * Dynamic GPU Parameters - Runtime Hardware Adaptation
 *
 * These macros provide unified access to GPU configuration parameters
 * from both host and device code. The configuration adapts at runtime
 * based on detected GPU capabilities for optimal performance.
 *
 * DESIGN PRINCIPLES:
 * - Host code: Direct variable access for configuration logic
 * - Device code: Constant memory access for high-performance kernel execution
 * - Unified interface: Same macro names work in both contexts
 * - Runtime adaptation: Parameters set based on actual hardware capabilities
 */
#ifdef __CUDA_ARCH__
// Device code: Access through constant memory for optimal performance
#define NUM_THREADS_PER_BLOCK (d_threads_per_block[0])
#define NUM_THREADS           (d_total_threads[0])
#else
// Host code: Direct variable access for configuration and management
#define NUM_THREADS_PER_BLOCK g_threads_per_block
#define NUM_THREADS           g_total_threads
#endif

#define NUM_BLOCKS g_num_blocks

////////////////////////////////////////////////////////////////////////////////
// TISSUE LAYER AND PHOTON DATA STRUCTURES

/**
 * LayerStruct - Optical Properties Definition for Tissue Layers
 *
 * Defines the optical and physical properties for each tissue layer
 * in the multi-layered medium. Memory-aligned for optimal GPU access.
 *
 * OPTICAL PARAMETERS:
 * - z_min/z_max: Layer boundaries in depth [cm]
 * - mutr: Transport mean free path = 1/(μs + μa) [cm]
 * - mua: Absorption coefficient [1/cm]
 * - g: Anisotropy factor [-1,1] for Henyey-Greenberg scattering
 * - n: Refractive index for Fresnel calculations
 *
 * MEMORY LAYOUT: 16-byte aligned for efficient GPU memory access
 */
typedef struct __align__(16) {
	float z_min; // Layer z_min [cm] - top boundary depth
	float z_max; // Layer z_max [cm] - bottom boundary depth
	float mutr;  // Reciprocal mu_total [cm] - transport mean free path
	float mua;   // Absorption coefficient [1/cm]
	float g;     // Anisotropy factor [-1,1] - forward scattering bias
	float n;     // Refractive index [-] - for Fresnel calculations
} LayerStruct;

/**
 * PhotonStruct - Complete Photon State Information
 *
 * Stores all information necessary to track a photon through the
 * multi-layered medium. Optimized for GPU thread-local storage.
 *
 * SPATIAL COORDINATES:
 * - (x,y,z): Current global position [cm]
 * - (dx,dy,dz): Unit direction vector (normalized)
 *
 * SIMULATION STATE:
 * - weight: Photon weight (integer representation for efficiency)
 * - layer: Current tissue layer index
 *
 * MEMORY LAYOUT: 16-byte aligned for coalesced GPU memory access
 */
typedef struct __align__(16) {
	float x;             // Global x coordinate [cm]
	float y;             // Global y coordinate [cm]
	float z;             // Global z coordinate [cm]
	float dx;            // (Global, normalized) x-direction cosine
	float dy;            // (Global, normalized) y-direction cosine
	float dz;            // (Global, normalized) z-direction cosine
	unsigned int weight; // Photon weight (integer representation)
	int layer;           // Current layer index [0 to n_layers-1]
} PhotonStruct;

/**
 * DetStruct - Detection Grid Configuration
 *
 * Defines the spatial and angular resolution for photon detection.
 * Used for Rd (reflectance), A (absorption), and Tt (transmittance)
 * measurements in cylindrical coordinates.
 *
 * GRID PARAMETERS:
 * - dr: Radial grid spacing [cm]
 * - dz: Depth grid spacing [cm]
 * - na: Number of angular bins for directional detection
 * - nr: Number of radial bins
 * - nz: Number of depth bins
 *
 * DETECTION ARRAYS:
 * - Reflectance: Rd[nr][na] - top surface detection
 * - Absorption: A[nr][nz] - volumetric absorption
 * - Transmittance: Tt[nr][na] - bottom surface detection
 */
typedef struct __align__(16) {
	float dr; // Detection grid resolution, r-direction [cm]
	float dz; // Detection grid resolution, z-direction [cm]

	int na;   // Number of grid elements in angular-direction [-]
	int nr;   // Number of grid elements in r-direction [-]
	int nz;   // Number of grid elements in z-direction [-]
} DetStruct;

/**
 * SimulationStruct - Complete Simulation Configuration
 *
 * Master configuration structure containing all parameters needed
 * to execute a multi-layered Monte Carlo photon transport simulation.
 *
 * SIMULATION PARAMETERS:
 * - number_of_photons: Total photons to launch
 * - ignoreAdetection: Flag to skip absorption detection (performance optimization)
 * - start_weight: Initial photon weight (typically 1.0 as integer)
 *
 * GEOMETRY CONFIGURATION:
 * - n_layers: Number of tissue layers
 * - layers: Array of optical properties per layer
 * - det: Detection grid configuration
 *
 * I/O CONFIGURATION:
 * - inp_filename: Input file with simulation parameters
 * - outp_filename: Output file for results
 *
 * PERFORMANCE TRACKING:
 * - begin, end: Timing markers for performance analysis
 * - AorB: Internal state flag
 */
typedef struct {
	unsigned long number_of_photons; // Total number of photons to simulate
	int ignoreAdetection;            // Skip absorption detection if true
	unsigned int n_layers;           // Number of tissue layers
	unsigned int start_weight;       // Initial photon weight (integer representation)
	char outp_filename[STR_LEN];     // Output file path
	char inp_filename[STR_LEN];      // Input file path
	long begin, end;                 // Timing markers
	char AorB;                       // Internal state flag
	DetStruct det;                   // Detection grid configuration
	LayerStruct* layers;             // Array of layer optical properties
} SimulationStruct;

/**
 * MemStruct - GPU Memory Management Structure
 *
 * Centralized management of all GPU memory allocations for the
 * Monte Carlo simulation. Provides organized access to device
 * memory pointers and ensures proper cleanup.
 *
 * PHOTON STATE ARRAYS:
 * - p: Array of photon structures (one per GPU thread)
 * - x: RNG state array for multiply-with-carry generators
 * - a: RNG multiplier array (safe primes for good statistical properties)
 *
 * THREAD MANAGEMENT:
 * - thread_active: Per-thread activity flags
 * - num_terminated_photons: Global counter for completed photons
 *
 * DETECTION ARRAYS:
 * - Rd_ra: Reflectance detection [radius][angle]
 * - A_rz: Absorption detection [radius][depth]
 * - Tt_ra: Transmittance detection [radius][angle]
 */
typedef struct {
	PhotonStruct* p;                      // Photon state array (one per thread)
	unsigned long long* x;                // RNG state array (MWC generators)
	unsigned int* a;                      // RNG multiplier array (safe primes)
	unsigned int* thread_active;          // Per-thread activity flags
	unsigned int* num_terminated_photons; // Global counter for terminated photons

	unsigned long long* Rd_ra;            // Reflectance detection array
	unsigned long long* A_rz;             // Absorption detection array
	unsigned long long* Tt_ra;            // Transmittance detection array
} MemStruct;

////////////////////////////////////////////////////////////////////////////////
// GPU CONSTANT MEMORY DECLARATIONS

/**
 * Device Constant Memory Variables
 *
 * These variables reside in GPU constant memory for high-performance
 * access from device kernels. Constant memory is cached and broadcast
 * to all threads simultaneously, making it ideal for frequently
 * accessed, read-only simulation parameters.
 *
 * THREAD CONFIGURATION:
 * - d_threads_per_block: Threads per block for kernel launches
 * - d_total_threads: Total GPU threads across all blocks
 *
 * SIMULATION PARAMETERS:
 * - num_photons_dc: Number of photons per thread
 * - n_layers_dc: Number of tissue layers
 * - start_weight_dc: Initial photon weight
 *
 * GEOMETRY DATA:
 * - layers_dc: Array of tissue layer properties
 * - det_dc: Detection grid configuration
 */

// GPU thread configuration in constant memory
__device__ __constant__ int d_threads_per_block[1];
__device__ __constant__ int d_total_threads[1];

// Simulation parameters in constant memory (high-frequency access)
__device__ __constant__ unsigned int num_photons_dc[1];
__device__ __constant__ unsigned int n_layers_dc[1];
__device__ __constant__ unsigned int start_weight_dc[1];
__device__ __constant__ LayerStruct layers_dc[MAX_LAYERS];
__device__ __constant__ DetStruct det_dc[1];

////////////////////////////////////////////////////////////////////////////////
// EXTERNAL VARIABLE DECLARATIONS - HOST CONFIGURATION

/**
 * Dynamic GPU Configuration Variables
 *
 * These variables are set at runtime based on detected GPU hardware
 * capabilities. They provide optimal thread configuration for the
 * specific GPU being used while maintaining compatibility across
 * different GPU architectures.
 *
 * CONFIGURATION STRATEGY:
 * - g_num_blocks: Set to maximize occupancy based on multiprocessor count
 * - g_threads_per_block: Optimized for memory coalescing and occupancy
 * - g_total_threads: Total concurrent threads for workload distribution
 *
 * USAGE:
 * These variables are initialized by the GPU detection and configuration
 * routine and then copied to constant memory for device kernel access.
 */
extern int g_num_blocks;        // Number of thread blocks for kernel launch
extern int g_threads_per_block; // Threads per block (typically 256-512)
extern int g_total_threads;     // Total threads across all blocks

#endif                          // CUDAMCML_H
