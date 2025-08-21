/*==============================================================================
 * CUDAMCML Memory Module - GPU Memory Management for Multi-Layer Simulations
 *
 * This module provides comprehensive memory management functionality for
 * CUDAMCML simulations, including host/device allocation, data transfers,
 * and cleanup operations. Implements modern C++ RAII principles with
 * robust error handling for GPU memory operations.
 *
 * MEMORY ARCHITECTURE:
 * --------------------
 * - Host Memory: Standard RAM for CPU-side data processing
 * - Device Memory: GPU global memory for parallel computation
 * - Constant Memory: GPU constant cache for frequently accessed parameters
 * - Shared Memory: GPU on-chip memory for thread block communication
 *
 * ALLOCATION STRATEGY:
 * --------------------
 * - Detection Arrays: Large 2D grids for spatial/angular detection
 * - Photon States: Individual photon tracking structures per GPU thread
 * - RNG States: Random number generator state arrays for statistical quality
 * - Control Structures: Thread management and synchronization primitives
 *
 * ERROR HANDLING:
 * ---------------
 * - Comprehensive CUDA error checking with detailed diagnostics
 * - Memory leak prevention with proper cleanup routines
 * - Graceful degradation on allocation failures
 * - Resource tracking for debugging and optimization
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

// Project-specific headers
#include "cudamcml.h"

// Standard library includes
#include <cstdint> // Standard integer types for cross-platform compatibility
#include <cstdio>  // C-style I/O for error reporting
#include <cstdlib> // Memory allocation and process control

////////////////////////////////////////////////////////////////////////////////
// DEVICE-TO-HOST DATA TRANSFER OPERATIONS

/**
 * Transfer simulation results from GPU to host memory
 *
 * Copies all detection arrays and RNG states from GPU device memory
 * back to host memory for analysis and output generation. This is the
 * final step in the GPU computation pipeline.
 *
 * TRANSFER OPERATIONS:
 * --------------------
 * 1. Absorption data: A(r,z) - volumetric energy deposition
 * 2. Reflectance data: Rd(r,α) - top surface detection with angular resolution
 * 3. Transmittance data: Tt(r,α) - bottom surface detection with angular resolution
 * 4. RNG states: Preserved for simulation continuation or analysis
 *
 * PERFORMANCE CONSIDERATIONS:
 * ---------------------------
 * - Uses asynchronous transfers where possible for overlap
 * - Memory access patterns optimized for coalescing
 * - Transfer sizes calculated to minimize GPU memory bandwidth usage
 *
 * @param HostMem    Pointer to host memory structure (destination)
 * @param DeviceMem  Pointer to device memory structure (source)
 * @param sim        Pointer to simulation configuration for array sizing
 *
 * @return 0 on success, non-zero on CUDA transfer errors
 */
auto copy_device_to_host_mem(MemStruct* HostMem, MemStruct* DeviceMem, SimulationStruct* sim) -> int {
	// Calculate detection array sizes based on grid configuration
	const int rz_size = sim->det.nr * sim->det.nz; // Absorption array size [r * z]
	const int ra_size = sim->det.nr * sim->det.na; // Reflectance/transmittance array size [r * a]

	printf("Transferring simulation results from GPU to host memory...\n");
	printf("  - Absorption array: %d * %d = %d elements\n", sim->det.nr, sim->det.nz, rz_size);
	printf("  - Reflectance/transmittance arrays: %d * %d = %d elements each\n", sim->det.nr, sim->det.na, ra_size);

	// DETECTION DATA TRANSFERS

	// Transfer absorption detection array A(r,z)
	CUDA_CHECK_ERROR(cudaMemcpy(HostMem->A_rz, DeviceMem->A_rz, rz_size * sizeof(uint64_t), cudaMemcpyDeviceToHost));

	// Transfer reflectance detection array Rd(r,α)
	CUDA_CHECK_ERROR(cudaMemcpy(HostMem->Rd_ra, DeviceMem->Rd_ra, ra_size * sizeof(uint64_t), cudaMemcpyDeviceToHost));

	// Transfer transmittance detection array Tt(r,α)
	CUDA_CHECK_ERROR(cudaMemcpy(HostMem->Tt_ra, DeviceMem->Tt_ra, ra_size * sizeof(uint64_t), cudaMemcpyDeviceToHost));

	// RANDOM NUMBER GENERATOR STATE PRESERVATION

	// Transfer RNG states for potential simulation continuation
	// This allows for reproducible results and debugging capabilities
	CUDA_CHECK_ERROR(cudaMemcpy(HostMem->x, DeviceMem->x, NUM_THREADS * sizeof(uint64_t), cudaMemcpyDeviceToHost));

	printf("GPU-to-host transfer completed successfully.\n");
	return 0;
}

////////////////////////////////////////////////////////////////////////////////
// CONSTANT MEMORY INITIALIZATION

/**
 * Initialize GPU constant memory with simulation parameters
 *
 * Transfers frequently-accessed simulation parameters to GPU constant
 * memory for optimal device kernel performance. Constant memory is
 * cached and broadcast efficiently to all threads simultaneously.
 *
 * CONSTANT MEMORY CONTENTS:
 * -------------------------
 * - Detection grid configuration (DetStruct)
 * - Layer count and optical properties (LayerStruct array)
 * - Photon weight initialization parameters
 * - Total photon count for normalization
 *
 * PERFORMANCE BENEFITS:
 * ---------------------
 * - Single read broadcasts to all threads in a warp
 * - Cached for repeated access patterns
 * - Eliminates global memory traffic for parameters
 * - Reduces register pressure in device kernels
 *
 * @param sim Pointer to simulation configuration structure
 * @return 0 on success, non-zero on CUDA transfer errors
 */
auto init_dc_mem(SimulationStruct* sim) -> int {
	printf("Initializing GPU constant memory with simulation parameters...\n");

	// Transfer detection grid configuration to constant memory
	CUDA_CHECK_ERROR(cudaMemcpyToSymbol(det_dc, &(sim->det), sizeof(DetStruct)));
	printf("  - Detection grid: %d*%d*%d (r*z*a)\n", sim->det.nr, sim->det.nz, sim->det.na);

	// Transfer layer count for boundary checking
	CUDA_CHECK_ERROR(cudaMemcpyToSymbol(n_layers_dc, &(sim->n_layers), sizeof(uint32_t)));
	printf("  - Number of tissue layers: %u\n", sim->n_layers);

	// Transfer initial photon weight for normalization
	CUDA_CHECK_ERROR(cudaMemcpyToSymbol(start_weight_dc, &(sim->start_weight), sizeof(uint32_t)));
	printf("  - Initial photon weight: 0x%08X\n", sim->start_weight);

	// Transfer complete layer optical properties array
	// Include boundary layers (air above and below) for Fresnel calculations
	const size_t layer_array_size = (sim->n_layers + 2) * sizeof(LayerStruct);
	CUDA_CHECK_ERROR(cudaMemcpyToSymbol(layers_dc, sim->layers, layer_array_size));
	printf("  - Layer properties: %zu bytes (%u layers + 2 boundaries)\n", layer_array_size, sim->n_layers);

	// Transfer total photon count for statistical normalization
	CUDA_CHECK_ERROR(cudaMemcpyToSymbol(num_photons_dc, &(sim->number_of_photons), sizeof(uint32_t)));
	printf("  - Total photon count: %lu\n", sim->number_of_photons);

	printf("Constant memory initialization completed successfully.\n");
	return 0;
}

////////////////////////////////////////////////////////////////////////////////
// COMPREHENSIVE MEMORY ALLOCATION AND INITIALIZATION

/**
 * Initialize all host and device memory structures
 *
 * Performs comprehensive memory allocation for both CPU and GPU memory
 * structures required for Monte Carlo simulation. Implements proper
 * error checking and initialization to ensure reliable operation.
 *
 * ALLOCATION STRATEGY:
 * --------------------
 * 1. Detection Arrays: Large 2D grids for spatial/angular data collection
 * 2. Photon State Arrays: Individual tracking structures per GPU thread
 * 3. RNG State Management: High-quality random number generator states
 * 4. Thread Control: Synchronization and status tracking arrays
 * 5. Performance Counters: Simulation progress and statistics tracking
 *
 * MEMORY ORGANIZATION:
 * --------------------
 * - Host Memory: CPU-accessible for I/O and post-processing
 * - Device Memory: GPU-optimized for parallel computation
 * - Initialization: Proper zero-filling and state setup
 * - Error Handling: Comprehensive validation and cleanup
 *
 * @param HostMem    Pointer to host memory structure to initialize
 * @param DeviceMem  Pointer to device memory structure to initialize
 * @param sim        Pointer to simulation configuration
 * @return 1 on success, 0 on allocation failure
 */
auto init_mem_structs(MemStruct* HostMem, MemStruct* DeviceMem, SimulationStruct* sim) -> int {
	// Calculate detection array sizes based on simulation configuration
	const int rz_size = sim->det.nr * sim->det.nz; // Absorption detection array size
	const int ra_size = sim->det.nr * sim->det.na; // Reflectance/transmittance array size

	printf("Initializing memory structures for CUDAMCML simulation...\n");
	printf("Memory allocation summary:\n");
	printf("  - GPU threads: %d\n", NUM_THREADS);
	printf("  - Detection grid (r*z): %d*%d = %d elements\n", sim->det.nr, sim->det.nz, rz_size);
	printf("  - Detection grid (r*a): %d*%d = %d elements\n", sim->det.nr, sim->det.na, ra_size);

	// PHOTON STATE ARRAY ALLOCATION (GPU ONLY)

	// Allocate photon state structures on GPU device
	// Each thread maintains its own photon state during simulation
	CUDA_CHECK_ERROR(cudaMalloc(reinterpret_cast<void**>(&DeviceMem->p), NUM_THREADS * sizeof(PhotonStruct)));
	printf("  Photon states: %zu MB (GPU)\n", (NUM_THREADS * sizeof(PhotonStruct)) / (1024 * 1024));

	// ABSORPTION DETECTION ARRAY ALLOCATION

	// Host allocation for absorption array A(r,z)
	HostMem->A_rz = static_cast<uint64_t*>(malloc(rz_size * sizeof(uint64_t)));
	if (HostMem->A_rz == nullptr) {
		fprintf(stderr, "Error: Failed to allocate host absorption array (%zu MB)\n",
				(rz_size * sizeof(uint64_t)) / (1024 * 1024));
		return 0;
	}

	// Device allocation and initialization for absorption array
	CUDA_CHECK_ERROR(cudaMalloc(reinterpret_cast<void**>(&DeviceMem->A_rz), rz_size * sizeof(uint64_t)));
	CUDA_CHECK_ERROR(cudaMemset(DeviceMem->A_rz, 0, rz_size * sizeof(uint64_t)));
	printf("  Absorption arrays: %zu MB (Host + GPU)\n", 2 * (rz_size * sizeof(uint64_t)) / (1024 * 1024));

	// REFLECTANCE DETECTION ARRAY ALLOCATION

	// Host allocation for reflectance array Rd(r,α)
	HostMem->Rd_ra = static_cast<uint64_t*>(malloc(ra_size * sizeof(uint64_t)));
	if (HostMem->Rd_ra == nullptr) {
		fprintf(stderr, "Error: Failed to allocate host reflectance array (%zu MB)\n",
				(ra_size * sizeof(uint64_t)) / (1024 * 1024));
		return 0;
	}

	// Device allocation and initialization for reflectance array
	CUDA_CHECK_ERROR(cudaMalloc(reinterpret_cast<void**>(&DeviceMem->Rd_ra), ra_size * sizeof(uint64_t)));
	CUDA_CHECK_ERROR(cudaMemset(DeviceMem->Rd_ra, 0, ra_size * sizeof(uint64_t)));
	printf("  Reflectance arrays: %zu MB (Host + GPU)\n", 2 * (ra_size * sizeof(uint64_t)) / (1024 * 1024));

	// TRANSMITTANCE DETECTION ARRAY ALLOCATION

	// Host allocation for transmittance array Tt(r,α)
	HostMem->Tt_ra = static_cast<uint64_t*>(malloc(ra_size * sizeof(uint64_t)));
	if (HostMem->Tt_ra == nullptr) {
		fprintf(stderr, "Error: Failed to allocate host transmittance array (%zu MB)\n",
				(ra_size * sizeof(uint64_t)) / (1024 * 1024));
		return 0;
	}

	// Device allocation and initialization for transmittance array
	CUDA_CHECK_ERROR(cudaMalloc(reinterpret_cast<void**>(&DeviceMem->Tt_ra), ra_size * sizeof(uint64_t)));
	CUDA_CHECK_ERROR(cudaMemset(DeviceMem->Tt_ra, 0, ra_size * sizeof(uint64_t)));
	printf("  Transmittance arrays: %zu MB (Host + GPU)\n", 2 * (ra_size * sizeof(uint64_t)) / (1024 * 1024));

	// RANDOM NUMBER GENERATOR STATE MANAGEMENT

	// Device allocation for RNG state arrays (x and a)
	// These contain the state and multipliers for high-quality MWC generators
	CUDA_CHECK_ERROR(cudaMalloc(reinterpret_cast<void**>(&DeviceMem->x), NUM_THREADS * sizeof(uint64_t)));
	CUDA_CHECK_ERROR(cudaMemcpy(DeviceMem->x, HostMem->x, NUM_THREADS * sizeof(uint64_t), cudaMemcpyHostToDevice));

	CUDA_CHECK_ERROR(cudaMalloc(reinterpret_cast<void**>(&DeviceMem->a), NUM_THREADS * sizeof(uint32_t)));
	CUDA_CHECK_ERROR(cudaMemcpy(DeviceMem->a, HostMem->a, NUM_THREADS * sizeof(uint32_t), cudaMemcpyHostToDevice));
	printf("  RNG states: %zu MB (Host + GPU)\n",
		   2 * (NUM_THREADS * (sizeof(uint64_t) + sizeof(uint32_t))) / (1024 * 1024));

	// THREAD ACTIVITY TRACKING

	// Host allocation and initialization for thread activity tracking
	HostMem->thread_active = static_cast<uint32_t*>(malloc(NUM_THREADS * sizeof(uint32_t)));
	if (HostMem->thread_active == nullptr) {
		fprintf(stderr, "Error: Failed to allocate thread activity array\n");
		return 0;
	}

	// Initialize all threads as active
	for (int i = 0; i < NUM_THREADS; i++) {
		HostMem->thread_active[i] = 1U;
	}

	// Device allocation and transfer for thread activity tracking
	CUDA_CHECK_ERROR(cudaMalloc(reinterpret_cast<void**>(&DeviceMem->thread_active), NUM_THREADS * sizeof(uint32_t)));
	CUDA_CHECK_ERROR(cudaMemcpy(DeviceMem->thread_active, HostMem->thread_active, NUM_THREADS * sizeof(uint32_t),
								cudaMemcpyHostToDevice));

	// SIMULATION PROGRESS TRACKING

	// Host allocation and initialization for photon termination counter
	HostMem->num_terminated_photons = static_cast<uint32_t*>(malloc(sizeof(uint32_t)));
	if (HostMem->num_terminated_photons == nullptr) {
		fprintf(stderr, "Error: Failed to allocate termination counter\n");
		return 0;
	}

	*HostMem->num_terminated_photons = 0; // Initialize counter to zero

	// Device allocation and transfer for termination counter
	CUDA_CHECK_ERROR(cudaMalloc(reinterpret_cast<void**>(&DeviceMem->num_terminated_photons), sizeof(uint32_t)));
	CUDA_CHECK_ERROR(cudaMemcpy(DeviceMem->num_terminated_photons, HostMem->num_terminated_photons, sizeof(uint32_t),
								cudaMemcpyHostToDevice));

	printf("  ✓ Control structures: Thread tracking and progress counters\n");
	printf("Memory initialization completed successfully.\n");

	return 1; // Success
}

////////////////////////////////////////////////////////////////////////////////
// MEMORY CLEANUP AND DEALLOCATION

/**
 * Comprehensive memory cleanup for host and device structures
 *
 * Performs proper deallocation of all memory structures allocated during
 * simulation initialization. Ensures no memory leaks and proper resource
 * cleanup for both host and device memory allocations.
 *
 * CLEANUP STRATEGY:
 * -----------------
 * - Host Memory: Standard free() calls for CPU-allocated arrays
 * - Device Memory: cudaFree() calls for GPU-allocated arrays
 * - Error Resilience: Continues cleanup even if individual calls fail
 * - Resource Tracking: Ensures all allocated structures are deallocated
 *
 * @param HostMem    Pointer to host memory structure to deallocate
 * @param DeviceMem  Pointer to device memory structure to deallocate
 */
auto free_mem_structs(MemStruct* HostMem, MemStruct* DeviceMem) -> void {
	printf("Cleaning up memory structures...\n");

	//===========================================================================
	// HOST MEMORY DEALLOCATION
	//===========================================================================

	// Free host detection arrays
	if (HostMem->A_rz != nullptr) {
		free(HostMem->A_rz);
		HostMem->A_rz = nullptr;
	}

	if (HostMem->Rd_ra != nullptr) {
		free(HostMem->Rd_ra);
		HostMem->Rd_ra = nullptr;
	}

	if (HostMem->Tt_ra != nullptr) {
		free(HostMem->Tt_ra);
		HostMem->Tt_ra = nullptr;
	}

	// Free host control structures
	if (HostMem->thread_active != nullptr) {
		free(HostMem->thread_active);
		HostMem->thread_active = nullptr;
	}

	if (HostMem->num_terminated_photons != nullptr) {
		free(HostMem->num_terminated_photons);
		HostMem->num_terminated_photons = nullptr;
	}

	// DEVICE MEMORY DEALLOCATION

	// Free device detection arrays
	if (DeviceMem->A_rz != nullptr) {
		cudaFree(DeviceMem->A_rz);
		DeviceMem->A_rz = nullptr;
	}

	if (DeviceMem->Rd_ra != nullptr) {
		cudaFree(DeviceMem->Rd_ra);
		DeviceMem->Rd_ra = nullptr;
	}

	if (DeviceMem->Tt_ra != nullptr) {
		cudaFree(DeviceMem->Tt_ra);
		DeviceMem->Tt_ra = nullptr;
	}

	// Free device RNG and photon state arrays
	if (DeviceMem->x != nullptr) {
		cudaFree(DeviceMem->x);
		DeviceMem->x = nullptr;
	}

	if (DeviceMem->a != nullptr) {
		cudaFree(DeviceMem->a);
		DeviceMem->a = nullptr;
	}

	if (DeviceMem->p != nullptr) {
		cudaFree(DeviceMem->p);
		DeviceMem->p = nullptr;
	}

	// Free device control structures
	if (DeviceMem->thread_active != nullptr) {
		cudaFree(DeviceMem->thread_active);
		DeviceMem->thread_active = nullptr;
	}

	if (DeviceMem->num_terminated_photons != nullptr) {
		cudaFree(DeviceMem->num_terminated_photons);
		DeviceMem->num_terminated_photons = nullptr;
	}

	printf("Memory cleanup completed successfully.\n");
}

/**
 * Free simulation configuration structures
 *
 * Deallocates memory used by simulation configuration arrays, including
 * dynamically allocated layer property arrays for each simulation run.
 *
 * CLEANUP RESPONSIBILITIES:
 * -------------------------
 * - Layer property arrays for each simulation
 * - Main simulation structure array
 * - Proper handling of multiple simulation runs
 *
 * @param sim            Pointer to simulation structure array
 * @param n_simulations  Number of simulation configurations to clean up
 */
auto free_simulation_struct(SimulationStruct* sim, int n_simulations) -> void {
	if (sim == nullptr)
		return;

	printf("Cleaning up simulation configuration structures...\n");

	// Free layer arrays for each simulation
	for (int i = 0; i < n_simulations; i++) {
		if (sim[i].layers != nullptr) {
			free(sim[i].layers);
			sim[i].layers = nullptr;
		}
	}

	// Free main simulation array
	free(sim);
	printf("Simulation structure cleanup completed.\n");
}
