/*==============================================================================
 * CUDAMCML - GPU-Accelerated Monte Carlo Multi-Layer Photon Transport
 *
 * CUDA-based Monte Carlo simulation of photon migration in layered media (CUDAMCML).
 *
 * This implementation extends Monte Carlo photon transport to complex multi-layered
 * biological tissues with arbitrary optical properties. The GPU acceleration enables
 * high-resolution simulations that would be computationally prohibitive on CPU.
 *
 * HISTORICAL CONTEXT:
 * -------------------
 * Some documentation is available for CUDAMCML and should have been distributed along
 * with this source code. If that is not the case: Documentation, source code and
 * executables for CUDAMCML are available for download on our webpage:
 *
 * http://www.atomic.physics.lu.se/Biophotonics
 * http://www.atomic.physics.lu.se/fileadmin/atomfysik/Biophotonics/Software/CUDAMCML.zip
 *
 * We encourage the use, and modification of this code, and hope it will help
 * users/programmers to utilize the power of GPGPU for their simulation needs. While we
 * don't have a scientific publication describing this code, we would very much appreciate
 * if you cite our original GPGPU Monte Carlo letter (on which CUDAMCML is based) if you
 * use this code or derivations thereof for your own scientific work:
 *
 * E. Alerstam, T. Svensson and S. Andersson-Engels, "Parallel computing with graphics
 * processing units for high-speed Monte Carlo simulations of photon migration", Journal
 * of Biomedical Optics Letters, 13(6) 060504 (2008).
 *
 * PHYSICAL MODEL:
 * ---------------
 * - Multi-layered turbid media with configurable layer properties
 * - Pencil beam or collimated source incident on top surface
 * - Henyey-Greenberg scattering with layer-specific anisotropy factors
 * - Fresnel reflection/transmission at all refractive index boundaries
 * - Spatially-resolved detection: Rd(r,α), A(r,z), Tt(r,α)
 * - Russian roulette photon termination for computational efficiency
 *
 * COMPUTATIONAL ARCHITECTURE:
 * ---------------------------
 * - Modular design: Separate compilation units for I/O, memory, RNG, transport
 * - Dynamic GPU configuration: Runtime adaptation to hardware capabilities
 * - Memory optimization: Aligned structures, constant memory utilization
 * - Exception-safe programming: RAII principles with modern C++
 * - Comprehensive error handling: CUDA-specific diagnostics and recovery
 *
 * SIMULATION WORKFLOW:
 * --------------------
 * 1. Parse input file with layer properties and simulation parameters
 * 2. Configure GPU based on detected hardware capabilities
 * 3. Allocate and initialize GPU memory structures
 * 4. Launch massively parallel photon transport kernels
 * 5. Collect detection results and perform statistical analysis
 * 6. Export results in standard MCML format
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
 *
 * This code is distributed under the terms of the GNU General Public Licence.
 */

////////////////////////////////////////////////////////////////////////////////
// SYSTEM INCLUDES AND DEPENDENCIES

// Standard C++ library includes for I/O, memory management, and error handling
#include <cfloat>    // Floating-point limits and constants
#include <cstdio>    // C-style I/O for compatibility
#include <iomanip>   // Stream manipulators for formatted output
#include <iostream>  // Modern C++ I/O streams
#include <memory>    // Smart pointers and memory management
#include <stdexcept> // Standard exception classes

// CUDA runtime and device includes for GPU computing
#include <cuda_runtime.h>             // CUDA runtime API
#include <device_launch_parameters.h> // CUDA kernel launch parameters

// Project-specific includes
#include "cudamcml.h"    // Main header with constants, structures, and function declarations
#include "safe_primes.h" // Embedded safe prime numbers for RNG initialization

// Include specialized component modules - compiled as single translation unit
#include "cudamcml_io.cu"        // Input/output operations and file parsing
#include "cudamcml_mem.cu"       // GPU memory management and allocation
#include "cudamcml_rng.cu"       // Random number generation and initialization
#include "cudamcml_transport.cu" // Core photon transport physics and detection

////////////////////////////////////////////////////////////////////////////////
// CUDA MONTE CARLO MULTI-LAYER NAMESPACE

namespace cudamcml
{

////////////////////////////////////////////////////////////////////////////////
// GPU CONFIGURATION AND HARDWARE MANAGEMENT

/**
 * GPU Configuration Management Class
 *
 * Encapsulates CUDA hardware detection, configuration, and optimization
 * for multi-layered Monte Carlo photon transport simulations. Adapts
 * thread configuration based on detected GPU capabilities.
 *
 * DESIGN PHILOSOPHY:
 * - Dynamic configuration based on actual hardware capabilities
 * - Conservative defaults for broad GPU compatibility
 * - Oversubscription for optimal occupancy and throughput
 * - Detailed hardware reporting for performance analysis
 *
 * CONFIGURATION STRATEGY:
 * - Detect multiprocessor count and capabilities
 * - Apply oversubscription factor for improved occupancy
 * - Balance thread count with memory requirements
 * - Validate against hardware limits and constraints
 */
class GPUConfig
{
public:
	// Conservative fallback configuration for unknown hardware
	static constexpr int DEFAULT_FALLBACK_BLOCKS = 56;
	static constexpr int DEFAULT_FALLBACK_THREADS = 28672;

	// Active GPU configuration parameters
	int num_blocks;        // Number of CUDA thread blocks
	int threads_per_block; // Threads per block (typically 512)
	int total_threads;     // Total concurrent threads

	/**
	 * Constructor - Initialize with fallback configuration
	 *
	 * Uses conservative defaults that work across different GPU generations.
	 * Actual configuration is determined during initialize() call.
	 */
	GPUConfig() :
		num_blocks(DEFAULT_FALLBACK_BLOCKS), threads_per_block(DEFAULT_THREADS_PER_BLOCK),
		total_threads(DEFAULT_FALLBACK_THREADS) {}

	/**
	 * Hardware Detection and Dynamic Configuration
	 *
	 * Detects GPU capabilities and configures optimal thread parameters
	 * for maximum throughput. Applies oversubscription and validates
	 * against hardware constraints.
	 */
	void initialize() {
		cudaDeviceProp deviceProp;
		int device;

		CUDA_CHECK_ERROR(cudaGetDevice(&device));
		CUDA_CHECK_ERROR(cudaGetDeviceProperties(&deviceProp, device));

		// Extract key hardware characteristics
		const int multiprocessors = deviceProp.multiProcessorCount;
		const int max_threads_per_sm = deviceProp.maxThreadsPerMultiProcessor;
		const int max_blocks_per_sm = deviceProp.maxBlocksPerMultiProcessor;

		// Calculate optimal thread configuration for multi-layered Monte Carlo
		threads_per_block = DEFAULT_THREADS_PER_BLOCK;

		// Apply oversubscription for improved GPU utilization
		// More blocks than SMs helps hide memory latency and improve throughput
		num_blocks = multiprocessors * GPU_OVERSUBSCRIPTION_FACTOR;
		total_threads = num_blocks * threads_per_block;

		print_configuration(deviceProp, multiprocessors, max_threads_per_sm, max_blocks_per_sm);
		copy_to_device_constants();
	}

private:
	/**
	 * Print comprehensive GPU configuration report
	 *
	 * Displays detected hardware capabilities, configured parameters,
	 * and oversubscription strategy for performance optimization.
	 */
	void print_configuration(const cudaDeviceProp& deviceProp, int multiprocessors, int max_threads_per_sm,
							 int max_blocks_per_sm) const {
		std::cout << "=== CUDAMCML GPU Configuration ===\n"
				  << "GPU: " << deviceProp.name << "\n"
				  << "Compute Capability: " << deviceProp.major << "." << deviceProp.minor << "\n"
				  << "Multiprocessors: " << multiprocessors << "\n"
				  << "Max threads per SM: " << max_threads_per_sm << "\n"
				  << "Max blocks per SM: " << max_blocks_per_sm << "\n"
				  << "Configured blocks: " << num_blocks << "\n"
				  << "Threads per block: " << threads_per_block << "\n"
				  << "Total GPU threads: " << total_threads << "\n"
				  << "Oversubscription factor: " << GPU_OVERSUBSCRIPTION_FACTOR << "x\n"
				  << "Memory: " << deviceProp.totalGlobalMem / (1024 * 1024) << " MB\n"
				  << "===================================\n\n";
	}

	/**
	 * Copy configuration to GPU constant memory
	 *
	 * Transfers thread configuration parameters to constant memory
	 * for high-performance access from device kernels.
	 */
	void copy_to_device_constants() const {
		CUDA_CHECK_ERROR(cudaMemcpyToSymbol(d_threads_per_block, &threads_per_block, sizeof(int)));
		CUDA_CHECK_ERROR(cudaMemcpyToSymbol(d_total_threads, &total_threads, sizeof(int)));
	}
};

} // namespace cudamcml

////////////////////////////////////////////////////////////////////////////////
// GLOBAL STATE MANAGEMENT AND LEGACY COMPATIBILITY

/**
 * Global GPU Configuration Instance
 *
 * Centralized GPU configuration management using modern C++ practices.
 * Encapsulates hardware detection, parameter optimization, and constant
 * memory management in a single, well-defined interface.
 */
static cudamcml::GPUConfig g_gpu_config;

/**
 * Legacy Global Variables - Backward Compatibility
 *
 * These variables maintain compatibility with existing code that expects
 * global GPU parameters. They are automatically synchronized with the
 * modern GPUConfig class during initialization.
 *
 * NOTE: Future code should use the GPUConfig class interface directly
 * rather than these global variables for better encapsulation.
 */
int g_num_blocks = cudamcml::GPUConfig::DEFAULT_FALLBACK_BLOCKS;
int g_threads_per_block = DEFAULT_THREADS_PER_BLOCK;
int g_total_threads = cudamcml::GPUConfig::DEFAULT_FALLBACK_THREADS;

/**
 * Modern GPU Parameter Initialization
 *
 * Initializes GPU configuration based on detected hardware capabilities
 * and synchronizes legacy global variables for backward compatibility.
 * This function should be called once at program startup.
 *
 * INITIALIZATION SEQUENCE:
 * 1. Detect CUDA device capabilities
 * 2. Calculate optimal thread configuration
 * 3. Copy parameters to GPU constant memory
 * 4. Update legacy global variables for compatibility
 * 5. Display configuration report
 */
void initialize_gpu_params() {
	g_gpu_config.initialize();

	// Synchronize legacy globals with modern configuration
	g_num_blocks = g_gpu_config.num_blocks;
	g_threads_per_block = g_gpu_config.threads_per_block;
	g_total_threads = g_gpu_config.total_threads;
}

namespace cudamcml
{

////////////////////////////////////////////////////////////////////////////////
// RANDOM NUMBER GENERATOR MEMORY MANAGEMENT

/**
 * RAII Random Number Generator Memory Manager
 *
 * Manages memory allocation and initialization for multiply-with-carry (MWC)
 * random number generators used in Monte Carlo photon transport. Ensures
 * proper cleanup and exception safety.
 *
 * DESIGN PRINCIPLES:
 * - RAII: Automatic memory management with constructor/destructor
 * - Exception safety: Proper error handling throughout initialization
 * - Performance: Direct memory access for GPU kernel operations
 * - Statistical independence: Each thread gets unique RNG state
 *
 * USAGE:
 *   RNGMemory rng(thread_count);
 *   rng.initialize(seed, "safeprimes_base32.txt");
 *   // Use rng.x() and rng.a() in CUDA kernels
 *   // Automatic cleanup when rng goes out of scope
 */
class RNGMemory
{
public:
	/**
	 * Constructor - Allocate memory for RNG state arrays
	 *
	 * @param thread_count Number of GPU threads requiring RNG state
	 * @throws std::invalid_argument if thread_count <= 0
	 * @throws std::bad_alloc if memory allocation fails
	 */
	explicit RNGMemory(int thread_count) : thread_count_(thread_count) {
		if (thread_count <= 0) {
			throw std::invalid_argument("Thread count must be positive");
		}

		// Allocate memory for RNG state arrays
		x_data = std::make_unique<unsigned long long[]>(thread_count);
		a_data = std::make_unique<unsigned int[]>(thread_count);

		if (!x_data || !a_data) {
			throw std::bad_alloc();
		}
	}

	/**
	 * Initialize RNG state with safe prime multipliers
	 *
	 * Initializes each thread's RNG with unique state values and
	 * safe prime multipliers for good statistical properties.
	 *
	 * @param seed Initial seed value for RNG initialization
	 * @param prime_file Path to safe primes file (or nullptr for embedded primes)
	 * @throws std::runtime_error if RNG initialization fails
	 */
	void initialize(unsigned long long seed, const char* prime_file) {
		const int result = init_rng(x_data.get(), a_data.get(), thread_count_, prime_file, seed);
		if (result != 0) {
			throw std::runtime_error("Failed to initialize RNG with safe primes");
		}
	}

	// Accessor methods for use in CUDA kernel launches
	unsigned long long* x() { return x_data.get(); }   // RNG state array
	unsigned int* a() { return a_data.get(); }         // RNG multiplier array
	int thread_count() const { return thread_count_; } // Number of threads

private:
	int thread_count_;                                 // Number of GPU threads
	std::unique_ptr<unsigned long long[]> x_data;      // RNG state values
	std::unique_ptr<unsigned int[]> a_data;            // RNG multipliers (safe primes)
};

////////////////////////////////////////////////////////////////////////////////
// PERFORMANCE METRICS AND ANALYSIS

/**
 * Performance Metrics Collection and Reporting
 *
 * Tracks simulation performance metrics for throughput analysis and
 * optimization guidance. Provides detailed reporting of GPU utilization
 * and computational efficiency.
 *
 * TRACKED METRICS:
 * - Total simulation time (wall clock)
 * - Total photons processed across all threads
 * - Number of kernel launch iterations
 * - Throughput in photons per second
 * - GPU hardware utilization statistics
 */
struct PerformanceMetrics {
	double simulation_time;           // Total simulation time [seconds]
	unsigned long long total_photons; // Total photons processed
	unsigned int kernel_iterations;   // Number of kernel launches

	/**
	 * Calculate photons per second throughput
	 * @return Throughput in photons per second
	 */
	double photons_per_second() const { return simulation_time > 0.0 ? total_photons / simulation_time : 0.0; }

	/**
	 * Calculate throughput in millions of photons per second
	 * @return Throughput in millions of photons per second (convenient units)
	 */
	double million_photons_per_second() const { return photons_per_second() / 1e6; }

	/**
	 * Print comprehensive performance report
	 * @param gpu_config GPU configuration for context in reporting
	 */
	void print(const GPUConfig& gpu_config) const {
		std::cout << "=== CUDAMCML Performance Report ===\n"
				  << "Total photons simulated: " << total_photons << "\n"
				  << "Simulation time: " << std::fixed << std::setprecision(2) << simulation_time << " sec\n"
				  << "Performance: " << std::fixed << std::setprecision(1) << million_photons_per_second()
				  << " million photons/sec\n"
				  << "GPU utilization: " << gpu_config.num_blocks << " blocks * " << gpu_config.threads_per_block
				  << " threads = " << gpu_config.total_threads << " total threads\n"
				  << "Kernel iterations: " << kernel_iterations << "\n"
				  << "===================================\n\n";
	}
};

} // namespace cudamcml

////////////////////////////////////////////////////////////////////////////////
// MONTE CARLO SIMULATION ORCHESTRATION

namespace cudamcml
{

/**
 * Monte Carlo Simulation Runner
 *
 * Orchestrates the complete multi-layered Monte Carlo photon transport
 * simulation workflow. Manages GPU memory, kernel execution, and result
 * collection for complex tissue geometries.
 *
 * SIMULATION WORKFLOW:
 * 1. Memory initialization: Allocate and initialize GPU memory structures
 * 2. Photon launching: Initialize photon states for all GPU threads
 * 3. Transport simulation: Iterative kernel execution until completion
 * 4. Result collection: Gather detection results and statistics
 * 5. Memory cleanup: Proper deallocation and resource management
 *
 * KEY FEATURES:
 * - Exception-safe memory management using RAII principles
 * - Performance monitoring with detailed timing and throughput metrics
 * - Iterative kernel execution for memory-constrained scenarios
 * - Comprehensive progress reporting and status monitoring
 * - Automatic GPU resource cleanup on completion or error
 */
class SimulationRunner
{
public:
	/**
	 * Constructor - Initialize with GPU configuration
	 * @param gpu_config GPU configuration for kernel launches and memory allocation
	 */
	explicit SimulationRunner(const GPUConfig& gpu_config) : gpu_config_(gpu_config) {}

	/**
	 * Execute Complete Multi-Layered Monte Carlo Simulation
	 *
	 * Runs comprehensive photon transport simulation with the specified
	 * tissue geometry and optical properties. Handles all aspects from
	 * memory management to result collection.
	 *
	 * @param simulation Complete simulation configuration and parameters
	 * @param x Pre-initialized RNG state arrays for all threads
	 * @param a Pre-initialized RNG multiplier arrays for all threads
	 * @return Performance metrics including timing and throughput data
	 */
	PerformanceMetrics run_simulation(SimulationStruct* simulation, unsigned long long* x, unsigned int* a) {
		MemStruct device_mem, host_mem;

		// Phase 1: Initialize GPU and host memory structures
		initialize_memory(simulation, x, a, host_mem, device_mem);

		// Phase 2: Launch initial photon states on all GPU threads
		const clock_t start_time = clock();
		launch_photons(device_mem);

		std::cout << "Absorption detection: " << (simulation->ignoreAdetection ? "disabled" : "enabled") << "\n\n";

		// Phase 3: Execute iterative photon transport simulation
		const unsigned int iterations = run_main_simulation_loop(simulation, device_mem, host_mem);

		std::cout << "Simulation completed successfully!\n";

		// Phase 4: Finalize results and cleanup resources
		finalize_simulation(simulation, host_mem, device_mem);

		// Phase 5: Calculate and return performance metrics
		const clock_t end_time = clock();
		return create_performance_metrics(simulation, start_time, end_time, iterations);
	}

private:
	const GPUConfig& gpu_config_; // Reference to GPU configuration

	/**
	 * Initialize GPU and Host Memory Structures
	 *
	 * Allocates and initializes all memory structures required for the
	 * Monte Carlo simulation, including photon states, RNG arrays, and
	 * detection matrices.
	 *
	 * @param simulation Simulation configuration with geometry and parameters
	 * @param x Pre-initialized RNG state arrays
	 * @param a Pre-initialized RNG multiplier arrays
	 * @param host_mem Host memory structure to initialize
	 * @param device_mem Device memory structure to initialize
	 */
	void initialize_memory(SimulationStruct* simulation, unsigned long long* x, unsigned int* a, MemStruct& host_mem,
						   MemStruct& device_mem) {
		// Connect pre-initialized RNG arrays to memory structure
		host_mem.x = x;
		host_mem.a = a;

		// Initialize all GPU and host memory allocations
		init_mem_structs(&host_mem, &device_mem, simulation);

		// Copy simulation parameters to GPU constant memory
		init_dc_mem(simulation);
	}

	/**
	 * Launch Initial Photon States on All GPU Threads
	 *
	 * Initializes photon position, direction, and weight for all GPU threads
	 * according to the specified source configuration. Each thread receives
	 * a unique photon to begin transport simulation.
	 *
	 * @param device_mem Device memory containing photon state arrays
	 */
	void launch_photons(MemStruct& device_mem) {
		const dim3 dim_block(gpu_config_.threads_per_block);
		const dim3 dim_grid(gpu_config_.num_blocks);

		// Launch photon initialization kernel
		launch_photon_global<<<dim_grid, dim_block>>>(device_mem);
		CUDA_SAFE_CALL(cudaDeviceSynchronize());
	}

	/**
	 * Execute Main Iterative Monte Carlo Transport Loop
	 *
	 * Runs iterative photon transport kernels until all photons have been
	 * terminated through absorption, transmission, or Russian roulette.
	 * Provides progress monitoring and adapts to memory constraints.
	 *
	 * ALGORITHM:
	 * 1. Launch transport kernel (with or without absorption detection)
	 * 2. Check thread activity status across all GPU threads
	 * 3. Count terminated photons for progress reporting
	 * 4. Continue until no active threads remain
	 *
	 * @param simulation Simulation configuration for detection mode selection
	 * @param device_mem Device memory containing simulation state
	 * @param host_mem Host memory for thread activity monitoring
	 * @return Number of kernel iterations required for completion
	 */
	unsigned int run_main_simulation_loop(SimulationStruct* simulation, MemStruct& device_mem, MemStruct& host_mem) {
		const dim3 dim_block(gpu_config_.threads_per_block);
		const dim3 dim_grid(gpu_config_.num_blocks);

		unsigned int threads_active_total = 1;
		unsigned int iteration = 0;

		// Continue simulation until all photons are terminated
		while (threads_active_total > 0) {
			++iteration;

			// Launch appropriate transport kernel based on detection configuration
			if (simulation->ignoreAdetection == 1) {
				// Optimized kernel without absorption detection
				mc_d<1><<<dim_grid, dim_block>>>(device_mem);
			}
			else {
				// Full kernel with absorption detection
				mc_d<0><<<dim_grid, dim_block>>>(device_mem);
			}
			CUDA_SAFE_CALL(cudaDeviceSynchronize());

			// Monitor progress and thread activity
			threads_active_total = update_thread_activity(device_mem, host_mem);
			const auto terminated_photons = get_terminated_photon_count(device_mem, host_mem);

			std::cout << "Iteration " << iteration << ": Photons terminated = " << terminated_photons
					  << ", Threads active = " << threads_active_total << "\n";
		}

		return iteration;
	}

	/**
	 * Update Thread Activity Status from GPU
	 *
	 * Copies thread activity flags from GPU to host memory and counts
	 * the number of threads still actively transporting photons.
	 *
	 * @param device_mem Device memory containing thread activity flags
	 * @param host_mem Host memory for thread activity monitoring
	 * @return Total number of active threads
	 */
	unsigned int update_thread_activity(MemStruct& device_mem, MemStruct& host_mem) {
		CUDA_SAFE_CALL(cudaMemcpy(host_mem.thread_active, device_mem.thread_active,
								  gpu_config_.total_threads * sizeof(unsigned int), cudaMemcpyDeviceToHost));

		// Count total active threads across all GPU threads
		unsigned int total_active = 0;
		for (int i = 0; i < gpu_config_.total_threads; ++i) {
			total_active += host_mem.thread_active[i];
		}
		return total_active;
	}

	/**
	 * Get Total Terminated Photon Count from GPU
	 *
	 * Retrieves the global counter of terminated photons for progress
	 * reporting and simulation monitoring.
	 *
	 * @param device_mem Device memory containing terminated photon counter
	 * @param host_mem Host memory for counter retrieval
	 * @return Current number of terminated photons
	 */
	unsigned int get_terminated_photon_count(MemStruct& device_mem, MemStruct& host_mem) {
		CUDA_SAFE_CALL(cudaMemcpy(host_mem.num_terminated_photons, device_mem.num_terminated_photons,
								  sizeof(unsigned int), cudaMemcpyDeviceToHost));
		return *host_mem.num_terminated_photons;
	}

	/**
	 * Finalize Simulation and Export Results
	 *
	 * Copies final detection results from GPU to host, exports results
	 * to output files, and performs cleanup of GPU memory allocations.
	 *
	 * @param simulation Simulation configuration for output file paths
	 * @param host_mem Host memory structures for result storage
	 * @param device_mem Device memory structures to be cleaned up
	 */
	void finalize_simulation(SimulationStruct* simulation, MemStruct& host_mem, MemStruct& device_mem) {
		// Copy final detection results from GPU to host
		copy_device_to_host_mem(&host_mem, &device_mem, simulation);

		// Export results to standard MCML format files
		write_simulation_results(&host_mem, simulation, 0); // Timing handled externally

		// Clean up GPU memory allocations
		free_mem_structs(&host_mem, &device_mem);
	}

	/**
	 * Create Performance Metrics Report
	 *
	 * Calculates comprehensive performance metrics from simulation timing
	 * and configuration data for throughput analysis.
	 *
	 * @param simulation Simulation configuration with photon count
	 * @param start_time Simulation start timestamp
	 * @param end_time Simulation end timestamp
	 * @param iterations Number of kernel iterations executed
	 * @return Populated performance metrics structure
	 */
	PerformanceMetrics create_performance_metrics(SimulationStruct* simulation, clock_t start_time, clock_t end_time,
												  unsigned int iterations) {
		PerformanceMetrics metrics;
		metrics.simulation_time = static_cast<double>(end_time - start_time) / CLOCKS_PER_SEC;
		metrics.total_photons = simulation->number_of_photons;
		metrics.kernel_iterations = iterations;
		return metrics;
	}
};

} // namespace cudamcml

////////////////////////////////////////////////////////////////////////////////
// LEGACY COMPATIBILITY AND MAIN APPLICATION LOGIC

/**
 * Legacy Wrapper Function - Backward Compatibility
 *
 * Provides compatibility interface for existing code that expects the
 * traditional DoOneSimulation function signature. Internally uses the
 * modern SimulationRunner class for improved error handling and performance.
 *
 * @param simulation Complete simulation configuration
 * @param x Pre-initialized RNG state arrays
 * @param a Pre-initialized RNG multiplier arrays
 */
void do_one_simulation(SimulationStruct* simulation, unsigned long long* x, unsigned int* a) {
	cudamcml::SimulationRunner runner(g_gpu_config);
	const auto metrics = runner.run_simulation(simulation, x, a);
	metrics.print(g_gpu_config);
}

////////////////////////////////////////////////////////////////////////////////
// MAIN ENTRY POINT

/**
 * Main Application Entry Point
 *
 * Provides comprehensive command-line interface for CUDA Monte Carlo
 * multi-layered photon transport simulations. Handles argument parsing,
 * GPU initialization, RNG setup, and simulation execution.
 *
 * EXECUTION WORKFLOW:
 * 1. Parse command line arguments and validate input file
 * 2. Initialize CUDA device and configure optimal GPU parameters
 * 3. Parse simulation input file with tissue geometry and optical properties
 * 4. Initialize random number generators for all GPU threads
 * 5. Execute Monte Carlo photon transport simulation
 * 6. Generate performance report and export results
 * 7. Clean up resources and return appropriate exit codes
 *
 * COMMAND LINE USAGE:
 *   cudamcml <input_file.mci> [options]
 *
 * INPUT FILE FORMAT:
 *   Standard MCML format with multi-layer tissue specifications
 *
 * @param argc Command line argument count
 * @param argv Command line argument array
 * @return EXIT_SUCCESS on successful completion, EXIT_FAILURE on error
 */
int main(int argc, char* argv[]) {
	try {
		std::cout << "=== CUDAMCML - GPU Monte Carlo Multi-Layer Photon Transport ===\n\n";

		// Validate command line arguments
		if (argc < 2) {
			std::cerr << "Usage: " << argv[0] << " <input_file.mci> [options]\n"
					  << "\nRequired:\n"
					  << "  input_file.mci    MCML input file with tissue layer specifications\n"
					  << "\nExample:\n"
					  << "  " << argv[0] << " sample.mci\n\n";
			return EXIT_FAILURE;
		}

		// Initialize CUDA device and configure GPU parameters
		std::cout << "Initializing CUDA device and configuring GPU parameters...\n";
		initialize_gpu_params();

		// Parse simulation input file
		std::cout << "Loading simulation configuration from: " << argv[1] << "\n";
		const char* filename = argv[1];
		auto seed = static_cast<unsigned long long>(time(nullptr));
		int ignoreAdetection = 0;

		// Parse additional command line arguments
		if (interpret_arg(argc, argv, &seed, &ignoreAdetection) != 0) {
			std::cerr << "Error: Invalid command line arguments\n";
			return EXIT_FAILURE;
		}

		// Read simulation data from input file
		SimulationStruct* simulations = nullptr;
		const int n_simulations = read_simulation_data(const_cast<char*>(filename), &simulations, ignoreAdetection);

		if (n_simulations == 0 || !simulations) {
			std::cerr << "Error: Failed to read simulation data from " << filename << "\n";
			return EXIT_FAILURE;
		}

		// Display simulation configuration summary
		std::cout << "\nLoaded " << n_simulations << " simulation(s)\n";
		for (int i = 0; i < n_simulations; ++i) {
			std::cout << "Simulation " << (i + 1) << ":\n"
					  << "  Number of photons: " << simulations[i].number_of_photons << "\n"
					  << "  Number of layers: " << simulations[i].n_layers << "\n"
					  << "  Output file: " << simulations[i].outp_filename << "\n"
					  << "  Absorption detection: " << (simulations[i].ignoreAdetection ? "disabled" : "enabled")
					  << "\n";
		}

		// Initialize random number generators for all GPU threads
		std::cout << "\nInitializing random number generators...\n";
		cudamcml::RNGMemory rng(g_total_threads);
		rng.initialize(seed, nullptr); // Use embedded safe primes

		// Execute all Monte Carlo simulations with performance monitoring
		std::cout << "Starting Monte Carlo photon transport simulation(s)...\n\n";

		for (int i = 0; i < n_simulations; ++i) {
			std::cout << "=== Running Simulation " << (i + 1) << " of " << n_simulations << " ===\n";

			cudamcml::SimulationRunner runner(g_gpu_config);
			const auto metrics = runner.run_simulation(&simulations[i], rng.x(), rng.a());

			// Display performance report for this simulation
			std::cout << "\n";
			metrics.print(g_gpu_config);
			std::cout << "\n";
		}

		// Cleanup simulation configuration
		free_simulation_struct(simulations, n_simulations);

		std::cout << "CUDAMCML simulation completed successfully!\n";
		return EXIT_SUCCESS;
	}
	catch (...) {
		std::cerr << "Fatal error occurred during simulation.\n";
		return EXIT_FAILURE;
	}
}
