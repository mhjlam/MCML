/*==============================================================================
 * CUDAMC - GPU-Accelerated Monte Carlo Photon Transport Simulation
 *
 * PHYSICAL MODEL:
 * ---------------
 * - Semi-infinite turbid medium with specified optical properties
 * - Isotropic point source at medium surface (z=0)
 * - Henyey-Greenberg phase function for anisotropic scattering
 * - Fresnel reflection/transmission at medium boundaries
 * - Time-resolved photon detection with binned histogram output
 * - Statistical independence via multiply-with-carry random number generation
 *
 * COMPUTATIONAL ARCHITECTURE:
 * ---------------------------
 * - GPU Implementation: CUDA kernels with massive parallelization
 * - CPU Reference: "Gold standard" validation with identical algorithms
 * - Modern C++: RAII memory management, exception handling, namespaces
 * - Performance Analysis: Comprehensive throughput and speedup metrics
 * - Static Configuration: Hardware-agnostic thread allocation
 *
 * SIMULATION FEATURES:
 * --------------------
 * - Concurrent photon histories on GPU (65,536 threads typical)
 * - Time-resolved detection with configurable bin resolution
 * - Comprehensive validation against CPU reference implementation
 * - Performance benchmarking with detailed throughput analysis
 * - Error handling with CUDA-specific diagnostics
 *
 * USAGE:
 * ------
 * Compile with nvcc and run directly. The simulation automatically:
 * 1. Detects and configures available CUDA hardware
 * 2. Runs GPU simulation with massive parallelization
 * 3. Validates results against CPU reference implementation
 * 4. Reports comprehensive performance metrics and speedup factors
 */

////////////////////////////////////////////////////////////////////////////////
// SYSTEM INCLUDES AND DEPENDENCIES

// Standard C++ library includes for I/O, memory management, and error handling
#include <cstdio>    // C-style I/O for compatibility
#include <iomanip>   // Stream manipulators for formatted output
#include <iostream>  // Modern C++ I/O streams
#include <memory>    // Smart pointers for RAII
#include <stdexcept> // Standard exception classes

// CUDA runtime and device includes for GPU computing
#include <cuda_runtime.h>             // CUDA runtime API
#include <device_launch_parameters.h> // CUDA kernel launch parameters
#include <math_constants.h>           // Mathematical constants (M_PI, etc.)

// Project-specific includes
#include "cudamc.h"      // Main header with constants and function declarations
#include "safe_primes.h" // Embedded prime numbers for RNG initialization

// Component modules - included as source for single compilation unit
#include "cudamc_gold_standard.c" // CPU reference implementation
#include "cudamc_transport.cu"    // GPU device functions and kernels

////////////////////////////////////////////////////////////////////////////////
// CUDA MONTE CARLO NAMESPACE

namespace cudamc
{

////////////////////////////////////////////////////////////////////////////////
// GPU CONFIGURATION AND HARDWARE MANAGEMENT

/**
 * GPU Configuration Management Class
 *
 * Encapsulates CUDA hardware detection, configuration, and optimization.
 * Uses static thread allocation to match global array sizes and ensure
 * consistent memory access patterns across different GPU architectures.
 *
 * DESIGN PHILOSOPHY:
 * - Static configuration for predictable memory footprint
 * - Hardware detection for informational purposes and validation
 * - Conservative thread allocation for broad hardware compatibility
 * - Detailed reporting for performance analysis and debugging
 */
class CPUConfig
{
public:
	// Fallback configuration constants for hardware compatibility
	static constexpr int DEFAULT_FALLBACK_BLOCKS = 128;
	static constexpr int DEFAULT_FALLBACK_THREADS = 65536;

	// Active GPU configuration parameters
	int num_blocks;        // Number of CUDA thread blocks
	int threads_per_block; // Threads per block (typically 256)
	int total_threads;     // Total concurrent threads

	/**
	 * Constructor - Initialize with static configuration
	 *
	 * Uses compile-time constants to ensure consistent memory allocation
	 * and avoid runtime configuration complexity across different GPUs.
	 */
	CPUConfig() : num_blocks(NUM_BLOCKS), threads_per_block(NUM_THREADS_PER_BLOCK), total_threads(NUM_THREADS) {}

	/**
	 * Hardware Detection and Configuration Validation
	 *
	 * Detects GPU capabilities and validates our static configuration
	 * against hardware limits. Provides detailed hardware information
	 * for performance analysis and troubleshooting.
	 */
	void initialize() {
		cudaDeviceProp device_prop;
		int device;

		CUDA_CHECK_ERROR(cudaGetDevice(&device));
		CUDA_CHECK_ERROR(cudaGetDeviceProperties(&device_prop, device));

		// Extract hardware characteristics for reporting
		const int multiprocessors = device_prop.multiProcessorCount;
		const int max_threads_per_sm = device_prop.maxThreadsPerMultiProcessor;
		const int max_blocks_per_sm = device_prop.maxBlocksPerMultiProcessor;

		// Use static configuration to match global array allocations
		// This ensures consistent behavior across different GPU models
		threads_per_block = NUM_THREADS_PER_BLOCK;
		num_blocks = NUM_BLOCKS;
		total_threads = NUM_THREADS;

		print_configuration(device_prop, multiprocessors, max_threads_per_sm, max_blocks_per_sm);
	}

private:
	/**
	 * Print comprehensive GPU configuration report
	 *
	 * Displays hardware capabilities, configured parameters, and
	 * utilization analysis for performance optimization guidance.
	 */
	void print_configuration(const cudaDeviceProp& device_prop, int multiprocessors, int max_threads_per_sm,
							 int max_blocks_per_sm) const {
		std::cout << "=== CUDAMC GPU Configuration ===\n"
				  << "GPU: " << device_prop.name << "\n"
				  << "Compute capability: " << device_prop.major << "." << device_prop.minor << "\n"
				  << "Multiprocessors: " << multiprocessors << "\n"
				  << "Max threads per SM: " << max_threads_per_sm << "\n"
				  << "Max blocks per SM: " << max_blocks_per_sm << "\n"
				  << "Configured blocks: " << num_blocks << "\n"
				  << "Threads per block: " << threads_per_block << "\n"
				  << "Total GPU threads: " << total_threads << "\n"
				  << "Configuration: Static (matching array sizes)\n"
				  << "=================================\n\n";
	}
};

////////////////////////////////////////////////////////////////////////////////
// RAII DEVICE MEMORY MANAGEMENT

/**
 * RAII Device Memory Wrapper Template
 *
 * Provides automatic GPU memory management using Resource Acquisition Is
 * Initialization (RAII) principles. Ensures proper cleanup even in the
 * presence of exceptions, preventing GPU memory leaks.
 *
 * FEATURES:
 * - Automatic allocation and deallocation
 * - Exception-safe memory management
 * - Move semantics for efficient transfers
 * - Type-safe memory operations
 * - Convenient host-device data transfer methods
 *
 * USAGE:
 * DeviceMemory<float> gpu_array(1000);           // Allocate 1000 floats on GPU
 * gpu_array.copy_from_host(host_data);           // Transfer from CPU
 * kernel<<<blocks, threads>>>(gpu_array.get());  // Use in kernel
 * gpu_array.copy_to_host(results);               // Transfer results back
 * // Automatic cleanup when gpu_array goes out of scope
 */
template<typename T>
class DeviceMemory
{
public:
	/**
	 * Constructor - Allocate GPU memory for specified number of elements
	 *
	 * @param count Number of elements to allocate (not bytes)
	 * @throws std::invalid_argument if count is zero
	 * @throws std::runtime_error if GPU allocation fails
	 */
	explicit DeviceMemory(size_t count) : count_(count), size_(count * sizeof(T)) {
		if (count == 0) {
			throw std::invalid_argument("Memory count must be positive");
		}

		CUDA_CHECK_ERROR(cudaMalloc(reinterpret_cast<void**>(&device_ptr_), size_));
	}

	/**
	 * Destructor - Automatic GPU memory deallocation
	 *
	 * Provides exception-safe cleanup. CUDA errors during deallocation
	 * are not propagated to avoid termination during stack unwinding.
	 */
	~DeviceMemory() {
		if (device_ptr_) {
			cudaFree(device_ptr_); // Ignore errors in destructor
		}
	}

	// Non-copyable to prevent accidental double-free errors
	DeviceMemory(const DeviceMemory&) = delete;
	DeviceMemory& operator=(const DeviceMemory&) = delete;

	/**
	 * Move constructor - Transfer ownership efficiently
	 *
	 * Enables returning DeviceMemory objects from functions and
	 * storing them in containers without unnecessary allocations.
	 */
	DeviceMemory(DeviceMemory&& other) noexcept :
		device_ptr_(other.device_ptr_), count_(other.count_), size_(other.size_) {
		other.device_ptr_ = nullptr;
		other.count_ = 0;
		other.size_ = 0;
	}

	/**
	 * Copy data from host to device
	 *
	 * @param host_data Pointer to host memory containing source data
	 * @throws std::runtime_error if CUDA memory copy fails
	 */
	void copy_from_host(const T* host_data) {
		CUDA_CHECK_ERROR(cudaMemcpy(device_ptr_, host_data, size_, cudaMemcpyHostToDevice));
	}

	/**
	 * Copy data from device to host
	 *
	 * @param host_data Pointer to host memory for storing results
	 * @throws std::runtime_error if CUDA memory copy fails
	 */
	void copy_to_host(T* host_data) const {
		CUDA_CHECK_ERROR(cudaMemcpy(host_data, device_ptr_, size_, cudaMemcpyDeviceToHost));
	}

	// Accessor methods for use in CUDA kernel launches and API calls
	T* get() const { return device_ptr_; }  // Get raw device pointer
	size_t count() const { return count_; } // Get element count
	size_t size() const { return size_; }   // Get size in bytes

private:
	T* device_ptr_ = nullptr;               // Raw CUDA device pointer
	size_t count_;                          // Number of elements allocated
	size_t size_;                           // Total size in bytes
};

////////////////////////////////////////////////////////////////////////////////
// PERFORMANCE METRICS AND BENCHMARKING

/**
 * Performance Metrics Collection and Analysis
 *
 * Comprehensive performance tracking for both GPU and CPU implementations,
 * enabling detailed analysis of acceleration benefits and computational
 * efficiency. Tracks multiple performance indicators beyond simple timing.
 *
 * METRICS TRACKED:
 * - Execution time (wall clock) for both GPU and CPU
 * - Total computational operations performed
 * - Number of photons that terminated (reached detector)
 * - Histogram contributions (time-resolved detections)
 * - Throughput in billions of operations per second (GOPS)
 * - GPU vs CPU speedup factor with load normalization
 *
 * ANALYSIS FEATURES:
 * - Normalized throughput calculations accounting for different workloads
 * - Comprehensive speedup analysis with proper operation counting
 * - Detailed performance reporting with hardware utilization metrics
 */
struct PerformanceMetrics {
	// Timing measurements (wall clock seconds)
	double gpu_time_seconds;
	double cpu_time_seconds;

	// Operation counting for throughput analysis
	double total_gpu_operations; // Total GPU computational operations
	double total_cpu_operations; // Total CPU computational operations

	// Physics simulation results for validation
	uint64_t gpu_photons_terminated;      // GPU photons reaching detector
	uint64_t cpu_photons_terminated;      // CPU photons reaching detector
	uint64_t gpu_histogram_contributions; // GPU time-resolved detections
	uint64_t cpu_histogram_contributions; // CPU time-resolved detections

	/**
	 * Calculate GPU throughput in billions of operations per second
	 * @return GOPS (billions of operations per second) for GPU execution
	 */
	double gpu_throughput_gops() const {
		return gpu_time_seconds > 0.0 ? total_gpu_operations / gpu_time_seconds / 1e9 : 0.0;
	}

	/**
	 * Calculate CPU throughput in billions of operations per second
	 * @return GOPS (billions of operations per second) for CPU execution
	 */
	double cpu_throughput_gops() const {
		return cpu_time_seconds > 0.0 ? total_cpu_operations / cpu_time_seconds / 1e9 : 0.0;
	}

	/**
	 * Calculate normalized GPU vs CPU speedup factor
	 *
	 * Accounts for different workload sizes by normalizing based on
	 * operations performed rather than just execution time.
	 *
	 * @return Speedup factor (how many times faster GPU is than CPU)
	 */
	double speedup_factor() const {
		return (cpu_time_seconds > 0.0 && gpu_time_seconds > 0.0)
				   ? (total_gpu_operations * cpu_time_seconds) / (total_cpu_operations * gpu_time_seconds)
				   : 0.0;
	}

	/**
	 * Print comprehensive performance analysis report
	 *
	 * Generates detailed performance report including individual GPU/CPU
	 * metrics, comparison analysis, and hardware utilization information.
	 *
	 * @param config GPU configuration for context in reporting
	 */
	void print(const CPUConfig& config) const {
		std::cout << "=== CUDAMC Performance Report ===\n";

		// GPU Performance Analysis
		std::cout << "\nGPU Performance:\n"
				  << "  Simulation time: " << std::fixed << std::setprecision(3) << gpu_time_seconds << " sec\n"
				  << "  Total operations: " << std::scientific << std::setprecision(2) << total_gpu_operations << "\n"
				  << "  Throughput: " << std::fixed << std::setprecision(2) << gpu_throughput_gops()
				  << " billion ops/sec\n"
				  << "  Photons terminated: " << gpu_photons_terminated << "\n"
				  << "  Histogram contributions: " << gpu_histogram_contributions << "\n";

		// CPU Performance Analysis
		std::cout << "\nCPU Performance:\n"
				  << "  Simulation time: " << std::fixed << std::setprecision(3) << cpu_time_seconds << " sec\n"
				  << "  Total operations: " << std::scientific << std::setprecision(2) << total_cpu_operations << "\n"
				  << "  Throughput: " << std::fixed << std::setprecision(2) << cpu_throughput_gops()
				  << " billion ops/sec\n"
				  << "  Photons terminated: " << cpu_photons_terminated << "\n"
				  << "  Histogram contributions: " << cpu_histogram_contributions << "\n";

		// Comparative Performance Analysis
		std::cout << "\nPerformance Comparison:\n"
				  << "  GPU vs CPU speedup: " << std::fixed << std::setprecision(1) << speedup_factor() << "x\n"
				  << "  GPU utilization: " << config.num_blocks << " blocks * " << config.threads_per_block
				  << " threads = " << config.total_threads << " total threads\n"
				  << "==================================\n\n";
	}
};

} // namespace cudamc

////////////////////////////////////////////////////////////////////////////////
// GLOBAL STATE MANAGEMENT AND RNG INITIALIZATION

// Global GPU configuration instance
static cudamc::CPUConfig g_gpu_config;

// Global RNG arrays for all simulation threads
// These arrays are shared between GPU and CPU implementations
uint32_t x_test[NUM_THREADS]; // RNG state values (lower 32 bits)
uint32_t c_test[NUM_THREADS]; // RNG carry values
uint32_t a_test[NUM_THREADS]; // RNG multiplier values (from safe primes)

/**
 * Initialize Random Number Generators for All Threads
 *
 * Sets up independent RNG streams for each simulation thread using
 * multiply-with-carry (MWC) generators with safe prime multipliers.
 * Ensures statistical independence between threads while maintaining
 * good random number quality.
 *
 * ALGORITHM:
 * 1. Load initial multiplier from embedded safe primes data
 * 2. Generate unique initial state for each thread using MWC sequence
 * 3. Assign unique safe prime multiplier to each thread
 * 4. Initialize carry values using normalized random values
 *
 * THREAD SAFETY:
 * Each thread gets independent RNG state, ensuring no correlation
 * between parallel Monte Carlo histories on different threads.
 */
void initialize_rng() {
	uint32_t begin = 0U;
	uint64_t x_init = 1ULL; // Initial seed for state generation
	uint32_t c_init = 0U;   // Initial carry value
	uint32_t for_a;

	// Load first safe prime as base multiplier for state generation
	begin = safeprimes_data[0].a;

	// Generate unique RNG state for each thread
	for (uint32_t i = 0; i < NUM_THREADS; i++) {
		// Generate next x value using MWC with base multiplier
		x_init = x_init * begin + c_init;
		c_init = x_init >> 32;           // Extract carry
		x_init = x_init & 0xffffffffULL; // Keep lower 32 bits
		x_test[i] = static_cast<uint32_t>(x_init);

		// Assign unique safe prime multiplier to this thread
		// Cycle through available safe primes to ensure coverage
		for_a = safeprimes_data[(i + 1) % 50000].a; // Use actual safeprimes count
		a_test[i] = for_a;

		// Generate carry value for this thread
		x_init = x_init * begin + c_init;
		c_init = x_init >> 32;
		x_init = x_init & 0xffffffffULL;
		// Scale carry to appropriate range for this multiplier
		c_test[i] = static_cast<uint32_t>((static_cast<double>(x_init) / UINT_MAX) * for_a);
	}
}

namespace cudamc
{

////////////////////////////////////////////////////////////////////////////////
// MONTE CARLO SIMULATION ORCHESTRATION

/**
 * Monte Carlo Simulation Runner
 *
 * Orchestrates comprehensive Monte Carlo photon transport simulations
 * on both GPU and CPU platforms. Manages memory allocation, kernel
 * execution, result validation, and performance analysis.
 *
 * SIMULATION WORKFLOW:
 * 1. GPU Simulation: Massively parallel photon histories on CUDA
 * 2. CPU Reference: Sequential validation using identical algorithms
 * 3. Results Comparison: Validation of GPU implementation accuracy
 * 4. Performance Analysis: Comprehensive throughput and speedup metrics
 *
 * KEY FEATURES:
 * - RAII memory management for exception safety
 * - Comprehensive error handling with detailed diagnostics
 * - Performance profiling with operation counting
 * - Results validation between GPU and CPU implementations
 * - Detailed progress reporting and logging
 */
class SimulationRunner
{
public:
	/**
	 * Constructor - Initialize with GPU configuration
	 * @param config GPU configuration parameters for kernel launch
	 */
	explicit SimulationRunner(const CPUConfig& config) : gpu_config_(config) {}

	/**
	 * Execute Complete Monte Carlo Simulation Suite
	 *
	 * Runs both GPU and CPU Monte Carlo simulations with identical
	 * parameters and algorithms. Provides comprehensive performance
	 * analysis and validation of results.
	 *
	 * @param x RNG state arrays (lower 32 bits per thread)
	 * @param c RNG carry arrays per thread
	 * @param a RNG multiplier arrays per thread
	 * @return Performance metrics for both GPU and CPU executions
	 */
	PerformanceMetrics run_monte_carlo(uint32_t* x, uint32_t* c, uint32_t* a) {
		PerformanceMetrics metrics {};

		// Execute GPU simulation with massive parallelization
		run_gpu_simulation(x, c, a, metrics);

		// Execute CPU reference implementation for validation
		run_cpu_simulation(x, c, a, metrics);

		return metrics;
	}

private:
	const CPUConfig& gpu_config_; // GPU configuration reference

	/**
	 * Execute GPU Monte Carlo Simulation
	 *
	 * Launches CUDA kernel with thousands of parallel threads, each
	 * simulating independent photon histories. Uses RAII for memory
	 * management and comprehensive error checking.
	 *
	 * EXECUTION PHASES:
	 * 1. Host memory allocation and initialization
	 * 2. Device memory allocation and data transfer
	 * 3. CUDA kernel launch with configured thread geometry
	 * 4. Result retrieval and performance measurement
	 * 5. Automatic cleanup via RAII
	 *
	 * @param x RNG state arrays
	 * @param c RNG carry arrays
	 * @param a RNG multiplier arrays
	 * @param metrics Output metrics structure to populate
	 */
	void run_gpu_simulation(uint32_t* x, uint32_t* c, uint32_t* a, PerformanceMetrics& metrics) {
		std::cout << "Running GPU simulation...\n";
		std::cout << "GPU simulation: " << gpu_config_.total_threads << " threads * " << NUM_STEPS_GPU
				  << " steps = " << std::scientific << (double)gpu_config_.total_threads * NUM_STEPS_GPU
				  << " total operations\n";

		// Allocate host memory for results using smart pointers
		auto num = std::make_unique<uint32_t[]>(gpu_config_.total_threads);
		auto hist = std::make_unique<uint32_t[]>(TEMP_SIZE);

		// Initialize time-resolved histogram bins
		for (uint32_t i = 0; i < TEMP_SIZE; i++) {
			hist[i] = 0;
		}

		// GPU memory management with RAII - automatic cleanup guaranteed
		DeviceMemory<uint32_t> x_device(gpu_config_.total_threads);
		DeviceMemory<uint32_t> c_device(gpu_config_.total_threads);
		DeviceMemory<uint32_t> a_device(gpu_config_.total_threads);
		DeviceMemory<uint32_t> num_device(gpu_config_.total_threads);
		DeviceMemory<uint32_t> hist_device(TEMP_SIZE);

		// Transfer RNG initialization data and histogram to GPU
		x_device.copy_from_host(x);
		c_device.copy_from_host(c);
		a_device.copy_from_host(a);
		hist_device.copy_from_host(hist.get());

		// Begin performance timing for GPU execution
		const clock_t time1 = clock();

		// Configure CUDA kernel launch parameters
		const dim3 dim_block(gpu_config_.threads_per_block); // Threads per block
		const dim3 dim_grid(gpu_config_.num_blocks);         // Number of blocks

		// Ensure GPU is synchronized before kernel launch
		CUDA_CHECK_ERROR(cudaDeviceSynchronize());

		std::cout << "Launching kernel with " << gpu_config_.num_blocks << " blocks * " << gpu_config_.threads_per_block
				  << " threads = " << gpu_config_.total_threads << " total threads\n";

		// Launch Monte Carlo kernel - each thread simulates independent photon histories
		// Kernel signature: mc(x_device, c_device, a_device, num_device, hist_device)
		mc<<<dim_grid, dim_block>>>(x_device.get(), c_device.get(), a_device.get(), num_device.get(),
									hist_device.get());

		// Comprehensive error checking for kernel launch and execution
		CUDA_CHECK_ERROR(cudaGetLastError());      // Check for launch errors
		CUDA_CHECK_ERROR(cudaDeviceSynchronize()); // Wait for completion and check execution errors

		// Retrieve simulation results from GPU to host memory
		num_device.copy_to_host(num.get());   // Per-thread photon termination counts
		hist_device.copy_to_host(hist.get()); // Time-resolved detection histogram

		// End performance timing
		const clock_t time2 = clock();

		// Calculate comprehensive GPU performance metrics
		metrics.gpu_time_seconds = static_cast<double>(time2 - time1) / CLOCKS_PER_SEC;
		metrics.total_gpu_operations = static_cast<double>(gpu_config_.total_threads) * NUM_STEPS_GPU;

		// Count total photons that reached the detector (terminated)
		metrics.gpu_photons_terminated = 0;
		for (uint32_t i = 0; i < gpu_config_.total_threads; i++) {
			metrics.gpu_photons_terminated += num[i];
		}

		// Count total time-resolved detection events across all histogram bins
		metrics.gpu_histogram_contributions = 0;
		for (uint32_t i = 0; i < TEMP_SIZE; i++) {
			metrics.gpu_histogram_contributions += hist[i];
		}

		// Display GPU time-resolved detection histogram
		std::cout << "\nGPU Histogram: ";
		for (uint32_t i = 0; i < TEMP_SIZE; i++) {
			std::cout << hist[i] << " ";
		}
		std::cout << "\n";
	}

	/**
	 * Execute CPU Reference Monte Carlo Simulation
	 *
	 * Runs identical Monte Carlo algorithm on CPU for validation and
	 * performance comparison. Uses the same RNG initialization and
	 * physics parameters to ensure comparable results.
	 *
	 * VALIDATION PURPOSE:
	 * - Verify correctness of GPU implementation
	 * - Provide performance baseline for speedup calculations
	 * - Debug algorithm differences between GPU and CPU
	 *
	 * @param x RNG state arrays (same as used for GPU)
	 * @param c RNG carry arrays (same as used for GPU)
	 * @param a RNG multiplier arrays (same as used for GPU)
	 * @param metrics Output metrics structure to populate
	 */
	void run_cpu_simulation(uint32_t* x, uint32_t* c, uint32_t* a, PerformanceMetrics& metrics) {
		std::cout << "\nRunning CPU simulation (this may take several minutes)...\n";
		std::cout << "CPU simulation: " << NUM_THREADS_CPU << " threads * " << NUM_STEPS_CPU
				  << " steps = " << std::scientific << (double)NUM_THREADS_CPU * NUM_STEPS_CPU << " total operations\n";

		// Allocate host arrays for CPU results
		auto num_h = std::make_unique<uint32_t[]>(NUM_THREADS_CPU);
		auto hist_h = std::make_unique<uint32_t[]>(TEMP_SIZE);

		// Initialize CPU histogram bins
		for (uint32_t i = 0; i < TEMP_SIZE; i++) {
			hist_h[i] = 0;
		}

		// Execute CPU Monte Carlo simulation with performance timing
		const clock_t time1 = clock();
		gs_mc(x, c, a, num_h.get(), hist_h.get()); // Gold standard CPU implementation
		const clock_t time2 = clock();

		// Calculate comprehensive CPU performance metrics
		metrics.cpu_time_seconds = static_cast<double>(time2 - time1) / CLOCKS_PER_SEC;
		metrics.total_cpu_operations = static_cast<double>(NUM_THREADS_CPU) * NUM_STEPS_CPU;

		// Count total photons that reached the detector on CPU
		metrics.cpu_photons_terminated = 0;
		for (uint32_t i = 0; i < NUM_THREADS_CPU; i++) {
			metrics.cpu_photons_terminated += num_h[i];
		}

		// Count total time-resolved detection events across all histogram bins
		metrics.cpu_histogram_contributions = 0;
		for (uint32_t i = 0; i < TEMP_SIZE; i++) {
			metrics.cpu_histogram_contributions += hist_h[i];
		}

		// Display CPU time-resolved detection histogram for comparison
		std::cout << "\nCPU Histogram: ";
		for (uint32_t i = 0; i < TEMP_SIZE; i++) {
			std::cout << hist_h[i] << " ";
		}
		std::cout << "\n";
	}
};

} // namespace cudamc

////////////////////////////////////////////////////////////////////////////////
// MAIN APPLICATION CONTROL AND COORDINATION

/**
 * Main Monte Carlo Simulation Coordinator
 *
 * Orchestrates the complete CUDA Monte Carlo photon transport simulation
 * workflow. Handles CUDA device management, simulation execution, and
 * comprehensive performance reporting.
 *
 * SIMULATION WORKFLOW:
 * 1. CUDA Device Detection and Initialization
 * 2. GPU Hardware Configuration and Optimization
 * 3. Parallel GPU Monte Carlo Simulation
 * 4. Sequential CPU Reference Validation
 * 5. Performance Analysis and Detailed Reporting
 *
 * ERROR HANDLING:
 * - CUDA device availability validation
 * - Comprehensive error reporting with exceptions
 * - Resource cleanup guaranteed via RAII
 *
 * @param x Pre-initialized RNG state arrays for all threads
 * @param c Pre-initialized RNG carry arrays for all threads
 * @param a Pre-initialized RNG multiplier arrays for all threads
 */
void run_monte_carlo_simulation(uint32_t* x, uint32_t* c, uint32_t* a) {
	try {
		// Validate CUDA device availability before proceeding
		int device_count;
		CUDA_CHECK_ERROR(cudaGetDeviceCount(&device_count));

		if (device_count == 0) {
			throw std::runtime_error("No CUDA devices found");
		}

		// Initialize GPU hardware configuration and display capabilities
		g_gpu_config.initialize();

		// Execute complete simulation suite (GPU + CPU validation)
		cudamc::SimulationRunner runner(g_gpu_config);
		const auto metrics = runner.run_monte_carlo(x, c, a);

		// Generate comprehensive performance analysis report
		metrics.print(g_gpu_config);

		std::cout << "Monte Carlo simulation completed successfully!\n";
	}
	catch (const std::exception& e) {
		std::cerr << "Error in Monte Carlo simulation: " << e.what() << "\n";
		throw; // Re-throw for higher-level handling
	}
}

////////////////////////////////////////////////////////////////////////////////
// MAIN ENTRY POINT

/**
 * Main Application Entry Point
 *
 * Provides comprehensive exception handling and coordinates the complete
 * CUDAMC simulation workflow from initialization through final reporting.
 *
 * EXECUTION PHASES:
 * 1. Display application banner and initialization
 * 2. Initialize random number generators for all threads
 * 3. Execute complete Monte Carlo simulation suite
 * 4. Handle any errors with detailed diagnostics
 * 5. Return appropriate exit codes
 *
 * @param argc Command line argument count (currently unused)
 * @param argv Command line arguments (currently unused)
 * @return EXIT_SUCCESS on successful completion, EXIT_FAILURE on error
 */
int main(int argc, char* argv[]) {
	try {
		std::cout << "=== CUDAMC Monte Carlo Photon Migration Simulation ===\n\n";

		// Initialize random number generators for all simulation threads
		initialize_rng();

		// Execute comprehensive Monte Carlo simulation and analysis
		run_monte_carlo_simulation(x_test, c_test, a_test);

		return EXIT_SUCCESS;
	}
	catch (const std::exception& e) {
		std::cerr << "Fatal error: " << e.what() << "\n";
		return EXIT_FAILURE;
	}
	catch (...) {
		std::cerr << "Unknown fatal error occurred.\n";
		return EXIT_FAILURE;
	}
}
