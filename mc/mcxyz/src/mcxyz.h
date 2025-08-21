/*
 * MCXYZ - Monte Carlo simulation of photon transport in 3D voxelized media
 *
 * COPYRIGHT AND DEVELOPMENT HISTORY:
 * ===================================
 *
 * ORIGINAL DEVELOPMENT (2010-2017):
 * ----------------------------------
 * Created:     2010, 2012
 * Authors:     Steven L. Jacques, Ting Li
 *              Oregon Health & Science University
 * Last Update: 2017
 *
 * Original Usage: mcxyz myname
 *   Input files:  myname_H.mci (header), myname_T.bin (tissue structure)
 *   Output files: myname_OP.m (properties), myname_F.bin (fluence)
 *
 * Development Log:
 * - 2010: Written by Ting Li based on Steve Jacques' mcsub.
 * - 2010: Updated to use Steve Jacques' FindVoxelFace.
 * - 2012: Reorganized by Steve Jacques; reads input files, outputs binary files.
 * - 2013: Update thanks to contributions from Marleen Keijzer, Scott Prahl, Lihong Wang and Ting Li.
 * - 2014: Update thanks to feedback by Angelina Ryzhov and Reheman Baikejiang.
 * - 2017: Update thanks to feedback by Anh Phong Tran.
 * - 2019: Final update to original version thanks to feedback by Anh Phong Tran.
 *
 * MODERNIZATION (2025):
 * ---------------------------------
 * Modernized by: GitHub Copilot Assistant
 * Objectives: Upgrade to modern C17 standards matching MCML 2.1+ architecture
 *
 * Key Improvements:
 * - Modular C17 architecture with clean separation of concerns
 * - Professional command-line interface with comprehensive help system
 * - Memory-safe allocation with bounds checking and error handling
 * - PCG32 random number generator for superior statistical properties
 * - Cross-platform build system with optimization support
 * - Real-time progress reporting and performance metrics
 * - Thread-safe design preparation for future parallelization
 * - Comprehensive input validation and error recovery
 * - Full backward compatibility with original file formats
 *
 * Preserved Elements:
 * - Original Monte Carlo physics algorithms and accuracy
 * - Complete compatibility with existing input (.mci/.bin) and output (.bin/.m) formats
 * - All original mathematical models and tissue interaction physics
 * - MATLAB integration via maketissue.m and lookmcxyz.m programs
 *
 * License: GNU General Public License v3.0
 * Copyright (C) 2010-2017 Steven L. Jacques, Ting Li, Oregon Health & Science University
 * Copyright (C) 2025 Modernization - maintaining original authorship and algorithms
 *
 * PHYSICAL MODEL:
 * ---------------
 * - 3D voxelized tissue geometries with arbitrary optical properties
 * - Configurable absorption, scattering, and anisotropy per tissue type
 * - Henyey-Greenberg scattering phase function with tissue-specific g values
 * - Multiple source types: uniform, Gaussian, isotropic, and rectangular
 * - Comprehensive boundary condition support (infinite, escape, surface-only)
 * - Weight-based photon termination using Russian roulette techniques
 *
 * DETECTION GEOMETRY:
 * -------------------
 * - 3D fluence rate detection: F(x,y,z) [W/cm²/W.delivered]
 * - Voxel-based absorption detection throughout volume
 * - Configurable detection grid resolution and extent
 * - Multiple source configurations and focusing options
 */

#ifndef MCXYZ_H
#define MCXYZ_H

////////////////////////////////////////////////////////////////////////////////
// SYSTEM INCLUDES AND DEPENDENCIES

// Standard C17 library includes
#include <float.h>    // Floating-point limits and constants
#include <limits.h>   // Integer limits and constants
#include <math.h>     // Mathematical functions
#include <stdalign.h> // Alignment support
#include <stdbool.h>  // Boolean type support
#include <stddef.h>   // Standard type definitions
#include <stdint.h>   // Fixed-width integer types
#include <stdio.h>    // Standard I/O functions
#include <stdlib.h>   // Memory allocation and utilities
#include <string.h>   // String manipulation functions
#include <time.h>     // Time functions

////////////////////////////////////////////////////////////////////////////////
// COMPILE-TIME CONFIGURATION

#ifdef __cplusplus
extern "C" {
#endif

////////////////////////////////////////////////////////////////////////////////
// PHYSICAL AND MATHEMATICAL CONSTANTS

/**
 * Mathematical constants for photon transport calculations
 *
 * High precision values ensure accurate trigonometric and geometric
 * computations throughout the Monte Carlo simulation.
 */
#define PI 3.14159265358979323846 // Pi - ratio of circumference to diameter

/**
 * Monte Carlo simulation control parameters
 *
 * These constants control photon behavior, termination criteria, and
 * numerical precision throughout the transport simulation.
 */
#define THRESHOLD          0.01f   // Weight threshold for Russian roulette
#define CHANCE             0.1f    // Survival probability in roulette
#define ALIVE              1       // Photon status: active
#define DEAD               0       // Photon status: terminated
#define VOXEL_STEP_EPSILON 1.0e-7f // Small step to cross voxel faces

////////////////////////////////////////////////////////////////////////////////
// SIMULATION LIMITS AND ARRAY SIZES

/**
 * Maximum supported dimensions and array sizes
 *
 * These limits ensure memory safety while supporting realistic tissue
 * geometries. Values can be increased for larger simulations if needed.
 */
#define MAX_TISSUE_TYPES       100 // Maximum number of tissue types
#define MAX_STRING_LENGTH      256 // Maximum string buffer size
#define MAX_FILENAME_LENGTH    512 // Maximum filename length
#define MAX_TISSUE_NAME_LENGTH 64  // Maximum tissue name length

////////////////////////////////////////////////////////////////////////////////
// ERROR HANDLING AND RETURN CODES

/**
 * Error code enumeration for consistent error handling
 *
 * All functions return these codes to indicate success or specific
 * failure modes. This enables proper error propagation and handling.
 */
typedef enum {
	MCXYZ_SUCCESS = 0,              // Operation completed successfully
	MCXYZ_ERROR_MEMORY_ALLOCATION,  // Memory allocation failed
	MCXYZ_ERROR_FILE_NOT_FOUND,     // Input file not found
	MCXYZ_ERROR_FILE_FORMAT,        // Invalid file format
	MCXYZ_ERROR_INVALID_PARAMETER,  // Parameter out of valid range
	MCXYZ_ERROR_SIMULATION_FAILURE, // Monte Carlo simulation failed
	MCXYZ_ERROR_OUTPUT_WRITE,       // Output file write failed
	MCXYZ_ERROR_UNKNOWN             // Unknown error occurred
} McxyzErrorCode;

////////////////////////////////////////////////////////////////////////////////
// SOURCE CONFIGURATION TYPES

/**
 * Source type enumeration for different photon launching modes
 *
 * Defines the supported source geometries and their associated
 * parameter requirements for photon initialization.
 */
typedef enum {
	SOURCE_UNIFORM = 0,    // Uniform flat-field beam
	SOURCE_GAUSSIAN = 1,   // Gaussian beam profile
	SOURCE_ISOTROPIC = 2,  // Isotropic point source
	SOURCE_RECTANGULAR = 3 // Rectangular source
} SourceType;

/**
 * Boundary condition enumeration for photon escape behavior
 *
 * Controls how photons behave when reaching simulation volume boundaries.
 * Different modes support various tissue geometry requirements.
 */
typedef enum {
	BOUNDARY_INFINITE = 0,      // Infinite medium (no escape)
	BOUNDARY_ESCAPE_ALL = 1,    // Escape at all boundaries
	BOUNDARY_ESCAPE_SURFACE = 2 // Escape at top surface only
} BoundaryType;

////////////////////////////////////////////////////////////////////////////////
// SIMULATION PARAMETER STRUCTURES

/**
 * Source Configuration Structure
 *
 * Contains all parameters necessary to define the photon source geometry,
 * position, and launching characteristics. Supports multiple source types
 * with appropriate parameter validation.
 */
typedef struct {
	SourceType type;       // Source type identifier
	BoundaryType boundary; // Boundary condition type

	// Source position and geometry [cm]
	float position_x; // Source x-coordinate
	float position_y; // Source y-coordinate
	float position_z; // Source z-coordinate

	// Focus and beam parameters [cm]
	float focus_x; // Focus point x-coordinate
	float focus_y; // Focus point y-coordinate
	float focus_z; // Focus point z-coordinate

	float radius;  // Beam radius at launch
	float waist;   // Beam waist at focus

	// Manual trajectory specification (when enabled)
	bool manual_trajectory; // Use manual direction specification
	float direction_x;      // Manual x-direction cosine
	float direction_y;      // Manual y-direction cosine
	float direction_z;      // Manual z-direction cosine
} SourceConfig;

/**
 * Voxel Grid Configuration Structure
 *
 * Defines the 3D voxelized simulation volume including dimensions,
 * resolution, and spatial extent. Used for both tissue geometry
 * and fluence detection arrays.
 */
typedef struct {
	// Grid dimensions [number of voxels]
	int size_x; // Number of voxels in x-direction
	int size_y; // Number of voxels in y-direction
	int size_z; // Number of voxels in z-direction

	// Voxel spacing [cm]
	float spacing_x; // Voxel width in x-direction
	float spacing_y; // Voxel width in y-direction
	float spacing_z; // Voxel width in z-direction

	// Computed properties
	int total_voxels; // Total number of voxels
	float volume_x;   // Total x-extent [cm]
	float volume_y;   // Total y-extent [cm]
	float volume_z;   // Total z-extent [cm]
} VoxelGrid;

/**
 * Tissue Optical Properties Structure
 *
 * Contains optical properties for a single tissue type including
 * absorption, scattering, and anisotropy parameters.
 */
typedef struct {
	char name[MAX_TISSUE_NAME_LENGTH]; // Human-readable tissue name
	float absorption;                  // Absorption coefficient [cm⁻¹]
	float scattering;                  // Scattering coefficient [cm⁻¹]
	float anisotropy;                  // Anisotropy parameter g [-]
	float refractive_index;            // Refractive index [-]
} TissueProperties;

/**
 * Complete Simulation Configuration Structure
 *
 * Master structure containing all parameters necessary to run a complete
 * Monte Carlo photon transport simulation including geometry, source,
 * optical properties, and runtime parameters.
 */
typedef struct {
	char name[MAX_STRING_LENGTH]; // Simulation identifier name

	// Runtime control parameters
	float simulation_time_minutes; // Requested simulation time [min]
	uint64_t target_photon_count;  // Target number of photons

	// Geometric configuration
	VoxelGrid grid;      // Voxel grid specification
	SourceConfig source; // Source configuration

	// Tissue optical properties
	int tissue_count;                           // Number of tissue types
	TissueProperties tissues[MAX_TISSUE_TYPES]; // Tissue property array

	// Memory pointers for simulation data
	uint8_t* tissue_volume; // Tissue type per voxel
	float* fluence_volume;  // Fluence rate per voxel [W/cm²/W]
} SimulationConfig;

////////////////////////////////////////////////////////////////////////////////
// PHOTON STATE STRUCTURE

/**
 * Photon State Structure
 *
 * Contains all information necessary to track a single photon through
 * the Monte Carlo transport process including position, direction,
 * weight, and current tissue properties.
 */
typedef struct {
	// Spatial coordinates [cm]
	double position_x; // Current x-position
	double position_y; // Current y-position
	double position_z; // Current z-position

	// Direction cosines [-]
	double direction_x; // x-direction cosine
	double direction_y; // y-direction cosine
	double direction_z; // z-direction cosine

	// Photon properties
	double weight; // Current photon weight
	int status;    // ALIVE or DEAD

	// Current voxel indices
	int voxel_x; // Current voxel x-index
	int voxel_y; // Current voxel y-index
	int voxel_z; // Current voxel z-index

	// Current tissue properties
	float absorption; // Current absorption coefficient
	float scattering; // Current scattering coefficient
	float anisotropy; // Current anisotropy parameter
} PhotonState;

////////////////////////////////////////////////////////////////////////////////
// PERFORMANCE AND STATISTICS STRUCTURES

/**
 * Performance Metrics Structure
 *
 * Tracks simulation performance including timing, throughput, and
 * resource utilization for optimization and analysis.
 */
typedef struct {
	clock_t start_time;          // Simulation start time
	clock_t end_time;            // Simulation end time
	double elapsed_seconds;      // Total elapsed time [s]

	uint64_t photons_completed;  // Number of photons processed
	uint64_t photons_per_second; // Throughput [photons/s]

	size_t memory_allocated;     // Peak memory usage [bytes]
	int steps_per_photon_avg;    // Average steps per photon
} PerformanceMetrics;

////////////////////////////////////////////////////////////////////////////////
// FUNCTION DECLARATIONS

/**
 * Core Simulation Functions
 *
 * Primary interface functions for running Monte Carlo photon transport
 * simulations. These functions provide the main user-facing API.
 */

/**
 * Run complete Monte Carlo photon transport simulation
 *
 * @param config Simulation configuration with all parameters
 * @param metrics Performance metrics structure (output)
 * @return Error code indicating success or failure mode
 */
McxyzErrorCode run_monte_carlo_simulation(SimulationConfig* config, PerformanceMetrics* metrics);

/**
 * Input/Output Functions
 *
 * Functions for reading simulation input files, validating parameters,
 * and writing output results in standard formats.
 */

/**
 * Load simulation configuration from input files
 *
 * @param basename Base filename (without extensions)
 * @param config Simulation configuration structure (output)
 * @return Error code indicating success or failure mode
 */
McxyzErrorCode load_simulation_config(const char* basename, SimulationConfig* config);

/**
 * Write simulation results to output files
 *
 * @param config Simulation configuration with results
 * @param metrics Performance metrics for reporting
 * @return Error code indicating success or failure mode
 */
McxyzErrorCode write_simulation_results(const SimulationConfig* config, const PerformanceMetrics* metrics);

/**
 * Memory Management Functions
 *
 * Safe memory allocation and deallocation with error checking
 * and automatic cleanup capabilities.
 */

/**
 * Allocate and initialize simulation memory structures
 *
 * @param config Simulation configuration requiring memory
 * @return Error code indicating success or failure mode
 */
McxyzErrorCode allocate_simulation_memory(SimulationConfig* config);

/**
 * Free all simulation memory structures
 *
 * @param config Simulation configuration with allocated memory
 */
void free_simulation_memory(SimulationConfig* config);

/**
 * Utility Functions
 *
 * Helper functions for coordinate transformations, validation,
 * and mathematical operations used throughout the simulation.
 */

/**
 * Convert world coordinates to voxel indices
 *
 * @param x World x-coordinate [cm]
 * @param y World y-coordinate [cm]
 * @param z World z-coordinate [cm]
 * @param grid Voxel grid specification
 * @param voxel_x Voxel x-index (output)
 * @param voxel_y Voxel y-index (output)
 * @param voxel_z Voxel z-index (output)
 * @return true if coordinates within valid range
 */
bool world_to_voxel_coords(double x, double y, double z, const VoxelGrid* grid, int* voxel_x, int* voxel_y,
						   int* voxel_z);

/**
 * Get linear voxel index from 3D coordinates
 *
 * @param voxel_x Voxel x-index
 * @param voxel_y Voxel y-index
 * @param voxel_z Voxel z-index
 * @param grid Voxel grid specification
 * @return Linear voxel index
 */
static inline int get_voxel_index(int voxel_x, int voxel_y, int voxel_z, const VoxelGrid* grid) {
	return (voxel_z * grid->size_y * grid->size_x) + (voxel_x * grid->size_y) + voxel_y;
}

/**
 * Validate simulation configuration parameters
 *
 * @param config Simulation configuration to validate
 * @return Error code indicating validation result
 */
McxyzErrorCode validate_simulation_config(const SimulationConfig* config);

/**
 * Random Number Generation Functions
 *
 * Modern random number generator with good statistical properties
 * and thread-safety for future parallelization.
 */

/**
 * Initialize random number generator with seed
 *
 * @param seed Seed value for reproducible sequences
 */
void init_random_generator(uint64_t seed);

/**
 * Generate random number in range [0,1)
 *
 * @return Uniformly distributed random number
 */
double generate_random_number(void);

////////////////////////////////////////////////////////////////////////////////
// INLINE UTILITY FUNCTIONS

/**
 * Mathematical helper functions for common operations
 */

static inline double min_double(double a, double b) {
	return (a < b) ? a : b;
}

static inline double max_double(double a, double b) {
	return (a > b) ? a : b;
}

static inline double min3_double(double a, double b, double c) {
	return min_double(a, min_double(b, c));
}

static inline int sign_double(double x) {
	return (x >= 0.0) ? 1 : -1;
}

static inline double square_double(double x) {
	return x * x;
}

////////////////////////////////////////////////////////////////////////////////
// MULTI-THREADED SIMULATION FUNCTIONS

/**
 * Multi-threaded Monte Carlo simulation with performance optimizations
 */
McxyzErrorCode run_monte_carlo_simulation_mt(SimulationConfig* config, PerformanceMetrics* metrics);

/**
 * Ultra-optimized Monte Carlo simulation with advanced performance features
 */
McxyzErrorCode run_monte_carlo_simulation_ultra(SimulationConfig* config, PerformanceMetrics* metrics);

/**
 * Optimized memory allocation with alignment for cache performance
 */
McxyzErrorCode allocate_simulation_memory_aligned(SimulationConfig* config);

#ifdef __cplusplus
}
#endif

#endif // MCXYZ_H
