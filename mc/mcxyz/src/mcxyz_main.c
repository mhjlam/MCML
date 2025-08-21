/*==============================================================================
 * MCXYZ - Monte Carlo photon transport in 3D voxelized media
 *
 * Main Program and Command-Line Interface Module
 *
 * This module provides the main program entry point and handles all command-line
 * interface functionality including argument parsing, help display, and program
 * flow control. Implements a modern CLI with comprehensive options.
 *
 * This file contains the main application logic including:
 * - Command line argument parsing with validation
 * - Help system with usage examples and parameter descriptions
 * - Simulation configuration and execution
 * - Performance reporting and result output
 * - Error handling and cleanup
 *
 * COPYRIGHT:
 * ----------
 * Original work (2010-2017): Steven L. Jacques, Ting Li (Oregon Health & Science University)
 * Modernization (2025): C17 standards, multi-threading, performance optimizations
 *
 * LICENSE:
 * --------
 * This software is distributed under the terms of the GNU General Public License v3.0
 * 
 * COMPILATION:
 * -----------
 * Requires C17 compiler with OpenMP support
 * gcc -std=c17 -fopenmp -O2 -march=native -o mcxyz mcxyz_main.c mcxyz_*.c -lm
 *
 *============================================================================*/

#include "mcxyz.h"

#include <errno.h>

////////////////////////////////////////////////////////////////////////////////
// PROGRAM INFORMATION AND VERSION

#define PROGRAM_NAME        "mcxyz"
#define PROGRAM_VERSION     "2.0.0"
#define PROGRAM_DESCRIPTION "Monte Carlo simulation of photon transport in 3D voxelized media"
#define PROGRAM_AUTHORS     "Steven L. Jacques, Ting Li (Oregon Health & Science University)"
#define PROGRAM_ORIGINAL    "2010-2017"
#define PROGRAM_MODERNIZED  "2025"

////////////////////////////////////////////////////////////////////////////////
// PROGRAM CONFIGURATION STRUCTURE

/**
 * Program Runtime Configuration
 *
 * Contains command line options and runtime parameters that control
 * program behavior beyond the Monte Carlo simulation parameters.
 */
typedef struct {
	bool verbose;                              // Enable verbose output
	bool quiet;                                // Suppress non-essential output
	bool show_help;                            // Show help message
	bool show_version;                         // Show version information
	bool use_multithreading;                   // Enable multi-threaded execution
	bool use_ultra_optimization;               // Enable ultra performance optimizations

	char input_basename[MAX_FILENAME_LENGTH];  // Input file basename
	char output_basename[MAX_FILENAME_LENGTH]; // Output file basename (optional)

	// Runtime overrides
	float time_override;      // Override simulation time (minutes)
	uint64_t photon_override; // Override photon count
	uint64_t random_seed;     // Random number generator seed
	int thread_count;         // Number of threads to use (0 = auto)

	bool has_time_override;   // Whether time override is specified
	bool has_photon_override; // Whether photon override is specified
	bool has_seed_override;   // Whether seed override is specified
} ProgramConfig;

////////////////////////////////////////////////////////////////////////////////
// HELP AND USAGE FUNCTIONS

/**
 * Display program version information
 */
static void show_version_info(void) {
	printf("%s version %s\n", PROGRAM_NAME, PROGRAM_VERSION);
	printf("%s\n", PROGRAM_DESCRIPTION);
	printf("\n");
	printf("Original Authors: %s\n", PROGRAM_AUTHORS);
	printf("Original Development: %s\n", PROGRAM_ORIGINAL);
	printf("Modernization: %s - Upgraded to C17 with modular architecture\n", PROGRAM_MODERNIZED);
	printf("\n");
	printf("Development History:\n");
	printf("  2010-2012: Created by Ting Li and Steven L. Jacques (OHSU)\n");
	printf("  2017:      Final update to original monolithic version\n");
	printf("  2025:      Modernized to C17 with professional CLI and modular design\n");
	printf("\n");
	printf("Copyright (C) %s Steven L. Jacques, Ting Li, Oregon Health & Science University\n", PROGRAM_ORIGINAL);
	printf("Copyright (C) %s Modernization - maintaining original authorship and algorithms\n", PROGRAM_MODERNIZED);
	printf("License: GNU General Public License v3.0\n");
	printf("This is free software; see the source for copying conditions.\n");
}

/**
 * Display comprehensive help message with usage examples
 */
static void show_help_message(const char* program_name) {
	printf("USAGE:\n");
	printf("  %s [OPTIONS] <input_basename>\n\n", program_name);

	printf("DESCRIPTION:\n");
	printf("  Monte Carlo simulation of photon transport in 3D voxelized media.\n");
	printf("  Simulates photon migration through complex tissue geometries with\n");
	printf("  arbitrary optical properties per voxel type.\n\n");

	printf("INPUT FILES:\n");
	printf("  <input_basename>_H.mci    Header file with simulation parameters\n");
	printf("  <input_basename>_T.bin    Binary tissue structure file\n\n");

	printf("OUTPUT FILES:\n");
	printf("  <input_basename>_F.bin    Fluence rate distribution [W/cm^2/W]\n");
	printf("  <input_basename>_props.m  Tissue optical properties (MATLAB)\n\n");

	printf("OPTIONS:\n");
	printf("  -h, --help              Show this help message and exit\n");
	printf("  -v, --version           Show version information and exit\n");
	printf("  -V, --verbose           Enable verbose output with detailed progress\n");
	printf("  -q, --quiet             Suppress non-essential output messages\n");
	printf("  -j, --threads <count>   Enable multi-threading with specified thread count (0=auto)\n");
	printf("  -u, --ultra             Enable ultra performance optimizations (SIMD, advanced)\n");
	printf("  -t, --time <minutes>    Override simulation time (minutes)\n");
	printf("  -n, --photons <count>   Override target photon count\n");
	printf("  -s, --seed <value>      Set random number generator seed\n");
	printf("  -o, --output <basename> Set output file basename\n\n");

	printf("EXAMPLES:\n");
	printf("  Basic simulation:\n");
	printf("    %s skinvessel\n\n", program_name);

	printf("  Run for specific time with verbose output:\n");
	printf("    %s --time 5.0 --verbose skinvessel\n\n", program_name);

	printf("  Use specific random seed for reproducibility:\n");
	printf("    %s --seed 12345 --photons 1000000 skinvessel\n\n", program_name);

	printf("  Quiet mode with custom output location:\n");
	printf("    %s --quiet --output results/vessel skinvessel\n\n", program_name);

	printf("SOURCE TYPES:\n");
	printf("  0 = Uniform flat-field beam\n");
	printf("  1 = Gaussian beam profile\n");
	printf("  2 = Isotropic point source\n");
	printf("  3 = Rectangular source\n\n");

	printf("BOUNDARY CONDITIONS:\n");
	printf("  0 = Infinite medium (no boundaries)\n");
	printf("  1 = Escape at all boundaries\n");
	printf("  2 = Escape at top surface only\n\n");
}

////////////////////////////////////////////////////////////////////////////////
// SIMPLE COMMAND LINE PARSING

/**
 * Check if argument matches short or long option
 */
static bool matches_option(const char* arg, const char short_opt, const char* long_opt) {
	if (arg[0] != '-') {
		return false;
	}

	// Short option: -h
	if (arg[1] != '-' && arg[1] == short_opt && arg[2] == '\0') {
		return true;
	}

	// Long option: --help
	if (arg[1] == '-' && strcmp(arg + 2, long_opt) == 0) {
		return true;
	}

	return false;
}

/**
 * Initialize program configuration with default values
 */
static void init_program_config(ProgramConfig* config) {
	memset(config, 0, sizeof(ProgramConfig));

	// Set defaults
	config->verbose = false;
	config->quiet = false;
	config->show_help = false;
	config->show_version = false;

	config->time_override = 0.0F;
	config->photon_override = 0;
	config->random_seed = 0;

	config->has_time_override = false;
	config->has_photon_override = false;
	config->has_seed_override = false;
}

/**
 * Parse command line arguments and populate program configuration
 */
static McxyzErrorCode parse_command_line(int argc, char* argv[], ProgramConfig* config) {
	int i = 1; // Start from first argument (skip program name)
	bool found_input = false;

	while (i < argc) {
		char* arg = argv[i];

		// Help options
		if (matches_option(arg, 'h', "help")) {
			config->show_help = true;
			return MCXYZ_SUCCESS;
		}

		// Version option
		if (matches_option(arg, 'v', "version")) {
			config->show_version = true;
			return MCXYZ_SUCCESS;
		}

		// Verbose option
		if (matches_option(arg, 'V', "verbose")) {
			config->verbose = true;
			i++;
			continue;
		}

		// Quiet option
		if (matches_option(arg, 'q', "quiet")) {
			config->quiet = true;
			i++;
			continue;
		}

		// Multi-threading option
		if (matches_option(arg, 'j', "threads")) {
			if (i + 1 >= argc) {
				fprintf(stderr, "Error: --threads requires a value\n");
				return MCXYZ_ERROR_INVALID_PARAMETER;
			}
			config->thread_count = atoi(argv[i + 1]);
			config->use_multithreading = true;
			if (config->thread_count < 0) {
				fprintf(stderr, "Error: Thread count must be non-negative (got %d)\n", config->thread_count);
				return MCXYZ_ERROR_INVALID_PARAMETER;
			}
			i += 2;
			continue;
		}

		// Ultra optimization option
		if (matches_option(arg, 'u', "ultra")) {
			config->use_ultra_optimization = true;
			config->use_multithreading = true; // Ultra implies multi-threading
			i++;
			continue;
		}

		// Time override option
		if (matches_option(arg, 't', "time")) {
			if (i + 1 >= argc) {
				fprintf(stderr, "Error: --time requires a value\n");
				return MCXYZ_ERROR_INVALID_PARAMETER;
			}
			config->time_override = (float)atof(argv[i + 1]);
			config->has_time_override = true;
			if (config->time_override <= 0.0F) {
				fprintf(stderr, "Error: Time override must be positive (got %.2f)\n", config->time_override);
				return MCXYZ_ERROR_INVALID_PARAMETER;
			}
			i += 2;
			continue;
		}

		// Photon count override option
		if (matches_option(arg, 'n', "photons")) {
			if (i + 1 >= argc) {
				fprintf(stderr, "Error: --photons requires a value\n");
				return MCXYZ_ERROR_INVALID_PARAMETER;
			}
			config->photon_override = (uint64_t)strtoull(argv[i + 1], NULL, 10);
			config->has_photon_override = true;
			if (config->photon_override == 0) {
				fprintf(stderr, "Error: Photon count must be positive (got %llu)\n",
						(unsigned long long)config->photon_override);
				return MCXYZ_ERROR_INVALID_PARAMETER;
			}
			i += 2;
			continue;
		}

		// Random seed option
		if (matches_option(arg, 's', "seed")) {
			if (i + 1 >= argc) {
				fprintf(stderr, "Error: --seed requires a value\n");
				return MCXYZ_ERROR_INVALID_PARAMETER;
			}
			config->random_seed = (uint64_t)strtoull(argv[i + 1], NULL, 10);
			config->has_seed_override = true;
			i += 2;
			continue;
		}

		// Output basename option
		if (matches_option(arg, 'o', "output")) {
			if (i + 1 >= argc) {
				fprintf(stderr, "Error: --output requires a value\n");
				return MCXYZ_ERROR_INVALID_PARAMETER;
			}
			if (strlen(argv[i + 1]) >= MAX_FILENAME_LENGTH) {
				fprintf(stderr, "Error: Output basename too long (max %d characters)\n", MAX_FILENAME_LENGTH - 1);
				return MCXYZ_ERROR_INVALID_PARAMETER;
			}
			strncpy(config->output_basename, argv[i + 1], MAX_FILENAME_LENGTH - 1);
			config->output_basename[MAX_FILENAME_LENGTH - 1] = '\0';
			i += 2;
			continue;
		}

		// Unknown option
		if (arg[0] == '-') {
			fprintf(stderr, "Error: Unknown option '%s'\n", arg);
			fprintf(stderr, "Try '%s --help' for more information.\n", argv[0]);
			return MCXYZ_ERROR_INVALID_PARAMETER;
		}

		// Positional argument (input basename)
		if (!found_input) {
			if (strlen(arg) >= MAX_FILENAME_LENGTH) {
				fprintf(stderr, "Error: Input basename too long (max %d characters)\n", MAX_FILENAME_LENGTH - 1);
				return MCXYZ_ERROR_INVALID_PARAMETER;
			}
			strncpy(config->input_basename, arg, MAX_FILENAME_LENGTH - 1);
			config->input_basename[MAX_FILENAME_LENGTH - 1] = '\0';
			found_input = true;
			i++;
			continue;
		}

		// Extra argument
		fprintf(stderr, "Warning: Extra argument ignored: '%s'\n", arg);
		i++;
	}

	// Check for required input basename
	if (!found_input) {
		fprintf(stderr, "Error: Input basename is required\n");
		fprintf(stderr, "Usage: %s [OPTIONS] <input_basename>\n", argv[0]);
		fprintf(stderr, "Try '%s --help' for more information.\n", argv[0]);
		return MCXYZ_ERROR_INVALID_PARAMETER;
	}

	// Use input basename as output basename if not specified
	if (strlen(config->output_basename) == 0) {
		strncpy(config->output_basename, config->input_basename, MAX_FILENAME_LENGTH - 1);
		config->output_basename[MAX_FILENAME_LENGTH - 1] = '\0';
	}

	// Validate option combinations
	if (config->verbose && config->quiet) {
		fprintf(stderr, "Warning: Both --verbose and --quiet specified. Using verbose mode.\n");
		config->quiet = false;
	}

	return MCXYZ_SUCCESS;
}

////////////////////////////////////////////////////////////////////////////////
// SIMULATION ORCHESTRATION FUNCTIONS

/**
 * Apply command line overrides to simulation configuration
 */
static void apply_program_overrides(ProgramConfig* prog_config, SimulationConfig* sim_config) {
	if (prog_config->has_time_override) {
		sim_config->simulation_time_minutes = prog_config->time_override;
	}

	if (prog_config->has_photon_override) {
		sim_config->target_photon_count = prog_config->photon_override;
	}

	// Update output basename if specified
	if (strcmp(prog_config->output_basename, prog_config->input_basename) != 0) {
		strncpy(sim_config->name, prog_config->output_basename, MAX_STRING_LENGTH - 1);
		sim_config->name[MAX_STRING_LENGTH - 1] = '\0';
	}
}

/**
 * Print simulation configuration summary
 */
static void print_simulation_summary(const SimulationConfig* config, const ProgramConfig* prog_config) {
	if (prog_config->quiet) {
		return;
	}

	printf("=== MCXYZ Simulation Configuration ===\n");
	printf("Simulation name: %s\n", config->name);
	printf("Target time: %.2f minutes\n", config->simulation_time_minutes);
	printf("Grid dimensions: %d x %d x %d voxels\n", config->grid.size_x, config->grid.size_y, config->grid.size_z);
	printf("Voxel spacing: %.4f x %.4f x %.4f cm\n", config->grid.spacing_x, config->grid.spacing_y,
		   config->grid.spacing_z);
	printf("Total volume: %.3f x %.3f x %.3f cm\n", config->grid.volume_x, config->grid.volume_y,
		   config->grid.volume_z);
	printf("Source type: %d\n", (int)config->source.type);
	printf("Boundary type: %d\n", (int)config->source.boundary);
	printf("Tissue types: %d\n", config->tissue_count);

	if (prog_config->verbose) {
		printf("\nTissue Properties:\n");
		for (int i = 1; i <= config->tissue_count; i++) {
			printf("  Type %d: μa=%.4f, μs=%.4f, g=%.4f\n", i, config->tissues[i].absorption,
				   config->tissues[i].scattering, config->tissues[i].anisotropy);
		}
	}

	printf("======================================\n\n");
}

/**
 * Handle error conditions with appropriate messages and cleanup
 */
static int handle_error(McxyzErrorCode error_code, const char* context, SimulationConfig* sim_config) {
	const char* error_message;

	switch (error_code) {
		case MCXYZ_SUCCESS: return 0;
		case MCXYZ_ERROR_MEMORY_ALLOCATION: error_message = "Memory allocation failed"; break;
		case MCXYZ_ERROR_FILE_NOT_FOUND: error_message = "Input file not found"; break;
		case MCXYZ_ERROR_FILE_FORMAT: error_message = "Invalid file format"; break;
		case MCXYZ_ERROR_INVALID_PARAMETER: error_message = "Invalid parameter"; break;
		case MCXYZ_ERROR_SIMULATION_FAILURE: error_message = "Simulation failed"; break;
		case MCXYZ_ERROR_OUTPUT_WRITE: error_message = "Output file write failed"; break;
		default: error_message = "Unknown error"; break;
	}

	fprintf(stderr, "Error in %s: %s\n", context, error_message);

	// Clean up simulation memory if allocated
	if (sim_config && sim_config->tissue_volume) {
		free_simulation_memory(sim_config);
	}

	return (int)error_code;
}

////////////////////////////////////////////////////////////////////////////////
// MAIN PROGRAM ENTRY POINT

/**
 * Main application entry point with comprehensive CLI support
 */
int main(int argc, char* argv[]) {
	ProgramConfig prog_config;
	SimulationConfig sim_config;
	PerformanceMetrics metrics;
	McxyzErrorCode result;

	// Initialize configurations
	init_program_config(&prog_config);
	memset(&sim_config, 0, sizeof(SimulationConfig));
	memset(&metrics, 0, sizeof(PerformanceMetrics));

	// Parse command line arguments
	result = parse_command_line(argc, argv, &prog_config);
	if (result != MCXYZ_SUCCESS) {
		return handle_error(result, "command line parsing", NULL);
	}

	// Handle help and version requests
	if (prog_config.show_help) {
		show_help_message(argv[0]);
		return 0;
	}

	if (prog_config.show_version) {
		show_version_info();
		return 0;
	}

	// Initialize random number generator
	uint64_t seed = prog_config.has_seed_override ? prog_config.random_seed : (uint64_t)time(NULL);
	init_random_generator(seed);

	if (prog_config.verbose) {
		printf("Random seed: %llu\n", (unsigned long long)seed);
	}

	// Load simulation configuration from input files
	if (!prog_config.quiet) {
		printf("Loading simulation configuration from '%s'...\n", prog_config.input_basename);
	}

	result = load_simulation_config(prog_config.input_basename, &sim_config);
	if (result != MCXYZ_SUCCESS) {
		return handle_error(result, "loading simulation configuration", &sim_config);
	}

	// Apply command line overrides
	apply_program_overrides(&prog_config, &sim_config);

	// Validate simulation configuration
	result = validate_simulation_config(&sim_config);
	if (result != MCXYZ_SUCCESS) {
		return handle_error(result, "configuration validation", &sim_config);
	}

	// Allocate simulation memory
	result = allocate_simulation_memory(&sim_config);
	if (result != MCXYZ_SUCCESS) {
		return handle_error(result, "memory allocation", &sim_config);
	}

	// Print simulation summary
	print_simulation_summary(&sim_config, &prog_config);

	// Run Monte Carlo simulation (choose single-threaded or multi-threaded)
	if (!prog_config.quiet) {
		printf("Starting Monte Carlo simulation...\n");
		if (prog_config.use_multithreading) {
			printf("Using multi-threaded execution\n");
		}
		printf("Target photons: %llu\n", (unsigned long long)sim_config.target_photon_count);
		printf("Target time: %.2f minutes\n\n", sim_config.simulation_time_minutes);
	}

	if (prog_config.use_ultra_optimization) {
		result = run_monte_carlo_simulation_ultra(&sim_config, &metrics);
	}
	else if (prog_config.use_multithreading) {
		result = run_monte_carlo_simulation_mt(&sim_config, &metrics);
	}
	else {
		result = run_monte_carlo_simulation(&sim_config, &metrics);
	}

	if (result != MCXYZ_SUCCESS) {
		return handle_error(result, "Monte Carlo simulation", &sim_config);
	}

	// Write simulation results
	if (!prog_config.quiet) {
		printf("Writing simulation results...\n");
	}

	result = write_simulation_results(&sim_config, &metrics);
	if (result != MCXYZ_SUCCESS) {
		return handle_error(result, "writing results", &sim_config);
	}

	// Print performance summary
	if (!prog_config.quiet) {
		printf("\n=== Simulation Complete ===\n");
		printf("Photons processed: %llu\n", (unsigned long long)metrics.photons_completed);
		printf("Simulation time: %.3f seconds\n", metrics.elapsed_seconds);
		printf("Performance: %.1f million photons/second\n", metrics.photons_per_second / 1e6);

		// Calculate memory usage
		size_t memory_used = sim_config.grid.total_voxels * (sizeof(uint8_t) + sizeof(float));
		printf("Memory used: %.2f MB\n", memory_used / (1024.0 * 1024.0));
		printf("===========================\n");
	}

	// Clean up and exit
	free_simulation_memory(&sim_config);

	if (prog_config.verbose) {
		printf("Simulation completed successfully.\n");
	}

	return 0;
}
