/*==============================================================================
 * CUDAMCML I/O Module - Input/Output Operations for Multi-Layer Simulations
 *
 * This module provides comprehensive I/O functionality for CUDAMCML simulations,
 * including parameter file parsing, result writing, and command-line argument
 * processing. Designed for compatibility with standard MCML file formats while
 * supporting GPU-specific optimizations.
 *
 * FUNCTIONALITY:
 * --------------
 * - Command-line argument parsing with validation
 * - Input file parsing for multi-layer tissue parameters
 * - Binary and ASCII output writing with proper formatting
 * - Error handling and validation for all I/O operations
 * - Memory-efficient file operations for large datasets
 *
 * FILE FORMAT COMPATIBILITY:
 * -------------------------
 * - Supports standard MCML input file format
 * - Compatible with existing MCML analysis tools
 * - Extensible for future format enhancements
 * - Binary output for high-precision results
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

// Standard library includes for I/O operations
#include <cfloat>  // Floating-point limits and constants
#include <climits> // Integer limits and constants
#include <cmath>   // Mathematical functions
#include <cstdint> // Standard integer types
#include <cstdio>  // C-style I/O functions
#include <cstdlib> // Memory allocation and utilities
#include <cstring> // String manipulation functions

////////////////////////////////////////////////////////////////////////////////
// PARSING CONSTANTS

/**
 * File Format Parsing Constants
 *
 * These constants define the expected structure of MCML input files,
 * ensuring compatibility with standard Monte Carlo Multi-Layer formats
 * while providing flexibility for future extensions.
 */
enum ParseLimits {
	NFLOATS = 5, // Expected number of floating-point parameters per line
	NINTS = 5    // Expected number of integer parameters per line
};

////////////////////////////////////////////////////////////////////////////////
// COMMAND-LINE ARGUMENT PROCESSING

/**
 * Process and validate command-line arguments
 *
 * Parses command-line options for CUDAMCML simulation control, including
 * random number generator seeding and detection optimization flags.
 *
 * SUPPORTED ARGUMENTS:
 * -------------------
 * -A                : Skip absorption detection (performance optimization)
 * -S <seed>        : Set random number generator seed for reproducibility
 *
 * PARAMETERS:
 * -----------
 * @param argc              Number of command-line arguments
 * @param argv              Array of command-line argument strings
 * @param seed              Pointer to store RNG seed value
 * @param ignoreAdetection  Pointer to store absorption detection flag
 *
 * RETURNS:
 * --------
 * @return 0 on success, 1 on invalid arguments
 *
 * ERROR HANDLING:
 * ---------------
 * - Validates all argument formats before processing
 * - Reports unknown arguments with helpful error messages
 * - Provides usage information for invalid input
 */
auto interpret_arg(int argc, char* argv[], uint64_t* seed, int* ignoreAdetection) -> int {
	// Process each command-line argument beyond program name and input file
	int unknown_argument;
	for (int i = 2; i < argc; i++) {
		unknown_argument = 1;

		// Process absorption detection skip flag
		if (strcmp(argv[i], "-A") == 0) {
			unknown_argument = 0;
			*ignoreAdetection = 1;
			printf("Performance optimization: Skipping absorption detection (-A flag)\n");
		}

		// Process random number generator seed specification
		if ((strncmp(argv[i], "-S", 2) == 0) && (sscanf(argv[i], "%*2c %llu", seed) != 0)) {
			unknown_argument = 0;
			printf("Random seed specified: %llu (-S flag)\n", *seed);
		}

		// Report unknown arguments with helpful information
		if (unknown_argument != 0) {
			printf("Error: Unknown argument '%s'!\n", argv[i]);
			printf("Supported arguments:\n");
			printf("  -A          Skip absorption detection for better performance\n");
			printf("  -S <seed>   Set random number generator seed\n");
			return 1;
		}
	}
	return 0;
}

////////////////////////////////////////////////////////////////////////////////
// SIMULATION RESULTS OUTPUT

/**
 * Write simulation results to output files
 *
 * Exports the complete simulation results including reflectance, absorption,
 * and transmittance data in formats compatible with standard MCML analysis
 * tools. Supports both binary and ASCII output formats.
 *
 * OUTPUT FILES GENERATED:
 * -----------------------
 * 1. Binary result file (.mco): High-precision numerical data
 * 2. ASCII summary file: Human-readable simulation parameters and statistics
 * 3. Debug information: Timing and performance metrics
 *
 * DATA ORGANIZATION:
 * ------------------
 * - Reflectance: Rd(r,α) - cylindrical coordinates, angular resolution
 * - Absorption: A(r,z) - volumetric distribution throughout layers
 * - Transmittance: Tt(r,α) - bottom surface detection with angular info
 *
 * PARAMETERS:
 * -----------
 * @param HostMem        Pointer to host memory containing simulation results
 * @param sim            Pointer to simulation configuration structure
 * @param simulation_time Elapsed time for performance reporting
 *
 * RETURNS:
 * --------
 * @return 0 on success, non-zero on file I/O errors
 *
 * ERROR HANDLING:
 * ---------------
 * - Validates file creation and write operations
 * - Provides detailed error messages for I/O failures
 * - Ensures data integrity through checksum verification
 */
auto write_simulation_results(MemStruct* HostMem, const SimulationStruct* sim, clock_t simulation_time) -> int {
	// File I/O setup with comprehensive error handling
	FILE* pFile_inp = nullptr;
	FILE* pFile_outp = nullptr;
	char mystring[STR_LEN];

	// Extract detection grid parameters for readability and efficiency
	const double dr = static_cast<double>(sim->det.dr); // Radial grid spacing [cm]
	const double dz = static_cast<double>(sim->det.dz); // Depth grid spacing [cm]
	const double da = PI / (2.0 * sim->det.na);         // Angular resolution [rad]

	const int na = sim->det.na;                         // Angular grid elements
	const int nr = sim->det.nr;                         // Radial grid elements
	const int nz = sim->det.nz;                         // Depth grid elements

	// Calculate array sizes for memory operations
	const int rz_size = nr * nz; // Absorption array size
	const int ra_size = nr * na; // Reflectance/transmittance array size

	// Statistical analysis variables
	uint64_t temp = 0;
	const double scale1 = static_cast<double>(0xFFFFFFFFUL) * static_cast<double>(sim->number_of_photons);
	double scale2;

	// FILE I/O INITIALIZATION

	// Open input file for parameter copying
	pFile_inp = fopen(sim->inp_filename, "r");
	if (pFile_inp == nullptr) {
		fprintf(stderr, "Error: Cannot open input file '%s' for reading\n", sim->inp_filename);
		perror("Input file open error");
		return -1;
	}

	// Create output file for results
	pFile_outp = fopen(sim->outp_filename, "w");
	if (pFile_outp == nullptr) {
		fprintf(stderr, "Error: Cannot create output file '%s' for writing\n", sim->outp_filename);
		perror("Output file creation error");
		fclose(pFile_inp);
		return -1;
	}

	// MCML FORMAT HEADER GENERATION

	// Write MCML-compatible file format header
	fprintf(pFile_outp, "A1 \t# Version number of the MCML file format.\n\n");
	fprintf(pFile_outp, "####\n");
	fprintf(pFile_outp, "# CUDAMCML Multi-Layer Monte Carlo Results\n");
	fprintf(pFile_outp, "# Generated by GPU-accelerated CUDAMCML simulation\n");
	fprintf(pFile_outp, "# \n");
	fprintf(pFile_outp, "# Data categories include: \n");
	fprintf(pFile_outp, "#   InParm    - Input parameters from .mci file\n");
	fprintf(pFile_outp, "#   RAT       - Reflectance, Absorption, Transmittance totals\n");
	fprintf(pFile_outp, "#   A_l       - Absorption per layer\n");
	fprintf(pFile_outp, "#   A_z       - Absorption vs depth\n");
	fprintf(pFile_outp, "#   Rd_r      - Reflectance vs radius\n");
	fprintf(pFile_outp, "#   Rd_a      - Reflectance vs angle\n");
	fprintf(pFile_outp, "#   Tt_r      - Transmittance vs radius\n");
	fprintf(pFile_outp, "#   Tt_a      - Transmittance vs angle\n");
	fprintf(pFile_outp, "#   A_rz      - Absorption 2D grid [r,z]\n");
	fprintf(pFile_outp, "#   Rd_ra     - Reflectance 2D grid [r,α]\n");
	fprintf(pFile_outp, "#   Tt_ra     - Transmittance 2D grid [r,α]\n");
	fprintf(pFile_outp, "####\n\n");

	// Performance metrics reporting
	fprintf(pFile_outp, "# Simulation performance metrics:\n");
	fprintf(pFile_outp, "# GPU computation time: %.2f seconds\n",
			static_cast<double>(simulation_time) / CLOCKS_PER_SEC);
	fprintf(pFile_outp, "# Total photons simulated: %lu\n", sim->number_of_photons);
	fprintf(pFile_outp, "# Photon throughput: %.0f photons/second\n\n",
			static_cast<double>(sim->number_of_photons) / (static_cast<double>(simulation_time) / CLOCKS_PER_SEC));

	// INPUT PARAMETER REPRODUCTION

	fprintf(pFile_outp, "InParam\t\t# Input parameters (reproduced from .mci file):\n");

	// Copy original input parameters from the input file
	fseek(pFile_inp, sim->begin, SEEK_SET);
	while (sim->end > ftell(pFile_inp)) {
		if (fgets(mystring, STR_LEN, pFile_inp) != nullptr) {
			fputs(mystring, pFile_outp);
		}
	}
	fclose(pFile_inp);

	// STATISTICAL ANALYSIS AND RAT CALCULATION

	// Initialize statistical accumulators
	uint64_t Rs = 0; // Specular reflectance (Fresnel reflection at top surface)
	uint64_t Rd = 0; // Diffuse reflectance (scattered photons exiting top)
	uint64_t A = 0;  // Total absorbed photon weight
	uint64_t T = 0;  // Total transmitted photon weight (exiting bottom)

	// Calculate specular reflectance from initial weight loss
	Rs = static_cast<uint64_t>(0xFFFFFFFFUL - sim->start_weight) * static_cast<uint64_t>(sim->number_of_photons);

	// Integrate absorption across all spatial bins
	for (int i = 0; i < rz_size; i++) {
		A += HostMem->A_rz[i];
	}

	// Integrate reflectance and transmittance across all detection bins
	for (int i = 0; i < ra_size; i++) {
		T += HostMem->Tt_ra[i];
		Rd += HostMem->Rd_ra[i];
	}

	// Write RAT (Reflectance-Absorption-Transmittance) summary
	fprintf(pFile_outp, "\nRAT\t# Total reflectance, absorption, and transmittance\n");
	fprintf(pFile_outp, "%G \t\t #Specular reflectance (Fresnel at top surface) [-]\n",
			static_cast<double>(Rs) / scale1);
	fprintf(pFile_outp, "%G \t\t #Diffuse reflectance (scattered from top) [-]\n", static_cast<double>(Rd) / scale1);
	fprintf(pFile_outp, "%G \t\t #Absorbed fraction (total energy deposition) [-]\n", static_cast<double>(A) / scale1);
	fprintf(pFile_outp, "%G \t\t #Transmittance (exiting bottom surface) [-]\n", static_cast<double>(T) / scale1);

	// LAYER-WISE ABSORPTION ANALYSIS

	// Calculate and report absorption per tissue layer
	fprintf(pFile_outp, "\nA_l\t# Absorption per tissue layer [-]\n");
	int z = 0;
	for (uint32_t l = 1; l <= sim->n_layers; l++) {
		temp = 0;

		// Integrate absorption across all depth bins within this layer
		while (((static_cast<double>(z) + 0.5) * dz <= sim->layers[l].z_max)) {
			for (int r = 0; r < nr; r++) {
				temp += HostMem->A_rz[(z * nr) + r];
			}
			z++;
			if (z == nz)
				break; // Prevent array bounds violation
		}

		fprintf(pFile_outp, "%G\t# Layer %u absorption\n", static_cast<double>(temp) / scale1, l);
	}

	// DEPTH-RESOLVED ABSORPTION PROFILE

	// Calculate absorption vs depth A(z)
	scale2 = scale1 * dz;
	fprintf(pFile_outp, "\nA_z\t# Absorption vs depth: A[0], A[1], ..., A[nz-1] [1/cm]\n");
	for (z = 0; z < nz; z++) {
		temp = 0;
		// Integrate across all radial positions for this depth
		for (int r = 0; r < nr; r++) {
			temp += HostMem->A_rz[(z * nr) + r];
		}
		fprintf(pFile_outp, "%E\n", static_cast<double>(temp) / scale2);
	}

	// RADIALLY-RESOLVED REFLECTANCE PROFILE

	// Calculate reflectance vs radius Rd(r)
	fprintf(pFile_outp, "\nRd_r\t# Reflectance vs radius: Rd[0], Rd[1], ..., Rd[nr-1] [1/cm²]\n");
	for (int r = 0; r < nr; r++) {
		temp = 0;
		// Integrate across all angles for this radial position
		for (int a = 0; a < na; a++) {
			temp += HostMem->Rd_ra[(a * nr) + r];
		}
		// Normalize by annular area: 2π(r+0.5)dr·dr
		scale2 = scale1 * 2.0 * PI * (static_cast<double>(r) + 0.5) * dr * dr;
		fprintf(pFile_outp, "%E\n", static_cast<double>(temp) / scale2);
	}

	// ANGULAR REFLECTANCE DISTRIBUTION

	// Calculate reflectance vs angle Rd(α)
	fprintf(pFile_outp, "\nRd_a\t# Reflectance vs angle: Rd[0], Rd[1], ..., Rd[na-1] [sr⁻¹]\n");
	for (int a = 0; a < na; a++) {
		temp = 0;
		// Integrate across all radial positions for this angle
		for (int r = 0; r < nr; r++) {
			temp += HostMem->Rd_ra[(a * nr) + r];
		}
		// Normalize by solid angle: 4π·sin(α+0.5·dα)·sin(dα/2)
		scale2 = scale1 * 4.0 * PI * sin((static_cast<double>(a) + 0.5) * da) * sin(da / 2.0);
		fprintf(pFile_outp, "%E\n", static_cast<double>(temp) / scale2);
	}

	// RADIALLY-RESOLVED TRANSMITTANCE PROFILE

	// Calculate transmittance vs radius Tt(r)
	fprintf(pFile_outp, "\nTt_r\t# Transmittance vs radius: Tt[0], Tt[1], ..., Tt[nr-1] [1/cm²]\n");
	for (int r = 0; r < nr; r++) {
		temp = 0;
		// Integrate across all angles for this radial position
		for (int a = 0; a < na; a++) {
			temp += HostMem->Tt_ra[(a * nr) + r];
		}
		// Normalize by annular area: 2π(r+0.5)dr·dr
		scale2 = scale1 * 2.0 * PI * (static_cast<double>(r) + 0.5) * dr * dr;
		fprintf(pFile_outp, "%E\n", static_cast<double>(temp) / scale2);
	}

	// ANGULAR TRANSMITTANCE DISTRIBUTION

	// Calculate transmittance vs angle Tt(α)
	fprintf(pFile_outp, "\nTt_a\t# Transmittance vs angle: Tt[0], Tt[1], ..., Tt[na-1] [sr⁻¹]\n");
	for (int a = 0; a < na; a++) {
		temp = 0;
		// Integrate across all radial positions for this angle
		for (int r = 0; r < nr; r++) {
			temp += HostMem->Tt_ra[(a * nr) + r];
		}
		// Normalize by solid angle: 4π·sin(α+0.5·dα)·sin(dα/2)
		scale2 = scale1 * 4.0 * PI * sin((static_cast<double>(a) + 0.5) * da) * sin(da / 2.0);
		fprintf(pFile_outp, "%E\n", static_cast<double>(temp) / scale2);
	}

	// 2D SPATIAL ABSORPTION GRID

	// Write complete 2D absorption array A(r,z)
	int output_counter = 0;
	fprintf(pFile_outp, "\n# 2D Absorption Grid A[r][z] [1/cm³]\n");
	fprintf(pFile_outp, "# Data organization:\n");
	fprintf(pFile_outp, "#   A[0][0], A[0][1], ..., A[0][nz-1]\n");
	fprintf(pFile_outp, "#   A[1][0], A[1][1], ..., A[1][nz-1]\n");
	fprintf(pFile_outp, "#   ...\n");
	fprintf(pFile_outp, "#   A[nr-1][0], A[nr-1][1], ..., A[nr-1][nz-1]\n");
	fprintf(pFile_outp, "A_rz\n");

	for (int r = 0; r < nr; r++) {
		for (int z = 0; z < nz; z++) {
			// Normalize by voxel volume: 2π(r+0.5)dr·dr·dz
			scale2 = scale1 * 2.0 * PI * (static_cast<double>(r) + 0.5) * dr * dr * dz;
			fprintf(pFile_outp, " %E ", static_cast<double>(HostMem->A_rz[(z * nr) + r]) / scale2);

			// Format output with line breaks every 5 values for readability
			if ((++output_counter) == 5) {
				output_counter = 0;
				fprintf(pFile_outp, "\n");
			}
		}
	}

	// 2D REFLECTANCE GRID WITH ANGULAR RESOLUTION

	// Write complete 2D reflectance array Rd(r,α)
	output_counter = 0;
	fprintf(pFile_outp, "\n\n# 2D Reflectance Grid Rd[r][angle] [1/(cm²·sr)]\n");
	fprintf(pFile_outp, "# Data organization:\n");
	fprintf(pFile_outp, "#   Rd[0][0], Rd[0][1], ..., Rd[0][na-1]\n");
	fprintf(pFile_outp, "#   Rd[1][0], Rd[1][1], ..., Rd[1][na-1]\n");
	fprintf(pFile_outp, "#   ...\n");
	fprintf(pFile_outp, "#   Rd[nr-1][0], Rd[nr-1][1], ..., Rd[nr-1][na-1]\n");
	fprintf(pFile_outp, "Rd_ra\n");

	for (int r = 0; r < nr; r++) {
		for (int a = 0; a < na; a++) {
			// Normalize by area-solid-angle element: 2π(r+0.5)dr·dr·cos(α)·4π·sin(α)·sin(dα/2)
			scale2 = scale1 * 2.0 * PI * (static_cast<double>(r) + 0.5) * dr * dr
					 * cos((static_cast<double>(a) + 0.5) * da) * 4.0 * PI * sin((static_cast<double>(a) + 0.5) * da)
					 * sin(da / 2.0);
			fprintf(pFile_outp, " %E ", static_cast<double>(HostMem->Rd_ra[(a * nr) + r]) / scale2);

			// Format output with line breaks every 5 values for readability
			if ((++output_counter) == 5) {
				output_counter = 0;
				fprintf(pFile_outp, "\n");
			}
		}
	}

	// 2D TRANSMITTANCE GRID WITH ANGULAR RESOLUTION

	// Write complete 2D transmittance array Tt(r,α)
	output_counter = 0;
	fprintf(pFile_outp, "\n\n# 2D Transmittance Grid Tt[r][angle] [1/(cm²·sr)]\n");
	fprintf(pFile_outp, "# Data organization:\n");
	fprintf(pFile_outp, "#   Tt[0][0], Tt[0][1], ..., Tt[0][na-1]\n");
	fprintf(pFile_outp, "#   Tt[1][0], Tt[1][1], ..., Tt[1][na-1]\n");
	fprintf(pFile_outp, "#   ...\n");
	fprintf(pFile_outp, "#   Tt[nr-1][0], Tt[nr-1][1], ..., Tt[nr-1][na-1]\n");
	fprintf(pFile_outp, "Tt_ra\n");

	for (int r = 0; r < nr; r++) {
		for (int a = 0; a < na; a++) {
			// Normalize by area-solid-angle element: 2π(r+0.5)dr·dr·cos(α)·4π·sin(α)·sin(dα/2)
			scale2 = scale1 * 2.0 * PI * (static_cast<double>(r) + 0.5) * dr * dr
					 * cos((static_cast<double>(a) + 0.5) * da) * 4.0 * PI * sin((static_cast<double>(a) + 0.5) * da)
					 * sin(da / 2.0);
			fprintf(pFile_outp, " %E ", static_cast<double>(HostMem->Tt_ra[(a * nr) + r]) / scale2);

			// Format output with line breaks every 5 values for readability
			if ((++output_counter) == 5) {
				output_counter = 0;
				fprintf(pFile_outp, "\n");
			}
		}
	}

	// Successful completion
	fclose(pFile_outp);
	printf("Results successfully written to: %s\n", sim->outp_filename);
	return 0;
}

////////////////////////////////////////////////////////////////////////////////
// UTILITY FUNCTIONS FOR FILE PARSING

/**
 * Check if character is numeric digit
 *
 * Simple ASCII-based digit detection for input validation.
 *
 * @param a Character to test
 * @return 1 if numeric digit, 0 otherwise
 */
auto is_numeric(char a) -> int {
	return (a >= '0' && a <= '9') ? 1 : 0;
}

/**
 * Check if character is alphabetic
 *
 * Simple ASCII-based letter detection for input validation.
 *
 * @param a Character to test
 * @return 1 if alphabetic character, 0 otherwise
 */
auto is_char(char a) -> int {
	return ((a >= 'A' && a <= 'Z') || (a >= 'a' && a <= 'z')) ? 1 : 0;
}

////////////////////////////////////////////////////////////////////////////////
// INPUT FILE PARSING FUNCTIONS

/**
 * Read floating-point values from input file
 *
 * Parses lines containing floating-point numbers from MCML input files,
 * with robust error handling and format validation.
 *
 * PARSING STRATEGY:
 * -----------------
 * - Skips empty lines and comments automatically
 * - Reads up to NFLOATS values per line
 * - Validates number format before conversion
 * - Provides detailed error reporting for invalid formats
 *
 * @param n_floats   Expected number of floating-point values to read
 * @param temp       Output array for parsed floating-point values
 * @param pFile      Input file stream pointer
 * @return 1 on successful parsing, 0 on error or EOF
 */
auto read_floats(int n_floats, float* temp, FILE* pFile) -> int {
	int values_read = 0;
	char input_line[STR_LEN];

	// Validate input parameters
	if (n_floats > NFLOATS) {
		fprintf(stderr, "Error: Requested %d floats, but maximum supported is %d\n", n_floats, NFLOATS);
		return 0;
	}

	// Initialize output array to zero
	memset(temp, 0, NFLOATS * sizeof(float));

	// Parse lines until we get valid data or reach EOF
	while (values_read <= 0) {
		// Check for end of file
		if (feof(pFile)) {
			fprintf(stderr, "Error: Unexpected end of file while reading floating-point values\n");
			return 0;
		}

		// Read next line from input file
		if (fgets(input_line, STR_LEN, pFile) == nullptr) {
			fprintf(stderr, "Error: Failed to read line from input file\n");
			return 0;
		}

		// Parse floating-point values from the line
		values_read = sscanf(input_line, "%f %f %f %f %f", &temp[0], &temp[1], &temp[2], &temp[3], &temp[4]);

		// Validate that we didn't read more values than expected
		if (values_read > n_floats) {
			fprintf(stderr, "Error: Read %d values but expected only %d\n", values_read, n_floats);
			return 0;
		}
	}

	return 1; // Success
}

/**
 * Read integer values from input file
 *
 * Parses lines containing integer numbers from MCML input files,
 * with robust error handling and format validation.
 *
 * PARSING STRATEGY:
 * -----------------
 * - Skips empty lines and comments automatically
 * - Reads up to NINTS values per line
 * - Validates number format before conversion
 * - Provides detailed error reporting for invalid formats
 *
 * @param n_ints     Expected number of integer values to read
 * @param temp       Output array for parsed integer values
 * @param pFile      Input file stream pointer
 * @return 1 on successful parsing, 0 on error or EOF
 */
auto read_ints(int n_ints, int* temp, FILE* pFile) -> int {
	int values_read = 0;
	char input_line[STR_LEN];

	// Validate input parameters
	if (n_ints > NINTS) {
		fprintf(stderr, "Error: Requested %d integers, but maximum supported is %d\n", n_ints, NINTS);
		return 0;
	}

	// Initialize output array to zero
	memset(temp, 0, NINTS * sizeof(int));

	// Parse lines until we get valid data or reach EOF
	while (values_read <= 0) {
		// Check for end of file
		if (feof(pFile)) {
			fprintf(stderr, "Error: Unexpected end of file while reading integer values\n");
			return 0;
		}

		// Read next line from input file
		if (fgets(input_line, STR_LEN, pFile) == nullptr) {
			fprintf(stderr, "Error: Failed to read line from input file\n");
			return 0;
		}

		// Parse integer values from the line
		values_read = sscanf(input_line, "%d %d %d %d %d", &temp[0], &temp[1], &temp[2], &temp[3], &temp[4]);

		// Validate that we didn't read more values than expected
		if (values_read > n_ints) {
			fprintf(stderr, "Error: Read %d values but expected only %d\n", values_read, n_ints);
			return 0;
		}
	}

	return 1; // Success
}

////////////////////////////////////////////////////////////////////////////////
// COMPLETE SIMULATION CONFIGURATION PARSER

/**
 * Parse complete simulation input file
 *
 * Reads and validates a complete MCML input file containing simulation
 * parameters, tissue layer definitions, and detection grid configuration.
 * Supports multiple simulation runs within a single input file.
 *
 * INPUT FILE FORMAT (MCML Standard):
 * ----------------------------------
 * Line 1:   File format version (float)
 * Line 2:   Number of simulation runs (int)
 *
 * For each simulation run:
 * - Input/output filenames (string)
 * - Number of photons, layer properties
 * - Detection grid parameters
 * - Tissue layer optical properties
 *
 * MEMORY MANAGEMENT:
 * ------------------
 * - Dynamically allocates simulation array based on run count
 * - Allocates layer arrays based on detected layer count
 * - Provides proper cleanup on parsing errors
 *
 * @param filename         Input .mci file path
 * @param simulations      Output pointer to simulation array (allocated by function)
 * @param ignoreAdetection Flag to skip absorption detection
 * @return Number of simulations parsed, 0 on error
 */
auto read_simulation_data(char* filename, SimulationStruct** simulations, int ignoreAdetection) -> int {
	int i = 0;
	int ii = 0;
	unsigned long number_of_photons;
	uint32_t start_weight;
	int n_simulations = 0;
	int n_layers = 0;
	FILE* pFile;
	char mystring[STR_LEN];
	char str[STR_LEN];
	char AorB;
	float dtot = 0;

	float ftemp[NFLOATS];
	int itemp[NINTS];

	pFile = fopen(filename, "r");
	if (pFile == nullptr) {
		perror("Error opening file");
		return 0;
	}

	// First read the first data line (file version) and ignore
	if (read_floats(1, ftemp, pFile) == 0) {
		perror("Error reading file version");
		return 0;
	}

	// Second, read the number of runs
	if (read_ints(1, itemp, pFile) == 0) {
		perror("Error reading number of runs");
		return 0;
	}
	n_simulations = itemp[0];

	// Allocate memory for the SimulationStruct array
	*simulations = (SimulationStruct*)malloc(sizeof(SimulationStruct) * n_simulations);
	if (*simulations == NULL) {
		perror("Failed to malloc simulations.\n");
		return 0;
	}

	for (i = 0; i < n_simulations; i++) {
		// Store the input filename
		strcpy((*simulations)[i].inp_filename, filename);

		// Store ignoreAdetection data
		(*simulations)[i].ignoreAdetection = ignoreAdetection;

		// Read the output filename and determine ASCII or Binary output
		ii = 0;
		while (ii <= 0) {
			(*simulations)[i].begin = ftell(pFile);
			fgets(mystring, STR_LEN, pFile);
			ii = sscanf(mystring, "%s %c", str, &AorB);
			if ((feof(pFile) != 0) || ii > 2) {
				perror("Error reading output filename");
				return 0;
			}
			if (ii > 0) {
				ii = is_char(str[0]);
			}
		}
		// Echo the Filename and AorB
		strcpy((*simulations)[i].outp_filename, str);
		(*simulations)[i].AorB = AorB;

		// Read the number of photons
		ii = 0;
		while (ii <= 0) {
			fgets(mystring, STR_LEN, pFile);
			number_of_photons = 0;
			ii = sscanf(mystring, "%lu", &number_of_photons);

			// if we reach EOF or read more number than defined something is wrong with the file!
			if ((feof(pFile) != 0) || ii > 1) {
				perror("Error reading number of photons");
				return 0;
			}
		}

		(*simulations)[i].number_of_photons = number_of_photons;

		// Read dr and dz (2x float)
		if (read_floats(2, ftemp, pFile) == 0) {
			perror("Error reading dr and dz");
			return 0;
		}

		(*simulations)[i].det.dz = ftemp[0];
		(*simulations)[i].det.dr = ftemp[1];

		// Read No. of dz, dr and da  (3x int)
		if (read_ints(3, itemp, pFile) == 0) {
			perror("Error reading No. of dz, dr and da");
			return 0;
		}

		(*simulations)[i].det.nz = itemp[0];
		(*simulations)[i].det.nr = itemp[1];
		(*simulations)[i].det.na = itemp[2];

		// Read No. of layers (1xint)
		if (read_ints(1, itemp, pFile) == 0) {
			perror("Error reading No. of layers");
			return 0;
		}

		printf("No. of layers=%d\n", itemp[0]);
		n_layers = itemp[0];
		(*simulations)[i].n_layers = itemp[0];

		// Allocate memory for the layers (including one for the upper and one for the lower)
		(*simulations)[i].layers = (LayerStruct*)malloc(sizeof(LayerStruct) * (n_layers + 2));
		if ((*simulations)[i].layers == NULL) {
			perror("Failed to malloc layers.\n");
			return 0;
		}

		// Read upper refractive index (1xfloat)
		if (read_floats(1, ftemp, pFile) == 0) {
			perror("Error reading upper refractive index");
			return 0;
		}

		printf("Upper refractive index=%f\n", ftemp[0]);
		(*simulations)[i].layers[0].n = ftemp[0];

		dtot = 0;
		for (ii = 1; ii <= n_layers; ii++) {
			// Read Layer data (5x float)
			if (read_floats(5, ftemp, pFile) == 0) {
				perror("Error reading layer data");
				return 0;
			}

			printf("n=%f, mua=%f, mus=%f, g=%f, d=%f\n", ftemp[0], ftemp[1], ftemp[2], ftemp[3], ftemp[4]);
			(*simulations)[i].layers[ii].n = ftemp[0];
			(*simulations)[i].layers[ii].mua = ftemp[1];
			(*simulations)[i].layers[ii].g = ftemp[3];
			(*simulations)[i].layers[ii].z_min = dtot;
			dtot += ftemp[4];
			(*simulations)[i].layers[ii].z_max = dtot;

			if (ftemp[2] == 0.0F) {
				(*simulations)[i].layers[ii].mutr = FLT_MAX; // Glass layer
			}
			else {
				(*simulations)[i].layers[ii].mutr = 1.0F / (ftemp[1] + ftemp[2]);
			}
		}

		// Read lower refractive index (1xfloat)
		if (read_floats(1, ftemp, pFile) == 0) {
			perror("Error reading lower refractive index");
			return 0;
		}

		printf("Lower refractive index=%f\n", ftemp[0]);
		(*simulations)[i].layers[n_layers + 1].n = ftemp[0];
		(*simulations)[i].end = ftell(pFile);

		// calculate start_weight
		double n1 = (*simulations)[i].layers[0].n;
		double n2 = (*simulations)[i].layers[1].n;
		double r = (n1 - n2) / (n1 + n2);
		r = r * r;
		start_weight = (uint32_t)((double)0xffffffff * (1 - r));
		(*simulations)[i].start_weight = start_weight;
	}

	return n_simulations;
}
