/*************
 * mcfluor.c - Modern Monte Carlo Fluorescence Simulation
 *
 * A Monte Carlo fluorescence simulation program that:
 *	1. Sets parameters for Monte Carlo runs to simulate fluorescence
 *	2. Calls the mcsub() routine to simulate:
 *		- Excitation into tissue
 *		- Emission due to uniform background fluorophore
 *		- Emission due to off-center fluorophore
 *	3. Saves the results into output files:
 *		- mcOUT101.dat = excitation
 *		- mcOUT102.dat = emission of uniform background fluorophore
 *		- mcOUT103.dat = emission of off-center fluorophore at a +x position
 *		- mcOUT104.dat = emission of off-center fluorophore at a -x position
 *
 * Original: Oct. 29, 2007. Steven L. Jacques
 * Modernized: August 2025 - Updated to C17 standard, refactored into structured
 *             functions, converted to use modern header-only mcsub.h library,
 *             improved memory management and error handling, enhanced comments,
 *             addressed clang-tidy warnings for better code quality
 *************/

#define MCSUB_IMPLEMENTATION
#include "mcsub.h"

#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// Constants
enum {
	NUM_BINS = 101,           /* number of bins, num_radial and num_depth, for z and r */
	TIMING_SAMPLE = 999,      /* number of photons for timing estimation */
	BACKGROUND_PHOTONS = 100, /* photons per background fluorescence point */
	MINUTES_PER_SECOND = 60,
	PROGRESS_INTERVAL = 10,
	FILE_EXCITATION = 101,
	FILE_BACKGROUND = 102,
	FILE_HETERO_POS = 103,
	FILE_HETERO_NEG = 104
};

// Physical constants
static const double ANISOTROPY_FORWARD = 0.90;
static const double ANISOTROPY_ISOTROPIC = 0.0;
static const double REFRACTIVE_INDEX_TISSUE = 1.33;
static const double REFRACTIVE_INDEX_AIR = 1.00;
static const double MILLION_PHOTONS = 1e6;

/**
 * Simulation parameters structure for organized parameter handling
 */
typedef struct {
	// File output
	int file_number;

	// Excitation properties
	double excitation_absorption; // muax - excitation absorption coeff. [cm^-1]
	double excitation_scattering; // musx - excitation scattering coeff. [cm^-1]
	double excitation_anisotropy; // gx - excitation anisotropy [dimensionless]
	double absorption_coeff;      // mua - absorption coefficient [cm^-1]
	double scattering_coeff;      // mus - scattering coefficient [cm^-1]
	double anisotropy_factor;     // g - anisotropy [dimensionless]
	double tissue_refrac_idx;     // n1 - refractive index of medium
	double external_refrac_idx;   // n2 - refractive index outside medium

	// Beam characteristics
	short beam_flag;                     // 0=collimated, 1=focused Gaussian, 2=isotropic pt
	double beam_radius;                  // radius - used if beam_flag = 0 or 1
	double beam_waist;                   // waist - used if beam_flag = 1
	double focus_depth;                  // zfocus - used if beam_flag = 1
	double source_x, source_y, source_z; // xs, ys, zs - used if beam_flag = 2
	int boundary_flag;                   // 0=infinite medium, 1=air/tissue boundary

	// Background fluorescence
	double fluor_absorption;         // muaf - fluorescence absorption coeff. [cm^-1]
	double fluor_scattering;         // musf - fluorescence scattering coeff. [cm^-1]
	double fluor_anisotropy;         // gf - fluorescence anisotropy [dimensionless]
	double extinction_concentration; // eC - ext. coeff. x conc of fluor [cm^-1]
	double energy_yield;             // Y - Energy yield for fluorescence [W/W]

	// Heterogeneity parameters
	double hetero_x, hetero_y, hetero_z;    // xh, yh, zh - heterogeneity position
	double hetero_extinction_concentration; // heC - extra eC of heterogeneity
	double hetero_yield;                    // hY - energy yield of heterogeneity
	double hetero_radius;                   // hrad - radius of spherical heterogeneity

	// Simulation parameters
	double num_runs;       // Number photons launched = num_runs x 1e6
	double radial_spacing; // dr - radial bin size [cm]
	double depth_spacing;  // dz - depth bin size [cm]

	// Control flags
	short enable_printout; // PRINTOUT - Enable/disable progress printing
} SimulationParams;

/**
 * Timing and results structure
 */
typedef struct {
	double start_time, finish_time1, finish_time2, finish_time3;
	double time_per_excitation;  // Time per excitation photon [min]
	double time_per_emission;    // Time per emission photon [min]
	double specular_reflectance; // S - Specular reflectance fraction
	double absorbed_fraction;    // A - Absorbed fraction
	double escaped_fraction;     // E - Escaped fraction
} SimulationResults;

/**
 * Array collection structure for better organization
 */
typedef struct {
	double *excitation_flux;       // Jx - excitation flux array
	double *fluorescence_flux;     // Jf - fluorescence flux array
	double *temp_flux;             // temp1 - temporary flux array
	double **excitation_fluence;   // Fx - excitation fluence matrix
	double **fluorescence_fluence; // Ff - fluorescence fluence matrix
	double **temp_fluence;         // temp2 - temporary fluence matrix
} SimulationArrays;

/**
 * Initialize default simulation parameters
 */
static SimulationParams init_default_params(void) {
	const SimulationParams params = {
        .file_number = 1,

		// Excitation parameters
		.excitation_absorption = 1.0,
		.excitation_scattering = 100.0,
		.excitation_anisotropy = ANISOTROPY_FORWARD,
		.absorption_coeff = 1.0,
		.scattering_coeff = 100.0,
		.anisotropy_factor = ANISOTROPY_FORWARD,
		.tissue_refrac_idx = REFRACTIVE_INDEX_TISSUE,
		.external_refrac_idx = REFRACTIVE_INDEX_AIR,

		// Beam characteristics
		.beam_flag = 0,
		.beam_radius = 0.0,
		.beam_waist = 0.0,
		.focus_depth = 0.0,
		.source_x = 0.0,
		.source_y = 0.0,
		.source_z = 0.0,
		.boundary_flag = 1,

		// Background fluorescence
		.fluor_absorption = 5.0,
		.fluor_scattering = 50.0,
		.fluor_anisotropy = ANISOTROPY_ISOTROPIC,
		.extinction_concentration = 1.0,
		.energy_yield = 1.0,

		// Heterogeneity
		.hetero_x = 0.2,
		.hetero_y = 0.0,
		.hetero_z = 0.3,
		.hetero_extinction_concentration = 0.1,
		.hetero_yield = 1.0,
		.hetero_radius = 0.01,

		// Other parameters
		.num_runs = 0.1,
		.radial_spacing = 0.0100,
		.depth_spacing = 0.0100,
		.enable_printout = 1
    };
	return params;
}

/**
 * Print simulation parameters summary
 */
static void print_parameters(const SimulationParams *params) {
	printf("----- USER CHOICES -----\n");
	printf("EXCITATION\n");
	printf("muax = %.3f\n", params->excitation_absorption);
	printf("musx = %.3f\n", params->excitation_scattering);
	printf("gx = %.3f\n", params->excitation_anisotropy);
	printf("n1 = %.3f\n", params->tissue_refrac_idx);
	printf("n2 = %.3f\n", params->external_refrac_idx);
	printf("beam_flag = %d\n", params->beam_flag);
	printf("radius = %.4f\n", params->beam_radius);
	printf("waist = %.4f\n", params->beam_waist);
	printf("zfocus = %.4f\n", params->focus_depth);
	printf("xs = %.4f\n", params->source_x);
	printf("ys = %.4f\n", params->source_y);
	printf("zs = %.4f\n", params->source_z);
	printf("BACKGROUND FLUORESCENCE\n");
	printf("muaf = %.3f\n", params->fluor_absorption);
	printf("musf = %.3f\n", params->fluor_scattering);
	printf("gf = %.3f\n", params->fluor_anisotropy);
	printf("eC = %.3f\n", params->extinction_concentration);
	printf("Y = %.3f\n", params->energy_yield);
	printf("FLUORESCENT HETEROGENEITY\n");
	printf("xh = %.4f\n", params->hetero_x);
	printf("yh = %.4f\n", params->hetero_y);
	printf("zh = %.4f\n", params->hetero_z);
	printf("heC = %.4f\n", params->hetero_extinction_concentration);
	printf("hY = %.4f\n", params->hetero_yield);
	printf("hrad = %.4f\n", params->hetero_radius);
	printf("OTHER\n");
	printf("Nruns = %.1f @1e6 photons/run\n", params->num_runs);
	printf("dr = %.4f\n", params->radial_spacing);
	printf("dz = %.4f\n", params->depth_spacing);
	printf("---------------\n\n");
}

/**
 * Allocate simulation arrays with error checking
 */
static bool allocate_arrays(SimulationArrays *arrays) {
	arrays->excitation_flux = mcsub_alloc_vector(1, NUM_BINS);
	arrays->fluorescence_flux = mcsub_alloc_vector(1, NUM_BINS);
	arrays->temp_flux = mcsub_alloc_vector(1, NUM_BINS);
	arrays->excitation_fluence = mcsub_alloc_matrix(1, NUM_BINS, 1, NUM_BINS);
	arrays->fluorescence_fluence = mcsub_alloc_matrix(1, NUM_BINS, 1, NUM_BINS);
	arrays->temp_fluence = mcsub_alloc_matrix(1, NUM_BINS, 1, NUM_BINS);

	return (arrays->excitation_flux != NULL && arrays->fluorescence_flux != NULL && arrays->temp_flux != NULL
			&& arrays->excitation_fluence != NULL && arrays->fluorescence_fluence != NULL
			&& arrays->temp_fluence != NULL);
}

/**
 * Free simulation arrays
 */
static void free_arrays(SimulationArrays *arrays) {
	if (arrays->excitation_flux != NULL) {
		mcsub_free_vector(arrays->excitation_flux, 1, NUM_BINS);
	}
	if (arrays->fluorescence_flux != NULL) {
		mcsub_free_vector(arrays->fluorescence_flux, 1, NUM_BINS);
	}
	if (arrays->temp_flux != NULL) {
		mcsub_free_vector(arrays->temp_flux, 1, NUM_BINS);
	}
	if (arrays->excitation_fluence != NULL) {
		mcsub_free_matrix(arrays->excitation_fluence, 1, NUM_BINS, 1, NUM_BINS);
	}
	if (arrays->fluorescence_fluence != NULL) {
		mcsub_free_matrix(arrays->fluorescence_fluence, 1, NUM_BINS, 1, NUM_BINS);
	}
	if (arrays->temp_fluence != NULL) {
		mcsub_free_matrix(arrays->temp_fluence, 1, NUM_BINS, 1, NUM_BINS);
	}
}

/**
 * Initialize arrays to zero
 */
static void initialize_arrays(const SimulationArrays *arrays) {
	const long num_radial = NUM_BINS;
	const long num_depth = NUM_BINS;

	for (long ir = 1; ir <= num_radial; ir++) {
		arrays->excitation_flux[ir] = 0.0;
		arrays->fluorescence_flux[ir] = 0.0;
		arrays->temp_flux[ir] = 0.0;

		for (long iz = 1; iz <= num_depth; iz++) {
			arrays->excitation_fluence[iz][ir] = 0.0;
			arrays->fluorescence_fluence[iz][ir] = 0.0;
			arrays->temp_fluence[iz][ir] = 0.0;
		}
	}
}

/**
 * Estimate execution time per photon for both excitation and emission
 */
static void estimate_timing(const SimulationParams *params, SimulationResults *results) {
	const long num_radial = NUM_BINS;
	const long num_depth = NUM_BINS;

	// Allocate temporary arrays for timing estimation
	SimulationArrays timing_arrays;
	if (!allocate_arrays(&timing_arrays)) {
		mcsub_error("Failed to allocate timing arrays");
		return;
	}

	printf("Estimating execution times...\n");

	// Time excitation with small sample
	const clock_t time_start = clock();
	mcsub(params->excitation_absorption, params->excitation_scattering, params->excitation_anisotropy,
		  params->tissue_refrac_idx, params->external_refrac_idx, num_radial, num_depth, params->radial_spacing,
		  params->depth_spacing, TIMING_SAMPLE, params->beam_flag, params->source_x, params->source_y, params->source_z,
		  params->boundary_flag, params->beam_radius, params->beam_waist, params->focus_depth,
		  timing_arrays.excitation_flux, timing_arrays.excitation_fluence, &results->specular_reflectance,
		  &results->absorbed_fraction, &results->escaped_fraction, 0);
	const clock_t time_end = clock();
	results->time_per_excitation =
		(double)(time_end - time_start) / CLOCKS_PER_SEC / MINUTES_PER_SECOND / TIMING_SAMPLE;
	printf("%.3e min/EX photon \n", results->time_per_excitation);

	// Time emission with small sample
	const double midpoint_depth = (num_depth / 2.0) * params->depth_spacing; // zs is midway
	const clock_t time_start2 = clock();
	mcsub(params->fluor_absorption, params->fluor_scattering, params->fluor_anisotropy, params->tissue_refrac_idx,
		  params->external_refrac_idx, num_radial, num_depth, params->radial_spacing, params->depth_spacing,
		  TIMING_SAMPLE, 2, 0, 0, midpoint_depth, params->boundary_flag, params->beam_radius, params->beam_waist,
		  params->focus_depth, timing_arrays.excitation_flux, timing_arrays.excitation_fluence,
		  &results->specular_reflectance, &results->absorbed_fraction, &results->escaped_fraction, 0);
	const clock_t time_end2 = clock();
	results->time_per_emission =
		(double)(time_end2 - time_start2) / CLOCKS_PER_SEC / MINUTES_PER_SECOND / TIMING_SAMPLE;
	printf("%.3e min/EM photon\n", results->time_per_emission);

	const double estimated_time = (results->time_per_excitation * MILLION_PHOTONS * params->num_runs)
								  + (results->time_per_emission * num_depth * num_radial * BACKGROUND_PHOTONS)
								  + (results->time_per_emission * MILLION_PHOTONS * params->num_runs);
	printf("\nTotal estimated completion time = %.2f min\n\n", estimated_time);

	// Clean up temporary arrays
	free_arrays(&timing_arrays);
}

/**
 * Run excitation simulation
 */
static void run_excitation(const SimulationParams *params, SimulationResults *results, const SimulationArrays *arrays) {
	const long num_radial = NUM_BINS;
	const long num_depth = NUM_BINS;
	const double num_photons = MILLION_PHOTONS * params->num_runs;

	printf("EXCITATION\n");
	printf("est. completion time = %.2f min\n\n", results->time_per_excitation * num_photons);

	mcsub(params->excitation_absorption, params->excitation_scattering, params->excitation_anisotropy,
		  params->tissue_refrac_idx, params->external_refrac_idx, num_radial, num_depth, params->radial_spacing,
		  params->depth_spacing, num_photons, params->beam_flag, params->source_x, params->source_y, params->source_z,
		  params->boundary_flag, params->beam_radius, params->beam_waist, params->focus_depth, arrays->excitation_flux,
		  arrays->excitation_fluence, &results->specular_reflectance, &results->absorbed_fraction,
		  &results->escaped_fraction, params->enable_printout);

	// Save excitation results
	mcsub_save_file(FILE_EXCITATION, arrays->excitation_flux, arrays->excitation_fluence, results->specular_reflectance,
					results->absorbed_fraction, results->escaped_fraction, params->absorption_coeff,
					params->scattering_coeff, params->anisotropy_factor, params->tissue_refrac_idx,
					params->external_refrac_idx, params->beam_flag, params->beam_radius, params->beam_waist,
					params->source_x, params->source_y, params->source_z, (short)num_radial, (short)num_depth,
					params->radial_spacing, params->depth_spacing, num_photons);

	results->finish_time1 = clock();
	printf("------------------------------------------------------\n");
	printf("Elapsed Time for excitation = %.3f min\n",
		   (results->finish_time1 - results->start_time) / CLOCKS_PER_SEC / MINUTES_PER_SECOND);
}

/**
 * Run background fluorescence simulation
 */
static void run_background_fluorescence(const SimulationParams *params,
										SimulationResults *results,
										const SimulationArrays *arrays) {
	const long num_radial = NUM_BINS;
	const long num_depth = NUM_BINS;

	printf("BACKGROUND FLUORESCENCE\n");
	printf("est. completion time = %.2f min\n\n",
		   results->time_per_emission * num_depth * num_radial * BACKGROUND_PHOTONS);

	/* Accumulate Monte Carlo fluorescence due to each bin source
	 * weighted by strength of absorbed excitation in each bin
	 * source, Fx[iz][ir]. Do not include the last bins [num_depth][num_radial]
	 * which are for overflow. */
	double total_photons = 0.0;

	for (long ir = 1; ir < num_radial; ir++) {
		double photons_current_row = 0.0;
		const double fluor_conversion = params->extinction_concentration * params->energy_yield * 2 * M_PI * (ir - 0.5)
										* params->radial_spacing * params->radial_spacing * params->depth_spacing;

		for (long iz = 1; iz < num_depth; iz++) {
			total_photons += BACKGROUND_PHOTONS;
			photons_current_row += BACKGROUND_PHOTONS;

			// Set to launch as isotropic point source at bin [iz][ir]
			const double outer_radius = ir * params->radial_spacing;
			const double inner_radius = (ir - 1) * params->radial_spacing;
			const double radius =
				(2.0 / 3.0) * (outer_radius * outer_radius + outer_radius * inner_radius + inner_radius * inner_radius)
				/ (outer_radius + inner_radius);
			const double source_x = radius;
			const double source_y = 0;
			const double source_z = (iz - 0.5) * params->depth_spacing;

			// Call Monte Carlo subroutine for fluorescence emission
			mcsub(params->fluor_absorption, params->fluor_scattering, params->fluor_anisotropy,
				  params->tissue_refrac_idx, params->external_refrac_idx, num_radial, num_depth, params->radial_spacing,
				  params->depth_spacing, BACKGROUND_PHOTONS, 2, source_x, source_y, source_z,
				  params->boundary_flag, // beam_flag=2 for isotropic point
				  params->beam_radius, params->beam_waist, params->focus_depth, arrays->temp_flux, arrays->temp_fluence,
				  &results->specular_reflectance, &results->absorbed_fraction, &results->escaped_fraction, 0);

			// Accumulate Monte Carlo results
			for (long iir = 1; iir <= num_radial; iir++) {
				arrays->fluorescence_flux[iir] +=
					arrays->temp_flux[iir] * arrays->excitation_fluence[iz][ir] * fluor_conversion;
				for (long iiz = 1; iiz <= num_depth; iiz++) {
					arrays->fluorescence_fluence[iiz][iir] +=
						arrays->temp_fluence[iiz][iir] * arrays->excitation_fluence[iz][ir] * fluor_conversion;
				}
			}
		}

		// Print progress for user
		if (ir < PROGRESS_INTERVAL || (ir % PROGRESS_INTERVAL) == 0) {
			printf("%.0f fluor photons \t@ ir = %ld \t total = %.0f\n", photons_current_row, ir, total_photons);
		}
	}

	printf("%.0f fluorescent photons total\n", total_photons);

	results->finish_time2 = clock();
	printf("Elapsed Time for background fluorescence = %.3f min\n",
		   (results->finish_time2 - results->finish_time1) / CLOCKS_PER_SEC / MINUTES_PER_SECOND);

	// Save background fluorescence results
	mcsub_save_file(FILE_BACKGROUND, arrays->fluorescence_flux, arrays->fluorescence_fluence,
					results->specular_reflectance, results->absorbed_fraction, results->escaped_fraction,
					params->absorption_coeff, params->scattering_coeff, params->anisotropy_factor,
					params->tissue_refrac_idx, params->external_refrac_idx, 2, params->beam_radius, params->beam_waist,
					0, 0, params->hetero_z, (short)num_radial, (short)num_depth, params->radial_spacing,
					params->depth_spacing, total_photons);
}

/**
 * Run heterogeneous fluorescence simulation for one side
 */
static void run_heterogeneous_fluorescence_side(const SimulationParams *params,
												const SimulationResults *results,
												const SimulationArrays *arrays,
												int file_num,
												bool positive_side) {
	const long num_radial = NUM_BINS;
	const long num_depth = NUM_BINS;

	// Initialize fluorescence arrays
	for (long ir = 1; ir <= num_radial; ir++) {
		arrays->fluorescence_flux[ir] = 0.0;
		for (long iz = 1; iz <= num_depth; iz++) {
			arrays->fluorescence_fluence[iz][ir] = 0.0;
		}
	}

	/* Convolve impulse response against fluorescent source at (hetero_x,hetero_y,hetero_z).
	 * Weight by Fx[iz][ir]*4/3*PI*hetero_radius^3*hetero_extinction_concentration*hetero_yield
	 * which is fluorescent power of heterogeneity [W/W]. */
	const double fluor_power = (4.0 / 3.0) * M_PI * params->hetero_radius * params->hetero_radius
							   * params->hetero_radius * params->hetero_extinction_concentration * params->hetero_yield;
	const long ir_hetero = (long)(sqrt((params->hetero_x * params->hetero_x) + (params->hetero_y * params->hetero_y))
								  / params->radial_spacing)
						   + 1;
	const long iz_hetero = (long)(params->hetero_z / params->depth_spacing) + 1;

	for (long iir = 1; iir <= num_radial; iir++) {
		const double observer_radius = iir * params->radial_spacing; // radial position of observer
		double hetero_observer_distance;

		if (positive_side) {
			hetero_observer_distance =
				sqrt(((observer_radius - params->hetero_x) * (observer_radius - params->hetero_x))
					 + (params->hetero_y * params->hetero_y)); // hetero-observer distance
		}
		else {
			hetero_observer_distance =
				sqrt(((observer_radius + params->hetero_x) * (observer_radius + params->hetero_x))
					 + (params->hetero_y * params->hetero_y)); // hetero-observer distance
		}

		long temp_index = (long)(hetero_observer_distance / params->radial_spacing) + 1;
		if (temp_index > num_radial) {
			temp_index = num_radial;
		}

		arrays->fluorescence_flux[iir] +=
			arrays->temp_flux[temp_index] * arrays->excitation_fluence[iz_hetero][ir_hetero] * fluor_power;

		for (long iiz = 1; iiz <= num_depth; iiz++) {
			arrays->fluorescence_fluence[iiz][iir] +=
				arrays->temp_fluence[iiz][temp_index] * arrays->excitation_fluence[iz_hetero][ir_hetero] * fluor_power;
		}
	}

	// Save heterogeneity fluorescence results
	const double num_photons = params->num_runs * MILLION_PHOTONS;
	mcsub_save_file(file_num, arrays->fluorescence_flux, arrays->fluorescence_fluence, results->specular_reflectance,
					results->absorbed_fraction, results->escaped_fraction, params->absorption_coeff,
					params->scattering_coeff, params->anisotropy_factor, params->tissue_refrac_idx,
					params->external_refrac_idx, 2, params->beam_radius, params->beam_waist, 0, 0, params->hetero_z,
					(short)num_radial, (short)num_depth, params->radial_spacing, params->depth_spacing, num_photons);
}

/**
 * Run heterogeneous fluorescence simulation
 */
static void run_heterogeneous_fluorescence(const SimulationParams *params,
										   SimulationResults *results,
										   const SimulationArrays *arrays) {
	const long num_radial = NUM_BINS;
	const long num_depth = NUM_BINS;
	const double num_photons = params->num_runs * MILLION_PHOTONS;

	printf("FLUORESCENT HETEROGENEITY\n");
	printf("est. completion time = %.3f min\n\n", results->time_per_emission * num_photons);

	/* Fluorescent heterogeneity at (hetero_x,hetero_y,hetero_z) as a small sphere with specified radius.
	 * Note that results are fluorescence_flux(x), F(z,x) in y=0 plane.
	 * Usually will let hetero_y = 0.
	 * Launch at source_x = 0, source_y = 0, source_z = hetero_z */

	// Generate fluorescent impulse response
	mcsub(params->fluor_absorption, params->fluor_scattering, params->fluor_anisotropy, params->tissue_refrac_idx,
		  params->external_refrac_idx, num_radial, num_depth, params->radial_spacing, params->depth_spacing,
		  num_photons, 2, 0, 0, params->hetero_z, params->boundary_flag, // beam_flag=2 for isotropic point
		  params->beam_radius, params->beam_waist, params->focus_depth, arrays->temp_flux, arrays->temp_fluence,
		  &results->specular_reflectance, &results->absorbed_fraction, &results->escaped_fraction,
		  params->enable_printout);

	// Process positive z side of response
	run_heterogeneous_fluorescence_side(params, results, arrays, FILE_HETERO_POS, true);

	// Process negative z side of response
	run_heterogeneous_fluorescence_side(params, results, arrays, FILE_HETERO_NEG, false);

	results->finish_time3 = clock();
	printf("Elapsed Time for fluorescent heterogeneity = %.3f min\n",
		   (results->finish_time3 - results->finish_time2) / CLOCKS_PER_SEC / MINUTES_PER_SECOND);
}

/**
 * Print final timing summary
 */
static void print_timing_summary(const SimulationParams *params, const SimulationResults *results) {
	printf("------------------------------------------------------\n");

	const time_t now = time(NULL);
	printf("%s\n", ctime(&now));

	const double estimated_time = (results->time_per_excitation * MILLION_PHOTONS * params->num_runs)
								  + (results->time_per_emission * NUM_BINS * NUM_BINS * BACKGROUND_PHOTONS)
								  + (results->time_per_emission * MILLION_PHOTONS * params->num_runs);
	const double actual_time = (results->finish_time3 - results->start_time) / CLOCKS_PER_SEC / MINUTES_PER_SECOND;

	printf("\nEstimated total completion time = %.2f min\n", estimated_time);
	printf("Actual total elapsed time = %.2f min\n", actual_time);
}

/* 
 * Main function to run the Monte Carlo fluorescence simulation
 */
int main(void) {
	// Initialize parameters and results structures
	const SimulationParams params = init_default_params();
	SimulationResults results = {0};
	SimulationArrays arrays = {0};

	// Set up timing
	results.start_time = clock();
	const time_t now = time(NULL);
	printf("%s\n", ctime(&now));

	// Print simulation parameters
	print_parameters(&params);

	// Allocate arrays using modern mcsub memory management
	if (!allocate_arrays(&arrays)) {
		mcsub_error("Failed to allocate memory for simulation arrays");
		return EXIT_FAILURE;
	}

	// Initialize arrays
	initialize_arrays(&arrays);

	// Estimate timing for completion
	estimate_timing(&params, &results);

	// Run excitation simulation
	run_excitation(&params, &results, &arrays);
	printf("------------------------------------------------------\n");

	// Run background fluorescence simulation
	run_background_fluorescence(&params, &results, &arrays);
	printf("------------------------------------------------------\n");

	// Run heterogeneous fluorescence simulation
	run_heterogeneous_fluorescence(&params, &results, &arrays);

	// Print timing summary
	print_timing_summary(&params, &results);

	// Clean up memory
	free_arrays(&arrays);

	return EXIT_SUCCESS;
}
