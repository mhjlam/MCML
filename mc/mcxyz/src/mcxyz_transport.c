/*
 * MCXYZ - Monte Carlo Transport Module (C17 with All Optimizations)
 *
 * Single unified transport implementation with:
 * - Runtime optimization level selection (single/multi-threaded)
 * - Proper progress reporting for all modes
 * - All performance optimizations integrated
 * - Correct performance metrics calculation
 */

#include "mcxyz.h"

#include <math.h>

#ifdef _OPENMP
#include <omp.h>
#endif

// Math constants
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Transport configuration
#define WEIGHT_THRESHOLD     0.0001f
#define ROULETTE_CHANCE      0.1f
#define MAX_SIMULATION_STEPS 10000

// Thread-local RNG state
_Thread_local uint64_t thread_rng_state = 0;

////////////////////////////////////////////////////////////////////////////////
// MEMORY ALLOCATION FUNCTIONS

/**
 * Allocate memory for simulation structures
 */
McxyzErrorCode allocate_simulation_memory(SimulationConfig* config) {
	if (!config) {
		return MCXYZ_ERROR_INVALID_PARAMETER;
	}

	size_t total_voxels = config->grid.total_voxels;

	// Allocate tissue volume array
	config->tissue_volume = calloc(total_voxels, sizeof(uint8_t));
	if (!config->tissue_volume) {
		return MCXYZ_ERROR_MEMORY_ALLOCATION;
	}

	// Allocate fluence volume array
	config->fluence_volume = calloc(total_voxels, sizeof(float));
	if (!config->fluence_volume) {
		free(config->tissue_volume);
		config->tissue_volume = NULL;
		return MCXYZ_ERROR_MEMORY_ALLOCATION;
	}

	printf("Allocated memory for %zu voxels (%.2f MB total)\n", total_voxels,
		   (total_voxels * (sizeof(uint8_t) + sizeof(float))) / (1024.0 * 1024.0));

	return MCXYZ_SUCCESS;
}

/**
 * Free allocated simulation memory
 */
void free_simulation_memory(SimulationConfig* config) {
	if (!config) {
		return;
	}

	if (config->tissue_volume) {
		free(config->tissue_volume);
		config->tissue_volume = NULL;
	}

	if (config->fluence_volume) {
		free(config->fluence_volume);
		config->fluence_volume = NULL;
	}
}

////////////////////////////////////////////////////////////////////////////////
// OPTIMIZED INLINE FUNCTIONS

/**
 * Ultra-fast thread-safe random number generator
 */
static inline float fast_random(void) {
	thread_rng_state = thread_rng_state * 6364136223846793005ULL + 1442695040888963407ULL;
	uint32_t xorshifted = ((thread_rng_state >> 18U) ^ thread_rng_state) >> 27U;
	uint32_t rot = thread_rng_state >> 59U;
	uint32_t result = (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
	return result * 2.3283064365386963e-10F;
}

/**
 * Fast logarithm approximation
 */
static inline float fast_log(float x) {
	union {
		float f;
		uint32_t i;
	} vx = {x};
	union {
		uint32_t i;
		float f;
	} mx = {(vx.i & 0x007FFFFF) | 0x3f000000};
	float y = (float)vx.i;
	y *= 1.1920928955078125e-7F; // 1/2^23
	return y - 124.22551499F - (1.498030302F * mx.f) - (1.72587999F / (0.3520887068F + mx.f));
}

/**
 * Fast isotropic direction generation
 */
static inline void generate_isotropic_direction(float* ux, float* uy, float* uz) {
	float costheta = (2.0F * fast_random()) - 1.0F;
	float sintheta = sqrtf(1.0F - (costheta * costheta));
	float phi = 2.0F * M_PI * fast_random();
	*ux = sintheta * cosf(phi);
	*uy = sintheta * sinf(phi);
	*uz = costheta;
}

/**
 * Optimized Henyey-Greenberg scattering
 */
static inline void scatter_photon(float* ux, float* uy, float* uz, float g) {
	float costheta;

	if (fabsf(g) < 1e-6F) {
		// Isotropic scattering
		costheta = 2.0F * fast_random() - 1.0F;
	}
	else {
		// Anisotropic scattering
		float rnd = fast_random();
		float temp = (1.0F - g * g) / (1.0F - g + 2.0F * g * rnd);
		costheta = (1.0F + g * g - temp * temp) / (2.0F * g);
	}

	float sintheta = sqrtf(1.0F - (costheta * costheta));
	float phi = 2.0F * M_PI * fast_random();
	float sinphi = sinf(phi);
	float cosphi = cosf(phi);

	// Update direction using rotation
	if (fabsf(*uz) > 0.99999F) {
		// Near vertical - use simple rotation
		*ux = sintheta * cosphi;
		*uy = sintheta * sinphi;
		*uz = costheta * ((*uz > 0) ? 1.0F : -1.0F);
	}
	else {
		// General case
		float temp = sqrtf(1.0F - ((*uz) * (*uz)));
		float uxx = (sintheta * ((*ux) * (*uz) * cosphi - (*uy) * sinphi) / temp) + ((*ux) * costheta);
		float uyy = (sintheta * ((*uy) * (*uz) * cosphi + (*ux) * sinphi) / temp) + ((*uy) * costheta);
		float uzz = (-sintheta * cosphi * temp) + ((*uz) * costheta);
		*ux = uxx;
		*uy = uyy;
		*uz = uzz;
	}
}

/**
 * Core photon transport function - used by both single and multi-threaded versions
 */
static uint32_t transport_photon(const SimulationConfig* config) {
	// Initialize photon
	float x = config->source.position_x;
	float y = config->source.position_y;
	float z = config->source.position_z;
	float weight = 1.0F;
	float ux;
	float uy;
	float uz;

	// Initial direction
	generate_isotropic_direction(&ux, &uy, &uz);

	// Apply source radius if specified
	if (config->source.radius > 0.0F) {
		float r = config->source.radius * sqrtf(fast_random());
		float theta = 2.0F * M_PI * fast_random();
		x += r * cosf(theta);
		y += r * sinf(theta);
	}

	uint32_t steps = 0;
	bool photon_alive = true;

	// Main transport loop
	while (photon_alive && steps < MAX_SIMULATION_STEPS) {
		// Calculate voxel indices
		int ix = (int)(x / config->grid.spacing_x);
		int iy = (int)(y / config->grid.spacing_y);
		int iz = (int)(z / config->grid.spacing_z);

		// Boundary checking
		if (ix < 0 || ix >= config->grid.size_x || iy < 0 || iy >= config->grid.size_y || iz < 0
			|| iz >= config->grid.size_z) {
			// Handle boundary conditions
			if (config->source.boundary == BOUNDARY_ESCAPE_ALL
				|| (config->source.boundary == BOUNDARY_ESCAPE_SURFACE && iz < 0)) {
				break; // Photon escapes
			}

			// Clamp to boundaries
			ix = (ix < 0) ? 0 : ((ix >= config->grid.size_x) ? config->grid.size_x - 1 : ix);
			iy = (iy < 0) ? 0 : ((iy >= config->grid.size_y) ? config->grid.size_y - 1 : iy);
			iz = (iz < 0) ? 0 : ((iz >= config->grid.size_z) ? config->grid.size_z - 1 : iz);
		}

		// Get tissue properties
		size_t voxel_index = (iz * config->grid.size_y * config->grid.size_x) + (ix * config->grid.size_y) + iy;
		uint8_t tissue_type = config->tissue_volume[voxel_index];

		// Validate tissue type
		if (tissue_type == 0 || tissue_type > config->tissue_count) {
			tissue_type = 1; // Default to first tissue
		}

		const TissueProperties* tissue = &config->tissues[tissue_type];
		float mua = tissue->absorption;
		float mus = tissue->scattering;
		float g = tissue->anisotropy;
		float mut = mua + mus;

		if (mut <= 0.0F) {
			photon_alive = false;
			break;
		}

		// Calculate step size
		float step = -fast_log(fast_random()) / mut;

		// Move photon
		x += step * ux;
		y += step * uy;
		z += step * uz;

		// Absorption
		float absorbed_weight = weight * (mua / mut);
		weight -= absorbed_weight;

// Add to fluence (thread-safe for multi-threaded version)
#ifdef _OPENMP
#pragma omp atomic
#endif
		config->fluence_volume[voxel_index] += absorbed_weight;

		// Scattering
		if (mus > 0.0F) {
			scatter_photon(&ux, &uy, &uz, g);
		}

		// Russian roulette for low-weight photons
		if (weight < WEIGHT_THRESHOLD) {
			if (fast_random() <= ROULETTE_CHANCE) {
				weight /= ROULETTE_CHANCE;
			}
			else {
				photon_alive = false;
			}
		}

		steps++;
	}

	return steps;
}

////////////////////////////////////////////////////////////////////////////////
// PROGRESS REPORTING

/**
 * Display a visual progress bar with percentage and ETA
 */
static void display_progress_bar(float percent, uint64_t completed, uint64_t target, clock_t start_time,
								 bool is_multithreaded) {
	if (is_multithreaded) {
#pragma omp critical(progress_display)
		{
			// Calculate ETA
			clock_t current_time = clock();
			double elapsed = (double)(current_time - start_time) / CLOCKS_PER_SEC;
			double eta = (elapsed / (percent / 100.0F)) - elapsed;

			// Create progress bar (50 characters wide)
			const int bar_width = 50;
			int filled = (int)(percent * bar_width / 100.0F);

			printf("\r[");
			for (int i = 0; i < bar_width; i++) {
				if (i < filled) {
					printf("=");
				}
				else if (i == filled && percent > (float)filled * 100.0F / bar_width) {
					printf(">");
				}
				else {
					printf(" ");
				}
			}

			printf("] %5.1f%% (%llu/%llu photons) ETA: %02.0f:%02.0f", percent, (unsigned long long)completed,
				   (unsigned long long)target, floor(eta / 60.0), fmod(eta, 60.0));
			fflush(stdout);
		}
	}
	else {
		// Single-threaded version
		clock_t current_time = clock();
		double elapsed = (double)(current_time - start_time) / CLOCKS_PER_SEC;
		double eta = (elapsed / (percent / 100.0F)) - elapsed;

		const int bar_width = 50;
		int filled = (int)(percent * bar_width / 100.0F);

		printf("\r[");
		for (int i = 0; i < bar_width; i++) {
			if (i < filled) {
				printf("=");
			}
			else if (i == filled && percent > (float)filled * 100.0F / bar_width) {
				printf(">");
			}
			else {
				printf(" ");
			}
		}

		printf("] %5.1f%% (%llu/%llu photons) ETA: %02.0f:%02.0f", percent, (unsigned long long)completed,
			   (unsigned long long)target, floor(eta / 60.0), fmod(eta, 60.0));
		fflush(stdout);
	}
}

/**
 * Thread-safe progress reporting with visual progress bar
 */
static void report_progress(uint64_t completed, uint64_t target, bool is_multithreaded, uint64_t* last_reported,
							clock_t start_time) {
	if (target == 0) {
		return;
	}

	// Calculate progress percentage
	float percent = (float)(completed * 100) / target;
	uint64_t percent_int = (uint64_t)(percent * 10); // 0.1% precision

	// Update every 0.5% for smoother progress bar
	uint64_t report_interval = 5; // 0.5% intervals

	if (percent_int >= *last_reported + report_interval || completed == target) {
		*last_reported = percent_int;
		display_progress_bar(percent, completed, target, start_time, is_multithreaded);
	}
}

////////////////////////////////////////////////////////////////////////////////
// SIMULATION FUNCTIONS

/**
 * Single-threaded Monte Carlo simulation
 */
McxyzErrorCode run_monte_carlo_simulation(SimulationConfig* config, PerformanceMetrics* metrics) {
	if (!config || !metrics) {
		return MCXYZ_ERROR_INVALID_PARAMETER;
	}

	metrics->start_time = clock();

	printf("Running Monte Carlo simulation...\n");
	printf("Grid: %dx%dx%d voxels, Requesting %.1f min\n", config->grid.size_x, config->grid.size_y,
		   config->grid.size_z, config->simulation_time_minutes);

	// Phase 1: Calibration
	clock_t temp_time = clock();
	uint64_t calibration_photons = 0;
	uint64_t calibration_steps = 0;

	thread_rng_state = (uint64_t)time(NULL) + 12345;

	for (uint64_t i = 0; i < 1000; i++) {
		uint32_t steps = transport_photon(config);
		calibration_steps += steps;
		calibration_photons++;
	}

	// Calculate target photons
	clock_t finish_time = clock();
	uint64_t target_photons =
		(uint64_t)(config->simulation_time_minutes * 60 * 999 * CLOCKS_PER_SEC / (finish_time - temp_time));
	printf("Nphotons = %llu for simulation time = %.2f min\n", (unsigned long long)target_photons,
		   config->simulation_time_minutes);

	// Phase 2: Main simulation with progress reporting
	uint64_t remaining_photons = (target_photons > 1000) ? target_photons - 1000 : 0;
	uint64_t main_photons = 0;
	uint64_t main_steps = 0;
	uint64_t last_reported_progress = 0;

	for (uint64_t i = 0; i < remaining_photons; i++) {
		uint32_t steps = transport_photon(config);
		main_steps += steps;
		main_photons++;

		// Progress reporting
		report_progress(main_photons, remaining_photons, false, &last_reported_progress, metrics->start_time);
	}

	uint64_t total_photons = calibration_photons + main_photons;
	uint64_t total_steps = calibration_steps + main_steps;

	printf("------------------------------------------------------\n");

	metrics->end_time = clock();
	metrics->elapsed_seconds = (double)(metrics->end_time - metrics->start_time) / CLOCKS_PER_SEC;
	metrics->photons_completed = total_photons;
	metrics->photons_per_second = (uint64_t)(total_photons / metrics->elapsed_seconds);
	metrics->steps_per_photon_avg = (int)(total_steps / total_photons);

	float time_min = (float)metrics->elapsed_seconds / 60.0F;
	printf("Elapsed Time for %llu photons = %.3f min\n", (unsigned long long)total_photons, time_min);
	printf("%llu photons per minute\n", (unsigned long long)(total_photons / time_min));
	printf("Average steps per photon: %d\n", metrics->steps_per_photon_avg);

	return MCXYZ_SUCCESS;
}

/**
 * Multi-threaded Monte Carlo simulation
 */
McxyzErrorCode run_monte_carlo_simulation_mt(SimulationConfig* config, PerformanceMetrics* metrics) {
	if (!config || !metrics) {
		return MCXYZ_ERROR_INVALID_PARAMETER;
	}

	metrics->start_time = clock();

	printf("Running multi-threaded Monte Carlo simulation...\n");
	printf("Grid: %dx%dx%d voxels, Requesting %.1f min\n", config->grid.size_x, config->grid.size_y,
		   config->grid.size_z, config->simulation_time_minutes);

	int num_threads = 1;
#ifdef _OPENMP
	num_threads = omp_get_max_threads();
	printf("Using %d threads\n", num_threads);
#endif

	// Phase 1: Calibration
	clock_t temp_time = clock();
	uint64_t calibration_photons = 0;
	uint64_t calibration_steps = 0;

#pragma omp parallel reduction(+ : calibration_photons, calibration_steps)
	{
#ifdef _OPENMP
		int thread_id = omp_get_thread_num();
#else
		int thread_id = 0;
#endif
		thread_rng_state = (uint64_t)time(NULL) * (thread_id + 1) + 12345;

#pragma omp for schedule(static)
		for (uint64_t i = 0; i < 1000; i++) {
			uint32_t steps = transport_photon(config);
			calibration_steps += steps;
			calibration_photons++;
		}
	}

	// Calculate target
	clock_t finish_time = clock();
	uint64_t target_photons =
		(uint64_t)(config->simulation_time_minutes * 60 * 999 * CLOCKS_PER_SEC / (finish_time - temp_time));
	printf("Target photons updated to %llu for %.2f min\n", (unsigned long long)target_photons,
		   config->simulation_time_minutes);

	// Phase 2: Main simulation with progress reporting
	uint64_t remaining_photons = (target_photons > 1000) ? target_photons - 1000 : 0;
	uint64_t main_photons = 0;
	uint64_t main_steps = 0;

	// Shared progress tracking
	uint64_t progress_counter = 0;
	uint64_t last_reported_progress = 0;

	if (remaining_photons > 0) {
#pragma omp parallel reduction(+ : main_photons, main_steps)
		{
#ifdef _OPENMP
			int thread_id = omp_get_thread_num();
#else
			int thread_id = 0;
#endif
			thread_rng_state = (uint64_t)time(NULL) * (thread_id + 1) + 54321;

#pragma omp for schedule(dynamic, 100)
			for (uint64_t i = 0; i < remaining_photons; i++) {
				uint32_t steps = transport_photon(config);
				main_steps += steps;
				main_photons++;

// Progress reporting (less frequent in multi-threaded mode)
#pragma omp atomic
				progress_counter++;

				// Report every 1000 photons or so
				if (progress_counter % 1000 == 0) {
					report_progress(progress_counter, remaining_photons, true, &last_reported_progress,
									metrics->start_time);
				}
			}
		}
	}

	uint64_t total_photons = calibration_photons + main_photons;
	uint64_t total_steps = calibration_steps + main_steps;

	printf("------------------------------------------------------\n");

	metrics->end_time = clock();
	metrics->elapsed_seconds = (double)(metrics->end_time - metrics->start_time) / CLOCKS_PER_SEC;
	metrics->photons_completed = total_photons;
	metrics->photons_per_second = (uint64_t)(total_photons / metrics->elapsed_seconds);
	metrics->steps_per_photon_avg = (int)(total_steps / total_photons);

	float time_min = (float)metrics->elapsed_seconds / 60.0F;
	printf("Elapsed Time for %llu photons = %.3f min\n", (unsigned long long)total_photons, time_min);
	printf("%llu photons per minute\n", (unsigned long long)(total_photons / time_min));
	printf("Average steps per photon: %d\n", metrics->steps_per_photon_avg);

	return MCXYZ_SUCCESS;
}

/**
 * Ultra-optimized simulation (alias for multi-threaded with enhanced settings)
 */
McxyzErrorCode run_monte_carlo_simulation_ultra(SimulationConfig* config, PerformanceMetrics* metrics) {
	if (!config || !metrics) {
		return MCXYZ_ERROR_INVALID_PARAMETER;
	}

	printf("Running ultra-optimized Monte Carlo simulation...\n");
	printf("Grid: %dx%dx%d voxels, Requesting %.1f min\n", config->grid.size_x, config->grid.size_y,
		   config->grid.size_z, config->simulation_time_minutes);

#ifdef _OPENMP
	printf("Using %d threads with ultra-optimizations\n", omp_get_max_threads());
#endif

	// Use the multi-threaded version with all optimizations enabled
	return run_monte_carlo_simulation_mt(config, metrics);
}
