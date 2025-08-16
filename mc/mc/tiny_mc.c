/*******************************************************************************
 *  tiny_mc.c
 *
 *  Tiny Monte Carlo by Scott Prahl (https://omlc.org)
 *  1 W Point Source Heating in Infinite Isotropic Scattering Medium
 *
 *  Updated for C17 compliance and improved readability. ML, 08/2025
 ******************************************************************************/

#include <float.h>
#include <math.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Simulation parameters */
typedef struct {
	double mu_a;              /* Absorption coefficient [1/cm] - must be non-zero */
	double mu_s;              /* Reduced scattering coefficient [1/cm] */
	double microns_per_shell; /* Thickness of spherical shells [microns] */
	int32_t shells;           /* Number of radial shells */
	int64_t photons;          /* Number of photons to simulate */
} SimParams;

/* Photon state */
typedef struct {
	double x, y, z; /* Position [cm] */
	double u, v, w; /* Direction cosines */
	double weight;  /* Photon weight */
} Photon;

/* Results storage */
typedef struct {
	double albedo;         /* Single scattering albedo */
	double shells_per_mfp; /* Shells per mean free path */
	double *heat;          /* Heat deposition array */
} Results;

/* C standard library random number generator wrapper */
double random_gen(char type, long seed, long *status);

/* Helper function to write formatted output to both file and console */
int out(FILE *file, const char *format, ...);

/* Function prototypes */
void launch_photon(Photon *photon);
void move_photon(Photon *photon, const SimParams *params);
void absorb_photon(Photon *photon, Results *results, const SimParams *params);
void scatter_photon(Photon *photon);
void roulette_photon(Photon *photon);
void print_results(FILE *target, const SimParams *params, const Results *results);

int main(void) {
	/* Simulation parameters */
	const SimParams params = {
		.mu_a = 2.0,               /* Absorption coefficient [1/cm] - must be non-zero */
		.mu_s = 20.0,              /* Reduced scattering coefficient [1/cm] */
		.microns_per_shell = 50.0, /* Shell thickness [microns] */
		.shells = 101,             /* Number of shells */
		.photons = 10000           /* Number of photons */
	};

	/* Initialize results structure */
	Results results = {0};
	results.heat = calloc(params.shells, sizeof(double));
	if (!results.heat) {
		fprintf(stderr, "Error: Cannot allocate memory for heat array\n");
		return 1;
	}

	/* Calculate derived parameters */
	results.albedo = params.mu_s / (params.mu_s + params.mu_a);
	results.shells_per_mfp = 1e4 / params.microns_per_shell / (params.mu_a + params.mu_s);

	/* Initialize random number generator */
	random_gen(0, 1, NULL);

	/* Open output file */
	FILE *target = fopen("tiny_mc.out", "w");
	if (!target) {
		fprintf(stderr, "Error: Cannot open output file\n");
		free(results.heat);
		return 1;
	}

	/* Run simulation */
	printf("Running Tiny Monte Carlo simulation...\n");
	printf("Photons: %lld, Shells: %d\n", (long long)params.photons, params.shells);

	for (int64_t i = 0; i < params.photons; i++) {
		Photon photon;
		launch_photon(&photon);

		while (photon.weight > 0) {
			move_photon(&photon, &params);
			absorb_photon(&photon, &results, &params);
			scatter_photon(&photon);
			roulette_photon(&photon);
		}

		/* Progress indicator */
		if ((i + 1) % 1000 == 0) {
			printf("Progress: %lld/%lld photons\n", (long long)(i + 1), (long long)params.photons);
		}
	}

	/* Output results */
	print_results(target, &params, &results);

	/* Cleanup */
	fclose(target);
	free(results.heat);

	printf("Simulation complete. Results saved to tiny_mc.out\n");
	return 0;
}

/* Initialize photon at point source */
void launch_photon(Photon *photon) {
	photon->x = 0.0;
	photon->y = 0.0;
	photon->z = 0.0;
	photon->u = 0.0;
	photon->v = 0.0;
	photon->w = 1.0;
	photon->weight = 1.0;
}

/* Move photon to next interaction site */
void move_photon(Photon *photon, const SimParams *params) {
	double rnd;
	while ((rnd = random_gen(1, 0, NULL)) <= 0.0) {} /* Ensure 0 < rnd <= 1 */
	const double t = -log(rnd);                     /* Step size in mean free paths */
	photon->x += t * photon->u;
	photon->y += t * photon->v;
	photon->z += t * photon->w;
}

/* Handle photon absorption and energy deposition */
void absorb_photon(Photon *photon, Results *results, const SimParams *params) {
	/* Calculate radial distance from source */
	const double r = sqrt(pow(photon->x, 2) + pow(photon->y, 2) + pow(photon->z, 2));

	/* Determine shell index */
	int32_t shell = (int32_t)(r * results->shells_per_mfp);
	if (shell >= params->shells) {
		shell = params->shells - 1; /* Far-field shell */
	}

	/* Deposit energy */
	const double absorbed = (1.0 - results->albedo) * photon->weight;
	results->heat[shell] += absorbed;
	photon->weight *= results->albedo;
}

/* Scatter photon isotropically */
void scatter_photon(Photon *photon) {
	/* Use rejection method to sample isotropic direction */
	double xi1, xi2, t;
	do {
		xi1 = 2.0 * random_gen(1, 0, NULL) - 1.0;
		xi2 = 2.0 * random_gen(1, 0, NULL) - 1.0;
		t = pow(xi1, 2) + pow(xi2, 2);
	}
	while (t > 1.0);

	/* Convert to direction cosines */
	photon->u = 2.0 * t - 1.0;
	const double sqrt_factor = sqrt((1.0 - pow(photon->u, 2)) / t);
	photon->v = xi1 * sqrt_factor;
	photon->w = xi2 * sqrt_factor;
}

/* Apply Russian roulette for low-weight photons */
void roulette_photon(Photon *photon) {
	const double threshold = 0.001;
	const double chance = 0.1;

	if (photon->weight < threshold) {
		if (random_gen(1, 0, NULL) > chance) {
			photon->weight = 0.0;     /* Terminate photon */
		}
		else {
			photon->weight /= chance; /* Increase weight */
		}
	}
}

/* Print simulation results */
void print_results(FILE *target, const SimParams *params, const Results *results) {
	out(target, "Tiny Monte Carlo by Scott Prahl (https://omlc.org)\n");
	out(target, "1 W Point Source Heating in Infinite Isotropic Scattering Medium\n\n");

	out(target, "Simulation Parameters:\n");
	out(target, "Absorption Coefficient:   %8.3f/cm\n", params->mu_a);
	out(target, "Reduced Scattering Coeff: %8.3f/cm\n", params->mu_s);
	out(target, "Shell thickness:          %8.1f microns\n", params->microns_per_shell);
	out(target, "Photons Simulated:        %8lld\n", (long long)params->photons);
	out(target, "Single Scattering Albedo: %8.5f\n", results->albedo);
	out(target, "\n%15s %20s\n", "Radius [microns]", "Heat [W/cm^3]");
	out(target, "%15s %20s\n", "---------------", "--------------------");

	double total_heat = 0.0;
	for (int32_t i = 0; i < params->shells; i++) {
		const double r_inner = i * params->microns_per_shell;
		const double r_outer = (i + 1) * params->microns_per_shell;
		const double r_center = (r_inner + r_outer) / 2.0;

		/* Calculate shell volume in cm^3 */
		const double volume = (4.0 / 3.0) * M_PI * (pow(r_outer * 1e-4, 3) - pow(r_inner * 1e-4, 3));

		/* Heat density [W/cm^3] */
		const double heat_density = (volume > 0) ? results->heat[i] / params->photons / volume : 0.0;

		total_heat += results->heat[i];

		if (heat_density > 1e-8 || i < 20) { /* Print first 20 shells and significant values */
			out(target, "%15.1f %20.8e\n", r_center, heat_density);
		}
	}

	out(target, "\nTotal absorbed energy: %12.8f\n", total_heat / params->photons);
	out(target, "Energy conservation check: %8.5f (should be close to %.5f)\n", total_heat / params->photons,
				1.0 - results->albedo);
}

/* Helper function to write formatted output to both file and console */
int out(FILE *file, const char *format, ...) {
	va_list args1, args2;
	int result;

	va_start(args1, format);
	va_start(args2, format);

	result = vfprintf(file, format, args1);
	vprintf(format, args2);

	va_end(args1);
	va_end(args2);

	return result;
}

/* C standard library random number generator wrapper */
double random_gen(char type, long seed, long *status) {
	static unsigned int current_seed = 1;

	switch (type) {
		case 0: {
			current_seed = (unsigned int)(seed < 0 ? -seed : seed);
			if (current_seed == 0) {
				current_seed = 1;
			}
			srand(current_seed);
			break;
		}
		case 1: {
			return (double)rand() / ((double)RAND_MAX + 1.0);
		}
		case 2:
			if (status) {
				status[0] = (long)current_seed;
			}
			break;
		case 3:
			if (status) {
				current_seed = (unsigned int)status[0];
				srand(current_seed);
			}
			break;
		default: fprintf(stderr, "Wrong parameter to RandomGen(): %d\n", type); break;
	}
	return 0.0;
}
