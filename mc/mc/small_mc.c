/*******************************************************************************
 *  small_mc.c
 *
 * 	Small Monte Carlo by Scott Prahl (https://omlc.org)
 *  1 W/cm^2 Uniform Illumination of Semi-Infinite Medium
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
	double mu_a;            /* Absorption coefficient [1/cm] */
	double mu_s;            /* Scattering coefficient [1/cm] */
	double g;               /* Scattering anisotropy [-1 <= g <= 1] */
	double n;               /* Index of refraction of medium */
	double microns_per_bin; /* Thickness of one bin layer [microns] */
	int32_t bins;           /* Number of depth bins */
	int64_t photons;        /* Number of photons to simulate */
} SimParams;

/* Photon state */
typedef struct {
	double x, y, z; /* Position [cm] */
	double u, v, w; /* Direction cosines */
	double weight;  /* Photon weight */
} Photon;

/* Results storage */
typedef struct {
	double rs;           /* Specular reflection */
	double rd;           /* Diffuse reflection */
	double albedo;       /* Single scattering albedo */
	double crit_angle;   /* Critical angle for total internal reflection */
	double bins_per_mfp; /* Bins per mean free path */
	double *heat;        /* Heat deposition array */
} Results;

/* C standard library random number generator wrapper */
double random_gen(char Type, long Seed, long *Status);

/* Helper function to write formatted output to both file and console */
int out(FILE *file, const char *format, ...);

/* Function prototypes */
void launch_photon(Photon *photon, const Results *results);
void bounce_photon(Photon *photon, Results *results, const SimParams *params);
void move_photon(Photon *photon, const SimParams *params);
void absorb_photon(Photon *photon, Results *results, const SimParams *params);
void scatter_photon(Photon *photon, const SimParams *params, Results *results);
void print_results(FILE *target, const SimParams *params, const Results *results);

int main(void) {
	/* Simulation parameters */
	const SimParams params = {
		.mu_a = 5.0,             /* Absorption coefficient [1/cm] */
		.mu_s = 95.0,            /* Scattering coefficient [1/cm] */
		.g = 0.5,                /* Scattering anisotropy */
		.n = 1.5,                /* Refractive index */
		.microns_per_bin = 20.0, /* Bin thickness [microns] */
		.bins = 101,             /* Number of bins */
		.photons = 100000        /* Number of photons */
	};

	/* Initialize results structure */
	Results results = {0};
	results.heat = calloc(params.bins, sizeof(double));
	if (!results.heat) {
		fprintf(stderr, "Error: Cannot allocate memory for heat array\n");
		return 1;
	}

	/* Calculate derived parameters */
	results.albedo = params.mu_s / (params.mu_s + params.mu_a);
	results.rs = pow((params.n - 1.0) / (params.n + 1.0), 2); /* Specular reflection */
	results.crit_angle = sqrt(1.0 - pow(1.0 / params.n, 2));  /* Critical angle cosine */
	results.bins_per_mfp = 1e4 / params.microns_per_bin / (params.mu_a + params.mu_s);

	/* Initialize random number generator */
	random_gen(0, 1, NULL);

	/* Open output file */
	FILE *target = fopen("small_mc.out", "w");
	if (!target) {
		fprintf(stderr, "Error: Cannot open output file\n");
		free(results.heat);
		return 1;
	}

	/* Run simulation */
	printf("Running Small Monte Carlo simulation...\n");
	printf("Photons: %lld, Bins: %d\n", (long long)params.photons, params.bins);

	for (int64_t i = 0; i < params.photons; i++) {
		Photon photon;
		launch_photon(&photon, &results);

		while (photon.weight > 0) {
			move_photon(&photon, &params);
			absorb_photon(&photon, &results, &params);
			scatter_photon(&photon, &params, &results);
		}

		/* Progress indicator */
		if ((i + 1) % 10000 == 0) {
			printf("Progress: %lld/%lld photons\n", (long long)(i + 1), (long long)params.photons);
		}
	}

	/* Output results */
	print_results(target, &params, &results);

	/* Cleanup */
	fclose(target);
	free(results.heat);

	printf("Simulation complete. Results saved to small_mc.out\n");
	return 0;
}

/* Initialize photon at source */
void launch_photon(Photon *photon, const Results *results) {
	photon->x = 0.0;
	photon->y = 0.0;
	photon->z = 0.0;
	photon->u = 0.0;
	photon->v = 0.0;
	photon->w = 1.0;
	photon->weight = 1.0 - results->rs;
}

/* Handle interaction with top surface */
void bounce_photon(Photon *photon, Results *results, const SimParams *params) {
	/* Reverse direction */
	photon->w = -photon->w;
	photon->z = -photon->z;

	/* Check for total internal reflection */
	if (photon->w <= results->crit_angle) {
		return;
	}

	/* Calculate Fresnel reflection */
	const double t = sqrt(1.0 - pow(params->n, 2) * (1.0 - pow(photon->w, 2))); /* cos of exit angle */
	const double temp1 = (photon->w - params->n * t) / (photon->w + params->n * t);
	const double temp = (t - params->n * photon->w) / (t + params->n * photon->w);
	const double rf = (pow(temp1, 2) + pow(temp, 2)) / 2.0;                     /* Fresnel reflection coefficient */

	/* Update diffuse reflection */
	results->rd += (1.0 - rf) * photon->weight;
	photon->weight -= (1.0 - rf) * photon->weight;
}

/* Move photon to next scattering or absorption event */
void move_photon(Photon *photon, const SimParams *params) {
	/* Step size - matches original small_mc approach */
	const double d = -log((random_gen(1, 0, NULL) * RAND_MAX + 1.0) / (RAND_MAX + 1.0));

	photon->x += d * photon->u;
	photon->y += d * photon->v;
	photon->z += d * photon->w;

	/* Check for surface interaction */
	if (photon->z <= 0) {
		/* At surface - will be handled by scatter_photon calling bounce_photon */
	}
}

/* Handle photon absorption */
void absorb_photon(Photon *photon, Results *results, const SimParams *params) {
	if (photon->z <= 0) {
		return; /* Don't absorb at surface */
	}

	/* Deposit energy in appropriate depth bin */
	const int32_t bin = (int32_t)(photon->z * results->bins_per_mfp);
	const int32_t safe_bin = (bin >= params->bins - 1) ? params->bins - 1 : bin;
	results->heat[safe_bin] += photon->weight * (1.0 - results->albedo);

	/* Update photon weight */
	photon->weight *= results->albedo;

	/* Russian roulette for low-weight photons */
	if (photon->weight < 0.001) {
		if (random_gen(1, 0, NULL) > 0.1) {
			photon->weight = 0.0;  /* Kill photon */
		}
		else {
			photon->weight /= 0.1; /* Increase weight to maintain energy */
		}
	}
}

/* Scatter photon into new direction */
void scatter_photon(Photon *photon, const SimParams *params, Results *results) {
	if (photon->z <= 0) {
		/* At surface - handle boundary */
		bounce_photon(photon, results, params);
		return;
	}

	/* Sample scattering angles using Henyey-Greenstein phase function */
	const double rnd1 = random_gen(1, 0, NULL);
	const double rnd2 = random_gen(1, 0, NULL);

	double costheta;
	if (fabs(params->g) < 1e-6) {
		/* Isotropic scattering */
		costheta = 2.0 * rnd1 - 1.0;
	}
	else {
		/* Henyey-Greenstein scattering */
		const double temp = (1.0 - pow(params->g, 2)) / (1.0 - params->g + 2.0 * params->g * rnd1);
		costheta = (1.0 + pow(params->g, 2) - pow(temp, 2)) / (2.0 * params->g);
	}

	const double sintheta = sqrt(1.0 - pow(costheta, 2));
	const double phi = 2.0 * M_PI * rnd2;
	const double cosphi = cos(phi);
	const double sinphi = sin(phi);

	/* Update direction using standard transformation */
	double uxx, uyy, uzz;
	if (fabs(photon->w) > (1.0 - DBL_EPSILON)) {
		/* Close to perpendicular */
		uxx = sintheta * cosphi;
		uyy = sintheta * sinphi;
		uzz = costheta * (photon->w >= 0.0 ? 1.0 : -1.0);
	}
	else {
		/* General case */
		const double temp = sqrt(1.0 - pow(photon->w, 2));
		uxx = sintheta * (photon->u * photon->w * cosphi - photon->v * sinphi) / temp + photon->u * costheta;
		uyy = sintheta * (photon->v * photon->w * cosphi + photon->u * sinphi) / temp + photon->v * costheta;
		uzz = -sintheta * cosphi * temp + photon->w * costheta;
	}

	photon->u = uxx;
	photon->v = uyy;
	photon->w = uzz;
}

/* Print simulation results */
void print_results(FILE *target, const SimParams *params, const Results *results) {
	out(target, "Small Monte Carlo by Scott Prahl (https://omlc.org)\n");
	out(target, "1 W/cm^2 Uniform Illumination of Semi-Infinite Medium\n\n");

	out(target, "Simulation Parameters:\n");
	out(target, "Scattering Coefficient:   %8.3f/cm\n", params->mu_s);
	out(target, "Absorption Coefficient:   %8.3f/cm\n", params->mu_a);
	out(target, "Anisotropy:               %8.3f\n", params->g);
	out(target, "Refractive Index:         %8.3f\n", params->n);
	out(target, "Photons Simulated:        %8lld\n", (long long)params->photons);
	out(target, "\nResults:\n");
	out(target, "Specular Reflection:      %10.5f\n", results->rs);
	out(target, "Backscattered Reflection: %10.5f\n", results->rd / params->photons);
	out(target, "\n%12s %20s\n", "Depth [microns]", "Heat [W/cm^3]");
	out(target, "%12s %20s\n", "-------------", "--------------------");

	for (int32_t i = 0; i < params->bins - 1; i++) {
		const double depth = i * params->microns_per_bin;
		const double heat_density = results->heat[i] / params->microns_per_bin * 1e4 / params->photons;
		out(target, "%12.0f %20.5f\n", depth, heat_density);
	}

	/* Extra bin for anything beyond measurement range */
	const double extra_heat = results->heat[params->bins - 1] / params->photons;
	out(target, "%12s %20.5f\n", "extra", extra_heat);
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

/* PCG random number generator wrapper */

/* PCG Random Number Generator State */
typedef struct {
	unsigned long long state;
	unsigned long long inc;
} pcg_state_t;

static pcg_state_t rng_state = {0x853c49e6748fea9bULL, 0xda3e39cb94b95bdbULL};

static unsigned int pcg32_random(void) {
	unsigned long long oldstate = rng_state.state;
	rng_state.state = oldstate * 6364136223846793005ULL + rng_state.inc;
	unsigned int xorshifted = (unsigned int)(((oldstate >> 18u) ^ oldstate) >> 27u);
	unsigned int rot = (unsigned int)(oldstate >> 59u);
	return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
}

static void pcg_seed(unsigned long long seed) {
	rng_state.state = 0U;
	rng_state.inc = (seed << 1u) | 1u;
	pcg32_random();
	rng_state.state += seed;
	pcg32_random();
}

double random_gen(char Type, long Seed, long *Status) {
	switch (Type) {
		case 0: {
			unsigned long long pcg_seed_val = (unsigned long long)(Seed < 0 ? -Seed : Seed);
			if (pcg_seed_val == 0) {
				pcg_seed_val = 1;
			}
			pcg_seed(pcg_seed_val);
			break;
		}
		case 1: {
			return (pcg32_random() >> 5) * 0x1.0p-27;
		}
		case 2:
			if (Status) {
				Status[0] = (long)(rng_state.state & 0xFFFFFFFF);
				Status[1] = (long)(rng_state.inc & 0xFFFFFFFF);
			}
			break;
		case 3:
			if (Status) {
				rng_state.state = (unsigned long long)Status[0];
				rng_state.inc = (unsigned long long)Status[1];
			}
			break;
		default: fprintf(stderr, "Wrong parameter to RandomGen(): %d\n", Type); break;
	}
	return 0.0;
}
