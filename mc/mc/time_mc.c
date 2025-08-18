/*******************************************************************************
 *  time_mc.c
 *
 *  Time-Resolved Monte Carlo by Scott Prahl (https://omlc.org)
 *  1 J Pulse Irradiation of Semi-Infinite Medium
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

#define LIGHTSPEED 2.997925E10 /* Speed of light in vacuum [cm/s] */

/* Simulation parameters */
typedef struct {
	double mu_a;       /* Absorption coefficient [1/cm] */
	double mu_s;       /* Scattering coefficient [1/cm] */
	double g;          /* Scattering anisotropy [-1 <= g <= 1] */
	double n;          /* Index of refraction of medium */
	double ps_per_bin; /* Picoseconds per bin for backscattered light */
	int32_t bins;      /* Number of time bins */
	int64_t photons;   /* Number of photons to simulate */
} SimParams;

/* Photon state */
typedef struct {
	double x, y, z; /* Position [cm] */
	double u, v, w; /* Direction cosines */
	double t;       /* Time [ps] */
	double weight;  /* Photon weight */
} Photon;

/* Results storage */
typedef struct {
	double rs;           /* Specular reflection */
	double rd;           /* Total diffuse reflection */
	double albedo;       /* Single scattering albedo */
	double crit_angle;   /* Critical angle for total internal reflection */
	double bins_per_mfp; /* Time bins per mean free path */
	double *refl;        /* Time-resolved reflection array */
} Results;

/* C standard library random number generator wrapper */
double random_gen(char type, long seed, long *status);

/* Helper function to write formatted output to both file and console */
int out(FILE *file, const char *format, ...);

/* Function prototypes */
void launch_photon(Photon *photon, const Results *results);
void bounce_photon(Photon *photon, Results *results, const SimParams *params);
void move_photon(Photon *photon, const SimParams *params);
void absorb_photon(Photon *photon, const SimParams *params, Results *results);
void scatter_photon(Photon *photon, const SimParams *params, Results *results);
void print_results(FILE *target, const SimParams *params, const Results *results);

int main(void) {
	/* Simulation parameters */
	const SimParams params = {
		.mu_a = 5.0,       /* Absorption coefficient [1/cm] */
		.mu_s = 95.0,      /* Scattering coefficient [1/cm] */
		.g = 0.5,          /* Scattering anisotropy */
		.n = 1.5,          /* Refractive index */
		.ps_per_bin = 0.2, /* Picoseconds per time bin */
		.bins = 201,       /* Number of time bins */
		.photons = 30000   /* Number of photons */
	};

	/* Initialize results structure */
	Results results = {0};
	results.refl = calloc(params.bins, sizeof(double));
	if (!results.refl) {
		fprintf(stderr, "Error: Cannot allocate memory for reflection array\n");
		return 1;
	}

	/* Calculate derived parameters */
	results.albedo = params.mu_s / (params.mu_s + params.mu_a);
	results.rs = pow((params.n - 1.0) / (params.n + 1.0), 2); /* Specular reflection */
	results.crit_angle = sqrt(1.0 - pow(1.0 / params.n, 2));  /* Critical angle cosine */
	results.bins_per_mfp = LIGHTSPEED / 1e12 / params.ps_per_bin / (params.mu_a + params.mu_s);

	/* Initialize random number generator */
	random_gen(0, 1, NULL);

	/* Open output file */
	FILE *target = fopen("time_mc.out", "w");
	if (!target) {
		fprintf(stderr, "Error: Cannot open output file\n");
		free(results.refl);
		return 1;
	}

	/* Run simulation */
	printf("Running Time-Resolved Monte Carlo simulation...\n");
	printf("Photons: %lld, Time bins: %d\n", (long long)params.photons, params.bins);

	for (int64_t i = 0; i < params.photons; i++) {
		Photon photon;
		launch_photon(&photon, &results);

		while (photon.weight > 0) {
			move_photon(&photon, &params);
			absorb_photon(&photon, &params, &results);
			scatter_photon(&photon, &params, &results);
		}

		/* Progress indicator */
		if ((i + 1) % 5000 == 0) {
			printf("Progress: %lld/%lld photons\n", (long long)(i + 1), (long long)params.photons);
		}
	}

	/* Output results */
	print_results(target, &params, &results);

	/* Cleanup */
	fclose(target);
	free(results.refl);

	printf("Simulation complete. Results saved to time_mc.out\n");
	return 0;
}

/* Initialize photon at source */
void launch_photon(Photon *photon, const Results *results) {
	photon->x = 0.0;
	photon->y = 0.0;
	photon->z = 0.0;
	photon->t = 0.0;
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

	/* Calculate Fresnel transmission */
	const double tt = sqrt(1.0 - pow(params->n, 2) * (1.0 - pow(photon->w, 2))); /* cos of exit angle */
	const double temp1 = (photon->w - params->n * tt) / (photon->w + params->n * tt);
	const double temp = (tt - params->n * photon->w) / (tt + params->n * photon->w);
	const double tf = 1.0 - (pow(temp1, 2) + pow(temp, 2)) / 2.0;                /* Fresnel transmission coefficient */

	/* Update total diffuse reflection */
	results->rd += tf * photon->weight;

	/* Calculate time bin (subtract time after passing surface) */
	const double exit_time = photon->t - photon->z * sqrt(1.0 - pow(photon->w, 2)) * params->n / LIGHTSPEED * 1e12;
	int32_t bin = (int32_t)(exit_time / params->ps_per_bin);
	if (bin >= params->bins) {
		bin = params->bins - 1;
	}
	if (bin >= 0) {
		results->refl[bin] += tf * photon->weight;
	}

	/* Reduce photon weight */
	photon->weight -= tf * photon->weight;
}

/* Move photon to next scattering or absorption event */
void move_photon(Photon *photon, const SimParams *params) {
	double rnd;
	while ((rnd = random_gen(1, 0, NULL)) <= 0.0) {} /* Ensure 0 < rnd <= 1 */

	const double s = -log(rnd);                      /* Step size in mean free paths */
	photon->x += s * photon->u;
	photon->y += s * photon->v;
	photon->z += s * photon->w;
	photon->t += s / LIGHTSPEED * 1e12 * params->n;  /* Update time [ps] */

	/* Check for surface interaction */
	if (photon->z <= 0) {
		/* Move back to surface and adjust time */
		const double step_to_surface = -photon->z / photon->w;
		photon->x -= step_to_surface * photon->u;
		photon->y -= step_to_surface * photon->v;
		photon->t -= step_to_surface / LIGHTSPEED * 1e12 * params->n;
		photon->z = 0.0;
	}
}

/* Handle photon absorption */
void absorb_photon(Photon *photon, const SimParams *params, Results *results) {
	(void)results; /* Suppress unused parameter warning */
	
	if (photon->z <= 0) {
		return; /* Don't absorb at surface */
	}

	const double absorb = photon->weight * params->mu_a / (params->mu_a + params->mu_s);
	photon->weight -= absorb;

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
	out(target, "Time-Resolved Monte Carlo by Scott Prahl (https://omlc.org)\n");
	out(target, "1 J Pulse Irradiation of Semi-Infinite Medium\n\n");

	out(target, "Simulation Parameters:\n");
	out(target, "Scattering Coefficient:   %8.3f/cm\n", params->mu_s);
	out(target, "Absorption Coefficient:   %8.3f/cm\n", params->mu_a);
	out(target, "Anisotropy:               %8.3f\n", params->g);
	out(target, "Refractive Index:         %8.3f\n", params->n);
	out(target, "Photons Simulated:        %8lld\n", (long long)params->photons);
	out(target, "Time resolution:          %8.3f ps/bin\n\n", params->ps_per_bin);

	out(target, "Results:\n");
	out(target, "Specular Reflection:      %10.5f\n", results->rs);
	out(target, "Total Diffuse Reflection: %10.5f\n", results->rd / params->photons);
	out(target, "\n%12s %20s\n", "Time [ps]", "Reflectance");
	out(target, "%12s %20s\n", "-------------", "--------------------");

	for (int32_t i = 0; i < params->bins; i++) {
		const double time = i * params->ps_per_bin;
		const double reflectance = results->refl[i] / params->photons;
		if (reflectance > 1e-8) { /* Only print non-negligible values */
			out(target, "%12.3f %20.8e\n", time, reflectance);
		}
	}
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

double random_gen(char type, long seed, long *status) {
	switch (type) {
		case 0: {
			unsigned long long pcg_seed_val = (unsigned long long)(seed < 0 ? -seed : seed);
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
			if (status) {
				status[0] = (long)(rng_state.state & 0xFFFFFFFF);
				status[1] = (long)(rng_state.inc & 0xFFFFFFFF);
			}
			break;
		case 3:
			if (status) {
				rng_state.state = (unsigned long long)status[0];
				rng_state.inc = (unsigned long long)status[1];
			}
			break;
		default: fprintf(stderr, "Wrong parameter to RandomGen(): %d\n", type); break;
	}
	return 0.0;
}
