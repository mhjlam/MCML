/*******************************************************************************
 *  trmc.c
 *
 *  Time-Resolved Monte Carlo simulation for infinite homogeneous medium.
 *  Monte Carlo simulation of time-resolved photon fluence rate in response
 *  to an impulse of energy delivered as an isotropic point source.
 *
 *  Results are recorded as F[ir][it]/Uo for multiple timepoints and 100
 *  radial positions. Output format:
 *  r [cm]   F/Uo @ time#1   F/Uo @ time#2   F/Uo @ time#3   F/Uo @ time#4
 *
 *  Original by Steven L. Jacques based on collaborative work with
 *  Lihong Wang, Scott Prahl, and Marleen Keijzer.
 *  Updated for C17 compliance and improved readability. ML, 08/2025
 ******************************************************************************/

#include <math.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define LIGHTSPEED        2.997925E10 /* Speed of light in vacuum [cm/s] */
#define ONE_MINUS_COSZERO 1.0E-12     /* For theta near 0 or PI */

/* Simulation parameters (matching original) */
typedef struct {
	double mus;            /* Scattering coefficient [1/cm] */
	double g;              /* Anisotropy factor [-1 <= g <= 1] */
	double nt;             /* Tissue refractive index */
	int64_t nphotons;      /* Number of photons to simulate */
	int32_t nr;            /* Number of radial bins */
	double radial_size;    /* Maximum radial distance [cm] */
	int32_t nt_pts;        /* Number of time points */
	double time_points[4]; /* Time points [ns] */
} SimParams;

/* Results storage */
typedef struct {
	double c;               /* Speed of light in medium [cm/s] */
	double dr;              /* Radial bin size [cm] */
	double lt[4];           /* Path lengths at timepoints [cm] */
	double lmax;            /* Maximum path length [cm] */
	double csph[101][4];    /* Photon concentration [ir][it] */
	double fluence[101][4]; /* Fluence rate [ir][it] */
} Results;

/* C standard library random number generator wrapper */
double random_gen(char type, long seed, long *status);

/* Helper function to write formatted output to both file and console */
int dual_printf(FILE *file, const char *format, ...);

int main(void) {
	/* Initialize simulation parameters (matching original exactly) */
	const SimParams params = {
		.mus = 100.0,                             /* Scattering coefficient [1/cm] */
		.g = 0.90,                                /* Anisotropy factor */
		.nt = 1.33,                               /* Tissue refractive index */
		.nphotons = 1000,                         /* Number of photons */
		.nr = 100,                                /* Number of radial bins */
		.radial_size = 6.0,                       /* Maximum radial size [cm] */
		.nt_pts = 4,                              /* Number of time points */
		.time_points = {0.050, 0.100, 0.500, 1.0} /* Time points [ns] */
	};

	/* Initialize results structure */
	Results results = {0};
	results.c = LIGHTSPEED / params.nt;          /* Speed of light in medium */
	results.dr = params.radial_size / params.nr; /* Radial bin size */

	/* Convert time points to path lengths */
	for (int32_t it = 0; it < params.nt_pts; it++) {
		results.lt[it] = params.time_points[it] * 1e-9 * results.c; /* Convert ns to cm */
	}
	results.lmax = results.lt[params.nt_pts - 1];                   /* Maximum path length */

	/* Initialize random number generator */
	random_gen(0, 1, NULL);

	/* Open output file */
	FILE *target = fopen("trmc.out", "w");
	if (!target) {
		fprintf(stderr, "Error: Cannot open output file\n");
		return 1;
	}

	printf("start\n");

	/* Main simulation loop */
	for (int64_t i_photon = 1; i_photon <= params.nphotons; i_photon++) {
		/* Launch photon at origin */
		bool photon_status = true;
		double L = 0.0;                   /* Total path length */
		int32_t it = 0;                   /* Time point index */

		double x = 0.0, y = 0.0, z = 0.0; /* Position */

		/* Randomly set isotropic initial direction */
		double costheta = 2.0 * random_gen(1, 0, NULL) - 1.0;
		double sintheta = sqrt(1.0 - costheta * costheta);
		double psi = 2.0 * M_PI * random_gen(1, 0, NULL);
		double ux = sintheta * cos(psi);
		double uy = sintheta * sin(psi);
		double uz = costheta;

		/* Propagate photon until death */
		do {
			/* Take step */
			double rnd;
			while ((rnd = random_gen(1, 0, NULL)) <= 0.0) {} /* Ensure 0 < rnd <= 1 */
			double s = -log(rnd) / params.mus;               /* Step size (scattering only) */

			/* Check if we reach a timepoint during this step */
			if (it < params.nt_pts && L + s >= results.lt[it]) {
				double s1 = results.lt[it] - L; /* Partial step to timepoint */
				double x1 = x + s1 * ux;        /* Position at timepoint */
				double y1 = y + s1 * uy;
				double z1 = z + s1 * uz;

				/* Record photon at this timepoint */
				double r = sqrt(x1 * x1 + y1 * y1 + z1 * z1); /* Radial position */
				int32_t ir = (int32_t)(r / results.dr);
				if (ir >= params.nr)
					ir = params.nr;                           /* Overflow bin */
				results.csph[ir][it] += 1.0;                  /* Drop photon into bin */

				/* Move to next timepoint */
				it++;
			}

			/* Complete the full step */
			x += s * ux;
			y += s * uy;
			z += s * uz;
			L += s;

			/* Scatter photon */
			double rnd1 = random_gen(1, 0, NULL);
			double rnd2 = random_gen(1, 0, NULL);

			/* Sample scattering angle using Henyey-Greenstein */
			double mu;                 /* cosine of scattering angle */
			if (fabs(params.g) < 1e-6) {
				mu = 2.0 * rnd1 - 1.0; /* Isotropic */
			}
			else {
				double temp = (1.0 - params.g * params.g) / (1.0 - params.g + 2.0 * params.g * rnd1);
				mu = (1.0 + params.g * params.g - temp * temp) / (2.0 * params.g);
			}

			double sintheta_new = sqrt(1.0 - mu * mu);
			double psi_new = 2.0 * M_PI * rnd2;
			double cospsi = cos(psi_new);
			double sinpsi = sin(psi_new);

			/* Update direction cosines */
			double uxx, uyy, uzz;
			if (fabs(uz) > (1.0 - ONE_MINUS_COSZERO)) {
				/* Close to perpendicular */
				uxx = sintheta_new * cospsi;
				uyy = sintheta_new * sinpsi;
				uzz = mu * (uz >= 0.0 ? 1.0 : -1.0);
			}
			else {
				/* General case */
				double temp = sqrt(1.0 - uz * uz);
				uxx = sintheta_new * (ux * uz * cospsi - uy * sinpsi) / temp + ux * mu;
				uyy = sintheta_new * (uy * uz * cospsi + ux * sinpsi) / temp + uy * mu;
				uzz = -sintheta_new * cospsi * temp + uz * mu;
			}

			ux = uxx;
			uy = uyy;
			uz = uzz;

			/* Check if photon should be terminated */
			if (L >= results.lmax) {
				photon_status = false;
			}
		}
		while (photon_status == true);
	}

	/* Convert concentration to fluence rate and save results */
	dual_printf(target, "number of photons = %f\n", (double)params.nphotons);
	dual_printf(target, "dr = %.5f [cm] \n", results.dr);
	dual_printf(target, "last row is overflow. Ignore last row.\n");
	dual_printf(target, "Output is fluence rate F [W/(cm2 s)].\n");

	/* Column headers */
	dual_printf(target, "r [cm] \t %5.3f ns \t %5.3f ns \t %5.3f ns \t %5.3f ns\n", params.time_points[0],
				params.time_points[1], params.time_points[2], params.time_points[3]);

	/* Data rows */
	for (int32_t ir = 0; ir < params.nr; ir++) {
		double r = (ir + 0.5) * results.dr;
		dual_printf(target, "%5.4f", r);

		double shell_volume = 4.0 * M_PI * r * r * results.dr; /* Spherical shell volume */

		for (int32_t it = 0; it < 4; it++) {
			results.fluence[ir][it] = results.c * results.csph[ir][it] / params.nphotons / shell_volume;
			dual_printf(target, "\t%5.3e", results.fluence[ir][it]);
		}
		dual_printf(target, "\n");
	}

	/* Cleanup */
	fclose(target);
	printf("finish\n");

	return 0;
}

/* Helper function to write formatted output to both file and console */
int dual_printf(FILE *file, const char *format, ...) {
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
		default: fprintf(stderr, "Wrong parameter to random_gen(): %d\n", type); break;
	}
	return 0.0;
}
