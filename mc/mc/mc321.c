/*******************************************************************************
 *  mc321.c
 *
 *  Monte Carlo simulation yielding spherical, cylindrical, and planar
 *    responses to an isotropic point source in an infinite homogeneous
 *    medium with no boundaries. This program is a minimal Monte Carlo
 *    program scoring photon distributions in spherical, cylindrical,
 *    and planar shells.
 *
 *  by Steven L. Jacques based on prior collaborative work
 *    with Lihong Wang, Scott Prahl, and Marleen Keijzer.
 *    partially funded by the NIH (R29-HL45045, 1991-1997) and
 *    the DOE (DE-FG05-91ER617226, DE-FG03-95ER61971, 1991-1999).
 *
 *  A published report illustrates use of the program:
 *    S. L. Jacques: "Light distributions from point, line, and plane
 *    sources for photochemical reactions and fluorescence in turbid
 *    biological tissues," Photochem. Photobiol. 67:23-32, 1998.
 *
 *  Trivial fixes to remove warnings. SAP, 11/2017
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

/* C standard library random number generator wrapper */
double random_gen(char Type, long Seed, long *Status);

/* Helper function to write formatted output to both file and console */
int out(FILE *file, const char *format, ...);

int main(void) {
	/****
	 * INPUT
	 * Input the optical properties
	 * Input the bin and array sizes
	 * Input the number of photons
	 ****/
	const struct {
		double mua;         /* absorption coefficient [cm^-1] */
		double mus;         /* scattering coefficient [cm^-1] */
		double g;           /* anisotropy [-] */
		double nt;          /* tissue index of refraction */
		double photons;     /* number of photons in simulation */
		double radial_size; /* maximum radial size */
		int32_t bins;       /* number of radial positions */
	} params = {.mua = 1.0, .mus = 0.0, .g = 0.90, .nt = 1.33, .photons = 10000, .radial_size = 10.0, .bins = 100};

	/* Arrays are automatically sized based on params.bins - no manual intervention needed */
	const double dr = params.radial_size / params.bins; /* cm */
	const double albedo = params.mus / (params.mus + params.mua);

	/* Arrays to store photon concentrations - dynamically sized based on params.bins */
	double Csph[params.bins + 1]; /* spherical   photon concentration [0..bins] */
	double Ccyl[params.bins + 1]; /* cylindrical photon concentration [0..bins] */
	double Cpla[params.bins + 1]; /* planar      photon concentration [0..bins] */

	/* Initialize arrays to zero */
	for (int32_t i = 0; i <= params.bins; i++) {
		Csph[i] = 0.0;
		Ccyl[i] = 0.0;
		Cpla[i] = 0.0;
	}

	/****
	 * INITIALIZATIONS
	 ****/
	random_gen(0, 1, NULL); /* Initialize the random number generator with a seed. */

	/****
	 * RUN
	 * Launch N photons, initializing each one before propagation.
	 ****/
	for (int32_t i_photon = 1; i_photon <= (int32_t)params.photons; i_photon++) {
		/****
		 * LAUNCH
		 * Initialize photon position and trajectory.
		 * Implements an isotropic point source.
		 ****/
		double W = 1.0;            /* set photon weight to one */
		bool photon_status = true; /* Launch an ALIVE photon */

		/* Photon position */
		struct {
			double x, y, z;
		} pos = {0, 0, 0}; /* Set photon position to origin. */

		/* Photon trajectory as cosines */
		double ux, uy, uz;

		/* Randomly set photon trajectory to yield an isotropic source. */
		const double costheta_init = 2.0 * random_gen(1, 0, NULL) - 1.0;
		const double sintheta_init = sqrt(1.0 - (costheta_init * costheta_init)); /* sintheta is always positive */
		const double psi_init = 2.0 * M_PI * random_gen(1, 0, NULL);
		ux = sintheta_init * cos(psi_init);
		uy = sintheta_init * sin(psi_init);
		uz = costheta_init;

		/****
		 * HOP_DROP_SPIN_CHECK
		 * Propagate one photon until it dies as determined by ROULETTE.
		 ****/
		do {
			/****
			 * HOP
			 * Take step to new position
			 * s = stepsize
			 * ux, uy, uz are cosines of current photon trajectory
			 ****/

			double rnd;
			while ((rnd = random_gen(1, 0, NULL)) <= 0.0) {}         /* yields 0 < rnd <= 1 */
			const double s = -log(rnd) / (params.mua + params.mus); /* Step size */
			pos.x += s * ux;                                        /* Update positions. */
			pos.y += s * uy;
			pos.z += s * uz;

			/****
			 * DROP
			 * Drop photon weight (W) into local bin.
			 ****/
			const double absorb = W * (1 - albedo); /* photon weight absorbed at this step */
			W -= absorb;                            /* decrement WEIGHT by amount absorbed */

			/* Spherical binning */
			{
				const double r =
					sqrt((pos.x * pos.x) + (pos.y * pos.y) + (pos.z * pos.z)); /* current spherical radial position */
				int32_t ir = (int32_t)(r / dr);                                /* ir = index to spatial bin */
				if (ir >= params.bins) {
					ir = params.bins;                                          /* far-field bin for r >= radial_size */
				}
				Csph[ir] += absorb;                                            /* DROP absorbed weight into bin */
			}

			/* Cylindrical binning */
			{
				const double r = sqrt((pos.x * pos.x) + (pos.y * pos.y)); /* current cylindrical radial position */
				int32_t ir = (int32_t)(r / dr);                           /* ir = index to spatial bin */
				if (ir >= params.bins) {
					ir = params.bins;                                     /* far-field bin for r >= radial_size */
				}
				Ccyl[ir] += absorb;                                       /* DROP absorbed weight into bin */
			}

			/* Planar binning */
			{
				const double r = fabs(pos.z);   /* current planar radial position */
				int32_t ir = (int32_t)(r / dr); /* ir = index to spatial bin */
				if (ir >= params.bins) {
					ir = params.bins;           /* far-field bin for r >= radial_size */
				}
				Cpla[ir] += absorb;             /* DROP absorbed weight into bin */
			}

			/****
			 * SPIN
			 * Scatter photon into new trajectory defined by theta and psi.
			 * Theta is specified by cos(theta), which is determined
			 * based on the Henyey-Greenstein scattering function.
			 * Convert theta and psi into cosines ux, uy, uz.
			 ****/

			/* Sample costheta */
			double costheta;
			const double rnd_scatter = random_gen(1, 0, NULL);
			if (params.g == 0.0) {
				costheta = 2.0 * rnd_scatter - 1.0;
			}
			else {
				const double temp = (1.0 - params.g * params.g) / (1.0 - params.g + 2 * params.g * rnd_scatter);
				costheta = (1.0 + params.g * params.g - temp * temp) / (2.0 * params.g);
			}
			const double sintheta = sqrt(1.0 - (costheta * costheta)); /* sqrt() is faster than sin(). */

			/* Sample psi */
			const double psi = 2.0 * M_PI * random_gen(1, 0, NULL);
			const double cospsi = cos(psi);
			const double sinpsi = (psi < M_PI) ? sqrt(1.0 - (cospsi * cospsi)) : -sqrt(1.0 - (cospsi * cospsi));

			/* New trajectory */
			double uxx, uyy, uzz;
			if (1 - fabs(uz) <= DBL_EPSILON) { /* close to perpendicular. */
				uxx = sintheta * cospsi;
				uyy = sintheta * sinpsi;
				uzz = costheta * (uz >= 0.0 ? 1.0 : -1.0);
			}
			else {                             /* usually use this option */
				const double temp = sqrt(1.0 - (uz * uz));
				uxx = sintheta * (ux * uz * cospsi - uy * sinpsi) / temp + ux * costheta;
				uyy = sintheta * (uy * uz * cospsi + ux * sinpsi) / temp + uy * costheta;
				uzz = -sintheta * cospsi * temp + uz * costheta;
			}

			/* Update trajectory */
			ux = uxx;
			uy = uyy;
			uz = uzz;

			/****
			 * CHECK ROULETTE
			 * If photon weight below THRESHOLD, then terminate photon using Roulette technique.
			 * Photon has CHANCE probability of having its weight increased by factor of 1/CHANCE,
			 * and 1-CHANCE probability of terminating.
			 ****/
			const double threshold = 0.01; /* Threshold for roulette */
			const double chance = 0.1;     /* Chance of surviving roulette */

			if (W < threshold) {
				if (random_gen(1, 0, NULL) <= chance) {
					W /= chance;
				}
				else {
					photon_status = false; /* Photon is DEAD */
				}
			}
		}
		while (photon_status == true);
	}

	/****
	 * SAVE
	 * Convert data to relative fluence rate [cm^-2] and save to file called "mc321.out".
	 ****/
	FILE *target = fopen("mc321.out", "w");
	if (!target) {
		fprintf(stderr, "Error: Cannot open output file\n");
		return 1;
	}

	/* print header */
	out(target, "Photons: %.0f\n", params.photons);
	out(target, "Radius: %.1f\n", params.radial_size);
	out(target, "Bins: %d\n", params.bins);
	out(target, "Bin size: %.5f [cm]\n\n", dr);

	/* print column titles */
	out(target, "%8s %14s %14s %14s\n", "r [cm]", "Fsph [1/cm2]", "Fcyl [1/cm2]", "Fpla [1/cm2]");
	out(target, "%8s %14s %14s %14s\n", "--------", "--------------", "--------------", "--------------");

	/* print data:  radial position, fluence rates for 3D, 2D, 1D geometries */
	/* Note: Skip the far-field bin (ir = NR) as it doesn't represent a specific radial range */
	for (int32_t ir = 0; ir < params.bins; ir++) {
		const double r = (ir + 0.5) * dr;

		/* Calculate shell volumes */
		const double sph_volume = 4.0 * M_PI * r * r * dr; /* per spherical shell */
		const double cyl_volume = 2.0 * M_PI * r * dr;     /* per cm length of cylinder */
		const double pla_volume = dr;                      /* per cm2 area of plane */

		const double Fsph = Csph[ir] / params.photons / sph_volume / params.mua;
		const double Fcyl = Ccyl[ir] / params.photons / cyl_volume / params.mua;
		const double Fpla = Cpla[ir] / params.photons / pla_volume / params.mua;
		out(target, "%8.5f %14.6f %14.6f %14.6f\n", r, Fsph, Fcyl, Fpla);
	}

	/* Report far-field bin separately if it contains significant data */
	if (Csph[params.bins] > 0 || Ccyl[params.bins] > 0 || Cpla[params.bins] > 0) {
		out(target, "%8s %14s %14s %14s\n", "--------", "--------------", "--------------", "--------------");
		out(target, "%8s %14s %14s %14s\n", "r [cm]", "Sph [weight]", "Cyl [weight]", "Pla [weight]");
		out(target, ">%7.4f %14.6f %14.6f %14.6f\n", params.radial_size, Csph[params.bins], Ccyl[params.bins],
					Cpla[params.bins]);
	}

	fclose(target);
	return 0;
}

/****
 * Helper function to write formatted output to both file and console
 * Returns the number of characters written to file (or negative on error)
 ****/
int out(FILE *file, const char *format, ...) {
	va_list args1, args2;
	int result;

	/* Initialize variable argument lists */
	va_start(args1, format);
	va_start(args2, format);

	/* Write to file */
	result = vfprintf(file, format, args1);

	/* Write to console */
	vprintf(format, args2);

	/* Clean up */
	va_end(args1);
	va_end(args2);

	return result;
}

/****
 *  PCG random number generator wrapper
 *  Maintains compatibility with the original random_gen interface
 *  while using the PCG algorithm for improved statistical properties.
 *
 *  When Type is 0, sets Seed as the seed. Make sure 0<Seed<32000.
 *  When Type is 1, returns a random number between 0 and 1.
 *  When Type is 2, gets the status of the generator.
 *  When Type is 3, restores the status of the generator.
 *
 *  The status is represented by two values: Status[0] = state, Status[1] = inc.
 ****/

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
		/* set seed */
		case 0: {
			unsigned long long pcg_seed_val = (unsigned long long)(seed < 0 ? -seed : seed);
			if (pcg_seed_val == 0) { /* avoid zero seed */
				pcg_seed_val = 1;
			}
			pcg_seed(pcg_seed_val);
			break;
		}
		/* get a random number */
		case 1: {
			/* Return random number in range [0, 1) */
			return (pcg32_random() >> 5) * 0x1.0p-27;
		}

		/* get status */
		case 2:
			if (status) {
				status[0] = (long)(rng_state.state & 0xFFFFFFFF);
				status[1] = (long)(rng_state.inc & 0xFFFFFFFF);
			}
			break;

		/* restore status */
		case 3:
			if (status) {
				rng_state.state = (unsigned long long)status[0];
				rng_state.inc = (unsigned long long)status[1];
			}
			break;

		/* default case for error handling */
		default: fprintf(stderr, "Wrong parameter to random_gen(): %d\n", type); break;
	}
	return 0.0;
}
