/**************************************************************************
 *	Copyright Univ. of Texas M.D. Anderson Cancer Center, 1992-1996.
 *	Program name: MCML.
 *	Launch, move, and record photon weight.
 ****/

#include "mcml.h"

/* 1=split photon, 0=statistical reflection. */
#define PARTIAL_REFLECTION 0

/* If 1-cos(theta) <= ONE_MINUS_COS_ZERO, fabs(theta) <= 1e-6 rad. */
/* If 1+cos(theta) <= ONE_MINUS_COS_ZERO, fabs(PI-theta) <= 1e-6 rad. */
#define ONE_MINUS_COS_ZERO 1.0E-12

/* If cos(theta) <= COS90D, theta >= PI/2 - 1e-6 rad. */
#define COS_90_DEGREES 1.0E-6

/**************************************************************************
 *	Modern thread-safe random number generator using a simple LCG.
 *	Maintains the original RandomGen interface for backward compatibility.
 *
 *	Uses a high-quality Linear Congruential Generator with parameters
 *	from Numerical Recipes that have good statistical properties.
 *	Much simpler and faster than the old algorithm.
 *
 *	When Type is 0, sets Seed as the seed.
 *	When Type is 1, returns a random number between 0 and 1.
 *	When Type is 2, gets the status of the generator.
 *	When Type is 3, restores the status of the generator.
 *
 *	Uses thread-local storage for thread safety.
 ****/
double RandomGen(char Type, long Seed, long* Status) {
	/* Thread-local LCG state - much simpler than the old algorithm */
	static _Thread_local unsigned long state = 1;

	/* set seed */
	if (Type == 0) {
		state = (Seed > 0) ? (unsigned long)Seed : 1;
		/* Warm up the generator */
		(void)RandomGen(1, 0, NULL);
		(void)RandomGen(1, 0, NULL);
	}
	/* get a number */
	else if (Type == 1) {
		/* High-quality LCG parameters from Numerical Recipes */
		state = (state * 1664525UL + 1013904223UL);

		/* Convert to double in range [0,1) using upper bits for better quality */
		return (double)(state >> 8) * (1.0 / 16777216.0);
	}
	/* get status */
	else if (Type == 2) {
		if (Status != NULL) {
			Status[0] = (long)state;
			/* Clear remaining array elements for consistency with old interface */
			for (int i = 1; i < 57; i++) {
				Status[i] = 0;
			}
		}
	}
	/* restore status */
	else if (Type == 3) {
		if (Status != NULL) {
			state = (unsigned long)Status[0];
		}
	}
	else {
		fprintf(stderr, "Wrong parameter to RandomGen(): %d\n", Type);
	}
	return 0.0;
}

/**************************************************************************
 *	Compute the specular reflectance.
 *
 *	If the first layer is a turbid medium, use the Fresnel reflection from
 *  the boundary between the top abmient medium and the first layer as the
 *  specular reflectance.
 *
 *	The subroutine assumes the Layerspecs array is correctly initialized.
 ****/
double Rspecular(LayerStru* Layerspecs_Ptr) {
	double r = (Layerspecs_Ptr[0].n - Layerspecs_Ptr[1].n) / (Layerspecs_Ptr[0].n + Layerspecs_Ptr[1].n);
	return (r * r);
}

/**************************************************************************
 *      Choose a new direction for photon propagation by sampling
 *	1. the polar deflection angle theta
 *	2. the azimuthal angle psi.
 *
 *      Note:
 *      theta: 0 - pi so sin(theta) is always positive
 *      Feel free to use sqrt() for cos(theta).
 *
 *      psi: 0 - 2pi
 *      for 0-pi:   sin(psi) is +
 *      for pi-2pi: sin(psi) is -
 ****/
void Spin(double g, PhotonStru* restrict Photon_Ptr) {
	if (!Photon_Ptr) {
		return;
	}

	/* Store current direction cosines */
	const double ux = Photon_Ptr->ux;
	const double uy = Photon_Ptr->uy;
	const double uz = Photon_Ptr->uz;

	/* sample theta. */
	const double rand1 = RandomGen(1, 0, NULL);
	double cost;
	if (g == 0.0) {
		cost = 2.0 * rand1 - 1.0; /* isotropic scattering */
	}
	else {
		const double temp = (1 - g * g) / (1 - g + 2 * g * rand1);
		cost = (1 + g * g - temp * temp) / (2 * g);
	}

	/* sqrt is faster than sin. */
	const double sint = sqrt(1.0 - (cost * cost));

	/* sample psi. */
	const double psi = 2.0 * M_PI * RandomGen(1, 0, NULL);
	const double cosp = cos(psi);
	/* sqrt is faster than sin. */
	const double sinp = (psi < M_PI) ? sqrt(1.0 - (cosp * cosp)) : -sqrt(1.0 - (cosp * cosp));

	/* update directional cosines. */
	/* close to perpendicular. */
	if (1 - fabs(uz) <= ONE_MINUS_COS_ZERO) {
		Photon_Ptr->ux = sint * cosp;
		Photon_Ptr->uy = sint * sinp;
		/* SIGN() is faster than division. */
		Photon_Ptr->uz = cost * SIGN(uz);
	}
	else {
		const double temp = sqrt(1.0 - (uz * uz));
		Photon_Ptr->ux = sint * (ux * uz * cosp - uy * sinp) / temp + ux * cost;
		Photon_Ptr->uy = sint * (uy * uz * cosp + ux * sinp) / temp + uy * cost;
		Photon_Ptr->uz = -sint * cosp * temp + uz * cost;
	}
}

/**************************************************************************
 *	Initialize a photon packet.
 *
 *	If an isotropic source is launched inside a glass layer, we check
 *	whether the photon will be total-internally reflected.  If it
 *	does, the photon is killed to avoid a infinite travelling inside
 *	the glass layer.
 ****/
void LaunchPhoton(double Rsp, InStru* In_Ptr, OutStru* Out_Ptr, PhotonStru* Photon_Ptr) {
	Photon_Ptr->w = 1.0 - Rsp;
	Photon_Ptr->alive = 1;
	Photon_Ptr->layer = (In_Ptr->slayer != 0) ? In_Ptr->slayer : 1;
	Photon_Ptr->s = 0;
	Photon_Ptr->sleft = 0;
	Photon_Ptr->scatters = 0;
	Photon_Ptr->time = 0;

	Photon_Ptr->x = 0.0;
	Photon_Ptr->y = 0.0;
	Photon_Ptr->z = In_Ptr->sz;
	Photon_Ptr->ux = 0.0;
	Photon_Ptr->uy = 0.0;
	Photon_Ptr->uz = 1.0;

	Out_Ptr->Ai = 0.0;
	Out_Ptr->Tbi = 0.0;
	Out_Ptr->Tdi = 0.0;
	Out_Ptr->Rbi = 0.0;
	Out_Ptr->Rdi = 0.0;

	if (In_Ptr->source_type == isotropic) {
		LayerStru lstru = In_Ptr->layerspecs[Photon_Ptr->layer];

		/* to avoid scoring into Rb or Tb. */
		Photon_Ptr->scatters++;

		/* isotropically scatter the photon. */
		Spin(0.0, Photon_Ptr);

		/* glass layer. */
		if (lstru.mua == 0.0 && lstru.mus == 0.0) {
			/* total internal reflection */
			if (fabs(Photon_Ptr->uz) <= lstru.cos_crit0 && fabs(Photon_Ptr->uz) <= lstru.cos_crit1) {
				Photon_Ptr->alive = 0;
			}
		}
	}
}

/**************************************************************************
 *	Move the photon S away in the current layer of medium.
 ****/
void Hop(PhotonStru* Photon_Ptr, double S, double n) {
	Photon_Ptr->x += S * Photon_Ptr->ux;
	Photon_Ptr->y += S * Photon_Ptr->uy;
	Photon_Ptr->z += S * Photon_Ptr->uz;
	Photon_Ptr->time += S * n * ONE_OVER_C;
}

/**************************************************************************
 *	Pick a step size in dimensionless unit for a photon packet.
 *	If the member s is zero, make a new step size
 *	with: -log(rnd).
 *	Otherwise, finish the leftover in s.
 ****/
void SetStepSize(PhotonStru* Photon_Ptr) {
	/* make a new step. */
	if (Photon_Ptr->s == 0.0) {
		/* avoid zero. */
		double rnd;
		while ((rnd = RandomGen(1, 0, NULL)) <= 0.0) {
			;
		}
		Photon_Ptr->s = -log(rnd);
	}
}

/**************************************************************************
 *	Return the distance between the photon position to the boundary along
 *  the photon direction.
 ****/
double PathToBoundary(PhotonStru* Photon_Ptr, InStru* In_Ptr) {
	/* length to boundary. */
	short layer = Photon_Ptr->layer;
	double uz = Photon_Ptr->uz;

	/* Distance to the boundary. */
	double path = DBL_MAX; /* infinity */

	/* path > 0. */
	if (uz > 0.0) {
		path = (In_Ptr->layerspecs[layer].z1 - Photon_Ptr->z) / uz;
	}
	/* path > 0. */
	else if (uz < 0.0) {
		path = (In_Ptr->layerspecs[layer].z0 - Photon_Ptr->z) / uz;
	}

	return (path);
}

/**************************************************************************
 *	Drop photon weight inside the tissue (not glass).
 *
 *      The photon is assumed alive.
 *
 *	The weight drop is dw = w*mua/(mua+mus).
 *
 *	The dropped weight is assigned to the absorption array
 *	elements.
 ****/
void Drop(InStru* In_Ptr, PhotonStru* Photon_Ptr, OutStru* Out_Ptr) {
	/* Absorbed weight. */
	double x = Photon_Ptr->x;
	double y = Photon_Ptr->y;

	short layer = Photon_Ptr->layer;
	RecordStru record = In_Ptr->record;

	/* Update photon weight. */
	double mua = In_Ptr->layerspecs[layer].mua;
	double mus = In_Ptr->layerspecs[layer].mus;
	double dwa = Photon_Ptr->w * mua / (mua + mus);
	Photon_Ptr->w -= dwa;

	/* Compute array indices. */
	short iz = 0;
	short ir = 0;
	short it = 0;
	if (record.A_rzt || record.A_zt || record.A_z || record.A_rz) {
		if (Photon_Ptr->z >= In_Ptr->zm) {
			iz = In_Ptr->nz - 1;
		}
		else {
			iz = (short)(Photon_Ptr->z / In_Ptr->dz);
		}
	}

	if (record.A_rzt || record.A_zt || record.A_t) {
		if (Photon_Ptr->time >= In_Ptr->tm) {
			it = In_Ptr->nt - 1;
		}
		else {
			it = (short)floor(Photon_Ptr->time / In_Ptr->dt);
		}
	}

	/* Assign dwa to an absorption array element. */

	/* Scattered. */
	if (Photon_Ptr->scatters) {
		if (record.A_rzt || record.A_rz) {
			double temp = sqrt((x * x) + (y * y));
			if (temp >= In_Ptr->rm) {
				ir = In_Ptr->nr - 1;
			}
			else {
				ir = (short)(temp / In_Ptr->dr);
			}
		}
		if (record.A_rzt) {
			Out_Ptr->A_rzt[ir][iz][it] += dwa;
		}
		if (record.A_rz) {
			Out_Ptr->A_rz[ir][iz] += dwa;
		}
	}

	/* Ballistic. */
	else {
		if (record.A_rzt) {
			Out_Ptr->Ab_zt[iz][it] += dwa;
		}
		if (record.A_rz) {
			Out_Ptr->Ab_z[iz] += dwa;
		}
	}

	if (record.A_zt) {
		Out_Ptr->A_zt[iz][it] += dwa;
	}
	if (record.A_z) {
		Out_Ptr->A_z[iz] += dwa;
	}

	if (record.A_t) {
		Out_Ptr->A_t[it] += dwa;
	}
	Out_Ptr->Ai += dwa;
}

/**************************************************************************
 *	The photon weight is small, and the photon packet tries
 *	to survive a roulette.
 ****/
void Roulette(PhotonStru* Photon_Ptr) {
	/* already dead. */
	if (Photon_Ptr->w == 0.0) {
		Photon_Ptr->alive = 0;
	}

	/* survived the roulette. */
	else if (RandomGen(1, 0, NULL) < CHANCE) {
		Photon_Ptr->w /= CHANCE;
	}
	else {
		Photon_Ptr->alive = 0;
	}
}

/**************************************************************************
 *	Compute the Fresnel reflectance.
 *
 *	Make sure that the cosine of the incident angle ai
 *	is positive, and the case when the angle is greater
 *	than the critical angle is ruled out.
 *
 *	cai: cosine of the incident angle ai.
 *	cat_Ptr: pointer to the cosine of the transmission
 *			 angle at.
 *
 * 	Avoid trigonometric function operations as much as
 *	possible, because they are computation-intensive.
 *
 * ni:  incident refractive index
 * nt:  transmit refractive index
 * cai: cosine of angle ai. +
 ****/
double RFresnel(double ni, double nt, double cai, double* cat_Ptr) {
	/* Matched boundary. */
	if (ni == nt) {
		*cat_Ptr = cai;
		return 0.0;
	}
	/* Normal incidence. */
	if (1 - cai <= ONE_MINUS_COS_ZERO) {
		*cat_Ptr = cai;
		double r = (nt - ni) / (nt + ni);
		return r * r;
	}
	/* Very slant incidence. */
	if (cai < COS_90_DEGREES) {
		*cat_Ptr = 0.0;
		return 1.0;
	}
	/* General incidence. */
	else {
		/* Sine of the angles ai & at. */
		double sai = sqrt(1 - cai * cai);
		double sat = ni * sai / nt;

		/* Total internal reflection. */
		if (sat >= 1.0) {
			*cat_Ptr = 0.0;
			return 1.0;
		}

		/* cosine of at. */
		double cat = sqrt(1 - sat * sat);
		*cat_Ptr = cat;

		/* cosines of ai+at & ai-at. */
		double cap = cai * cat - sai * sat; /* c+ = cc - ss. */
		double cam = cai * cat + sai * sat; /* c- = cc + ss. */

		/* sines of ai+at & ai-at. */
		double sap = sai * cat + cai * sat; /* s+ = sc + cs. */
		double sam = sai * cat - cai * sat; /* s- = sc - cs. */

		/* arranged for speed. */
		return 0.5 * sam * sam * (cam * cam + cap * cap) / (sap * sap * cam * cam);
	}
	return 0.0;
}

/**************************************************************************
 *	Record the photon weight exiting the first layer(uz<0),
 *	to the reflection array.
 *
 *	Update the photon weight as well.
 ****/
void RecordR(double Refl, InStru* In_Ptr, PhotonStru* Photon_Ptr, OutStru* Out_Ptr) { /* reflectance. */
	double x = Photon_Ptr->x;
	double y = Photon_Ptr->y;
	RecordStru record = In_Ptr->record;

	short ir = 0;
	short ia = 0;
	short it = 0; /* index to r & angle. */
	if (record.Rd_rat || record.Rd_at || record.Rd_rt || record.Rd_t) {
		if (Photon_Ptr->time >= In_Ptr->tm) {
			it = In_Ptr->nt - 1;
		}
		else {
			it = (short)floor(Photon_Ptr->time / In_Ptr->dt);
		}
	}

	/* Scattered. */
	if (Photon_Ptr->scatters) {
		if (record.Rd_rat || record.Rd_rt || record.Rd_ra || record.Rd_r) {
			double temp = sqrt((x * x) + (y * y));
			if (temp >= In_Ptr->rm) {
				ir = In_Ptr->nr - 1;
			}
			else {
				ir = (short)(temp / In_Ptr->dr);
			}
		}
		if (record.Rd_rat || record.Rd_at || record.Rd_ra || record.Rd_a) {
			if ((ia = (short)(acos(-Photon_Ptr->uz) / In_Ptr->da) > In_Ptr->na - 1)) {
				ia = In_Ptr->na - 1;
			}
		}

		/* Assign photon weight to the reflection array element. */
		if (record.Rd_rat) {
			Out_Ptr->Rd_rat[ir][ia][it] += Photon_Ptr->w * (1.0 - Refl);
		}
		if (record.Rd_ra) {
			Out_Ptr->Rd_ra[ir][ia] += Photon_Ptr->w * (1.0 - Refl);
		}

		if (record.Rd_rt) {
			Out_Ptr->Rd_rt[ir][it] += Photon_Ptr->w * (1.0 - Refl);
		}
		if (record.Rd_r) {
			Out_Ptr->Rd_r[ir] += Photon_Ptr->w * (1.0 - Refl);
		}

		if (record.Rd_at) {
			Out_Ptr->Rd_at[ia][it] += Photon_Ptr->w * (1.0 - Refl);
		}
		if (record.Rd_a) {
			Out_Ptr->Rd_a[ia] += Photon_Ptr->w * (1.0 - Refl);
		}
		if (record.Rd_t) {
			Out_Ptr->Rd_t[it] += Photon_Ptr->w * (1.0 - Refl);
		}
		Out_Ptr->Rdi += Photon_Ptr->w * (1.0 - Refl);
	}

	/* Ballistic. */
	else {
		Out_Ptr->Rbi += Photon_Ptr->w * (1.0 - Refl);
	}

	Photon_Ptr->w *= Refl;
}

/**************************************************************************
 *	Record the photon weight exiting the last layer (uz > 0), no matter
 *  whether the layer is glass or not, to the transmittance array.
 *
 *	Update the photon weight as well.
 ****/
void RecordT(double Refl, InStru* In_Ptr, PhotonStru* Photon_Ptr, OutStru* Out_Ptr) {
	double x = Photon_Ptr->x;
	double y = Photon_Ptr->y;
	RecordStru record = In_Ptr->record;

	/* Index to r & angle. */
	short ir = 0;
	short ia = 0;
	short it = 0;
	if (record.Td_rat || record.Td_at || record.Td_rt || record.Td_t) {
		if (Photon_Ptr->time >= In_Ptr->tm) {
			it = In_Ptr->nt - 1;
		}
		else {
			it = (short)floor(Photon_Ptr->time / In_Ptr->dt);
		}
	}

	/* Scattered. */
	if (Photon_Ptr->scatters) {
		if (record.Td_rat || record.Td_rt || record.Td_ra || record.Td_r) {
			double temp = sqrt((x * x) + (y * y));
			if (temp >= In_Ptr->rm) {
				ir = In_Ptr->nr - 1;
			}
			else {
				ir = (short)(temp / In_Ptr->dr);
			}
		}
		if (record.Td_rat || record.Td_at || record.Td_ra || record.Td_a) {
			if ((ia = (short)(acos(Photon_Ptr->uz) / In_Ptr->da) > In_Ptr->na - 1)) {
				ia = In_Ptr->na - 1;
			}
		}

		/* Assign photon weight to the transmittance array element. */
		if (record.Td_rat) {
			Out_Ptr->Td_rat[ir][ia][it] += Photon_Ptr->w * (1.0 - Refl);
		}
		if (record.Td_ra) {
			Out_Ptr->Td_ra[ir][ia] += Photon_Ptr->w * (1.0 - Refl);
		}

		if (record.Td_rt) {
			Out_Ptr->Td_rt[ir][it] += Photon_Ptr->w * (1.0 - Refl);
		}
		if (record.Td_r) {
			Out_Ptr->Td_r[ir] += Photon_Ptr->w * (1.0 - Refl);
		}

		if (record.Td_at) {
			Out_Ptr->Td_at[ia][it] += Photon_Ptr->w * (1.0 - Refl);
		}
		if (record.Td_a) {
			Out_Ptr->Td_a[ia] += Photon_Ptr->w * (1.0 - Refl);
		}

		if (record.Td_t) {
			Out_Ptr->Td_t[it] += Photon_Ptr->w * (1.0 - Refl);
		}
		Out_Ptr->Tdi += Photon_Ptr->w * (1.0 - Refl);
	}

	/* Collimated. */
	else {
		Out_Ptr->Tbi += Photon_Ptr->w * (1.0 - Refl);
	}

	Photon_Ptr->w *= Refl;
}

/**************************************************************************
 *	Decide whether the photon will be transmitted or
 *	reflected on the upper boundary (uz<0) of the current
 *	layer.
 *
 *	If "layer" is the first layer, the photon packet will
 *	be partially transmitted and partially reflected if
 *	PARTIAL_REFLECTION is set to 1,
 *	or the photon packet will be either transmitted or
 *	reflected determined statistically if PARTIAL_REFLECTION
 *	is set to 0.
 *
 *	Record the transmitted photon weight as reflection.
 *
 *	If the "layer" is not the first layer and the photon
 *	packet is transmitted, move the photon to "layer-1".
 *
 *	Update the photon parmameters.
 ****/
void CrossUpOrNot(InStru* In_Ptr, PhotonStru* Photon_Ptr, OutStru* Out_Ptr) {
	/* z directional cosine. */
	double uz = Photon_Ptr->uz;

	/* cosines of transmission alpha. always * +. */
	double uz1 = 0.0;

	short layer = Photon_Ptr->layer;
	double ni = In_Ptr->layerspecs[layer].n;
	double nt = In_Ptr->layerspecs[layer - 1].n;

	/* Get reflectance. If 1.0, then total internal reflection. */
	double r = (-uz <= In_Ptr->layerspecs[layer].cos_crit0) ? 1.0 : RFresnel(ni, nt, -uz, &uz1);

#if PARTIAL_REFLECTION
	if (layer == 1 && r < 1.0) { /* partially transmitted. */
		Photon_Ptr->uz = -uz1;   /* escaped photon. */
		RecordR(r, In_Ptr, Photon_Ptr, Out_Ptr);
		Photon_Ptr->uz = -uz; /* reflected photon. */
	}
	else if ((double)RandomGen(1, 0, NULL) > r) { /* transmitted to layer-1. */
		Photon_Ptr->layer--;
		Photon_Ptr->ux *= ni / nt;
		Photon_Ptr->uy *= ni / nt;
		Photon_Ptr->uz = -uz1;
	}
	else /* reflected. */
		Photon_Ptr->uz = -uz;
#else
	/* transmitted to layer-1. */
	if (RandomGen(1, 0, NULL) > r) {
		if (layer == 1) {
			Photon_Ptr->uz = -uz1;
			RecordR(0.0, In_Ptr, Photon_Ptr, Out_Ptr);
			Photon_Ptr->alive = 0; /* escaped. */
		}
		else {
			Photon_Ptr->layer--;
			Photon_Ptr->ux *= ni / nt;
			Photon_Ptr->uy *= ni / nt;
			Photon_Ptr->uz = -uz1;
		}
	}
	/* reflected. */
	else {
		Photon_Ptr->uz = -uz;
	}
#endif
}

/**************************************************************************
 *	Decide whether the photon will be transmitted  or be
 *	reflected on the bottom boundary (uz>0) of the current
 *	layer.
 *
 *	If the photon is transmitted, move the photon to
 *	"layer+1". If "layer" is the last layer, record the
 *	transmitted weight as transmittance. See comments for
 *	CrossUpOrNot.
 *
 *	Update the photon parmameters.
 ****/
void CrossDnOrNot(InStru* In_Ptr, PhotonStru* Photon_Ptr, OutStru* Out_Ptr) {
	/* z directional cosine. */
	double uz = Photon_Ptr->uz;

	/* cosines of transmission alpha. */
	double uz1 = 0.0;

	short layer = Photon_Ptr->layer;
	double ni = In_Ptr->layerspecs[layer].n;
	double nt = In_Ptr->layerspecs[layer + 1].n;

	/* Get reflectance. If 1.0, then total internal reflection. */
	double r = (uz <= In_Ptr->layerspecs[layer].cos_crit1) ? 1.0 : RFresnel(ni, nt, uz, &uz1);

#if PARTIAL_REFLECTION
	if (layer == In_Ptr->num_layers && r < 1.0) {
		Photon_Ptr->uz = uz1;
		RecordT(r, In_Ptr, Photon_Ptr, Out_Ptr);
		Photon_Ptr->uz = -uz;
	}
	else if ((double)RandomGen(1, 0, NULL) > r) { /* transmitted to layer+1. */
		Photon_Ptr->layer++;
		Photon_Ptr->ux *= ni / nt;
		Photon_Ptr->uy *= ni / nt;
		Photon_Ptr->uz = uz1;
	}
	else /* reflected. */
		Photon_Ptr->uz = -uz;
#else
	/* transmitted to layer+1. */
	if (RandomGen(1, 0, NULL) > r) {
		if (layer == In_Ptr->num_layers) {
			Photon_Ptr->uz = uz1;
			RecordT(0.0, In_Ptr, Photon_Ptr, Out_Ptr);

			/* escaped. */
			Photon_Ptr->alive = 0;
		}
		else {
			Photon_Ptr->layer++;
			Photon_Ptr->ux *= ni / nt;
			Photon_Ptr->uy *= ni / nt;
			Photon_Ptr->uz = uz1;
		}
	}
	/* reflected. */
	else {
		Photon_Ptr->uz = -uz;
	}
#endif
}

/**************************************************************************
 *	Set a step size if the previous step has finished.
 *
 *	If the step size fits in the current layer, move the photon,
 *	drop some weight, choose a new photon direction for propagation.
 *
 *	If the step size is long enough for the photon to
 *	hit an interface, this step is divided into three steps.
 *	First, move the photon to the boundary free of
 *	absorption or scattering.
 *	Second, update the step size to the unfinished step size.
 *	Third, decide whether the photon is reflected or transmitted.
 ****/
void HopDropSpin(InStru* In_Ptr, PhotonStru* Photon_Ptr, OutStru* Out_Ptr) {
	LayerStru layer_struct = In_Ptr->layerspecs[Photon_Ptr->layer];
	double mut = layer_struct.mua + layer_struct.mus;
	SetStepSize(Photon_Ptr);

	/* distance between photon and boundary cm. */
	double path = PathToBoundary(Photon_Ptr, In_Ptr);

	/* hit boundary. */
	if (path * mut <= Photon_Ptr->s) {
		/* move to boundary plane. */
		Hop(Photon_Ptr, path, layer_struct.n);

		/* update s. */
		Photon_Ptr->s -= path * mut;

		if (Photon_Ptr->uz < 0.0) {
			CrossUpOrNot(In_Ptr, Photon_Ptr, Out_Ptr);
		}
		else {
			CrossDnOrNot(In_Ptr, Photon_Ptr, Out_Ptr);
		}
	}
	/* fit in layer. */
	else {
		Hop(Photon_Ptr, Photon_Ptr->s / mut, layer_struct.n);

		/* update s. */
		Photon_Ptr->s = 0;
		Drop(In_Ptr, Photon_Ptr, Out_Ptr);
		Spin(In_Ptr->layerspecs[Photon_Ptr->layer].g, Photon_Ptr);
		Photon_Ptr->scatters++;
	}
}

/**************************************************************************
 *	Trace a photon, then compute the 0D constants including A, R, T and
 *	their standard errors.
 ****/
void TracePhoton(InStru* In_Ptr, PhotonStru* Photon_Ptr, OutStru* Out_Ptr) {
	do {
		HopDropSpin(In_Ptr, Photon_Ptr, Out_Ptr);
		if (Photon_Ptr->alive && Photon_Ptr->w < In_Ptr->Wth) {
			Roulette(Photon_Ptr);
		}
	}
	while (Photon_Ptr->alive);

	Out_Ptr->A += Out_Ptr->Ai;
	Out_Ptr->Ae += Out_Ptr->Ai * Out_Ptr->Ai;

	Out_Ptr->Tb += Out_Ptr->Tbi;
	Out_Ptr->Tbe += Out_Ptr->Tbi * Out_Ptr->Tbi;
	Out_Ptr->Td += Out_Ptr->Tdi;
	Out_Ptr->Tde += Out_Ptr->Tdi * Out_Ptr->Tdi;

	Out_Ptr->Rb += Out_Ptr->Rbi;
	Out_Ptr->Rbe += Out_Ptr->Rbi * Out_Ptr->Rbi;
	Out_Ptr->Rd += Out_Ptr->Rdi;
	Out_Ptr->Rde += Out_Ptr->Rdi * Out_Ptr->Rdi;
}
