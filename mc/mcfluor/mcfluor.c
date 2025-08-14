/*************
 * mcfluor.c
 * A calling program that
 *	1. sets parameters for Monte Carlo runs to simulate fluorescence
 *	2. calls the mcsub() routine to simulate
 *		excition into tissue
 *		emission due to uniform background fluorophore
 *		emission due to off-center fluorophore
 *	3. saves the results into output files
 *		mcOUT101.dat = excitation
 *		mcOUT102.dat = emission of uniform background fluorophore
 *		mcOUT103.dat = emission of off-center fluorophore at a +x position
 *		mcOUT104.dat = emission of off-center fluorophore at a -x position
 * USES
 *	mcsubLIB.c
 *	mcsubLIB.h
 * Output can be read by MATLAB program  <lookmcsubfluor.m>
 *
 * Oct. 29, 2007. Steven L. Jacques
 *************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "mcsubLIB.h"

/*************************/
/**** USER CHOICES *******/
/*************************/
#define BINS        101      /* number of bins, NZ and NR, for z and r */
/*************************/
/*************************/


/**********************
 * MAIN PROGRAM
 *********************/
int main() {

/*************************/
/**** USER CHOICES ******/
/*************************/
/* number of file for saving */
int      Nfile = 1;		/* saves as mcOUTi.dat, where i = Nfile */
/* excitation */
double muax = 1.0;		/* excitation absorption coeff. [cm^-1] */
double musx = 100.0;	/* excitation scattering coeff. [cm^-1] */
double gx = 0.90;		/* excitation anisotropy [dimensionless] */
double mua  = 1.0;      /* excitation absorption coeff. [cm^-1] */
double mus  = 100;      /* excitation scattering coeff. [cm^-1] */
double g    = 0.90;     /* excitation anisotropy [dimensionless] */
double n1 = 1.33;		/* refractive index of medium */
double n2 = 1.00;		/* refractive index outside medium */
short  mcflag = 0;		/* 0 = collimated, 1 = focused Gaussian, 2 = isotropic pt */
double radius = 0.0;	/* used if mcflag = 0 or 1 */
double waist = 0.0;		/* used if mcflag = 1 */
double zfocus = 0.0;	/* used if mcflag = 1 */
double xs = 0.0;		/* used if mcflag = 2 */
double ys = 0.0;		/* used if mcflag = 2 */
double zs = 0.0;		/* used if mcflag = 2 */
int    boundaryflag = 1;/* 0 = infinite medium, 1 = air/tissue surface boundary */
double S;				/* specular reflectance at air/tissue boundary */
double A;				/* total fraction of light absorbed by tissue */
double E;				/* total fraction of light escaping tissue */
short  PRINTOUT = 1;	/* PRINTOUT = 1 enables printout of # of photons launched */
/* background fluorescence */
double muaf = 5.0;		/* fluorescence absorption coeff. [cm^-1] */
double musf = 50.0;		/* fluorescence scattering coeff. [cm^-1] */
double gf = 0.0;		/* fluorescence anisotropy [dimensionless] */
double eC = 1.0;		/* ext. coeff. x conc of fluor [cm^-1] */
double Y = 1.0;			/* Energy yield for fluorescence [W/W] */
/* heterogeneity */
double xh = 0.2;		/* heterogeneity */
double yh = 0.0;		/* heterogeneity */
double zh = 0.3;		/* heterogeneity */
double heC = 0.1;		/* extra eC of heterogeneity */
double hY = 1.0;		/* energy yield of heterogeneity */
double hrad = 0.01;		/* radius of spherical heterogeneity */
/* other parameters */
double Nruns = 0.1;		/* number photons launched = Nruns x 1e6 */
double dr = 0.0100;		/* radial bin size [cm] */
double dz = 0.0100;		/* depth bin size [cm] */
/*************************/
/*************************/
char label[1];
double PI = 3.1415926;
double Nphotons;
long ir, iz, iir, iiz, ii;
double temp, temp3, temp4, r, r1, r2; /* dummy variables */
double timeperEx, timeperEm;  /* min per photon */
double start_time, finish_time1, finish_time2, finish_time3; /* for clock() */
double timeA, timeB;
time_t now;
double *Jx, *Jf, *temp1;
double **Fx, **Ff, **temp2;
long NR = BINS; /* number of radial bins */
long NZ = BINS; /* number of depth bins */

Jx		= AllocVector(1,BINS);
Jf		= AllocVector(1,BINS);
temp1	= AllocVector(1,BINS);
Fx		= AllocMatrix(1, BINS, 1, BINS); /* for absorbed excitation */
Ff		= AllocMatrix(1, BINS, 1, BINS); /* for absorbed fluor */
temp2	= AllocMatrix(1, BINS, 1, BINS); /* dummy matrix */


start_time = clock();
now = time(NULL);
printf("%s\n", ctime(&now));

if (1) { /* Switch printout ON=1 or OFF=0 */
	/* print out summary of parameters to user */
	printf("----- USER CHOICES -----\n");
	printf("EXCITATION\n");
	printf("muax = %1.3f\n",muax);
	printf("musx = %1.3f\n",musx);
	printf("gx = %1.3f\n",gx);
	printf("n1 = %1.3f\n",n1);
	printf("n2 = %1.3f\n",n2);
	printf("mcflag = %d\n",mcflag);
	printf("radius = %1.4f\n",radius);
	printf("waist = %1.4f\n",waist);
	printf("zfocus = %1.4f\n",radius);
	printf("xs = %1.4f\n",xs);
	printf("ys = %1.4f\n",ys);
	printf("zs = %1.4f\n",zs);
	printf("BACKGROUND FLUORESCENCE\n");
	printf("muaf = %1.3f\n",muaf);
	printf("musf = %1.3f\n",musf);
	printf("gf = %1.3f\n",gf);
	printf("eC = %1.3f\n",eC);
	printf("Y = %1.3f\n",Y);
	printf("FLUORESCENT HETEROGENEITY\n");
	printf("xh = %1.4f\n",xh);
	printf("yh = %1.4f\n",yh);
	printf("zh = %1.4f\n",zh);
	printf("heC = %1.4f\n",heC);
	printf("hY = %1.4f\n",hY);
	printf("hrad = %1.4f\n",hrad);
	printf("OTHER\n");
	printf("Nruns = %1.1f @1e6 photons/run\n", Nruns);
	printf("dr = %1.4f\n",dr);
	printf("dz = %1.4f\n",dz);
	printf("---------------\n\n");
}

/* Initialize arrays */
for (ir=1; ir<=NR; ir++) {
	Jx[ir] = 0.0;
	Jf[ir] = 0.0;
	temp1[ir] = 0.0;
	for (iz=1; iz<=NR; iz++) {
		Fx[iz][ir] = 0.0;
		Ff[iz][ir] = 0.0;
		temp2[iz][ir] = 0.0;
	}
}

//printf("CLOCKS_PER_SEC = %ld\n", CLOCKS_PER_SEC);
/*********************
* Time estimate for completion
*********************/
/* EXCITATION */
timeA = clock();
PRINTOUT = 0;
mcsub(	muax, musx, gx, n1, n2,   /* CALL THE MONTE CARLO SUBROUTINE */
	NR, NZ, dr, dz, 999,
	mcflag, xs, ys, zs, boundaryflag,
	radius, waist, zfocus,
	Jx, Fx, &S, &A, &E,  /* returns Jx, Fx, S, A, E */
	PRINTOUT);         
timeB = clock();
timeperEx = (timeB - timeA)/CLOCKS_PER_SEC/60/999; /* min per photon EX */
printf("%5.3e min/EX photon \n", timeperEx);

/* EMISSION */
temp = NZ/2*dz; /* zs is midway */
timeA = clock();
PRINTOUT = 0;
mcsub(	muaf, musf, gf, n1, n2,   /* CALL THE MONTE CARLO SUBROUTINE */
	NR, NZ, dr, dz, 999,
	2, 0, 0, temp, boundaryflag,
	radius, waist, zfocus,
	Jx, Fx, &S, &A, &E,         /* returns Jx, Fx, S, A, E */
	PRINTOUT);
timeB = clock();
timeperEm = (timeB - timeA)/CLOCKS_PER_SEC/60/999; /* min per photon EM */
printf("%5.3e min/EM photon\n", timeperEm);

printf("\nTotal estimated completion time = %5.2f min\n\n", \
	timeperEx*1e6*Nruns + timeperEm*NZ*NR*100 + timeperEm*1e6*Nruns);

/*********************
* EXCITATION
*********************/
Nphotons = 1e6*Nruns;
printf("EXCITATION\n");
printf("est. completion time = %5.2f min\n\n", timeperEx*Nphotons);
PRINTOUT = 1;
mcsub(	muax, musx, gx, n1, n2,   /* CALL THE MONTE CARLO SUBROUTINE */
		NR, NZ, dr, dz, Nphotons,
		mcflag, xs, ys, zs, boundaryflag,
		radius, waist, zfocus,
		Jx, Fx, &S, &A, &E,         /* returns Jx, Fx, S, A, E */
		PRINTOUT);

/* SAVE EXCITATION */
Nfile = 101; // mcOUT101 = excitation
SaveFile(	Nfile, Jx, Fx, S, A, E,       // save to "mcOUTi.dat", i = Nfile
			mua, mus, g, n1, n2, 
			mcflag, radius, waist, xs, ys, zs,
			NR, NZ, dr, dz, Nphotons);  

printf("------------------------------------------------------\n");
finish_time1 = clock();
printf("------------------------------------------------------\n");
printf("Elapsed Time for excitation = %5.3f min\n",
(double)(finish_time1-start_time)/CLOCKS_PER_SEC/60);
now = time(NULL);
printf("%s\n", ctime(&now));


/*********************
* BACKGROUND FLUORESCENCE
*********************/
Nphotons = 100;  // photons launched per iso point in background
printf("BACKGROUND FLUORESCENCE\n");
printf("est. completion time = %5.2f min\n\n", timeperEm*NZ*NR*Nphotons);

/* Accumulate Monte Carlo fluorescence due to each bin source
weighted by strength of absorbed excitation in each bin
source, Fx[iz][ir]. Do not include the last bins [NZ][NR]
which are for overflow. */
temp = 0.0; /* count total number of photons launched */
for (ir=1; ir<NR; ir++) {
	temp3 = 0.0; /* number of photons launched in current [ir] row */
	temp4 = eC*Y*2*PI*(ir - 0.5)*dr*dr*dz; /* fluorescence conversion */

	for (iz=1; iz<NZ; iz++) {
		temp += Nphotons;
		temp3 += Nphotons;
	
		/* Set to launch as isotropic point source at bin [iz][ir] */
		r2 = ir*dr;
		r1 = (ir-1)*dr;
		r = 2.0/3*(r2*r2 + r2*r1 + r1*r1)/(r2 + r1);
		xs = r;
		ys = 0;
		zs = (iz - 0.5)*dz;
		/* CALL THE MONTE CARLO SUBROUTINE */
		mcflag = 2; // isotropic point source of fluorescence
		PRINTOUT = 0; // disable printout
		mcsub(	muaf, musf, gf, n1, n2,   /* CALL THE MONTE CARLO SUBROUTINE */
				NR, NZ, dr, dz, Nphotons,
				mcflag, xs, ys, zs, boundaryflag,
				radius, waist, zfocus,
				temp1, temp2, &S, &A, &E,
				PRINTOUT);
				
		/* Accumulate Monte Carlo results */
		for (iir=1; iir<=NR; iir++) {
			Jf[iir] += temp1[iir]*Fx[iz][ir]*temp4;
			for (iiz=1; iiz<=NZ; iiz++)
				Ff[iiz][iir] += temp2[iiz][iir]*Fx[iz][ir]*temp4;
		} /* end iir */
	} /* end iz */
	
	/* Print out progress for user */
	if (ir < 10) {
		printf("%1.0f fluor photons \t@ ir = %ld \t total = %1.0f\n",
		temp3,ir, temp);
		}
	else if (fmod((double)ir,10)==0) {
		printf("%1.0f fluor photons \t@ ir = %ld \t total = %1.0f\n",
		temp3,ir, temp);
		}
} /* end ir */

printf("%1.0f fluor photons \t@ ir = %ld \t total = %1.0f\n", temp3,ir, temp);
printf("%1.0f fluorescent photons total\n",temp);
finish_time2 = clock();
printf("Elapsed Time for background fluorescence = %5.3f min\n",
(double)(finish_time2-finish_time1)/CLOCKS_PER_SEC/60);

/* SAVE BACKGROUND FLUORESCENCE */
Nfile = 102; // mcOUT102 = background fluorescence
SaveFile(	Nfile, Jf, Ff, S, A, E,       // save to "mcOUTi.dat", i = Nfile
			mua, mus, g, n1, n2, 
			mcflag, radius, waist, xs, ys, zs,
			NR, NZ, dr, dz, Nphotons);  
printf("------------------------------------------------------\n");

/******************************
* HETEROGENEOUS FLUORESCENCE *
******************************/
Nphotons = Nruns*1e6;
printf("FLUORESCENT HETEROGENEITY\n");
printf("est. completion time = %5.3f min\n\n", timeperEm*Nphotons);

/* Fluorescent heterogeneity at (xh,yh,zh)
* as a small sphere with specified radius.
* Note that results are Jf(x), F(z,x) in y=0 plane.
* Usually will let yh = 0.
* Launch at xs = 0, ys = 0, zs = zs */
/* CALL THE MONTE CARLO SUBROUTINE -> fluorescent impulse response */
xs = 0; ys = 0; zs = zh;
mcflag = 2; /* isotropic pt source */
PRINTOUT = 1;
mcsub(	muaf, musf, gf, n1, n2,   /* CALL THE MONTE CARLO SUBROUTINE */
		NR, NZ, dr, dz, Nphotons,
		mcflag, xs, ys, zs, boundaryflag,
		radius, waist, zfocus,
		temp1, temp2, &S, &A, &E,          /* returns Jx, Fx, S, A, E */
		PRINTOUT);

// POSITIVE Z side of response
/* Initialize arrays Jf[ ] and Ff[ ][ ] */
for (ir=1; ir<=NR; ir++) {
	Jf[ir] = 0.0;
	for (iz=1; iz<=NR; iz++)
		Ff[iz][ir] = 0.0;
}
/* Convolve impulse response against fluorescent source at (xh,yh,zh).
* Weight by Fx[iz][ir]*4/3*PI*hrad*hrad*hrad*heC*hY
* which is fluorescent power of heterogeneity [W/W]. */
temp4 = 4.0/3*PI*hrad*hrad*hrad*heC*hY;
ir = (long)(sqrt( xh*xh + yh*yh )/dr) + 1; /* ir, heterog-origin */
iz = (long)(zh/dz) + 1; /* iz, for Fx[iz][ir] */
for (iir=1; iir<=NR; iir++) { /* for Jf[iir], Ff[iiz][ir] */
	r2 = iir*dr; /* radial position of observer */
	//r1 = (iir-1)*dr;
	//r = 2.0/3*(r2*r2 + r2*r1 + r1*r1)/(r2 + r1);
	r = sqrt( (r2 - xh)*(r2 - xh) + yh*yh ); /* heterog-observer */
	ii = (long)(r/dr) + 1; /* for temp1[ii], temp2[iz][ii] */
	if (ii > NR) ii = NR;
	Jf[iir] += temp1[ii]*Fx[iz][ir]*temp4;
	for (iiz=1; iiz<=NZ; iiz++) /* for Ff[iiz][ir], temp2[iz][ii] */
		Ff[iiz][iir] += temp2[iiz][ii]*Fx[iz][ir]*temp4;
} /* end iir */

/* SAVE HETEROGENEITY FLUORESCENCE */
Nfile = 103; // mcOUT103 = positive z side of heterogeneous fluorescence
SaveFile(	Nfile, Jf, Ff, S, A, E,  // save to "mcOUTi.dat", i = Nfile
			mua, mus, g, n1, n2, 
			mcflag, radius, waist, xs, ys, zs,
			NR, NZ, dr, dz, Nphotons);  

// NEGATIVE Z side of response
/* Initialize arrays Jf[ ] and Ff[ ][ ] */
for (ir=1; ir<=NR; ir++) {
	Jf[ir] = 0.0;
	for (iz=1; iz<=NR; iz++)
		Ff[iz][ir] = 0.0;
}
/* Convolve impulse response against fluorescent source at (xh,yh,zh).
* Weight by Fx[iz][ir]*4/3*PI*hrad*hrad*hrad*heC*hY
* which is fluorescent power of heterogeneity [W/W]. */
temp4 = 4.0/3*PI*hrad*hrad*hrad*heC*hY;
ir = (long)(sqrt( xh*xh + yh*yh )/dr) + 1; /* ir, heterog-origin */
iz = (long)(zh/dz) + 1; /* iz, for Fx[iz][ir] */
for (iir=1; iir<=NR; iir++) { /* for Jf[iir], Ff[iiz][ir] */
	r2 = iir*dr; /* radial position of observer */
	//r1 = (iir-1)*dr;
	//r = 2.0/3*(r2*r2 + r2*r1 + r1*r1)/(r2 + r1);
	r = sqrt( (r2 + xh)*(r2 + xh) + yh*yh ); /* heterog-observer */
	ii = (long)(r/dr) + 1; /* for temp1[ii], temp2[iz][ii] */
	if (ii > NR) ii = NR;
	Jf[iir] += temp1[ii]*Fx[iz][ir]*temp4;
	for (iiz=1; iiz<=NZ; iiz++) /* for Ff[iiz][ir], temp2[iz][ii] */
		Ff[iiz][iir] += temp2[iiz][ii]*Fx[iz][ir]*temp4;
} /* end iir */

/* SAVE HETEROGENEITY FLUORESCENCE */
Nfile = 104; // mcOUT104 = negative z side of heterogeneous fluorescence
SaveFile(	Nfile, Jf, Ff, S, A, E,  // save to "mcOUTi.dat", i = Nfile
			mua, mus, g, n1, n2, 
			mcflag, radius, waist, xs, ys, zs,
			NR, NZ, dr, dz, Nphotons);  

finish_time3 = clock();
printf("Elapsed Time for fluorescent heterogeneity = %5.3f min\n", \
	(double)(finish_time3-finish_time2)/CLOCKS_PER_SEC/60);
printf("------------------------------------------------------\n");
printf("------------------------------------------------------\n");
now = time(NULL);
printf("%s\n", ctime(&now));
printf("\nEstimated total completion time = %5.2f min\n\n", \
	timeperEx*1e6*Nruns + timeperEm*NZ*NR*100 + timeperEm*1e6*Nruns);
printf("Actual total elapsed time = %5.2f min\n", \
	(double)(finish_time3-start_time)/CLOCKS_PER_SEC/60);

FreeVector(Jx, 1, BINS);
FreeVector(Jf, 1, BINS);
FreeVector(temp1, 1, BINS);
FreeMatrix(Fx, 1, BINS, 1, BINS);
FreeMatrix(Ff, 1, BINS, 1, BINS);
FreeMatrix(temp2, 1, BINS, 1, BINS);
return(1);
}
