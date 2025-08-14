
/********************************************
 *  trmc.c    , in ANSI Standard C programing language
 *  Time-Resolved Monte Carlo.  Dec. 1998.
 *
 *  Monte Carlo simulation of time-resolved photon fluence rate
 *    in response to an impulse of energy at time zero delivered as an
 *    isotropic point source.  Photon propagation is 3D in spherical
 *    coordinates. The instantaneous relative fluence rate, 
 *    F(r,t)/Uo [cm^-2 s^-1]  where F(r,t) = [J cm^-2 s^-1] and Uo = [J].
 *    For the calculation, the value of Uo is assumed to be unity [1 J].
 *    Propagation is in an infinite homogeneous medium
 *    with no boundaries. This program is a minimal Monte Carlo 
 *    program.
 *
 *  Results are recorded as F[ir][it]/Uo for each of four timepoints
 *    t = T[it], it = 0-3, and 100 radial positions, r = (ir + 0.5)*dr.
 *    Results are placed in output file "trmc.out" as columns:
 *    r [cm]   F/Uo @ time#1   F/Uo @ time#2   F/Uo @ time#3   F/Uo @ time#4
 *
 *  by Steven L. Jacques based on prior collaborative work 
 *    with Lihong Wang, Scott Prahl, and Marleen Keijzer.
 *    partially funded by the NIH (R29-HL45045, 1991-1997) and  
 *    the DOE (DE-FG05-91ER617226, DE-FG03-95ER61971, 1991-1999).
 *
 **********/

#include <math.h>
#include <stdio.h>

#define	PI          3.1415926
#define	LIGHTSPEED	2.997925E10 /* in vacuo speed of light [cm/s] */
#define ALIVE       1   		/* if photon not yet terminated */
#define DEAD        0    		/* if photon is to be terminated */
#define THRESHOLD   0.01		/* used in roulette */
#define CHANCE      0.1  		/* used in roulette */
#define COS90D      1.0E-6
     /* If cos(theta) <= COS90D, theta >= PI/2 - 1e-6 rad. */
#define ONE_MINUS_COSZERO 1.0E-12
     /* If 1-cos(theta) <= ONE_MINUS_COSZERO, fabs(theta) <= 1e-6 rad. */
     /* If 1+cos(theta) <= ONE_MINUS_COSZERO, fabs(PI-theta) <= 1e-6 rad. */
#define SIGN(x)           ((x)>=0 ? 1:-1)
#define InitRandomGen    (double) RandomGen(0, 1, NULL)
     /* Initializes the seed for the random number generator. */     
#define RandomNum        (double) RandomGen(1, 0, NULL)
     /* Calls for a random number from the randum number generator. */

/* DECLARE FUNCTION */
double RandomGen(char Type, long Seed, long *Status);  
     /* Random number generator */


int main() {

/* Propagation parameters */
double	x, y, z;    /* photon position */
double	ux, uy, uz; /* photon trajectory as cosines */
double  uxx, uyy, uzz;	/* temporary values used during SPIN */
double	s;          /* step sizes. s = -log(RND)/mus [cm] */
double	costheta;   /* cos(theta) */
double  sintheta;   /* sin(theta) */
double	cospsi;     /* cos(psi) */
double  sinpsi;     /* sin(psi) */
double	psi;        /* azimuthal angle */
double	i_photon;   /* current photon */
short   photon_status;  /* flag = ALIVE=1 or DEAD=0 */

/* time and pathlength variables */
short	NTpts;      /* number of timepoints */
double  T[4];       /* timepoints at which concentration is mapped */
double  LT[4];      /* total pathlengths at timepoints */
double  Lmax;       /* maximum timepoint and pathlength considered */
double  L;          /* total photon pathlength */
short	it;         /* index to timepoint */

/* other variables */
double  c;          /* speed of light in medium */
double	Csph[101][4];  /* photon concentration C[ir=0..100][it=0-3] */
double	F[101][4];  /* fluence rate */
double	mus;        /* scattering coefficient [cm^-1] */
double	g;          /* anisotropy [-] */
double	nt;         /* tissue index of refraction */
double	Nphotons;   /* number of photons in simulation */
short	NR;         /* number of radial positions */
double	radial_size;  /* maximum radial size */
double	r;          /* radial position */
double  dr;         /* radial bin size */
short	ir;         /* index to position */
double  shellvolume;  /* volume of shell at radial position r */

/* dummy variables */
double  rnd;            /* assigned random value 0-1 */
double  s1, x1, y1, z1; /* temporary values */
short	i, j;           /* dummy indices */
double	u, temp, sum;   /* dummy variables */
FILE*	target;         /* pointer to output file */

printf("start\n");


/**** INPUT
*   Input the number of photons
*   Input the optical properties
*   Input the bin and array sizes 
*   Input the time settings 
*****/

Nphotons    = 1E3; /* set number of photons in simulation */
/* Optical properties: */
mus         = 100;  /* cm^-1 */
g           = 0.90;  
nt          = 1.33;
/* Spatial bin parameters */
NR          = 100;	 /* set number of bins.  */
radial_size = 6.0;   /* cm, total range over which bins extend */
dr          = radial_size/NR;  /* cm */
/* Time point parameters: */
NTpts    = 4;        /* number of timepoints */
T[0]        = 0.050e-9; /* 1st timepoint */
T[1]        = 0.100e-9; /* 2nd timepoint */
T[2]        = 0.500e-9; /* 3rd timepoint */
T[3]        = 1e-9;     /* 4   timepoint */

/**** INITIALIZATIONS 
*****/
dr = radial_size/NR;      /* cm,  incremental bin size */
c = LIGHTSPEED/nt;        /* lightspeed within medium */
i_photon = 0;             /* zero photon counter */
InitRandomGen;            /* initialize seed of random number generator */

for (it=0; it<NTpts; it++) 
	LT[it] = T[it]*c;     /* pathlengths LT[it] associated with T[it] */
Lmax  = LT[NTpts-1];      /* last total pathlength considered */

for (ir=0; ir<=NR; ir++)  /* clear data array */
	for (it=0; it<NTpts; it++)
   		Csph[ir][it] = 0;
   		
   
/**** RUN
*   Launch N photons, initializing each one before propagation.
*****/
do {


/**** LAUNCH 
*   Initialize photon position and trajectory.
*   Implements an isotropic point source.
*****/
i_photon += 1;	            /* increment photon count */

photon_status = ALIVE;      /* Launch an ALIVE photon */
L = 0;                      /* set photon's accumulated pathlength to zero */
it = 0;                     /* set pointer to timepoint T[1] */

x = 0;                      /* Set photon position to origin. */
y = 0;
z = 0;

/* Randomly set photon trajectory to yield an isotropic source. */
costheta = 2.0*RandomNum - 1.0;   
sintheta = sqrt(1.0 - costheta*costheta);	/* sintheta is always positive */
psi = 2.0*PI*RandomNum;
ux = sintheta*cos(psi);
uy = sintheta*sin(psi);
uz = costheta;


/* HOP/DROP_SPIN_CHECK DO-WHILE LOOP
*   Propagate one photon until it dies.
*******/
do {


/**** HOP
*   Take step to new position
*   s = stepsize
*   ux, uy, uz are cosines of current photon trajectory
*****/
  while ((rnd = RandomNum) <= 0.0);   /* yields 0 < rnd <= 1. 
                                         If rnd == 0, retry. */
  s = -log(rnd)/mus;                  /* Step size.  Note: log() is base e */

  if (L + s >= LT[it]) {
		s1 = LT[it] - L;              /* partial step to   position at T[it] */
		x1 = x + s1*ux;               /* move to temporary position at T[it] */
		y1 = y + s1*uy;
		z1 = z + s1*uz;
  	
		/******************* DROP *********
		*	Register photon position into local bin C[ir][it].
		*	Are scoring the concentration of photons at time T[it].
		*   Any loss of photon energy due to absorption can be later accounted for
		*	  by the term exp(-mua*c*T[it]).
		*****/
		r = sqrt(x1*x1 + y1*y1 + z1*z1); /* current spherical radial position */
		ir = (short)(r/dr);           /* ir = index to spatial bin */
		if (ir >= NR) ir = NR;        /* last bin is for overflow */
		Csph[ir][it] += 1;            /* DROP photon into bin */
		
		it += 1;                      /* increment pointer to next timepoint */		
		}

  x += s*ux;	/* Update positions by taking full step s. */
  y += s*uy;
  z += s*uz;
  L += s;       /* Update total photon pathlength */
  

/**** SPIN 
*   Scatter photon into new trajectory defined by theta and psi.
*   Theta is specified by cos(theta), which is determined 
*   based on the Henyey-Greenstein scattering function.
*   Convert theta and psi into cosines ux, uy, uz. 
*****/
  /* Sample for costheta */
  rnd = RandomNum;
     if (g == 0.0)
        costheta = 2.0*rnd - 1.0;
     else {
        double temp = (1.0 - g*g)/(1.0 - g + 2*g*rnd);
        costheta = (1.0 + g*g - temp*temp)/(2.0*g);
        }
  sintheta = sqrt(1.0 - costheta*costheta); /* sqrt() is faster than sin(). */

  /* Sample psi. */
  psi = 2.0*PI*RandomNum;
  cospsi = cos(psi);
  if (psi < PI)
    sinpsi = sqrt(1.0 - cospsi*cospsi);     /* sqrt() is faster than sin(). */
  else
    sinpsi = -sqrt(1.0 - cospsi*cospsi);

  /* New trajectory. */
  if (1 - fabs(uz) <= ONE_MINUS_COSZERO) {      /* close to perpendicular. */
    uxx = sintheta * cospsi;
    uyy = sintheta * sinpsi;
    uzz = costheta * SIGN(uz);   /* SIGN() is faster than division. */
    } 
  else {					/* usually use this option */
    temp = sqrt(1.0 - uz * uz);
    uxx = sintheta * (ux * uz * cospsi - uy * sinpsi) / temp + ux * costheta;
    uyy = sintheta * (uy * uz * cospsi + ux * sinpsi) / temp + uy * costheta;
    uzz = -sintheta * cospsi * temp + uz * costheta;
    }
    
  /* Update trajectory */
  ux = uxx;
  uy = uyy;
  uz = uzz;


/**** CHECK if photon ALIVE 
*****/
if (L >= Lmax)
   photon_status = DEAD;


} /* end HOP/DROP_SPIN_CHECK loop */
while (photon_status == ALIVE);

  /* If photon dead, then launch new photon. */
} /* end RUN */
while (i_photon < Nphotons);


/**** SAVE
*   Convert data to photon concentration [cm^-3] and save to file called "trmc.out".
*****/
target = fopen("trmc.out", "w");

/* print header */
fprintf(target, "number of photons = %f\n", Nphotons);
fprintf(target, "dr = %5.5f [cm] \n", dr);
fprintf(target, "last row is overflow. Ignore last row.\n");
fprintf(target, "Output is fluence rate F [W/(cm2 s)].\n");

/* print column titles */
fprintf(target, "r [cm] \t %5.3f ns \t %5.3f ns \t %5.3f ns \t %5.3f ns\n", 
                 T[0]*1e9, T[1]*1e9, T[2]*1e9, T[3]*1e9);

/* print data:  radial position, Photon fluence rate @ T[it]. */
for (ir=0; ir<NR; ir++) {        /* Don't printout the overflow bin. */
  	r = (ir + 0.5)*dr;
    fprintf(target, "%5.4f", r);
  	shellvolume = 4.0*PI*r*r*dr; /* per spherical shell */
  	for (it=0; it<4; it++) {
    	F[ir][it] = c*Csph[ir][it]/Nphotons/shellvolume;
    	fprintf(target, "\t%5.3e", F[ir][it]);
    	}
    fprintf(target, "\n");
  	}

fclose(target);


printf("finish\n");

} /* end of main */

 

/* SUBROUTINES */

/**************************************************************************
 *	RandomGen
 *      A random number generator that generates uniformly
 *      distributed random numbers between 0 and 1 inclusive.
 *      The algorithm is based on:
 *      W.H. Press, S.A. Teukolsky, W.T. Vetterling, and B.P.
 *      Flannery, "Numerical Recipes in C," Cambridge University
 *      Press, 2nd edition, (1992).
 *      and
 *      D.E. Knuth, "Seminumerical Algorithms," 2nd edition, vol. 2
 *      of "The Art of Computer Programming", Addison-Wesley, (1981).
 *
 *      When Type is 0, sets Seed as the seed. Make sure 0<Seed<32000.
 *      When Type is 1, returns a random number.
 *      When Type is 2, gets the status of the generator.
 *      When Type is 3, restores the status of the generator.
 *
 *      The status of the generator is represented by Status[0..56].
 *
 *      Make sure you initialize the seed before you get random
 *      numbers.
 ****/
#define MBIG 1000000000
#define MSEED 161803398
#define MZ 0
#define FAC 1.0E-9

double RandomGen(char Type, long Seed, long *Status){
  static long i1, i2, ma[56];   /* ma[0] is not used. */
  long        mj, mk;
  short       i, ii;

  if (Type == 0) {              /* set seed. */
    mj = MSEED - (Seed < 0 ? -Seed : Seed);
    mj %= MBIG;
    ma[55] = mj;
    mk = 1;
    for (i = 1; i <= 54; i++) {
      ii = (21 * i) % 55;
      ma[ii] = mk;
      mk = mj - mk;
      if (mk < MZ)
        mk += MBIG;
      mj = ma[ii];
    }
    for (ii = 1; ii <= 4; ii++)
      for (i = 1; i <= 55; i++) {
        ma[i] -= ma[1 + (i + 30) % 55];
        if (ma[i] < MZ)
          ma[i] += MBIG;
      }
    i1 = 0;
    i2 = 31;
  } else if (Type == 1) {       /* get a number. */
    if (++i1 == 56)
      i1 = 1;
    if (++i2 == 56)
      i2 = 1;
    mj = ma[i1] - ma[i2];
    if (mj < MZ)
      mj += MBIG;
    ma[i1] = mj;
    return (mj * FAC);
  } else if (Type == 2) {       /* get status. */
    for (i = 0; i < 55; i++)
      Status[i] = ma[i + 1];
    Status[55] = i1;
    Status[56] = i2;
  } else if (Type == 3) {       /* restore status. */
    for (i = 0; i < 55; i++)
      ma[i + 1] = Status[i];
    i1 = Status[55];
    i2 = Status[56];
  } else
    puts("Wrong parameter to RandomGen().");
  return (0);
}
#undef MBIG
#undef MSEED
#undef MZ
#undef FAC

