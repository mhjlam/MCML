char   t1[80] = "Small Monte Carlo by Scott Prahl (https://omlc.org)";
char   t2[80] = "1 W/cm^2 Uniform Illumination of Semi-Infinite Medium";

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define BINS 101

double mu_a = 5;			/* Absorption Coefficient in 1/cm */
double mu_s = 95;			/* Scattering Coefficient in 1/cm */
double g = 0.5;				/* Scattering Anisotropy -1<=g<=1 */
double n = 1.5;				/* Index of refraction of medium */
double microns_per_bin = 20;/* Thickness of one bin layer */
long   i, photons = 100000;
double x,y,z,u,v,w,weight;
double rs, rd, bit, albedo, crit_angle, bins_per_mfp, heat[BINS];

void launch() /* Start the photon */
{
	x = 0.0; y = 0.0; z = 0.0;		  
	u = 0.0; v = 0.0; w = 1.0;		
	weight = 1.0 - rs;
}

void bounce () /* Interact with top surface */
{
double t, temp, temp1,rf;
	w = -w;
	z = -z;
	if (w <= crit_angle) return;  			/* total internal reflection */	

	t       = sqrt(1.0-n*n*(1.0-w*w));    	/* cos of exit angle */
	temp1   = (w - n*t)/(w + n*t);
	temp    = (t - n*w)/(t + n*w);
	rf      = (temp1*temp1+temp*temp)/2.0;	/* Fresnel reflection */
	rd     += (1.0-rf) * weight;
	weight -= (1.0-rf) * weight;
}

void move() /* move to next scattering or absorption event */
{
double d = -log((rand()+1.0)/(RAND_MAX+1.0));
	x += d * u;
	y += d * v;
	z += d * w;  
	if ( z<=0 ) bounce();
}

void absorb () /* Absorb light in the medium */
{
int bin=z*bins_per_mfp;

	if (bin >= BINS) bin = BINS-1;	
	heat[bin] += (1.0-albedo)*weight;
	weight *= albedo;
	if (weight < 0.001){ /* Roulette */
		bit -= weight;
		if (rand() > 0.1*RAND_MAX) weight = 0; else weight /= 0.1;
		bit += weight;
	}
}

void scatter() /* Scatter photon and establish new direction */
{
double x1, x2, x3, t, mu;

	for(;;) {								/*new direction*/
		x1=2.0*rand()/RAND_MAX - 1.0; 
		x2=2.0*rand()/RAND_MAX - 1.0; 
		if ((x3=x1*x1+x2*x2)<=1) break;
	}	
	if (g==0) {  /* isotropic */
		u = 2.0 * x3 -1.0;
		v = x1 * sqrt((1-u*u)/x3);
		w = x2 * sqrt((1-u*u)/x3);
		return;
	} 

	mu = (1-g*g)/(1-g+2.0*g*rand()/RAND_MAX);
	mu = (1 + g*g-mu*mu)/2.0/g;
	if ( fabs(w) < 0.9 ) {	
		t = mu * u + sqrt((1-mu*mu)/(1-w*w)/x3) * (x1*u*w-x2*v);
		v = mu * v + sqrt((1-mu*mu)/(1-w*w)/x3) * (x1*v*w+x2*u);
		w = mu * w - sqrt((1-mu*mu)*(1-w*w)/x3) * x1;
	} else {
		t = mu * u + sqrt((1-mu*mu)/(1-v*v)/x3) * (x1*u*v + x2*w);
		w = mu * w + sqrt((1-mu*mu)/(1-v*v)/x3) * (x1*v*w - x2*u);
		v = mu * v - sqrt((1-mu*mu)*(1-v*v)/x3) * x1;
	}
	u = t;
}

void print_results() /* Print the results */
{
int i;
	printf("%s\n%s\n\nScattering = %8.3f/cm\nAbsorption = %8.3f/cm\n",t1,t2,mu_s,mu_a);
	printf("Anisotropy = %8.3f\nRefr Index = %8.3f\nPhotons    = %8ld",g,n,photons);
	printf("\n\nSpecular Refl      = %10.5f\nBackscattered Refl = %10.5f",rs,rd/(bit+photons));
	printf("\n\n Depth         Heat\n[microns]     [W/cm^3]\n");

	for (i=0;i<BINS-1;i++){
		printf("%6.0f    %12.5f\n",i*microns_per_bin, heat[i]/microns_per_bin*1e4/(bit+photons));
	}
	printf(" extra    %12.5f\n",heat[BINS-1]/(bit+photons));
}

int main ()
{
	albedo = mu_s / (mu_s + mu_a);
	rs = (n-1.0)*(n-1.0)/(n+1.0)/(n+1.0);	/* specular reflection */
	crit_angle = sqrt(1.0-1.0/n/n);			/* cos of critical angle */
	bins_per_mfp = 1e4/microns_per_bin/(mu_a+mu_s);
	
	for (i = 1; i <= photons; i++){
		launch ();
		while (weight > 0) {
			move ();
			absorb ();
			scatter ();
		}
	}	
	print_results();
	return 0;
}
