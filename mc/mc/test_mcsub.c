/*
 * test_mcsub.c - Example usage of mcsub.h single-header library
 * 
 * This program demonstrates Monte Carlo simulation of photon transport in biological tissue.
 * It shows how to:
 *   1. Set up tissue optical properties and simulation parameters
 *   2. Allocate memory for result arrays  
 *   3. Run the Monte Carlo simulation
 *   4. Save results to file
 *   5. Clean up allocated memory
 * 
 * The simulation models a collimated beam incident on tissue with scattering and absorption.
 * Energy conservation is verified: Specular + Absorbed + Escaped = 1.0
 */

#define MCSUB_IMPLEMENTATION    // Must be defined before including mcsub.h
#include "mcsub.h"

int main(void) {
    printf("MCSUB Monte Carlo Photon Transport Simulation\\n");
    printf("==============================================\\n");
    
    /* Tissue optical properties - typical biological tissue values */
    double mua = 1.0;      /* absorption coefficient [cm^-1] */
    double mus = 100.0;    /* scattering coefficient [cm^-1] */
    double g = 0.90;       /* anisotropy factor [0-1] - 0.9 is highly forward scattering */
    double n1 = 1.4;       /* refractive index of tissue */
    double n2 = 1.4;       /* refractive index outside medium */
    
    /* Simulation grid - defines spatial resolution */
    short NR = 101;        /* number of radial bins */
    short NZ = 101;        /* number of depth bins */
    double dr = 0.0020;    /* radial bin size [cm] - 20 microns */
    double dz = 0.0020;    /* depth bin size [cm] - 20 microns */
    
    /* Incident beam configuration */
    short mcflag = 0;      /* beam type: 0=collimated, 1=Gaussian, 2=point source */
    double xs = 0, ys = 0, zs = 0;  /* source position [cm] (used for point sources) */
    int boundaryflag = 1;  /* 1=air/tissue boundary with Fresnel reflection, 0=infinite medium */
    double radius = 0.0;   /* beam radius [cm] - 0 for pencil beam */
    double waist = 0.10;   /* waist of Gaussian beam [cm] (unused for collimated) */
    double zfocus = 1.0;   /* focus depth [cm] */
    
    /* Simulation parameters */
    double Nphotons = 1e4; /* number of photons (small for quick test) */
    
    /* Allocate arrays */
    double *J = mcsub_alloc_vector(1, NR);
    double **F = mcsub_alloc_matrix(1, NZ, 1, NR);
    
    /* Results */
    double S, A, E;
    
    printf("Modern MCSUB Test - Monte Carlo Photon Transport\n");
    printf("================================================\n");
    printf("Tissue: mua=%g, mus=%g, g=%g, n=%g\n", mua, mus, g, n1);
    printf("Grid: %d x %d, dr=%g cm, dz=%g cm\n", NR, NZ, dr, dz);
    printf("Beam: %s, %g photons\n", 
           mcflag == 0 ? "collimated" : mcflag == 1 ? "Gaussian" : "point source",
           Nphotons);
    printf("\nRunning simulation...\n");
    
    /* Run Monte Carlo simulation */
    mcsub(mua, mus, g, n1, n2, NR, NZ, dr, dz, Nphotons,
          mcflag, xs, ys, zs, boundaryflag, radius, waist, zfocus,
          J, F, &S, &A, &E, 1);  /* PRINTOUT=1 for progress */
    
    printf("\nResults:\n");
    printf("--------\n");
    printf("Specular reflection: %6.4f\n", S);
    printf("Absorbed fraction:   %6.4f\n", A);
    printf("Escaped fraction:    %6.4f\n", E);
    printf("Total (check):       %6.4f\n", S+A+E);
    
    /* Save results to file */
    mcsub_save_file(1, J, F, S, A, E, mua, mus, g, n1, n2,
                    mcflag, radius, waist, xs, ys, zs,
                    NR, NZ, dr, dz, Nphotons);
    
    /* Clean up memory */
    mcsub_free_vector(J, 1, NR);
    mcsub_free_matrix(F, 1, NZ, 1, NR);
    
    printf("\nTest completed successfully!\n");
    return 0;
}
