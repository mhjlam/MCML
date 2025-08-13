/**************************************************************************
 *  Copyright Univ. of Texas M.D. Anderson Cancer Center, 1992.
 *  Program name: MCML.
 *  Monte Carlo simulation of light transport in
 *  multi-layered turbid media in ANSI Standard C.
 *
 *  This header file is shared by both MCML and CONV.
 ****
 *  Starting Date:  10/1991.
 *  Current Date:   07/1993.
 *
 *  Lihong Wang, Ph.D.
 *  Steven L. Jacques, Ph.D.
 *  Laser Biology Research Laboratory - 17
 *  M.D. Anderson Cancer Center
 *  University of Texas
 *  1515 Holcombe Blvd.
 *  Houston, Texas 77030
 *  USA
 *
 *  Liqiong Zheng, B.S.
 *  Department of Computer Science
 *  University of Houston
 *  Houston, Texas
 *
 *  This program was based on:
 *  (1) The Pascal code written by Marleen Keijzer and
 *  Steven L. Jacques in this laboratory in 1989, which
 *  deals with multi-layered turbid media.
 *
 *  (2) Algorithm for semi-infinite turbid medium by
 *  S.A. Prahl, M. Keijzer, S.L. Jacques, A.J. Welch,
 *  SPIE Institute Series Vol. IS 5 (1989), and by
 *  A.N. Witt, The Astrophysical Journal Supplement
 *  Series 35, 1-6 (1977).
 *
 *  Major modifications in version 1.x include:
 *  . Conformation to ANSI Standard C.
 *  . Removal of limit on number of array elements,
 *    because arrays in this program are dynamically
 *    allocated. This means that the program can accept
 *    any number of layers or gridlines as long as the
 *    memory permits.
 *  . Avoiding global variables whenever possible.  This
 *    program has not used global variables so far.
 *  . Grouping variables logically using structures.
 *  . Top-down design, keep each subroutine clear & short.
 *  . Angularly resolving reflectance and transmittance.
 *
 *  Major modifications in version 2.0 include:
 *  . Allowing interactive input of parameters.
 *  . Allowing to select which quantities to score.
 *  . Time-resolved simulation.
 *  . Adjustable source position (z).
 *  . Supporting isotropic sources.
 *  . Simulation time control in addition to photon control.
 *  . Computing standard errors of some physical quantities.
 *  . Allowing continuation simulations to reduce standard errors.
 *  Major modifications in version 2.1 include:
 *  . C17 standard compliance with modern headers and types.
 *  . Enhanced buffer safety with bounds-checked input functions.
 *  . Performance optimizations including contiguous memory allocation.
 *  . Replace unsafe scanf/strcpy with fgets and strncpy variants.
 *  . Restrict qualifiers for compiler optimization hints.
 *  . Using standard C17 bool instead of custom types.
 *  . Improved mathematical precision with high-precision constants.
 *  . Modernized memory management with registry-based allocation tracking.
 *  . Fixed buffer overflow vulnerabilities in array dimension specifications.
 ****
 *  General Naming Conventions:
 *  Preprocessor names: all capital letters,
 *  e.g. #define PREPROCESSORS
 *  Globals: first letter of each word is capital, no
 *  underscores,
 *  e.g. short GlobalVar;
 *  Dummy variables:  first letter of each word is capital,
 *  and words are connected by underscores,
 *  e.g. void NiceFunction(char Dummy_Var);
 *  Local variables:  all lower cases, words may be connected
 *  by underscores,
 *  e.g. short local_var;
 *  Function names or data types:  same as Globals.
 ****
 *  Dimension of length: cm.
 *  Dimension of time:   ps.
 ****/

#include <ctype.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* Use high-precision mathematical constants */
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

enum {
	STRLEN = 256 /* String length. */
};

#define SIGN(x) ((x) >= 0 ? 1 : -1)

/********************************* Stuctures *****************************/

enum source_type { pencil, isotropic };

/****
 *  Structure used to describe a photon packet.
 ****/
typedef struct {
	double x, y, z;    /* Cartesian coordinates.[cm] */
	double ux, uy, uz; /* directional cosines of a photon. */
	double w;          /* weight. */
	bool alive;        /* 1/0 if photon is alive/terminated. */
	short layer;       /* index to layer where photon is. */
	double s;          /* current step size. [cm]. */
	double sleft;      /* step size left. dimensionless [-]. */
	long scatters;     /* number of scatterings. */
	double time;       /* flight time [picosec]. */
} PhotonStru;

/****
 *   Specify which quantity is to be scored .
 *
 *   Data categories:
 *   Rd_rat                  Td_rat                  A_rzt
 *   Rd_ra   Rd_rt   Rd_at   Td_ra   Td_rt   Rd_at   A_rz    A_zt
 *   Rd_r    Rd_a    Rd_t    Td_r    Td_a    Td_t    A_z     A_t
 ****/
typedef struct { /* use bit field to save space. */
	unsigned int Rd_rat : 1;
	unsigned int Rd_ra : 1;
	unsigned int Rd_rt : 1;
	unsigned int Rd_at : 1;
	unsigned int Rd_r : 1;
	unsigned int Rd_a : 1;
	unsigned int Rd_t : 1;

	unsigned int Td_rat : 1;
	unsigned int Td_ra : 1;
	unsigned int Td_rt : 1;
	unsigned int Td_at : 1;
	unsigned int Td_r : 1;
	unsigned int Td_a : 1;
	unsigned int Td_t : 1;

	unsigned int A_rzt : 1;
	unsigned int A_rz : 1;
	unsigned int A_zt : 1;
	unsigned int A_z : 1;
	unsigned int A_t : 1;
} RecordStru;

/****
 *  Structure used to describe the geometry and optical
 *  properties of a layer.
 *  z0 and z1 are the z coordinates for the upper boundary
 *  and lower boundary respectively.
 *
 *  cos_crit0 and cos_crit1 are the cosines of the
 *  critical angle of total internal reflection for the
 *  upper boundary and lower boundary respectively.
 *  They are set to zero if no total internal reflection
 *  exists.
 *  They are used for computation speed.
 ****/
typedef struct {
	char medium[STRLEN]; /* name of the medium. */
	double z0, z1;       /* z coordinates of a layer. [cm] */
	double n;            /* refractive index of a layer. */
	double mua;          /* absorption coefficient. [1/cm] */
	double mus;          /* scattering coefficient. [1/cm] */
	double g;            /* anisotropy. */
	double cos_crit0;
    double cos_crit1;
} LayerStru;

/****
 *  Input parameters for each independent run.
 *
 *  z and r are for the cylindrical coordinate system. [cm]
 *  a is for the angle alpha between the photon exiting
 *  direction and the surface normal. [radian]
 *
 *  The grid line separations in z, r, and alpha
 *  directions are dz, dr, and da respectively.  The numbers
 *  of grid lines in z, r, and alpha directions are
 *  nz, nr, and na respectively.
 *
 *  The member layerspecs will point to an array of
 *  structures which store parameters of each layer.
 *  This array has (number_layers + 2) elements. One
 *  element is for a layer.
 *  The layers 0 and (num_layers + 1) are for top ambient
 *  medium and the bottom ambient medium respectively.
 *
 *  out_fformat is output-file format.  Use 'A' for ASCII,
 *  and 'B' for binary.
 *
 *  control_bit:
 *    1 - photon number only.
 *    2 - time limit only.
 *    3 - both.
 ****/
typedef struct {
	char out_fname[STRLEN]; /* output file name. */
	char out_fformat;       /* output file format. */
	long num_photons;       /* to be traced. */
	long num_seconds;       /* computation time limit. */
	long add_num_photons;   /* addtional photon number. */
	long add_num_seconds;   /* addtional computation time. */
	char control_bit;       /* control of simulation termination. */
	double Wth;             /* play roulette if photon weight < Wth. */
	enum source_type type;  /* beam type. */
	double sz;              /* z coordinate of source. */
	short slayer;           /* put source in slayer. */
	char smedium[STRLEN];   /* medium name of slayer. */
	long ran_seed;          /* random number seed. */
	double dz;              /* z grid separation.[cm] */
	double dr;              /* r grid separation.[cm] */
	double da;              /* alpha grid separation. [radian] */
	double dt;              /* time grid separation.[ps] */
	short nz;               /* array range 0..nz-1. */
	short nr;               /* array range 0..nr-1. */
	short na;               /* array range 0..na-1. */
	short nt;               /* array range 0..nt-1. */
	double zm;              /* maximun z. [cm] */
	double rm;              /* maximum r.[cm] */
	double am;              /* maximum alpha. [radian] */
	double tm;              /* maximum time. [ps] */
	short num_layers;       /* number of layers. */
	LayerStru *layerspecs;  /* layer parameters. */
	RecordStru record;      /* scored quantity */
	short num_media;        /* number of media. */
	LayerStru *mediumlist;  /* media parameters. */
	short num_runs;         /* number of runs. */
} InStru;

/****
 *  Structures for scoring physical quantities.
 *  z and r represent z and r coordinates of the
 *  cylindrical coordinate system. [cm]
 *  a is the angle alpha between the photon exiting
 *  direction and the normal to the surfaces. [radian]
 *  R represents reflectance, T transmission, A absorption.
 *
 *  See comments of the InStru.
 *  See manual for the physcial quantities.
 ****/
typedef struct {
	double ***Rd_rat; /* diffuse reflectance. [1/(cm2 sr ps)] */
	double **Rd_ra;   /* [1/(cm2 sr)] */
	double **Rd_rt;   /* [1/sr ps] */
	double **Rd_at;   /* [1/cm2 ps] */
	double *Rd_r;     /* [1/cm2] */
	double *Rd_a;     /* [1/sr] */
	double *Rd_t;     /* [1/ps] */
	double Rd;        /* total diffuse reflectance. [-] */
	double Rde;       /* standard error for Rd. [-] */
	double Rdi;       /* Rd of the i-th photon. [-] */
	double Rb;        /* ballistic reflectance. [-] */
	double Rbe;       /* standard error for Rb. [-] */
	double Rbi;       /* Rb of the i-th photon. [-] */
	double Rsp;       /* specular reflectance. [-] */
	double ***Td_rat; /* diffuse transmittance. [1/(cm2 sr ps)] */
	double **Td_ra;   /* [1/(cm2 sr)] */
	double **Td_rt;   /* [1/sr ps] */
	double **Td_at;   /* [1/cm2 ps] */
	double *Td_r;     /* [1/cm2] */
	double *Td_a;     /* [1/sr] */
	double *Td_t;     /* [1/ps] */
	double Td;        /* total diffuse transmittance. [-] */
	double Tde;       /* standard error for Td. [-] */
	double Tdi;       /* Td of the i-th photon. [-] */
	double Tb;        /* ballistic transmittance. [-] */
	double Tbe;       /* standard error for Tb. [-] */
	double Tbi;       /* Tb of the i-th photon. [-] */
	double ***A_rzt;  /* absorption. [1/(cm3 ps)] */
	double **A_rz;    /* [1/cm3] */
	double **A_zt;    /* [1/(cm ps)] */
	double *A_z;      /* [1/cm] */
	double *A_t;      /* [1/ps] */
	double **Ab_zt;   /* ballistic absorption. [1/(cm ps)] */
	double *Ab_z;     /* [1/cm] */
	double A;         /* total absorption. [-] */
	double Ae;        /* standard error for A. [-] */
	double Ai;        /* A of the i-th photon. [-] */
} OutStru;

/**************************************************************************
 *	Modern array structures for internal management.
 *  These maintain metadata while returning raw pointers for compatibility.
 ****/
typedef struct {
	double* raw_ptr;  /* The pointer returned to caller */
	size_t size;      /* Allocated size */
	ptrdiff_t offset; /* Index offset for access */
} Array1D;

typedef struct {
	double** raw_ptr;     /* The pointer returned to caller */
	double** data;        /* Row pointers */
	size_t rows;          /* Number of rows */
	size_t cols;          /* Number of columns */
	ptrdiff_t row_offset; /* Row offset */
	ptrdiff_t col_offset; /* Column offset */
} Array2D;

typedef struct {
	double*** raw_ptr;          /* The pointer returned to caller */
	double*** data;             /* 3D pointer structure */
	size_t dim1, dim2, dim3;    /* Dimensions */
	ptrdiff_t off1, off2, off3; /* Offsets */
} Array3D;

/**************************************************************************
 *	Function prototypes for dynamic memory allocation and
 *	release of arrays and matrices. These provide modern memory management
 *	with safety checks while maintaining backward compatibility.
 ****/
double* AllocArray1D(ptrdiff_t nl, ptrdiff_t nh);
double** AllocArray2D(ptrdiff_t nrl, ptrdiff_t nrh, ptrdiff_t ncl, ptrdiff_t nch);
double*** AllocArray3D(ptrdiff_t n1l, ptrdiff_t n1h, ptrdiff_t n2l, ptrdiff_t n2h, ptrdiff_t n3l, ptrdiff_t n3h);
void FreeArray1D(const double* array);
void FreeArray2D(double** array);
void FreeArray3D(double*** array);
void ErrorExit(const char* error_text);
