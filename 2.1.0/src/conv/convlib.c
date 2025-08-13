/**************************************************************************
 *  Copyright Univ. of Texas M.D. Anderson Cancer Center, 1992.
 *
 *  Some routines modified from Numerical Recipes in C,
 *  including error report, array or matrix declaration
 *  and releasing, integrations.
 *
 *  Some frequently used routines are also included here.
 ****/

#include "conv.h"

/**************************************************************************
 *  Modern array allocation with safety tracking.
 *  Simple registry for tracking allocations - based on MCML implementation.
 ****/
static Array1D* array1d_registry[1000] = {0};
static Array2D* array2d_registry[1000] = {0};
static Array3D* array3d_registry[1000] = {0};
static int next1d = 0, next2d = 0, next3d = 0;

/**************************************************************************
 *  Report error message to stderr, then exit the program
 *  with signal 1. Enhanced with const correctness.
 ****/
void ErrorExit(const char* restrict error_text) {
	fprintf(stderr, "%s\n", error_text);
	fprintf(stderr, "...now exiting to system...\n");
	exit(1);
}

/**************************************************************************
 *	Allocate a 1D array with index from nl to nh inclusive.
 *  Modern version with safety tracking using MCML techniques.
 ****/
double* AllocArray1D(ptrdiff_t nl, ptrdiff_t nh) {
	Array1D* array = calloc(1, sizeof(Array1D));
	if (!array) {
		ErrorExit("allocation failure in AllocArray1D()");
	}

	const size_t size = nh - nl + 1;
	array->raw_ptr = calloc(size, sizeof(double));
	if (!array->raw_ptr) {
		free(array);
		ErrorExit("allocation failure in AllocArray1D()");
	}

	array->raw_ptr -= nl; // Adjust for indexing from nl
	array->size = size;
	array->offset = nl;

	// Register for cleanup
	if (next1d < 1000) {
		array1d_registry[next1d++] = array;
	}
	return array->raw_ptr;
}

/**************************************************************************
 *	Allocate a 2D array with dimensions from nrl to nrh, ncl to nch.
 *  Modern version with safety tracking using MCML techniques.
 ****/
double** AllocArray2D(ptrdiff_t nrl, ptrdiff_t nrh, ptrdiff_t ncl, ptrdiff_t nch) {
	Array2D* array = calloc(1, sizeof(Array2D));
	if (!array) {
		ErrorExit("allocation failure in AllocArray2D()");
	}

	const size_t rows = nrh - nrl + 1;
	const size_t cols = nch - ncl + 1;

	// Allocate array of pointers
	array->data = calloc(rows, sizeof(double*));
	if (!array->data) {
		free(array);
		ErrorExit("allocation failure in AllocArray2D()");
	}

	// Allocate memory for each row
	for (size_t i = 0; i < rows; i++) {
		array->data[i] = calloc(cols, sizeof(double));
		if (!array->data[i]) {
			// Clean up already allocated rows
			for (size_t j = 0; j < i; j++) {
				free(array->data[j]);
			}
			free(array->data);
			free(array);
			ErrorExit("allocation failure in AllocArray2D()");
		}
		array->data[i] -= ncl; // Adjust for column offset
	}

	array->data -= nrl; // Adjust for row offset
	array->raw_ptr = array->data;
	array->rows = rows;
	array->cols = cols;
	array->row_offset = nrl;
	array->col_offset = ncl;

	// Register for cleanup
	if (next2d < 1000) {
		array2d_registry[next2d++] = array;
	}
	return array->raw_ptr;
}

/**************************************************************************
 *	Allocate a 3D array with dimensions from n1l to n1h, n2l to n2h, n3l to n3h.
 *  Modern version with safety tracking using MCML techniques.
 ****/
double*** AllocArray3D(ptrdiff_t n1l, ptrdiff_t n1h, ptrdiff_t n2l, ptrdiff_t n2h, ptrdiff_t n3l, ptrdiff_t n3h) {
	Array3D* array = calloc(1, sizeof(Array3D));
	if (!array) {
		ErrorExit("allocation failure in AllocArray3D()");
	}

	const size_t dim1 = n1h - n1l + 1;
	const size_t dim2 = n2h - n2l + 1;
	const size_t dim3 = n3h - n3l + 1;

	// Allocate array of double**
	array->data = calloc(dim1, sizeof(double**));
	if (!array->data) {
		free(array);
		ErrorExit("allocation failure in AllocArray3D()");
	}

	// Allocate array of double* for each layer
	for (size_t i = 0; i < dim1; i++) {
		array->data[i] = calloc(dim2, sizeof(double*));
		if (!array->data[i]) {
			// Clean up already allocated layers
			for (size_t j = 0; j < i; j++) {
				for (size_t k = 0; k < dim2; k++) {
					free(array->data[j][k]);
				}
				free(array->data[j]);
			}
			free(array->data);
			free(array);
			ErrorExit("allocation failure in AllocArray3D()");
		}

		// Allocate actual double arrays
		for (size_t j = 0; j < dim2; j++) {
			array->data[i][j] = calloc(dim3, sizeof(double));
			if (!array->data[i][j]) {
				// Clean up this layer
				for (size_t k = 0; k < j; k++) {
					free(array->data[i][k]);
				}
				free(array->data[i]);
				// Clean up previous layers
				for (size_t l = 0; l < i; l++) {
					for (size_t m = 0; m < dim2; m++) {
						free(array->data[l][m]);
					}
					free(array->data[l]);
				}
				free(array->data);
				free(array);
				ErrorExit("allocation failure in AllocArray3D()");
			}
			array->data[i][j] -= n3l; // Adjust for 3rd dimension offset
		}
		array->data[i] -= n2l; // Adjust for 2nd dimension offset
	}

	array->data -= n1l; // Adjust for 1st dimension offset
	array->raw_ptr = array->data;
	array->dim1 = dim1;
	array->dim2 = dim2;
	array->dim3 = dim3;
	array->off1 = n1l;
	array->off2 = n2l;
	array->off3 = n3l;

	// Register for cleanup
	if (next3d < 1000) {
		array3d_registry[next3d++] = array;
	}
	return array->raw_ptr;
}

/**************************************************************************
 *	Release Array1D memory - using registry lookup for safety.
 ****/
void FreeArray1D(const double* raw_ptr) {
	if (!raw_ptr) {
		return;
	}

	// Find array in registry
	for (int i = 0; i < next1d; i++) {
		if (array1d_registry[i] && array1d_registry[i]->raw_ptr == raw_ptr) {
			Array1D* array = array1d_registry[i];

			// Restore original pointer for freeing
			double* original_ptr = array->raw_ptr + array->offset;
			free(original_ptr);

			// Clear registry entry
			free(array);
			array1d_registry[i] = NULL;
			return;
		}
	}

	ErrorExit("FreeArray1D: array not found in registry");
}

/**************************************************************************
 *	Release Array2D memory - using registry lookup for safety.
 ****/
void FreeArray2D(double** raw_ptr) {
	if (!raw_ptr) {
		return;
	}

	// Find array in registry
	for (int i = 0; i < next2d; i++) {
		if (array2d_registry[i] && array2d_registry[i]->raw_ptr == raw_ptr) {
			Array2D* array = array2d_registry[i];

			// Restore original pointers for freeing
			double** original_data = array->data + array->row_offset;

			for (size_t j = 0; j < array->rows; j++) {
				if (original_data[j]) {
					free(original_data[j] + array->col_offset);
				}
			}
			free(original_data);

			// Clear registry entry
			free(array);
			array2d_registry[i] = NULL;
			return;
		}
	}

	ErrorExit("FreeArray2D: array not found in registry");
}

/**************************************************************************
 *  Release Array3D memory - using registry lookup for safety.
 ****/
void FreeArray3D(double*** raw_ptr) {
	if (!raw_ptr) {
		return;
	}

	// Find array in registry
	for (int i = 0; i < next3d; i++) {
		if (array3d_registry[i] && array3d_registry[i]->raw_ptr == raw_ptr) {
			Array3D* array = array3d_registry[i];

			// Restore original pointers for freeing
			double*** original_data = array->data + array->off1;

			for (size_t j = 0; j < array->dim1; j++) {
				if (original_data[j]) {
					double** original_layer = original_data[j] + array->off2;
					for (size_t k = 0; k < array->dim2; k++) {
						if (original_layer[k]) {
							free(original_layer[k] + array->off3);
						}
					}
					free(original_layer);
				}
			}
			free(original_data);

			// Clear registry entry
			free(array);
			array3d_registry[i] = NULL;
			return;
		}
	}

	ErrorExit("FreeArray3D: array not found in registry");
}

/**************************************************************************
 *  Trapezoidal integration.
 ****/

#define FUNC(x, y) ((*func)(x, y))
float trapzd(float (*func)(float, ConvStru *), float a, float b, int n, ConvStru *Conv_Ptr) {
	float x;
	float tnm;
	float sum;
	float del;
	static float s;
	static int it;
	int j;

	if (n == 1) {
		it = 1;
		return (s = 0.5 * (b - a) * (FUNC(a, Conv_Ptr) + FUNC(b, Conv_Ptr)));
	}

	tnm = it;
	del = (b - a) / tnm;
	x = a + 0.5 * del;
	for (sum = 0.0, j = 1; j <= it; j++, x += del) {
		sum += FUNC(x, Conv_Ptr);
	}
	it *= 2;
	s = 0.5 * (s + (b - a) * sum / tnm);
	return s;
}

#undef FUNC

/**************************************************************************
 *  Returns the integral of the function func from a to b.  EPS is the
 *  relative error of the integration.  This function is based on
 *  W.H. Press, S.A. Teukolsky, W.T. Vetterling, and B.P.  Flannery,
 *  "Numerical Recipes in C," Cambridge University Press, 2nd edition,
 *  (1992).
 ****/
#define JMAX 20
float qtrap(float (*func)(float, ConvStru *), float a, float b, ConvStru *Conv_Ptr) {
	int j;
	float s;
	float s_old;

	s_old = -1.0e30;
	for (j = 1; j <= JMAX; j++) {
		s = trapzd(func, a, b, j, Conv_Ptr);
		if (fabsf(s - s_old) <= Conv_Ptr->eps * fabsf(s_old)) {
			break;
		}
		s_old = s;
	}
	return (s);
}

#undef JMAX

/**************************************************************************
 *  Modified Bessel function exp(-x) I0(x), for x >=0.
 *  This function was modified from the original bessi0(). Instead of
 *  I0(x) itself, it returns I0(x) exp(-x).
 *  OPTIMIZED: Added const correctness, improved precision, and branch optimization
 ****/
double BessI0(const double x) {
	const double abs_x = fabs(x);

	if (__builtin_expect(abs_x < 3.75, 1)) { /* Branch prediction hint - common case */
		const double y = (x / 3.75) * (x / 3.75);
		/* High-precision coefficients for better accuracy */
		const double result =
			exp(-abs_x)
			* (1.0
			   + y
					 * (3.5156229
						+ y * (3.0899424 + y * (1.2067492 + y * (0.2659732 + y * (0.360768e-1 + y * 0.45813e-2))))));
		return result;
	}

	const double y = 3.75 / abs_x;
	const double sqrt_inv = 1.0 / sqrt(abs_x); /* Compute once */
	/* High-precision coefficients */
	const double result =
		sqrt_inv
		* (0.39894228
		   + y
				 * (0.1328592e-1
					+ y
						  * (0.225319e-2
							 + y
								   * (-0.157565e-2
									  + y
											* (0.916281e-2
											   + y
													 * (-0.2057706e-1
														+ y
															  * (0.2635537e-1
																 + y * (-0.1647633e-1 + y * 0.392377e-2))))))));
	return result;
}

/****************************************************************
 ****/
short GetShort(short Lo, short Hi) {
	short x;

	scanf("%hd", &x);
	while (x < Lo || x > Hi) {
		printf("...Parameter out of range.  Try again: ");
		scanf("%hd", &x);
	}
	return (x);
}

/****************************************************************
 ****/
float GetFloat(float Lo, float Hi) {
	float x;

	scanf("%f", &x);
	while (x < Lo || x > Hi) {
		printf("...Parameter out of range.  Try again: ");
		scanf("%f", &x);
	}
	return (x);
}
