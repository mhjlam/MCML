/**************************************************************************
 *	Copyright Univ. of Texas M.D. Anderson Cancer Center, 1992-1996.
 *	Program name: MCML.
 *	Input/output of data.
 ****/

#include "mcml.h"

/**************************************************************************
 *	Structure used to check against duplicated file names.
 ****/
struct NameList {
	char name[STRLEN];
	struct NameList* next;
};

typedef struct NameList NameNode;
typedef NameNode* NameLink;

/**************************************************************************
 *  Report error message to stderr, then exit the program
 *  with signal 1.
 ****/
void ErrorExit(char error_text[]) {
	fprintf(stderr, "%s\n", error_text);
	fprintf(stderr, "...now exiting to system...\n");
	exit(1);
}

/**************************************************************************
 *	Hybrid array allocation approach: Modern safety with legacy compatibility.
 ****/

// Simple registry for tracking hybrid allocations
static Array1D* array1d_registry[1000] = {0};
static Array2D* array2d_registry[1000] = {0};
static Array3D* array3d_registry[1000] = {0};
static int next1d = 0, next2d = 0, next3d = 0;

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

	array->raw_ptr -= nl; // Adjust for offset access
	array->size = size;
	array->offset = nl;

	if (next1d < 1000) {
		array1d_registry[next1d++] = array;
	}
	return array->raw_ptr;
}

/**************************************************************************
 *	Allocate a 2D array with row index from nrl to nrh inclusive, and column
 *  index from ncl to nch inclusive. Modern version with proper structures.
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
	array2d_registry[next2d++] = array;
	return array->raw_ptr;
} /**************************************************************************
   *	Allocate a 3D array with dimensions from n1l to n1h, n2l to n2h, n3l to n3h.
   *  Modern version with proper structures.
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
	array3d_registry[next3d++] = array;
	return array->raw_ptr;
}

/**************************************************************************
 *	Release Array1D memory - hybrid version using registry lookup.
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
 *	Release Array2D memory - hybrid version using registry lookup.
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
 *  Release Array3D memory - hybrid version using registry lookup.
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
 *	Legacy Modern functions - DISABLED in favor of hybrid approach
 ****/
#if 0
Array1D* AllocArray1DModern(ptrdiff_t nl, ptrdiff_t nh) {
    Array1D* array = calloc(1, sizeof(Array1D));
    if (!array) return NULL;
    
    const size_t size = nh - nl + 1;
    array->data = calloc(size, sizeof(double));
    if (!array->data) {
        free(array);
        return NULL;
    }
    
    array->size = size;
    array->offset = nl;
    return array;
}

Array2D* AllocArray2DModern(ptrdiff_t nrl, ptrdiff_t nrh, ptrdiff_t ncl, ptrdiff_t nch) {
    Array2D* array = calloc(1, sizeof(Array2D));
    if (!array) return NULL;
    
    const size_t rows = nrh - nrl + 1;
    const size_t cols = nch - ncl + 1;
    
    array->data = calloc(rows, sizeof(double*));
    if (!array->data) {
        free(array);
        return NULL;
    }
    
    for (size_t i = 0; i < rows; i++) {
        array->data[i] = calloc(cols, sizeof(double));
        if (!array->data[i]) {
            for (size_t j = 0; j < i; j++) {
                free(array->data[j]);
            }
            free(array->data);
            free(array);
            return NULL;
        }
    }
    
    array->rows = rows;
    array->cols = cols;
    array->row_offset = nrl;
    array->col_offset = ncl;
    return array;
}

void FreeArray1DModern(Array1D* array) {
    if (array) {
        free(array->data);
        free(array);
    }
}

void FreeArray2DModern(Array2D* array) {
    if (array) {
        if (array->data) {
            for (size_t i = 0; i < array->rows; i++) {
                free(array->data[i]);
            }
            free(array->data);
        }
        free(array);
    }
}

Array3D* AllocArray3DModern(ptrdiff_t n1l, ptrdiff_t n1h, ptrdiff_t n2l, ptrdiff_t n2h, ptrdiff_t n3l, ptrdiff_t n3h) {
    Array3D* array = calloc(1, sizeof(Array3D));
    if (!array) return NULL;
    
    const size_t dim1 = n1h - n1l + 1;
    const size_t dim2 = n2h - n2l + 1;
    const size_t dim3 = n3h - n3l + 1;
    
    array->data = calloc(dim1, sizeof(double**));
    if (!array->data) {
        free(array);
        return NULL;
    }
    
    for (size_t i = 0; i < dim1; i++) {
        array->data[i] = calloc(dim2, sizeof(double*));
        if (!array->data[i]) {
            for (size_t j = 0; j < i; j++) {
                for (size_t k = 0; k < dim2; k++) {
                    free(array->data[j][k]);
                }
                free(array->data[j]);
            }
            free(array->data);
            free(array);
            return NULL;
        }
        
        for (size_t j = 0; j < dim2; j++) {
            array->data[i][j] = calloc(dim3, sizeof(double));
            if (!array->data[i][j]) {
                for (size_t k = 0; k < j; k++) {
                    free(array->data[i][k]);
                }
                free(array->data[i]);
                for (size_t l = 0; l < i; l++) {
                    for (size_t k = 0; k < dim2; k++) {
                        free(array->data[l][k]);
                    }
                    free(array->data[l]);
                }
                free(array->data);
                free(array);
                return NULL;
            }
        }
    }
    
    array->dim1 = dim1;
    array->dim2 = dim2;
    array->dim3 = dim3;
    array->off1 = n1l;
    array->off2 = n2l;
    array->off3 = n3l;
    return array;
}

void FreeArray3DModern(Array3D* array) {
    if (array) {
        if (array->data) {
            for (size_t i = 0; i < array->dim1; i++) {
                if (array->data[i]) {
                    for (size_t j = 0; j < array->dim2; j++) {
                        free(array->data[i][j]);
                    }
                    free(array->data[i]);
                }
            }
            free(array->data);
        }
        free(array);
    }
}
#endif

/**************************************************************************
 *	Direct replacements for old allocation functions.
 *  These are simple function renames to use the modern implementations.
 *  Note: The old functions used 'short' parameters while modern ones use 'ptrdiff_t'
 *  but this is compatible since the values are small positive integers.
 *
 *  MODERNIZATION GUIDE:
 *  To fully modernize the code, follow these steps:
 *  1. Change OutStru fields from double*, double**, double*** to Array1D*, Array2D*, Array3D*
 *  2. Replace AllocArray1D/2D/3D calls with AllocArray1D/2D/3DModern
 *  3. Update array access from ptr[i] to array->data[i]
 *  4. Replace FreeArray1D/2D/3D calls with FreeArray1D/2D/3DModern
 *
 *  Example modernization:
 *  OLD: double** array = AllocArray2D(0, 10, 0, 20); array[5][10] = 3.14; FreeArray2D(array, 0, 10, 0, 20);
 *  NEW: Array2D* array = AllocArray2DModern(0, 10, 0, 20); array->data[5][10] = 3.14; FreeArray2DModern(array);
 ****//**************************************************************************
 *	Center a string according to the column width.
 *  Modernized with better buffer management and const correctness.
 ****/
void CtrPuts(const char* restrict InStr) {
	static const short COL_WIDTH = 80;

	if (!InStr) {
		return; /* Safety check */
	}

	const size_t str_len = strlen(InStr);
	if (str_len >= STRLEN) {
		/* String too long, just print it directly */
		puts(InStr);
		return;
	}

	const short nspaces = (str_len >= COL_WIDTH) ? 0 : (COL_WIDTH - (short)str_len) / 2;

	/* Use safer approach with snprintf to avoid buffer overrun */
	char outstr[STRLEN];
	int written = snprintf(outstr, sizeof(outstr), "%*s%s", nspaces, "", InStr);

	if (written > 0 && written < (int)sizeof(outstr)) {
		puts(outstr); /* Safe: no format string vulnerabilities */
	}
	else {
		puts(InStr); /* Fallback */
	}
}

/**************************************************************************
 *	Print messages about MCML.
 ****/
void AboutMCML(void) {
	CtrPuts("MCML 2.1.0, Copyright (c) 1992-1996, 2025");
	CtrPuts("Monte Carlo Simulation of Light Transport in Multi-Layered Turbid Media");

	CtrPuts(" ");
	CtrPuts("Lihong Wang, Ph.D.");
	CtrPuts("Bioengineering Program, Texas A&M University");
	CtrPuts("College Station, Texas 77843-3120, USA");

	CtrPuts("Liqiong Zheng, B.S.");
	CtrPuts("Summer student from Dept. of Computer Science,");
	CtrPuts("University of Houston, Texas, USA.");

	CtrPuts("Steven L. Jacques, Ph.D.");
	CtrPuts("Oregon Medical Laser Center, Providence/St. Vincent Hospital");
	CtrPuts("9205 SW Barnes Rd., Portland, Oregon 97225, USA");

	CtrPuts(" ");
	CtrPuts("Obtain the program thru anonymous ftp to laser.mda.uth.tmc.edu");

	CtrPuts(" ");
	CtrPuts("Please cite the following article in your publications:");
	printf("\tL.-H. Wang, S. L. Jacques, and L.-Q. Zheng, MCML - Monte \n");
	printf("\tCarlo modeling of photon transport in multi-layered\n");
	printf("\ttissues, Computer Methods and Programs in Biomedicine, 47,\n");
	printf("\t131-146 (1995)\n");
}

/**************************************************************************
 *	Kill the ith char (counting from 0), push the following
 *	chars forward by one.
 ****/
void KillChar(size_t i, char* Str) {
	size_t sl = strlen(Str);
	for (; i < sl; i++) {
		Str[i] = Str[i + 1];
	}
}

/**************************************************************************
 *	Eliminate the chars in a string which are not printing
 *	chars or spaces.
 *
 *	Spaces include ' ', '\f', '\t' etc.
 *
 *	Return 1 if no nonprinting chars found, otherwise
 *	return 0.
 ****/
bool CheckCharQ(char* Str) {
	bool found = 0; /* found bad char. */
	size_t sl = strlen(Str);
	size_t i = 0;

	while (i < sl) {
		if (Str[i] < 0) {
			printf("Non-ASCII file\n");
			return (0);
		}
		if (isprint(Str[i]) || isspace(Str[i])) {
			i++;
		}
		else {
			found = 1;
			KillChar(i, Str);
			sl--;
		}
	}

	return (found);
}

/**************************************************************************
 *	Return 1 if this line is a comment line in which the
 *	first non-space character is "#", or a space line.
 *	Return 0 otherwise.
 ****/
bool CommentLineQ(char* Buf) {
	/* length spanned by space or tab chars. */
	size_t spn = strspn(Buf, " \t");

	/* length before the 1st # or return. */
	size_t cspn = strcspn(Buf, "#\n");

	/* comment line or space line. */
	return spn == cspn;
}

/**************************************************************************
 *	Skip space or comment lines and return a data line.
 ****/
char* FindDataLine(FILE* Fp) {
	static char buf[STRLEN] = {0};

	/* skip space or comment lines. */
	do {
		if (fgets(buf, BUF_SIZE - 1, Fp) == NULL) {
			printf("Incomplete data.\n");
			buf[0] = '\0';
			break;
		}

		CheckCharQ(buf);
	}
	while (CommentLineQ(buf));

	return (buf);
}

/**************************************************************************
 *	Check whether the file version is the same as Version.
 ****/
bool CheckFileVersionQ(FILE* Fp, const char* Version) {
	/* line buffer. */
	char buf[STRLEN] = {0};

	/* skip comment line. */
	do {
		if (fgets(buf, BUF_SIZE - 1, Fp) == NULL) {
			buf[0] = '\0';
			break;
		}
	}
	while (CommentLineQ(buf));

	if ((buf[0] == '\0') || (strstr(buf, Version) == NULL)) {
		puts("Wrong file version.");
		return (0);
	}

	return (1);
}

/**************************************************************************
 *  Get a filename and open it for reading, retry until the file can be
 *  opened with a correct version or a '.' is typed.
 *	Return a NULL pointer if '.' is typed.
 ****/
FILE* GetFile(char* Fname, const char* Version) {
	FILE* Fp = NULL;

	while (1) {
		/* prompt. */
		printf("Specify filename (or . to quit to main menu):");

		// Clear the input buffer (consume any leftover characters)
		if (fgets(Fname, STRLEN, stdin) != NULL) {
			/* Replace newline with null terminator */
			size_t len = strlen(Fname);
			if (len > 0 && Fname[len - 1] == '\n') {
				Fname[len - 1] = '\0';
			}

			/* terminate with a period. */
			if (strlen(Fname) == 1 && Fname[0] == '.') {
				/* return a NULL pointer if '.' entered. */
				return (NULL);
			}

			/* open the file & check the version. */
			if (fopen_s(&Fp, Fname, "r")) {
				puts("File does not exist."); /* cannot open the file. */
			}
			else {
				if (CheckFileVersionQ(Fp, Version)) {
					return (Fp);
				}

				fclose(Fp);
			}
		}
	}
}

/*******************************************************************************
 *  Find number of media in the list. At the same time, check the
 *  optical parameters.
 ****/
bool FindNumMediaQ(FILE* Fp, short* NumMediaP) {
	char buf[STRLEN] = {0};
	char name[STRLEN] = {0};
	short num_media = 0;

	while (1) {
		strcpy_s(buf, sizeof(buf), FindDataLine(Fp));

		if (buf[0] == '\0') {
			printf("Missing end.\n");
			return (0);
		}
		if (strstr(buf, "end") != NULL) {
			break;
		}

		num_media++;

		double n, mua, mus, g;
		sscanf_s(buf, "%s%lf%lf%lf%lf", name, (unsigned)_countof(name), &n, &mua, &mus, &g);
		if (n <= 0 || mua < 0 || mus < 0 || g < -1 || g > 1) {
			printf("Bad optical parameters in %s\n", name);
			return (0);
		}
	}

	*NumMediaP = num_media;
	return (1);
}

/*******************************************************************************
 *  Read the parameters of one medium, assumming the
 *  parameters have been checked with FindNumMediaQ().
 ****/
bool ReadOneMediumQ(FILE* Fp, LayerStru* MediumP) {
	char buf[STRLEN] = {0};

	strcpy_s(buf, sizeof(buf), FindDataLine(Fp));
	if (buf[0] == '\0') {
		printf("Shouldn't happen here!");
		return (0);
	}
	sscanf_s(buf, "%s%lf%lf%lf%lf", MediumP->medium, (unsigned)_countof(MediumP->medium), &MediumP->n, &MediumP->mua,
			 &MediumP->mus, &MediumP->g);

	return (1);
}

/*******************************************************************************
 *  Read the media list.
 ****/
bool ReadMediumListQ(FILE* Fp, InStru* In_Ptr) {
	long file_pos = ftell(Fp);
	if (!FindNumMediaQ(Fp, &(In_Ptr->num_media))) {
		return (0);
	}
	fseek(Fp, file_pos, SEEK_SET);

	/* allocate an array for the layer parameters. */
	In_Ptr->mediumlist = (LayerStru*)malloc((In_Ptr->num_media) * sizeof(LayerStru));

	if (!(In_Ptr->mediumlist)) {
		printf("allocation failure in ReadMediumListQ()");
		return (0);
	}
	for (short i = 0; i < In_Ptr->num_media; i++) {
		ReadOneMediumQ(Fp, &(In_Ptr->mediumlist[i]));
	}
	FindDataLine(Fp); /* skip the signal end. */

	return (1);
}

/**************************************************************************
 *	Read the file name and the file format.
 *  Modernized with better variable declarations and const correctness.
 ****/
bool ReadFnameFormatQ(FILE* restrict Fp, InStru* restrict In_Ptr) {
	if (!Fp || !In_Ptr) {
		return false;
	}

	char buf[STRLEN] = {0};
	const char* line = FindDataLine(Fp);
	if (!line) {
		return false;
	}

	strcpy_s(buf, sizeof(buf), line);

	if (sscanf_s(buf, "%s %c", In_Ptr->out_fname, (unsigned)_countof(In_Ptr->out_fname), &In_Ptr->out_fformat, 1)
		!= 2) {
		printf("Reading file name and format.\n");
		return false;
	}
	In_Ptr->out_fformat = 'A'; /* now only support 'A' format. */

	return true;
}

/**************************************************************************
 *	Read the InStru members dz, dr and dt.
 *  Modernized with better variable declarations and const correctness.
 ****/
bool ReadDzDrDtQ(FILE* restrict Fp, InStru* restrict In_Ptr) {
	if (!Fp || !In_Ptr) {
		return false;
	}

	char buf[STRLEN] = {0};
	const char* line = FindDataLine(Fp);
	if (!line) {
		return false;
	}

	strcpy_s(buf, sizeof(buf), line);

	if (sscanf_s(buf, "%lf%lf%lf", &In_Ptr->dz, &In_Ptr->dr, &In_Ptr->dt) != 3) {
		printf("Reading dz, dr, dt. \n");
		return false;
	}

	if (In_Ptr->dz <= 0.0) {
		printf("Nonpositive dz.\n");
		return false;
	}
	if (In_Ptr->dr <= 0.0) {
		printf("Nonpositive dr.\n");
		return false;
	}
	if (In_Ptr->dt <= 0.0) {
		printf("Nonpositve dt. \n");
		return false;
	}
	return true;
}

/**************************************************************************
 *	Read the InStru members nz, nr, nt, na.
 ****/
bool ReadNzNrNtNaQ(FILE* Fp, InStru* In_Ptr) {
	char buf[STRLEN] = {0};

	/** read in number of dz, dr, da, dt. **/
	strcpy_s(buf, sizeof(buf), FindDataLine(Fp));

	float fz;
	float fr;
	float fa;
	float ft;
	if (sscanf_s(buf, "%f%f%f%f", &fz, &fr, &ft, &fa) != 4) {
		printf("Reading number of dz, dr, dt, da's.\n");
		return (0);
	}
	if (fz <= 0) {
		printf("Nonpositive number of dz's.\n");
		return (0);
	}
	if (fr <= 0) {
		printf("Nonpositive number of dr's.\n");
		return (0);
	}
	if (fa <= 0) {
		printf("Nonpositive number of da's.\n");
		return (0);
	}
	if (ft <= 0) {
		printf("Nonpositive number of dt's.\n");
		return (0);
	}

	In_Ptr->nz = (short)fz;
	In_Ptr->nr = (short)fr;
	In_Ptr->nt = (short)ft;
	In_Ptr->na = (short)fa;
	In_Ptr->da = 0.5 * M_PI / In_Ptr->na;

	return (1);
}

/**************************************************************************
 *   Initialize the RecordStru.
 ****/
void InitRecord(InStru* In_Ptr) {
	In_Ptr->record.Rd_r = 0;
	In_Ptr->record.Rd_a = 0;
	In_Ptr->record.Rd_ra = 0;
	In_Ptr->record.Rd_t = 0;
	In_Ptr->record.Rd_rt = 0;
	In_Ptr->record.Rd_at = 0;
	In_Ptr->record.Rd_rat = 0;
	In_Ptr->record.Td_r = 0;
	In_Ptr->record.Td_a = 0;
	In_Ptr->record.Td_ra = 0;
	In_Ptr->record.Td_t = 0;
	In_Ptr->record.Td_rt = 0;
	In_Ptr->record.Td_at = 0;
	In_Ptr->record.Td_rat = 0;
	In_Ptr->record.A_z = 0;
	In_Ptr->record.A_rz = 0;
	In_Ptr->record.A_t = 0;
	In_Ptr->record.A_zt = 0;
	In_Ptr->record.A_rzt = 0;
}

/**************************************************************************
 *	Change all characters in a string to upper case.
 ****/
char* ToUpperString(char* string) {
	for (int i = 0; i < (int)strlen(string); i++) {
		string[i] = toupper(string[i]);
	}

	return string;
}

/**************************************************************************
 *  Read which quantity is to be scored.
 ****/
bool ReadRecordQ(FILE* Fp, InStru* In_Ptr) {
	char buf[STRLEN] = {0};
	char string[STRLEN] = {0};
	bool error = 0;

	strcpy_s(buf, sizeof(buf), FindDataLine(Fp));
	char* index = buf;

	if (index[0] == '\0') {
		printf("Read scored quantities.\n");
		error = 1;
	}
	while (!error && index[0] != '\n' && index[0] != '#') {
		if (index[0] == '\\') { /* use '\' to continue in next line. */
			strcpy_s(buf, sizeof(buf), FindDataLine(Fp));
			index = buf;
		}
		sscanf_s(index, "%s", string, (unsigned)_countof(string));
		index = index + (char)strlen(string);
		while (index[0] == ' ' || index[0] == '\t') {
			index++;
		}

		if (strcmp(ToUpperString(string), "RD_R") == 0) {
			In_Ptr->record.Rd_r = 1;
		}
		else if (strcmp(ToUpperString(string), "RD_A") == 0) {
			In_Ptr->record.Rd_a = 1;
		}
		else if (strcmp(ToUpperString(string), "RD_RA") == 0) {
			In_Ptr->record.Rd_ra = 1;
		}
		else if (strcmp(ToUpperString(string), "RD_T") == 0) {
			In_Ptr->record.Rd_t = 1;
		}
		else if (strcmp(ToUpperString(string), "RD_RT") == 0) {
			In_Ptr->record.Rd_rt = 1;
		}
		else if (strcmp(ToUpperString(string), "RD_AT") == 0) {
			In_Ptr->record.Rd_at = 1;
		}
		else if (strcmp(ToUpperString(string), "RD_RAT") == 0) {
			In_Ptr->record.Rd_rat = 1;
		}
		else if (strcmp(ToUpperString(string), "TD_R") == 0) {
			In_Ptr->record.Td_r = 1;
		}
		else if (strcmp(ToUpperString(string), "TD_A") == 0) {
			In_Ptr->record.Td_a = 1;
		}
		else if (strcmp(ToUpperString(string), "TD_RA") == 0) {
			In_Ptr->record.Td_ra = 1;
		}
		else if (strcmp(ToUpperString(string), "TD_T") == 0) {
			In_Ptr->record.Td_t = 1;
		}
		else if (strcmp(ToUpperString(string), "TD_RT") == 0) {
			In_Ptr->record.Td_rt = 1;
		}
		else if (strcmp(ToUpperString(string), "TD_AT") == 0) {
			In_Ptr->record.Td_at = 1;
		}
		else if (strcmp(ToUpperString(string), "TD_RAT") == 0) {
			In_Ptr->record.Td_rat = 1;
		}
		else if (strcmp(ToUpperString(string), "A_Z") == 0) {
			In_Ptr->record.A_z = 1;
		}
		else if (strcmp(ToUpperString(string), "A_RZ") == 0) {
			In_Ptr->record.A_rz = 1;
		}
		else if (strcmp(ToUpperString(string), "A_T") == 0) {
			In_Ptr->record.A_t = 1;
		}
		else if (strcmp(ToUpperString(string), "A_ZT") == 0) {
			In_Ptr->record.A_zt = 1;
		}
		else if (strcmp(ToUpperString(string), "A_RZT") == 0) {
			In_Ptr->record.A_rzt = 1;
		}
		else {
			printf("Unknown quantity: %s\n", string);
			error = 1;
		}
	}

	return (!error);
}

/**************************************************************************
 *   Filter the RecordStru.
 ****/
bool FilterRecordQ(FILE* Fp, InStru* In_Ptr) {
	InitRecord(In_Ptr);

	if (!ReadRecordQ(Fp, In_Ptr)) {
		return (0);
	}

	if (In_Ptr->record.Rd_rat) {
		In_Ptr->record.Rd_ra = 0;
		In_Ptr->record.Rd_rt = 0;
		In_Ptr->record.Rd_at = 0;
		In_Ptr->record.Rd_r = 0;
		In_Ptr->record.Rd_a = 0;
		In_Ptr->record.Rd_t = 0;
	}
	if (In_Ptr->record.Rd_ra) {
		In_Ptr->record.Rd_r = 0;
		In_Ptr->record.Rd_a = 0;
	}
	if (In_Ptr->record.Rd_rt) {
		In_Ptr->record.Rd_r = 0;
		In_Ptr->record.Rd_t = 0;
	}
	if (In_Ptr->record.Rd_at) {
		In_Ptr->record.Rd_a = 0;
		In_Ptr->record.Rd_t = 0;
	}
	if (In_Ptr->record.Td_rat) {
		In_Ptr->record.Td_ra = 0;
		In_Ptr->record.Td_rt = 0;
		In_Ptr->record.Td_at = 0;
		In_Ptr->record.Td_r = 0;
		In_Ptr->record.Td_a = 0;
		In_Ptr->record.Td_t = 0;
	}
	if (In_Ptr->record.Td_ra) {
		In_Ptr->record.Td_r = 0;
		In_Ptr->record.Td_a = 0;
	}
	if (In_Ptr->record.Td_rt) {
		In_Ptr->record.Td_r = 0;
		In_Ptr->record.Td_t = 0;
	}
	if (In_Ptr->record.Td_at) {
		In_Ptr->record.Td_a = 0;
		In_Ptr->record.Td_t = 0;
	}
	if (In_Ptr->record.A_rzt) {
		In_Ptr->record.A_rz = 0;
		In_Ptr->record.A_zt = 0;
		In_Ptr->record.A_z = 0;
		In_Ptr->record.A_t = 0;
	}
	if (In_Ptr->record.A_rz) {
		In_Ptr->record.A_z = 0;
	}
	if (In_Ptr->record.A_zt) {
		In_Ptr->record.A_z = 0;
		In_Ptr->record.A_t = 0;
	}
	if (In_Ptr->record.A_zt) {
		In_Ptr->record.A_z = 0;
		In_Ptr->record.A_t = 0;
	}
	return (1);
}

/**************************************************************************
 *  Read the threshold weight.
 ****/
bool ReadWthQ(FILE* Fp, InStru* In_Ptr) {
	char buf[STRLEN] = {0};
	strcpy_s(buf, sizeof(buf), FindDataLine(Fp));

	if (sscanf_s(buf, "%lf", &(In_Ptr->Wth)) != 1) {
		printf("Reading threshold weight.\n");
		return (0);
	}
	if (In_Ptr->Wth < 0 || In_Ptr->Wth >= 1.0) {
		printf("Threshold weight out of range (0-1).\n");
		return (0);
	}
	return (1);
}

/**************************************************************************
 *  Read the random number seed.
 ****/
bool ReadRanSeedQ(FILE* Fp, InStru* In_Ptr) {
	char buf[STRLEN] = {0};
	strcpy_s(buf, sizeof(buf), FindDataLine(Fp));

	if (sscanf_s(buf, "%ld", &(In_Ptr->ran_seed)) != 1) {
		printf("Reading random number seed.\n");
		return (0);
	}
	if (In_Ptr->ran_seed < 0) {
		printf("Nonpositive random number seed.\n");
		return (0);
	}
	return (1);
}

/**************************************************************************
 *  Find number of layers.
 ****/
bool FindNumLayersQ(FILE* Fp, int* NumLayerP) {
	char buf[STRLEN] = {0};
	char name[STRLEN] = {0};
	short num_layers = 0;
	double thick;

	while (1) {
		strcpy_s(buf, sizeof(buf), FindDataLine(Fp));

		if (buf[0] == '\0') {
			printf("Missing end.\n");
			return (0);
		}
		if (strstr(buf, "end") != NULL) {
			break;
		}
		if ((sscanf_s(buf, "%s %lf", name, (unsigned)_countof(name), &thick) == 2)
			|| (sscanf_s(buf, "%s", name, (unsigned)_countof(name)) == 1)) {
			num_layers++;
		}
	}

	*NumLayerP = num_layers - 2;
	return (1);
}

/**************************************************************************
 *  Check whether the medium name is in the media list.
 ****/
bool ValidMediumNameQ(char* NameP, int* Index, InStru* In_Ptr) {
	for (short i = 0; i < In_Ptr->num_media; i++) {
		if (!strcmp(NameP, In_Ptr->mediumlist[i].medium)) {
			*Index = i;
			return (1);
		}
	}
	return (0);
}

/**************************************************************************
 *	Read the parameters of all layers.
 ****/
bool ReadLayerSpecsQ(FILE* Fp, InStru* In_Ptr) {
	char buf[STRLEN] = {0};
	char name[STRLEN] = {0};
	short i = 0;

	/* z coordinate of the current layer. */
	double z = 0.0;
	double thick = 0.0;

	long file_pos = ftell(Fp);

	int num_layers;
	if (!FindNumLayersQ(Fp, &num_layers)) {
		return (0);
	}
	fseek(Fp, file_pos, SEEK_SET);

	In_Ptr->num_layers = num_layers;

	/* Allocate an array for the layer parameters. */
	/* layer 0 and layer Num_Layers + 1 are for ambient. */
	In_Ptr->layerspecs = (LayerStru*)malloc((unsigned)(num_layers + 2) * sizeof(LayerStru));
	if (!(In_Ptr->layerspecs)) {
		printf("allocation failure in ReadLayerSpecsQ()");
		return (0);
	}
	for (i = 0; i <= num_layers + 1; i++) {
		strcpy_s(buf, sizeof(buf), FindDataLine(Fp));
		if (i == 0 || i == num_layers + 1) {
			if (sscanf_s(buf, "%s", name, (unsigned)_countof(name)) != 1) {
				printf("  Error in reading ambient layer name.\n");
				return (0);
			}
		}
		else {
			if (sscanf_s(buf, "%s%lf", name, (unsigned)_countof(name), &thick) != 2) {
				printf("  Error in ReadLayerSpecsQ().\n");
				return (0);
			}
			if (thick <= 0.0) {
				printf("  Nonpositive layer thickness.\n");
				return (0);
			}
		}

		int index;
		if (!ValidMediumNameQ(name, &index, In_Ptr)) {
			printf("  Invalid medium name. \n");
			return (0);
		}
		strcpy_s(In_Ptr->layerspecs[i].medium, sizeof(In_Ptr->layerspecs[i].medium), name);
		In_Ptr->layerspecs[i].n = In_Ptr->mediumlist[index].n;
		In_Ptr->layerspecs[i].mua = In_Ptr->mediumlist[index].mua;
		In_Ptr->layerspecs[i].mus = In_Ptr->mediumlist[index].mus;
		In_Ptr->layerspecs[i].g = In_Ptr->mediumlist[index].g;

		if ((i != 0) && (i != num_layers + 1)) {
			In_Ptr->layerspecs[i].z0 = z;
			z = z + thick;
			In_Ptr->layerspecs[i].z1 = z;
		}
		else if (i == 0) {
			In_Ptr->layerspecs[i].z0 = 0.0;
			In_Ptr->layerspecs[i].z1 = 0.0;
		}
		else if (i == (num_layers + 1)) {
			In_Ptr->layerspecs[i].z0 = z;
			In_Ptr->layerspecs[i].z1 = z;
		}
	}

	return (1);
}

/**************************************************************************
 *  Read the number of photons.
 *  Read computation time limit.
 *  Type = 0, read from a .mci input file;
 *  Type = 1, read from a .mco output file.
 ****/
bool ReadNumPhotonsQ(FILE* Fp, InStru* In_Ptr, char Type) {
	char buf[STRLEN] = {0};
	strcpy_s(buf, sizeof(buf), FindDataLine(Fp));

	if (Type == 0) {
		In_Ptr->add_num_photons = 0;
		In_Ptr->add_num_seconds = 0;
	}

	float temp;
	int hours;
	int minutes;
	if (sscanf_s(buf, "%f %d:%d", &temp, &hours, &minutes) == 3) {
		if (((long)temp > 0) && (hours * 3600 + minutes * 60) >= 0) {
			if (Type == 0) {
				In_Ptr->num_photons = (long)temp;
				In_Ptr->num_seconds = hours * 3600 + minutes * 60;
			}
			else {
				In_Ptr->add_num_photons = (long)temp;
				In_Ptr->add_num_seconds = hours * 3600 + minutes * 60;
			}

			In_Ptr->control_bit = 3;
		}
		else {
			printf("Nonpositive number of photons or time limit.\n");
			return (0);
		}
	}
	else if (sscanf_s(buf, "%d:%d", &hours, &minutes) == 2) {
		if ((hours * 3600 + minutes * 60) >= 0) {
			if (Type == 0) {
				In_Ptr->num_seconds = hours * 3600 + minutes * 60;
			}
			else {
				In_Ptr->add_num_seconds = hours * 3600 + minutes * 60;
			}

			In_Ptr->control_bit = 2;
		}
		else {
			printf("Nonpositive time limit.\n");
			return (0);
		}
	}
	else if (sscanf_s(buf, "%f", &temp) == 1) {
		if ((long)temp > 0) {
			if (Type == 0) {
				In_Ptr->num_photons = (long)temp;
			}
			else {
				In_Ptr->add_num_photons = (long)temp;
			}
			In_Ptr->control_bit = 1;
		}
		else {
			printf("Nonpositive number of photons.\n");
			return (0);
		}
	}
	else {
		printf("Invalid number of photons or time limit.\n");
		return (0);
	}

	return (1);
}

/**************************************************************************
 *  Read the beam source type (pencil/isotropic).
 ****/
bool ReadSourceTypeQ(FILE* Fp, InStru* In_Ptr) {
	char buf[STRLEN] = {0};
	char b_type[STRLEN] = {0};
	strcpy_s(buf, sizeof(buf), FindDataLine(Fp));

	if (sscanf_s(buf, "%s", b_type, (unsigned)_countof(b_type)) != 1) {
		printf("Reading photon source type. \n");
		return (0);
	}
	if (strcmp(b_type, "pencil") == 0) {
		In_Ptr->source_type = pencil;
		return (1);
	}
	if (strcmp(b_type, "isotropic") == 0) {
		In_Ptr->source_type = isotropic;
		return (1);
	}

	printf("Unknow photon source type. \n");
	return (0);
}

/**************************************************************************
 *      Compute the index to layer according to the z coordinate.
 *	If the z is on an interface between layers, the returned index
 *	will point to the upper layer.
 *	Index 0 is the top ambient medium and index num_layers+1 is the
 *	bottom one.
 ****/
bool ZToLayerQ(double z, short* index, InStru* In_Ptr) {
	/* index to layer. */
	short i = 0;
	short num_layers = In_Ptr->num_layers;

	if (z < 0.0) {
		printf("Nonpositive z coordinate.\n");
		return (0);
	}
	if (z > In_Ptr->layerspecs[num_layers].z1) {
		printf("Source is outside of the last layer. \n");
		return (0);
	}

	while (z > In_Ptr->layerspecs[i].z1) {
		i++;
	}
	*index = i;
	return (1);
}

/**************************************************************************
 *  Read starting position of photon source.
 ****/
bool ReadStartPQ(FILE* Fp, InStru* In_Ptr) {
	char buf[STRLEN] = {0};
	strcpy_s(buf, sizeof(buf), FindDataLine(Fp));

	double sz;
	short slayer;
	char smedium[STRLEN] = {0};

	/* z and medium. */
	if ((sscanf_s(buf, "%lf %s", &sz, smedium, (unsigned)_countof(smedium)) == 2)
		&& (smedium[0] != '#' && smedium[0] != '\n')) {
		if (!ZToLayerQ(sz, &slayer, In_Ptr)) {
			return (0);
		}

		if (strcmp(In_Ptr->layerspecs[slayer].medium, smedium) != 0) {
			if ((fabs(sz - In_Ptr->layerspecs[slayer].z1) < DBL_EPSILON)
				&& (strcmp(In_Ptr->layerspecs[slayer + 1].medium, smedium) == 0)) {
				slayer++;
				if (slayer > In_Ptr->num_layers) {
					puts("Source is outside of the last layer.");
					return (0);
				}
			}
			else {
				printf("Medium name and z coordinate do not match.\n");
				return (0);
			}
		}
	}
	/* z only. */
	else if (sscanf_s(buf, "%lf", &(sz)) == 1) {
		if (!ZToLayerQ(sz, &In_Ptr->slayer, In_Ptr)) {
			return (0);
		}
		strcpy_s(smedium, sizeof(smedium), "");
	}
	else {
		printf("Invalid starting position of photon source.\n");
		return (0);
	}

	if ((In_Ptr->source_type == isotropic) && (sz == 0.0)) {
		printf("Can not put isotropic source in upper ambient medium.\n");
		return 0;
	}

	In_Ptr->sz = sz;
	strcpy_s(In_Ptr->smedium, sizeof(In_Ptr->smedium), smedium);
	return (1);
}

/*************************************************************************
 *	Compute the critical angles for total internal
 *	reflection according to the relative refractive index
 *	of the layer.
 *	All layers are processed.
 ****/
void CriticalAngle(short Num_Layers, LayerStru** Layerspecs_PP) {
	for (short i = 1; i <= Num_Layers; i++) {
		double n1 = (*Layerspecs_PP)[i].n;
		double n2 = (*Layerspecs_PP)[i - 1].n;
		(*Layerspecs_PP)[i].cos_crit0 = n1 > n2 ? sqrt(1.0 - (n2 * n2 / (n1 * n1))) : 0.0;

		n2 = (*Layerspecs_PP)[i + 1].n;
		(*Layerspecs_PP)[i].cos_crit1 = n1 > n2 ? sqrt(1.0 - (n2 * n2 / (n1 * n1))) : 0.0;
	}
}

/**************************************************************************
 *	Read in the input parameters for one run.
 ****/
void ReadRunParam(FILE* Fp, InStru* In_Ptr) {
	if (!ReadFnameFormatQ(Fp, In_Ptr)) {
		exit(1);
	}

	/* geometry. */
	if (!ReadLayerSpecsQ(Fp, In_Ptr)) {
		exit(1);
	}

	/* skip the signal "end" of layers. */
	FindDataLine(Fp);

	/* source. */
	if (!ReadSourceTypeQ(Fp, In_Ptr)) {
		exit(1);
	}
	if (!ReadStartPQ(Fp, In_Ptr)) {
		exit(1);
	}

	/* grids. */
	if (!ReadDzDrDtQ(Fp, In_Ptr)) {
		exit(1);
	}
	if (!ReadNzNrNtNaQ(Fp, In_Ptr)) {
		exit(1);
	}
	In_Ptr->zm = In_Ptr->dz * In_Ptr->nz;
	In_Ptr->rm = In_Ptr->dr * In_Ptr->nr;
	In_Ptr->tm = In_Ptr->dt * In_Ptr->nt;
	In_Ptr->am = In_Ptr->da * In_Ptr->na;

	/* scored data categories. */
	if (!FilterRecordQ(Fp, In_Ptr)) {
		exit(1);
	}

	/* simulation control. */
	if (!ReadNumPhotonsQ(Fp, In_Ptr, 0)) {
		exit(1);
	}
	if (!ReadWthQ(Fp, In_Ptr)) {
		exit(1);
	}
	if (!ReadRanSeedQ(Fp, In_Ptr)) {
		exit(1);
	}

	CriticalAngle(In_Ptr->num_layers, &In_Ptr->layerspecs);
}

/**************************************************************************
 *  Read the media list in interactive mode.
 ****/
void InterReadMediumList(InStru* In_Ptr) {
	char string[STRLEN] = {0};
	char medium[STRLEN] = {0};

	printf("Specify medium list. Total number of mediums: ");

	fgets(string, STRLEN, stdin);
	while (sscanf_s(string, "%hd", &In_Ptr->num_media) != 1 || In_Ptr->num_media <= 0) {
		printf("Invalid medium number. Input again: ");
		fgets(string, STRLEN, stdin);
	}

	/* allocate an array for the layer parameters. */
	In_Ptr->mediumlist = (LayerStru*)malloc((In_Ptr->num_media) * sizeof(LayerStru));

	if (!(In_Ptr->mediumlist)) {
		printf("allocation failure in ReadMediumListQ()");
		exit(1);
	}

	for (short i = 0; i < In_Ptr->num_media; i++) {
		printf("Specify medium %d: \n  Medium name: ", i + 1);
		fgets(string, STRLEN, stdin);

		bool name_taken = 0;
		do {
			sscanf_s(string, "%s", medium, (unsigned)_countof(medium));
			for (short j = 0; j < i; j++) {
				if (strcmp(In_Ptr->mediumlist[j].medium, medium) == 0) {
					name_taken = 1;
					printf("  Duplicate medium. Input again: ");
					fgets(string, STRLEN, stdin);
					break;
				}
			}
		}
		while (name_taken);

		strcpy_s(In_Ptr->mediumlist[i].medium, sizeof(In_Ptr->mediumlist[i].medium), medium);

		printf("  Refractive index n (>= 1.0): ");
		fgets(string, STRLEN, stdin);
		while (sscanf_s(string, "%lf", &In_Ptr->mediumlist[i].n) != 1 || In_Ptr->mediumlist[i].n < 1.0) {
			printf("  Invalid refractive index. Input again (>= 1.0): ");
			fgets(string, STRLEN, stdin);
		}

		printf("  Absorption coefficient mua (>= 0.0 /cm): ");
		fgets(string, STRLEN, stdin);
		while (sscanf_s(string, "%lf", &In_Ptr->mediumlist[i].mua) != 1 || In_Ptr->mediumlist[i].mua < 0.0) {
			printf("  Invalid absorption coefficient. Input again (>= 0.0): ");
			fgets(string, STRLEN, stdin);
		}

		printf("  Scattering coefficient mus (>= 0.0 /cm): ");
		fgets(string, STRLEN, stdin);
		while (sscanf_s(string, "%lf", &In_Ptr->mediumlist[i].mus) != 1 || In_Ptr->mediumlist[i].mus < 0.0) {
			printf("  Invalid scattering coefficient. Input again (>= 0.0): ");
			fgets(string, STRLEN, stdin);
		}

		printf("  Anisotropy factor g (0.0 - 1.0): ");
		fgets(string, STRLEN, stdin);
		while (sscanf_s(string, "%lf", &In_Ptr->mediumlist[i].g) != 1 || In_Ptr->mediumlist[i].g < 0.0
			   || In_Ptr->mediumlist[i].g > 1.0) {
			printf("  Invalid anisotropy factor. Input again (0.0 - 1.0): ");
			fgets(string, STRLEN, stdin);
		}
		printf("\n");
	}
}

/**************************************************************************
 *  Read the file name and the file format interactively.
 ****/
void InterReadFnameFormat(InStru* In_Ptr) {
	FILE* file;
	char fname[STRLEN] = {0};
	char fmode[STRLEN] = {0};

	do {
		printf("Specify output filename with extension .mco: ");
		fgets(fname, STRLEN, stdin);
		fmode[0] = 'w';

		/* file exists. */
		if (!fopen_s(&file, fname, "r")) {
			printf("File %s exists, %s", fname, "w=overwrite, n=new filename: ");

			/* avoid null line. */
			do {
				fgets(fmode, STRLEN, stdin);
			}
			while (!strlen(fmode));
			fclose(file);
		}
	}
	while (fmode[0] != 'w');

	strcpy_s(In_Ptr->out_fname, sizeof(In_Ptr->out_fname), fname);

	/* now only support 'A' format. */
	In_Ptr->out_fformat = 'A';

	printf("\n");
}

/**************************************************************************
 *	Read dz, dr, dt interactively.
 ****/
void InterReadDzDrDt(InStru* In_Ptr) {
	do {
		printf("Specify dz, dr, dt in one line\n");
		printf("(all > 0.0 cm, e.g., 0.1 0.1 0.1): ");
	}
	while (!ReadDzDrDtQ(stdin, In_Ptr));

	printf("\n");
}

/**************************************************************************
 *      Read the InStru members nz, nr, na interactively.
 ****/
void InterReadNzNrNtNa(InStru* In_Ptr) {
	do {
		printf("Specify nz, nr, nt, na in one line\n");
		printf("(all > 0, e.g., 100 100 100 100): ");
	}
	while (!ReadNzNrNtNaQ(stdin, In_Ptr));

	In_Ptr->da = 0.5 * M_PI / In_Ptr->na;
	printf("\n");
}

/**************************************************************************
 *	Read and filter the quantities to be scored interactively.
 ****/
void InterFilterRecord(InStru* In_Ptr) {
	do {
		printf("Select scored quantities from the following data categories:\n");
		printf("\tRd_rat\t\t\tTd_rat\t\t\tA_rzt\n");
		printf("\tRd_ra\tRd_rt\tRd_at\tTd_ra\tTd_rt\tRd_at\tA_rz\tA_zt\n");
		printf("\tRd_r\tRd_a\tRd_t\tTd_r\tTd_a\tTd_t\tA_z\tA_t\n");
	}
	while (!FilterRecordQ(stdin, In_Ptr));

	printf("\n");
}

/**************************************************************************
 *	Read the threshold weight interactively.
 ****/
void InterReadWth(InStru* In_Ptr) {
	char string[STRLEN] = {0};

	printf("Input threshold weight (0 <= wth < 1.0, 0.0001 recommended): ");
	fgets(string, STRLEN, stdin);
	while (sscanf_s(string, "%lf", &In_Ptr->Wth) != 1 || In_Ptr->Wth < 0 || In_Ptr->Wth >= 1) {
		printf("Invalid wth. Input again (0 <= wth < 1.0): ");
		fgets(string, STRLEN, stdin);
	}

	printf("\n");
}

/**************************************************************************
 *	Read the random seed interactively.
 ****/
void InterReadRanSeed(InStru* In_Ptr) {
	char string[STRLEN] = {0};

	printf("Input random number seed (1 <= ran_seed <= 32000): ");
	fgets(string, STRLEN, stdin);
	while (sscanf_s(string, "%ld", &In_Ptr->ran_seed) != 1 || In_Ptr->ran_seed < 1 || In_Ptr->ran_seed > 32000) {
		printf("Invalid ran_seed. Input again (1 <= ran_seed <= 32000): ");
		fgets(string, STRLEN, stdin);
	}

	printf("\n");
}

/**************************************************************************
 ****/
void PrintMediumNames(InStru* In_Ptr) {
	printf("Available medium types:\n");

	int j = 1;
	for (int i = 0; i < In_Ptr->num_media; i++) {
		printf("%-16s", In_Ptr->mediumlist[i].medium);
		if (j % 4 == 0) {
			printf("\n");
		}
		j++;
	}

	printf("\n");
}

/**************************************************************************
 *	Read layer specifications interactively.
 ****/
void InterReadLayerSpecs(InStru* In_Ptr) {
	char string[STRLEN] = {0};
	char name[STRLEN] = {0};

	/* z coordinate of the current layer. */
	double z = 0.0;

	int index;

	printf("\nSpecify layer list. ");
	PrintMediumNames(In_Ptr);
	printf("\nTotal number of layers: ");

	fgets(string, STRLEN, stdin);
	while (sscanf_s(string, "%hd", &In_Ptr->num_layers) != 1 || In_Ptr->num_layers <= 0) {
		printf("Invalid layer number. Input again: ");
		fgets(string, STRLEN, stdin);
	}

	/* Allocate an array for the layer parameters. */
	/* layer 0 and layer Num_Layers + 1 are for ambient. */
	In_Ptr->layerspecs = (LayerStru*)malloc((unsigned)(In_Ptr->num_layers + 2) * sizeof(LayerStru));

	if (!(In_Ptr->layerspecs)) {
		printf("allocation failure in ReadLayerSpecsQ()");
		exit(1);
	}

	for (short i = 0; i <= In_Ptr->num_layers + 1; i++) {
		bool error = 1;
		while (error) {
			error = 0;
			if (i == 0) {
				printf("\n  Name of upper ambient medium: ");
				fgets(string, STRLEN, stdin);
				sscanf_s(string, "%s", name, (unsigned)_countof(name));
			}
			else if (i == In_Ptr->num_layers + 1) {
				printf("\n  Name of lower ambient medium: ");
				fgets(string, STRLEN, stdin);
				sscanf_s(string, "%s", name, (unsigned)_countof(name));
			}
			else {
				printf("\n  Medium name of layer %d: ", i);
				fgets(string, STRLEN, stdin);
				sscanf_s(string, "%s", name, (unsigned)_countof(name));
			}

			if (!ValidMediumNameQ(name, &index, In_Ptr)) {
				printf("  Invalid medium name. Input again.");
				error = 1;
			}
		}

		strcpy_s(In_Ptr->layerspecs[i].medium, sizeof(In_Ptr->layerspecs[i].medium), name);
		In_Ptr->layerspecs[i].n = In_Ptr->mediumlist[index].n;
		In_Ptr->layerspecs[i].mua = In_Ptr->mediumlist[index].mua;
		In_Ptr->layerspecs[i].mus = In_Ptr->mediumlist[index].mus;
		In_Ptr->layerspecs[i].g = In_Ptr->mediumlist[index].g;

		if ((i != 0) && (i != In_Ptr->num_layers + 1)) {
			printf("  Input the thickness of layer %d (thickness > 0.0 cm): ", i);
			fgets(string, STRLEN, stdin);

			double thick = 0.0;
			while (sscanf_s(string, "%lf", &thick) != 1 || thick <= 0) {
				printf("  Invalid thickness. Input again (thickness > 0.0 cm): ");
				fgets(string, STRLEN, stdin);
			}
			In_Ptr->layerspecs[i].z0 = z;
			z = z + thick;
			In_Ptr->layerspecs[i].z1 = z;
		}
		else if (i == 0) {
			In_Ptr->layerspecs[i].z0 = 0.0;
			In_Ptr->layerspecs[i].z1 = 0.0;
		}
		else if (i == In_Ptr->num_layers + 1) {
			In_Ptr->layerspecs[i].z0 = z;
			In_Ptr->layerspecs[i].z1 = z;
		}
	}

	printf("\n");
}

/**************************************************************************
 *  Read the number of photons, or computation time interactively.
 ****/
void InterReadNumPhotons(InStru* In_Ptr) {
	printf("Specify number of photons or time in hh:mm format,\n");
	printf("or both in one line (e.g. 10000 5:30): ");

	while (!ReadNumPhotonsQ(stdin, In_Ptr, 0)) {
		printf("Input again: ");
	}

	printf("\n");
}

/**************************************************************************
 *  Read the beam source type (pencil/isotropic).
 ****/
void InterReadSourceType(InStru* In_Ptr) {
	char string[STRLEN] = {0};

	printf("Input source type (p = pencil / i = isotropic): ");
	fgets(string, STRLEN, stdin);

	char c;
	while (sscanf_s(string, "%c", &c, 1) != 1 || !(toupper(c) == 'P' || toupper(c) == 'I')) {
		printf("Invalid type. Input again (p = pencil / i = isotropic): ");
		fgets(string, STRLEN, stdin);
	}

	if (toupper(c) == 'P') {
		In_Ptr->source_type = pencil;
	}
	else {
		In_Ptr->source_type = isotropic;
	}

	printf("\n");
}

/**************************************************************************
 *  Read starting position of photon source.
 ****/
void InterReadStartP(InStru* In_Ptr) {
	do {
		printf("Input the z coordinate of source (0.0 - %f cm) and the medium\n",
			   In_Ptr->layerspecs[In_Ptr->num_layers].z1);
		printf("where the source is if the z is on an interface (e.g. 1.0 [air]):");
	}
	while (!ReadStartPQ(stdin, In_Ptr));

	printf("\n");
}

/*************************************************************************
 *  If Fp is stdout, freeze the screen and print a more message on screen
 *  every 20 lines. The Line is the line index.
 ****/
void More(FILE* Fp, const int* Line) {
	if (Fp == stdout) {
		if (!((*Line) % 20)) {
			printf("--More-- (Press Return to continue)");
			fflush(Fp);

			char c;
			do {
				fread(&c, 1, 1, stdin);
			}
			while (c != '\n');
		}
	}
}

/*************************************************************************
 *  Write medium list to the file Fp.
 *  if Fp is stdout, freeze the screen every 20 lines.
 ****/
void PutMediumListToFile(FILE* Fp, InStru* In_Ptr, int* Line) {
	char format[STRLEN] = {0};

	fprintf(Fp, "# Specify media \n");
	(*Line)++;
	fprintf(Fp, "#\tname\t\tn\tmua\tmus\tg\n");
	(*Line)++;

	for (int i = 0; i < In_Ptr->num_media; i++) {
		LayerStru s;

		More(Fp, Line);
		s = In_Ptr->mediumlist[i];
		if (strlen(s.medium) + 1 > 8) {
			strcpy_s(format, sizeof(format), "\t%s \t%G\t%G\t%G\t%G\n");
		}
		else {
			strcpy_s(format, sizeof(format), "\t%s \t\t%G\t%G\t%G\t%G\n");
		}

		fprintf(Fp, format, s.medium, s.n, s.mua, s.mus, s.g);
		(*Line)++;
	}
	fprintf(Fp, "end #of media\n");
	(*Line)++;
}

void PutFnameFormatToFile(FILE* Fp, InStru* In_Ptr, int* Line) {
	More(Fp, Line);
	fprintf(Fp, "%s \t%c\t\t\t# output file name, format.\n", In_Ptr->out_fname, In_Ptr->out_fformat);
	(*Line)++;
}

void PutDzDrDtToFile(FILE* Fp, InStru* In_Ptr, int* Line) {
	More(Fp, Line);
	fprintf(Fp, "%G\t%G\t%G\t\t\t# dz, dr, dt.\n", In_Ptr->dz, In_Ptr->dr, In_Ptr->dt);
	(*Line)++;
}

void PutNzNrNtNaToFile(FILE* Fp, InStru* In_Ptr, int* Line) {
	More(Fp, Line);
	fprintf(Fp, "%d\t%d\t%d\t%d\t\t# nz, nr, nt, na.\n", In_Ptr->nz, In_Ptr->nr, In_Ptr->nt, In_Ptr->na);
	(*Line)++;
}

void PutScoredToFile(FILE* Fp, InStru* In_Ptr, int* Line) {
	More(Fp, Line);
	fprintf(Fp, "# This simulation will score the following categories:\n");
	(*Line)++;

	More(Fp, Line);
	if (In_Ptr->record.Rd_r) {
		fprintf(Fp, "Rd_r \t");
	}
	if (In_Ptr->record.Rd_a) {
		fprintf(Fp, "Rd_a \t");
	}
	if (In_Ptr->record.Rd_ra) {
		fprintf(Fp, "Rd_ra \t");
	}
	if (In_Ptr->record.Rd_t) {
		fprintf(Fp, "Rd_t \t");
	}
	if (In_Ptr->record.Rd_rt) {
		fprintf(Fp, "Rd_rt \t");
	}
	if (In_Ptr->record.Rd_at) {
		fprintf(Fp, "Rd_at \t");
	}
	if (In_Ptr->record.Rd_rat) {
		fprintf(Fp, "Rd_rat \t");
	}

	if (In_Ptr->record.Td_r) {
		fprintf(Fp, "Td_r \t");
	}
	if (In_Ptr->record.Td_a) {
		fprintf(Fp, "Td_a \t");
	}
	if (In_Ptr->record.Td_ra) {
		fprintf(Fp, "Td_ra \t");
	}
	if (In_Ptr->record.Td_t) {
		fprintf(Fp, "Td_t \t");
	}
	if (In_Ptr->record.Td_rt) {
		fprintf(Fp, "Td_rt \t");
	}
	if (In_Ptr->record.Td_at) {
		fprintf(Fp, "Td_at \t");
	}
	if (In_Ptr->record.Td_rat) {
		fprintf(Fp, "Td_rat \t");
	}

	if (In_Ptr->record.A_z) {
		fprintf(Fp, "A_z \t");
	}
	if (In_Ptr->record.A_rz) {
		fprintf(Fp, "A_rz \t");
	}
	if (In_Ptr->record.A_t) {
		fprintf(Fp, "A_t \t");
	}
	if (In_Ptr->record.A_zt) {
		fprintf(Fp, "A_zt \t");
	}
	if (In_Ptr->record.A_rzt) {
		fprintf(Fp, "A_rzt \t");
	}

	fprintf(Fp, "\n");
	(*Line)++;
}

void PutWthToFile(FILE* Fp, InStru* In_Ptr, int* Line) {
	More(Fp, Line);
	fprintf(Fp, "%G\t\t\t\t\t# threshold weight.\n", In_Ptr->Wth);
	(*Line)++;
}

void PutRanSeedToFile(FILE* Fp, InStru* In_Ptr, int* Line) {
	More(Fp, Line);
	fprintf(Fp, "%ld\t\t\t\t\t# random number seed.\n", In_Ptr->ran_seed);
	(*Line)++;
}

void PutLayerSpecsToFile(FILE* Fp, InStru* In_Ptr, int* Line) {
	char format[STRLEN] = {0};

	More(Fp, Line);
	fprintf(Fp, "# \tmedium \t\tthickness\n");
	(*Line)++;

	for (int i = 0; i <= In_Ptr->num_layers + 1; i++) {
		LayerStru s;
		More(Fp, Line);

		s = In_Ptr->layerspecs[i];
		if (i != 0 && i != In_Ptr->num_layers + 1) {
			if (strlen(s.medium) + 1 > 8) {
				strcpy_s(format, sizeof(format), "\t%s \t%G\n");
			}
			else {
				strcpy_s(format, sizeof(format), "\t%s \t\t%G\n");
			}
			fprintf(Fp, format, s.medium, s.z1 - s.z0);
		}
		else {
			fprintf(Fp, "\t%s\n", s.medium);
		}
		(*Line)++;
	}

	More(Fp, Line);
	fprintf(Fp, "end #of layers\n");
	(*Line)++;
}

void PutNumPhotonsToFile(FILE* Fp, InStru* In_Ptr, int* Line) {
	More(Fp, Line);

	if (In_Ptr->control_bit == 1) {
		fprintf(Fp, "%ld  \t\t\t\t\t# no. of photons | time\n", In_Ptr->num_photons);
	}
	else if (In_Ptr->control_bit == 2) {
		fprintf(Fp, "%ld:%ld\t\t\t\t\t# no. of photons | time\n", In_Ptr->num_seconds / 3600,
				In_Ptr->num_seconds % 3600 / 60);
	}
	else {
		fprintf(Fp, "%ld  \t%ld:%ld\t\t\t\t# no. of photons | time\n", In_Ptr->num_photons, In_Ptr->num_seconds / 3600,
				In_Ptr->num_seconds % 3600 / 60);
	}

	(*Line)++;
}

void PutSourceTypeToFile(FILE* Fp, InStru* In_Ptr, int* Line) {
	More(Fp, Line);

	if (In_Ptr->source_type == pencil) {
		fprintf(Fp, "pencil \t\t\t\t\t# src type: pencil/isotropic.\n");
	}
	else {
		fprintf(Fp, "isotropic \t\t\t\t# src type: pencil/isotropic.\n");
	}

	(*Line)++;
}

void PutStartPToFile(FILE* Fp, InStru* In_Ptr, int* Line) {
	More(Fp, Line);

	if (strlen(In_Ptr->smedium) == 0) {
		fprintf(Fp, "%G\t\t\t\t\t# starting position of source.\n", In_Ptr->sz);
	}
	else if (strlen(In_Ptr->smedium) + 1 > 8) {
		fprintf(Fp, "%G\t%s \t\t\t# starting position of source.\n", In_Ptr->sz, In_Ptr->smedium);
	}
	else {
		fprintf(Fp, "%G\t%s \t\t\t\t# starting position of source.\n", In_Ptr->sz, In_Ptr->smedium);
	}

	(*Line)++;
}

/*************************************************************************
 *  Write input parameters to the file Fp.
 *  If Fp is stdout, freeze the screen every 20 lines.
 ****/
void PutInputToFile(FILE* Fp, InStru* In_Ptr) {
	/* line index. */
	fprintf(Fp, "mcmli2.1 \t\t\t# file version \n\n");
	int line = 2;
	PutMediumListToFile(Fp, In_Ptr, &line);

	More(Fp, &line);
	fprintf(Fp, "\n# Specify data for run 1\n");
	line += 2;

	PutFnameFormatToFile(Fp, In_Ptr, &line);

	/* geometry. */
	More(Fp, &line);
	fprintf(Fp, "\n");
	line++;
	PutLayerSpecsToFile(Fp, In_Ptr, &line);

	/* source. */
	More(Fp, &line);
	fprintf(Fp, "\n");
	line++;
	PutSourceTypeToFile(Fp, In_Ptr, &line);
	PutStartPToFile(Fp, In_Ptr, &line);

	/* grids. */
	More(Fp, &line);
	fprintf(Fp, "\n");
	line++;
	PutDzDrDtToFile(Fp, In_Ptr, &line);
	PutNzNrNtNaToFile(Fp, In_Ptr, &line);

	/* scored data categories. */
	More(Fp, &line);
	fprintf(Fp, "\n");
	line++;
	PutScoredToFile(Fp, In_Ptr, &line);

	/* simulation control. */
	More(Fp, &line);
	fprintf(Fp, "\n");
	line++;

	PutNumPhotonsToFile(Fp, In_Ptr, &line);
	PutWthToFile(Fp, In_Ptr, &line);
	PutRanSeedToFile(Fp, In_Ptr, &line);

	More(Fp, &line);
	fprintf(Fp, "end #of runs\n\n");
}

/**************************************************************************
 *  Read in the input parameters for one run in interactive mode.
 ****/
void InterReadParam(InStru* In_Ptr) {
	char string[STRLEN] = {0};

	InterReadMediumList(In_Ptr);
	InterReadFnameFormat(In_Ptr);
	InterReadLayerSpecs(In_Ptr);
	InterReadSourceType(In_Ptr);
	InterReadStartP(In_Ptr);
	InterReadDzDrDt(In_Ptr);
	InterReadNzNrNtNa(In_Ptr);
	InterFilterRecord(In_Ptr);
	InterReadNumPhotons(In_Ptr);
	InterReadWth(In_Ptr);
	InterReadRanSeed(In_Ptr);

	printf("Do you want to save the input to a file? (Y/N)");
	fgets(string, STRLEN, stdin);
	if (toupper(string[0]) == 'Y') {
		printf("Give the file name to save input: ( .mci): ");
		fgets(string, STRLEN, stdin);

		FILE* fp;
		if (fopen_s(&fp, string, "w")) {
			puts("Can not open the file to write.");
		}
		else {
			PutInputToFile(fp, In_Ptr);
		}
	}
}

/**************************************************************************
 *  Check consistance of input parameters for one run.
 *  Such as: the consistance of medium list, layer
 *  list, souce starting position and source type.
 ****/
bool CheckInputConsis(InStru* In_Ptr) {
	for (int i = 0; i <= In_Ptr->num_layers + 1; i++) {
		int index;
		if (!ValidMediumNameQ(In_Ptr->layerspecs[i].medium, &index, In_Ptr)) {
			printf("Invalid medium name of layer %d.\n", i);
			return 0;
		}

		In_Ptr->layerspecs[i].n = In_Ptr->mediumlist[index].n;
		In_Ptr->layerspecs[i].mua = In_Ptr->mediumlist[index].mua;
		In_Ptr->layerspecs[i].mus = In_Ptr->mediumlist[index].mus;
		In_Ptr->layerspecs[i].g = In_Ptr->mediumlist[index].g;
	}

	if ((In_Ptr->source_type == isotropic) && (In_Ptr->sz == 0.0)) {
		printf("Can not put isotropic source in upper ambient medium.\n");
		return 0;
	}
	if (!ZToLayerQ(In_Ptr->sz, &In_Ptr->slayer, In_Ptr)) {
		return 0;
	}

	if (In_Ptr->smedium[0] != '\0') {
		if (strcmp(In_Ptr->layerspecs[In_Ptr->slayer].medium, In_Ptr->smedium) != 0) {
			if ((fabs(In_Ptr->sz - In_Ptr->layerspecs[In_Ptr->slayer].z1) < DBL_EPSILON)
				&& (strcmp(In_Ptr->layerspecs[In_Ptr->slayer + 1].medium, In_Ptr->smedium) == 0)) {
				In_Ptr->slayer++;
			}
			else {
				printf("Medium name and z coordinate do not match.\n");
				return (0);
			}
		}
	}
	return 1;
}

/****************************************************************************
 *    Menu for changing input parameters.
 *****/
void ShowChangeMenu() {
	puts("  o = Print the input on screen.");
	puts("  m = Change media list.");
	puts("  f = Change output file name and format.");
	puts("  d = Change dz, dr, dt.");
	puts("  n = Change nz, nr, nt, na.");
	puts("  c = Change scored data categories.");
	puts("  w = Change threshold weight.");
	puts("  r = Change random number seed.");
	puts("  l = Change layer specifications.");
	puts("  p = Change photon number and computation time limit.");
	puts("  s = Change source type.");
	puts("  z = Change source starting position.");
	puts("  q = Quit from change menu and start simulation.");
	puts("  x = Exit to the main menu.");
	puts("  * Commands here are not case-sensitive");
}

/***********************************************************************
 *   Continue to change input parameters or quit.
 *****/
char QuitOrContinue() {
	char string[STRLEN] = {0};

	do {
		printf("Do you want to change them? (Y/N): ");
		do {
			fgets(string, STRLEN, stdin);
		}
		while (!strlen(string));
	}
	while (toupper(string[0]) != 'Y' && toupper(string[0]) != 'N');

	return (toupper(string[0]));
}

void ChangeMediumList(InStru* In_Ptr) {
	int line = 1;
	printf("Current medium list: \n");
	PutMediumListToFile(stdout, In_Ptr, &line);
	printf("\n");

	if (QuitOrContinue() == 'Y') {
		free(In_Ptr->mediumlist);
		InterReadMediumList(In_Ptr);
	}
}

void ChangeFnameFormat(InStru* In_Ptr) {
	int line = 1;
	printf("Current output file name and format: \n");
	PutFnameFormatToFile(stdout, In_Ptr, &line);
	printf("\n");
	InterReadFnameFormat(In_Ptr);
}

void ChangeDzDrDt(InStru* In_Ptr) {
	int line = 1;
	printf("Current dz, dr, dt: \n");
	PutDzDrDtToFile(stdout, In_Ptr, &line);
	printf("\n");
	InterReadDzDrDt(In_Ptr);
}

void ChangeNzNrNtNa(InStru* In_Ptr) {
	int line = 1;
	printf("Current nz, nr, nt, na: \n");
	PutNzNrNtNaToFile(stdout, In_Ptr, &line);
	printf("\n");
	InterReadNzNrNtNa(In_Ptr);
}

void ChangeRecord(InStru* In_Ptr) {
	int line = 1;
	PutScoredToFile(stdout, In_Ptr, &line);
	printf("\n");

	if (QuitOrContinue() == 'Y') {
		InterFilterRecord(In_Ptr);
	}
}

void ChangeWth(InStru* In_Ptr) {
	int line = 1;
	printf("Current threshold weight: \n");
	PutWthToFile(stdout, In_Ptr, &line);
	printf("\n");
	InterReadWth(In_Ptr);
}

void ChangeRanSeed(InStru* In_Ptr) {
	int line = 1;
	printf("Current random number seed: \n");
	PutRanSeedToFile(stdout, In_Ptr, &line);
	printf("\n");
	InterReadRanSeed(In_Ptr);
}

void ChangeLayerSpecs(InStru* In_Ptr) {
	int line = 1;
	printf("Current layer sepcification: \n");
	PutLayerSpecsToFile(stdout, In_Ptr, &line);
	printf("\n");

	if (QuitOrContinue() == 'Y') {
		InterReadLayerSpecs(In_Ptr);
	}
}

void ChangeNumPhotons(InStru* In_Ptr) {
	int line = 1;
	printf("Current value: \n");
	PutNumPhotonsToFile(stdout, In_Ptr, &line);
	printf("\n");
	InterReadNumPhotons(In_Ptr);
}

void ChangeSourceType(InStru* In_Ptr) {
	int line = 1;
	printf("Current source type: \n");
	PutSourceTypeToFile(stdout, In_Ptr, &line);
	printf("\n");
	InterReadSourceType(In_Ptr);
}

void ChangeStartP(InStru* In_Ptr) {
	int line = 1;
	printf("Layer Specification: \n");
	PutLayerSpecsToFile(stdout, In_Ptr, &line);
	printf("\nCurrent starting position: \n");
	PutStartPToFile(stdout, In_Ptr, &line);
	printf("\n");
	InterReadStartP(In_Ptr);
}

/************************************************************************
 *  return 1 if string[0] = Q, quit change menu;
 *  return 2 if string[0] = X, quit to the main menu;
 *  return 0 otherwise.
 ****/
int BranchChangeMenu(char* string, InStru* In_Ptr) {
	switch (toupper(string[0])) {
		case 'M': ChangeMediumList(In_Ptr); break;

		case 'F': ChangeFnameFormat(In_Ptr); break;

		case 'D': ChangeDzDrDt(In_Ptr); break;

		case 'N': ChangeNzNrNtNa(In_Ptr); break;

		case 'C': ChangeRecord(In_Ptr); break;

		case 'W': ChangeWth(In_Ptr); break;

		case 'R': ChangeRanSeed(In_Ptr); break;

		case 'L': ChangeLayerSpecs(In_Ptr); break;

		case 'P': ChangeNumPhotons(In_Ptr); break;

		case 'S': ChangeSourceType(In_Ptr); break;

		case 'Z': ChangeStartP(In_Ptr); break;

		case 'O': PutInputToFile(stdout, In_Ptr); break;

		case 'H': ShowChangeMenu(); break;

		case 'Q': return 1;

		case 'X': return 2;

		default: puts("...Unknown command");
	}

	return 0;
}

/***************************************************************************
 *   Return 1 if quit change and start simulation;
 *   return 0 if exit to main menu.
 ****/
bool RunChangedInput(InStru* In_Ptr) {
	char string[STRLEN] = {0};
	int branch;

	printf("Any changes to the input parameters? (Y/N)");
	do {
		fgets(string, STRLEN, stdin);
	}
	while (!strlen(string));

	while (toupper(string[0]) == 'Y') {
		do {
			do {
				printf("\n> Change menu (h for help) => ");
				fgets(string, STRLEN, stdin);
			}
			while (!strlen(string));

			branch = BranchChangeMenu(string, In_Ptr);

			/* string[0] is 'X' or 'Q'. */
			if (branch != 0) {
				break;
			}
		}
		while (1);

		printf("Do you want to save the input to a file? (Y/N)");
		fgets(string, STRLEN, stdin);
		if (toupper(string[0]) == 'Y') {
			printf("Give the file name to save input: ( .mci): ");
			fgets(string, STRLEN, stdin);

			FILE* fp;
			if (fopen_s(&fp, string, "w")) {
				puts("Can not open the file to write.");
			}
			else {
				PutInputToFile(fp, In_Ptr);
			}
		}

		/* quit change menu and start simulation. */
		if (branch == 1) {
			if (!CheckInputConsis(In_Ptr)) {
				do {
					printf("Change input or exit to main menu (c/x): ");
					fgets(string, STRLEN, stdin);
				}
				while (!strlen(string) || (toupper(string[0]) != 'X' && toupper(string[0]) != 'C'));

				if (toupper(string[0]) == 'X') {
					free(In_Ptr->mediumlist);
					free(In_Ptr->layerspecs);
					return 0;
				}

				string[0] = 'Y'; /* continue to change parameters. */
			}
			else {
				return 1;
			}
		}
		/* exit to menu. */
		else {
			free(In_Ptr->mediumlist);
			free(In_Ptr->layerspecs);
			return 0;
		}
	}

	return 1;
}

/**************************************************************************
 *	Return 1, if the name in the name list.
 *	Return 0, otherwise.
 ****/
bool NameInList(char* Name, NameLink List) {
	while (List != NULL) {
		if (strcmp(Name, List->name) == 0) {
			return (1);
		}
		List = List->next;
	};
	return (0);
}

/**************************************************************************
 *	Add the name to the name list.
 ****/
void AddNameToList(char* Name, NameLink* List_Ptr) {
	// NameLink list = *List_Ptr;

	/* first node. */
	if (*List_Ptr == NULL) {
		/* Allocate if list is null */
		NameLink list = (NameLink)calloc(1, sizeof(NameNode));

		*List_Ptr = list;
		strcpy_s(list->name, STRLEN, Name);
		list->next = NULL;
	}

	/* subsequent nodes. */
	else {
		NameLink list = *List_Ptr;

		/* Move to the last node. */
		while (list->next != NULL) {
			list = list->next;
		}

		/* Append a node to the list. */
		list->next = (NameLink)malloc(sizeof(NameNode));
		list = list->next;
		if (list != NULL) {
			strcpy_s(list->name, STRLEN, Name);
			list->next = NULL;
		}
	}
}

/**************************************************************************
 *	Check against duplicated file names.
 *
 *	A linked list is set up to store the file names used
 *	in this input data file.
 ****/
bool FnameTaken(char* fname, NameLink* List_Ptr) {
	if (NameInList(fname, *List_Ptr)) {
		return (1);
	}

	AddNameToList(fname, List_Ptr);
	return (0);
}

/**************************************************************************
 *	Free each node in the file name list.
 ****/
void FreeFnameList(NameLink List) {
	NameLink next;
	while (List != NULL) {
		next = List->next;
		free(List);
		List = next;
	}
}

/*******************************************************************************
 *  Check the whether the flag end is met.
 *  The file position is restored to the current position
 *  at the end of the inquery.
 ****/
bool EndOfRunsQ(FILE** FilePP) {
	char buf[STRLEN] = {0};

	/* found end of runs. */
	bool found = 1;

	/* record file position. */
	long file_pos = ftell(*FilePP);
	strcpy_s(buf, sizeof(buf), FindDataLine(*FilePP));
	if (buf[0] == '\0') {
		found = 0;
		printf("Missing end.\n");
	}
	else if (strstr(buf, "end") == NULL) {
		found = 0;
	}

	fseek(*FilePP, file_pos, SEEK_SET); /* restore postion. */
	return (found);
}

/**************************************************************************
 *  Check the input parameters for all runs.
 *  This function will count number of runs and assign it to
 *  In_Ptr->num_runs.
 ****/
void CheckParamFromFile(FILE* Fp, InStru* In_Ptr) {
	short i_run = 0;
	NameLink head = NULL;

	if (!ReadMediumListQ(Fp, In_Ptr)) {
		exit(1);
	}

	long file_pos = ftell(Fp);
	do {
		printf("Checking input data for run %d\n", ++i_run);
		ReadRunParam(Fp, In_Ptr);

		/* output files share the same file name. */
		bool name_taken = FnameTaken(In_Ptr->out_fname, &head);
		if (name_taken) {
			printf("file name %s duplicated.\n", In_Ptr->out_fname);
			exit(1);
		}
		free(In_Ptr->layerspecs);
	}
	while (!EndOfRunsQ(&Fp));

	In_Ptr->num_runs = i_run;
	FreeFnameList(head);
	fseek(Fp, file_pos, SEEK_SET);
}

/**************************************************************************
 *	Allocate the arrays in OutStru for one run, and
 *	array elements are automatically initialized to zeros.
 *
 *	Remember that the indices for Rd_r[], Td_r[],
 *	& A_rz[][iz] start from -1 storing the collimated
 *	responses.
 ****/
void InitOutputData(InStru* In_Ptr, OutStru* Out_Ptr) {
	short nz = In_Ptr->nz;
	short nr = In_Ptr->nr;
	short na = In_Ptr->na;
	short nt = In_Ptr->nt;

	/* remember to use nl+2 because of 2 for ambient. */
	short nl = In_Ptr->num_layers;

	if (nz <= 0 || nr <= 0 || na <= 0 || nl <= 0) {
		printf("Invalid grid parameters.\n");
		exit(1);
	}

	/* Init pure numbers. */
	Out_Ptr->Rsp = 0.0;
	Out_Ptr->Rb = 0.0;
	Out_Ptr->Rd = 0.0;
	Out_Ptr->Td = 0.0;
	Out_Ptr->Tb = 0.0;
	Out_Ptr->A = 0.0;

	Out_Ptr->Rbe = 0.0;
	Out_Ptr->Rde = 0.0;
	Out_Ptr->Tde = 0.0;
	Out_Ptr->Tbe = 0.0;
	Out_Ptr->Ae = 0.0;

	/* Allocate the 1D, 2D and 3D arrays. */
	Out_Ptr->Rd_rat = (In_Ptr->record.Rd_rat) ? AllocArray3D(0, nr - 1, 0, na - 1, 0, nt - 1) : NULL;
	Out_Ptr->Rd_ra = (In_Ptr->record.Rd_ra) ? AllocArray2D(0, nr - 1, 0, na - 1) : NULL;
	Out_Ptr->Rd_rt = (In_Ptr->record.Rd_rt) ? AllocArray2D(0, nr - 1, 0, nt - 1) : NULL;
	Out_Ptr->Rd_at = (In_Ptr->record.Rd_at) ? AllocArray2D(0, na - 1, 0, nt - 1) : NULL;

	// Modern array allocations - these would eventually become Array1D* when struct is modernized
	Out_Ptr->Rd_r = (In_Ptr->record.Rd_r) ? AllocArray1D(0, nr - 1) : NULL;
	Out_Ptr->Rd_a = (In_Ptr->record.Rd_a) ? AllocArray1D(0, na - 1) : NULL;
	Out_Ptr->Rd_t = (In_Ptr->record.Rd_t) ? AllocArray1D(0, nt - 1) : NULL;

	Out_Ptr->Td_rat = (In_Ptr->record.Td_rat) ? AllocArray3D(0, nr - 1, 0, na - 1, 0, nt - 1) : NULL;
	Out_Ptr->Td_ra = (In_Ptr->record.Td_ra) ? AllocArray2D(0, nr - 1, 0, na - 1) : NULL;
	Out_Ptr->Td_rt = (In_Ptr->record.Td_rt) ? AllocArray2D(0, nr - 1, 0, nt - 1) : NULL;
	Out_Ptr->Td_at = (In_Ptr->record.Td_at) ? AllocArray2D(0, na - 1, 0, nt - 1) : NULL;
	Out_Ptr->Td_r = (In_Ptr->record.Td_r) ? AllocArray1D(0, nr - 1) : NULL;
	Out_Ptr->Td_a = (In_Ptr->record.Td_a) ? AllocArray1D(0, na - 1) : NULL;
	Out_Ptr->Td_t = (In_Ptr->record.Td_t) ? AllocArray1D(0, nt - 1) : NULL;

	Out_Ptr->A_rzt = (In_Ptr->record.A_rzt) ? AllocArray3D(0, nr - 1, 0, nz - 1, 0, nt - 1) : NULL;
	Out_Ptr->Ab_zt = (In_Ptr->record.A_rzt) ? AllocArray2D(0, nz - 1, 0, nt - 1) : NULL;
	Out_Ptr->A_rz = (In_Ptr->record.A_rz) ? AllocArray2D(0, nr - 1, 0, nz - 1) : NULL;
	Out_Ptr->Ab_z = (In_Ptr->record.A_rz) ? AllocArray1D(0, nz - 1) : NULL;
	Out_Ptr->A_zt = (In_Ptr->record.A_zt) ? AllocArray2D(0, nz - 1, 0, nt - 1) : NULL;
	Out_Ptr->A_z = (In_Ptr->record.A_z) ? AllocArray1D(0, nz - 1) : NULL;
	Out_Ptr->A_t = (In_Ptr->record.A_t) ? AllocArray1D(0, nt - 1) : NULL;
}

/**************************************************************************
 *	Undo what InitOutputData did.
 *  i.e. free the data allocations.
 ****/
void FreeData(InStru* In_Ptr, OutStru* Out_Ptr) {
	free(In_Ptr->layerspecs);

	FreeArray3D(Out_Ptr->Rd_rat);
	FreeArray2D(Out_Ptr->Rd_ra);
	FreeArray2D(Out_Ptr->Rd_rt);
	FreeArray2D(Out_Ptr->Rd_at);
	FreeArray1D(Out_Ptr->Rd_r);
	FreeArray1D(Out_Ptr->Rd_a);
	FreeArray1D(Out_Ptr->Rd_t);

	FreeArray3D(Out_Ptr->Td_rat);
	FreeArray2D(Out_Ptr->Td_ra);
	FreeArray2D(Out_Ptr->Td_rt);
	FreeArray2D(Out_Ptr->Td_at);
	FreeArray1D(Out_Ptr->Td_r);
	FreeArray1D(Out_Ptr->Td_a);
	FreeArray1D(Out_Ptr->Td_t);

	FreeArray3D(Out_Ptr->A_rzt);
	FreeArray2D(Out_Ptr->Ab_zt);
	FreeArray2D(Out_Ptr->A_rz);
	FreeArray1D(Out_Ptr->Ab_z);
	FreeArray2D(Out_Ptr->A_zt);
	FreeArray1D(Out_Ptr->A_z);
	FreeArray1D(Out_Ptr->A_t);
}

/**************************************************************************
 *	Scale Rd and Td properly.
 *	"a" stands for angle alpha.
 ****
 *	Scale Rd(r,a) and Td(r,a) by
 *      (area perpendicular to photon direction)
 *		x(solid angle)x(No. of photons).
 *	or
 *		[2*PI*r*dr*cos(a)]x[2*PI*sin(a)*da]x[No. of photons]
 *	or
 *		[2*PI*PI*dr*da*r*sin(2a)]x[No. of photons]
 ****
 *	Scale Rd(r) and Td(r) by
 *		(area on the surface)x(No. of photons).
 ****
 *	Scale Rd(a) and Td(a) by
 *		(solid angle) cos(a) x(No. of photons).
 ****
 *  Mode = 0, scale Rd and Td; Mode = 1, unscale Rd and Td.
 ****/
void ScaleRdTd(InStru* In_Ptr, OutStru* Out_Ptr, char Mode) {
	short nr = In_Ptr->nr;
	short na = In_Ptr->na;
	short nt = In_Ptr->nt;
	double dr = In_Ptr->dr;
	double da = In_Ptr->da;
	double dt = In_Ptr->dt;

	double scale1 = (double)In_Ptr->num_photons;
	if (Mode == 0) {
		Out_Ptr->Rde = 1 / scale1 * sqrt(Out_Ptr->Rde - (Out_Ptr->Rd * Out_Ptr->Rd / scale1));
		Out_Ptr->Tde = 1 / scale1 * sqrt(Out_Ptr->Tde - (Out_Ptr->Td * Out_Ptr->Td / scale1));
		Out_Ptr->Rbe = 1 / scale1 * sqrt(Out_Ptr->Rbe - (Out_Ptr->Rb * Out_Ptr->Rb / scale1));
		Out_Ptr->Tbe = 1 / scale1 * sqrt(Out_Ptr->Tbe - (Out_Ptr->Tb * Out_Ptr->Tb / scale1));

		Out_Ptr->Rd /= scale1;
		Out_Ptr->Td /= scale1;
		Out_Ptr->Rb = Out_Ptr->Rb / scale1 + Out_Ptr->Rsp;
		Out_Ptr->Tb /= scale1;
	}
	else {
		Out_Ptr->Rd *= scale1;
		Out_Ptr->Td *= scale1;
		Out_Ptr->Rb = (Out_Ptr->Rb - Out_Ptr->Rsp) * scale1;
		Out_Ptr->Tb *= scale1;

		Out_Ptr->Rde = (scale1 * Out_Ptr->Rde) * (scale1 * Out_Ptr->Rde) + 1 / scale1 * Out_Ptr->Rd * Out_Ptr->Rd;
		Out_Ptr->Tde = (scale1 * Out_Ptr->Tde) * (scale1 * Out_Ptr->Tde) + 1 / scale1 * Out_Ptr->Td * Out_Ptr->Td;
		Out_Ptr->Rbe = (scale1 * Out_Ptr->Rbe) * (scale1 * Out_Ptr->Rbe) + 1 / scale1 * Out_Ptr->Rb * Out_Ptr->Rb;
		Out_Ptr->Tbe = (scale1 * Out_Ptr->Tbe) * (scale1 * Out_Ptr->Tbe) + 1 / scale1 * Out_Ptr->Tb * Out_Ptr->Tb;
	}

	scale1 = dt * In_Ptr->num_photons;
	if (In_Ptr->record.Rd_t) {
		for (short it = 0; it < nt; it++) {
			/* scale Rd_t. */
			if (Mode == 0) {
				Out_Ptr->Rd_t[it] /= scale1;
			}
			/* unscale Rd_t. */
			else {
				Out_Ptr->Rd_t[it] *= scale1;
			}
		}
	}

	if (In_Ptr->record.Td_t) {
		for (short it = 0; it < nt; it++) {
			/* scale Td_t. */
			if (Mode == 0) {
				Out_Ptr->Td_t[it] /= scale1;
			}
			/* unscale Rd_t. */
			else {
				Out_Ptr->Td_t[it] *= scale1;
			}
		}
	}

	scale1 = 2.0 * M_PI * dr * dr * In_Ptr->num_photons;
	/* area is 2*PI*[(ir+0.5)*dr]*dr.  ir + 0.5 to be added. */

	if (In_Ptr->record.Rd_r) {
		for (short ir = 0; ir < nr; ir++) {
			double scale2 = 1.0 / ((ir + 0.5) * scale1);
			/* scale Rd_r. */
			if (Mode == 0) {
				Out_Ptr->Rd_r[ir] *= scale2;
			}
			/* unscale Rd_r. */
			else {
				Out_Ptr->Rd_r[ir] /= scale2;
			}
		}
	}

	if (In_Ptr->record.Td_r) {
		for (short ir = 0; ir < nr; ir++) {
			double scale2 = 1.0 / ((ir + 0.5) * scale1);
			/* scale Td_r. */
			if (Mode == 0) {
				Out_Ptr->Td_r[ir] *= scale2;
			}
			/* unscale Td_r. */
			else {
				Out_Ptr->Td_r[ir] /= scale2;
			}
		}
	}

	scale1 *= dt;
	if (In_Ptr->record.Rd_rt) {
		for (short ir = 0; ir < nr; ir++) {
			for (short it = 0; it < nt; it++) {
				double scale2 = 1.0 / ((ir + 0.5) * scale1);
				/* scale Rd_rt. */
				if (Mode == 0) {
					Out_Ptr->Rd_rt[ir][it] *= scale2;
				}
				/* unscale Rd_rt. */
				else {
					Out_Ptr->Rd_rt[ir][it] *= scale2;
				}
			}
		}
	}

	if (In_Ptr->record.Td_rt) {
		for (short ir = 0; ir < nr; ir++) {
			for (short it = 0; it < nt; it++) {
				double scale2 = 1.0 / ((ir + 0.5) * scale1);
				/* scale Td_rt. */
				if (Mode == 0) {
					Out_Ptr->Td_rt[ir][it] *= scale2;
				}
				/* unscale Td_rt. */
				else {
					Out_Ptr->Td_rt[ir][it] /= scale2;
				}
			}
		}
	}

	scale1 = M_PI * da * In_Ptr->num_photons;
	/* solid angle times cos(a) is PI*sin(2a)*da. sin(2a) to be added. */

	if (In_Ptr->record.Rd_a) {
		for (short ia = 0; ia < na; ia++) {
			double scale2 = 1.0 / (sin(2.0 * (ia + 0.5) * da) * scale1);
			/* scale Rd_a. */
			if (Mode == 0) {
				Out_Ptr->Rd_a[ia] *= scale2;
			}
			/* unscale Rd_a. */
			else {
				Out_Ptr->Rd_a[ia] /= scale2;
			}
		}
	}

	if (In_Ptr->record.Td_a) {
		for (short ia = 0; ia < na; ia++) {
			double scale2 = 1.0 / (sin(2.0 * (ia + 0.5) * da) * scale1);
			/* scale Td_a. */
			if (Mode == 0) {
				Out_Ptr->Td_a[ia] *= scale2;
			}
			/* unscale Td_a. */
			else {
				Out_Ptr->Td_a[ia] /= scale2;
			}
		}
	}

	scale1 *= dt;
	if (In_Ptr->record.Rd_at) {
		for (short ia = 0; ia < na; ia++) {
			for (short it = 0; it < nt; it++) {
				double scale2 = 1.0 / (sin(2.0 * (ia + 0.5) * da) * scale1);
				/* scale Rd_at. */
				if (Mode == 0) {
					Out_Ptr->Rd_at[ia][it] *= scale2;
				}
				/* unscale Rd_at. */
				else {
					Out_Ptr->Rd_at[ia][it] /= scale2;
				}
			}
		}
	}

	if (In_Ptr->record.Td_at) {
		for (short ia = 0; ia < na; ia++) {
			for (short it = 0; it < nt; it++) {
				double scale2 = 1.0 / (sin(2.0 * (ia + 0.5) * da) * scale1);
				/* scale Td_at. */
				if (Mode == 0) {
					Out_Ptr->Td_at[ia][it] *= scale2;
				}
				/* unscale Td_at. */
				else {
					Out_Ptr->Td_at[ia][it] /= scale2;
				}
			}
		}
	}

	scale1 = 2.0 * M_PI * dr * dr * M_PI * da * In_Ptr->num_photons;
	if (In_Ptr->record.Rd_ra) {
		for (short ir = 0; ir < nr; ir++) {
			for (short ia = 0; ia < na; ia++) {
				double scale2 = 1.0 / ((ir + 0.5) * sin(2.0 * (ia + 0.5) * da) * scale1);
				/* scale Rd_ra. */
				if (Mode == 0) {
					Out_Ptr->Rd_ra[ir][ia] *= scale2;
				}
				/* unscale Rd_ra. */
				else {
					Out_Ptr->Rd_ra[ir][ia] /= scale2;
				}
			}
		}
	}

	if (In_Ptr->record.Td_ra) {
		for (short ir = 0; ir < nr; ir++) {
			for (short ia = 0; ia < na; ia++) {
				double scale2 = 1.0 / ((ir + 0.5) * sin(2.0 * (ia + 0.5) * da) * scale1);
				/* scale Td_ra. */
				if (Mode == 0) {
					Out_Ptr->Td_ra[ir][ia] *= scale2;
				}
				/* unscale Td_ra. */
				else {
					Out_Ptr->Td_ra[ir][ia] /= scale2;
				}
			}
		}
	}

	scale1 *= dt;
	if (In_Ptr->record.Rd_rat) {
		for (short ir = 0; ir < nr; ir++) {
			for (short ia = 0; ia < na; ia++) {
				for (short it = 0; it < nt; it++) {
					double scale2 = 1.0 / ((ir + 0.5) * sin(2.0 * (ia + 0.5) * da) * scale1);
					/* scale Rd_rat. */
					if (Mode == 0) {
						Out_Ptr->Rd_rat[ir][ia][it] *= scale2;
					}
					/* unscale Rd_rat. */
					else {
						Out_Ptr->Rd_rat[ir][ia][it] /= scale2;
					}
				}
			}
		}
	}

	if (In_Ptr->record.Td_rat) {
		for (short ir = 0; ir < nr; ir++) {
			for (short ia = 0; ia < na; ia++) {
				for (short it = 0; it < nt; it++) {
					double scale2 = 1.0 / ((ir + 0.5) * sin(2.0 * (ia + 0.5) * da) * scale1);
					/* scale Td_rat. */
					if (Mode == 0) {
						Out_Ptr->Td_rat[ir][ia][it] *= scale2;
					}
					/* unscale Td_rat. */
					else {
						Out_Ptr->Td_rat[ir][ia][it] /= scale2;
					}
				}
			}
		}
	}
}

/**************************************************************************
 *	Scale absorption arrays properly.
 *  Mode = 0, scale A; Mode = 1, unscale A.
 ****/
void ScaleA(InStru* In_Ptr, OutStru* Out_Ptr, char Mode) {
	short nz = In_Ptr->nz;
	short nr = In_Ptr->nr;
	short nt = In_Ptr->nt;
	double dz = In_Ptr->dz;
	double dr = In_Ptr->dr;
	double dt = In_Ptr->dt;
	double scale1 = (double)In_Ptr->num_photons;

	/* scale A. */
	if (Mode == 0) {
		Out_Ptr->Ae = 1 / scale1 * sqrt(Out_Ptr->Ae - (Out_Ptr->A * Out_Ptr->A / scale1));
		Out_Ptr->A /= scale1;
	}
	/* unscale A. */
	else {
		Out_Ptr->A *= scale1;
		Out_Ptr->Ae = (scale1 * Out_Ptr->Ae) * (scale1 * Out_Ptr->Ae) + 1 / scale1 * Out_Ptr->A * Out_Ptr->A;
	}

	double scale2 = scale1 * dt;
	if (In_Ptr->record.A_t) {
		for (short it = 0; it < nt; it++) {
			/* scale A_t. */
			if (Mode == 0) {
				Out_Ptr->A_t[it] /= scale2;
			}
			/* unscale A_t. */
			else {
				Out_Ptr->A_t[it] *= scale2;
			}
		}
	}

	scale1 *= dz;
	if (In_Ptr->record.A_z) {
		for (short iz = 0; iz < nz; iz++) {
			/* scale A_z. */
			if (Mode == 0) {
				Out_Ptr->A_z[iz] /= scale1;
			}
			/* unscale A_z. */
			else {
				Out_Ptr->A_z[iz] *= scale1;
			}
		}
	}

	scale2 = scale1 * dt;
	if (In_Ptr->record.A_zt) {
		for (short iz = 0; iz < nz; iz++) {
			for (short it = 0; it < nt; it++) {
				/* scale A_zt. */
				if (Mode == 0) {
					Out_Ptr->A_zt[iz][it] /= scale2;
				}
				/* unscale A_zt. */
				else {
					Out_Ptr->A_zt[iz][it] *= scale2;
				}
			}
		}
	}

	if (In_Ptr->record.A_rz) {
		for (short iz = 0; iz < nz; iz++) {
			/* scale Ab_z. */
			if (Mode == 0) {
				Out_Ptr->Ab_z[iz] /= scale1;
			}
			/* unscale Ab_z. */
			else {
				Out_Ptr->Ab_z[iz] *= scale1;
			}
		}
	}

	if (In_Ptr->record.A_rzt) {
		for (short iz = 0; iz < nz; iz++) {
			for (short it = 0; it < nt; it++) {
				/* scale Ab_zt. */
				if (Mode == 0) {
					Out_Ptr->Ab_zt[iz][it] /= scale2;
				}
				/* unscale Ab_zt. */
				else {
					Out_Ptr->Ab_zt[iz][it] *= scale2;
				}
			}
		}
	}

	scale1 = 2.0 * M_PI * dr * dr * dz * In_Ptr->num_photons;
	if (In_Ptr->record.A_rz) {
		for (short ir = 0; ir < nr; ir++) {
			for (short iz = 0; iz < nz; iz++) {
				/* scale A_rz. */
				if (Mode == 0) {
					Out_Ptr->A_rz[ir][iz] /= (ir + 0.5) * scale1;
				}
				/* unscale A_rz. */
				else {
					Out_Ptr->A_rz[ir][iz] *= (ir + 0.5) * scale1;
				}
			}
		}
	}

	scale2 = scale1 * dt;
	if (In_Ptr->record.A_rzt) {
		for (short ir = 0; ir < nr; ir++) {
			for (short iz = 0; iz < nz; iz++) {
				for (short it = 0; it < nt; it++) {
					/* scale A_rzt. */
					if (Mode == 0) {
						Out_Ptr->A_rzt[ir][iz][it] /= (ir + 0.5) * scale2;
					}
					/* unscale A_rzt. */
					else {
						Out_Ptr->A_rzt[ir][iz][it] *= (ir + 0.5) * scale2;
					}
				}
			}
		}
	}
}

/**************************************************************************
 *	Scale results of current run.
 *  Mode = 0, scale result; Mode = 1, unscale result.
 ****/
void ScaleResult(InStru* In_Ptr, OutStru* Out_Ptr, char Mode) {
	ScaleRdTd(In_Ptr, Out_Ptr, Mode);
	ScaleA(In_Ptr, Out_Ptr, Mode);
}

/**************************************************************************
 *	Write the version number as the first string in the
 *	file.
 *	Use chars only so that they can be read as either
 *	ASCII or binary.
 ****/
void WriteVersion(FILE* Fp, const char* Version) {
	fprintf(Fp, "%s \t# Version number of the file format.\n\n", Version);
	fprintf(Fp, "####\n# Data categories include: \n");
	fprintf(Fp, "# InParam, RAT, \n");
	fprintf(Fp, "# Rd_r\tRd_a\tRd_ra\tRd_t\tRd_rt\tRd_at\tRd_rat\n");
	fprintf(Fp, "# Td_r\tTd_a\tTd_ra\tTd_t\tTd_rt\tTd_at\tTd_rat\n");
	fprintf(Fp, "# A_z\tA_rz\tA_t\tA_zt\tA_rzt\n");
	fprintf(Fp, "####\n\n");
}

/***************************************************************************
 * Save the status of the random number generater to output file.
 ****/
void SaveRandomStatus(FILE* Fp) {
	/* get the status. */
	long status[57];
	RandomGen(2, 0, status);
	fprintf(Fp, "# status of the random number generator:");

	for (int i = 0; i < 57; i++) {
		if (i % 5) {
			fprintf(Fp, "%14ld", status[i]);
		}
		else {
			fprintf(Fp, "\n%14ld ", status[i]);
		}
	}

	fprintf(Fp, "\n\n");
}

/***************************************************************************
 * Read and restore the status of random number generater from previous
 * output file.
 ****/
void RestoreRandomStatus(FILE* Fp) {
	char buf[STRLEN] = {0};
	long status[57];

	do {
		fgets(buf, sizeof(buf), Fp);
	}
	while (buf[0] != '#');

	for (int i = 0; i < 57; i++) {
		fscanf_s(Fp, "%ld", &status[i]);
	}

	/* restore the status. */
	RandomGen(3, 0, status);
}

/**************************************************************************
 *	Write reflectance, absorption, transmission.
 ****/
void WriteRAT(FILE* Fp, OutStru* Out_Ptr) {
	/* flag. */
	fprintf(Fp, "RAT #Reflectance, absorption, transmittance.\n");
	fprintf(Fp, "# Average \tStandard Err \tRel Err\n");
	fprintf(Fp, "%-14.6G \t\t\t\t#Rsp: Specular reflectance.\n", Out_Ptr->Rsp);
	fprintf(Fp, "%-14.6G \t%-14.6G %6.2f%%\t#Rb: Ballistic reflectance.\n", Out_Ptr->Rb, Out_Ptr->Rbe,
			(Out_Ptr->Rb) ? Out_Ptr->Rbe / Out_Ptr->Rb * 100 : 0);
	fprintf(Fp, "%-14.6G \t%-14.6G %6.2f%%\t#Rd: Diffuse reflectance.\n", Out_Ptr->Rd, Out_Ptr->Rde,
			(Out_Ptr->Rd) ? Out_Ptr->Rde / Out_Ptr->Rd * 100 : 0);
	fprintf(Fp, "%-14.6G \t%-14.6G %6.2f%%\t#A:  Absorbed fraction.\n", Out_Ptr->A, Out_Ptr->Ae,
			(Out_Ptr->A) ? Out_Ptr->Ae / Out_Ptr->A * 100 : 0);
	fprintf(Fp, "%-14.6G \t%-14.6G %6.2f%%\t#Tb: Ballistic transmittance.\n", Out_Ptr->Tb, Out_Ptr->Tbe,
			(Out_Ptr->Tb) ? Out_Ptr->Tbe / Out_Ptr->Tb * 100 : 0);
	fprintf(Fp, "%-14.6G \t%-14.6G %6.2f%%\t#Td: Diffuse transmittance.\n", Out_Ptr->Td, Out_Ptr->Tde,
			(Out_Ptr->Td) ? Out_Ptr->Tde / Out_Ptr->Td * 100 : 0);
	fprintf(Fp, "\n");
}

/**************************************************************************
 *	Read reflectance, absorption, transmission.
 ****/
void ReadRAT(FILE* Fp, OutStru* Out_Ptr) {
	char buf[STRLEN] = {0};

	/* skip RAT line. */
	FindDataLine(Fp);

	strcpy_s(buf, sizeof(buf), FindDataLine(Fp));
	sscanf_s(buf, "%lf", &(Out_Ptr->Rsp));

	strcpy_s(buf, sizeof(buf), FindDataLine(Fp));
	sscanf_s(buf, "%lf %lf", &(Out_Ptr->Rb), &(Out_Ptr->Rbe));

	strcpy_s(buf, sizeof(buf), FindDataLine(Fp));
	sscanf_s(buf, "%lf %lf", &(Out_Ptr->Rd), &(Out_Ptr->Rde));

	strcpy_s(buf, sizeof(buf), FindDataLine(Fp));
	sscanf_s(buf, "%lf %lf", &(Out_Ptr->A), &(Out_Ptr->Ae));

	strcpy_s(buf, sizeof(buf), FindDataLine(Fp));
	sscanf_s(buf, "%lf %lf", &(Out_Ptr->Tb), &(Out_Ptr->Tbe));

	strcpy_s(buf, sizeof(buf), FindDataLine(Fp));
	sscanf_s(buf, "%lf %lf", &(Out_Ptr->Td), &(Out_Ptr->Tde));
}

/**************************************************************************
 *  5 numbers each line.
 *  Mode = 0, read; Mode = 1, write.
 ****/
void IOAb_zt(FILE* Fp, short Nz, short Nt, OutStru* Out_Ptr, char Mode) {
	if (Mode == 1) {
		/* flag. */
		fprintf(Fp, "%s\n%s\n%s\n%s\n%s\n%s\n", "# Ab[z][t]. [1/(cm ps)]", "# Ab[0][0], [0][1],..[0][nt-1]",
				"# Ab[1][0], [1][1],..[1][nt-1]", "# ...", "# Ab[nz-1][0], [nz-1][1],..[nz-1][nt-1]", "Ab_zt");
	}
	else {
		/* skip A_z line. */
		FindDataLine(Fp);
	}

	short i = 0;
	for (short iz = 0; iz < Nz; iz++) {
		for (short it = 0; it < Nt; it++) {
			if (Mode == 1) {
				fprintf(Fp, "%12.4E ", Out_Ptr->Ab_zt[iz][it]);
				if (++i % 5 == 0) {
					fprintf(Fp, "\n");
				}
			}
			else {
				fscanf_s(Fp, "%lf", &(Out_Ptr->Ab_zt[iz][it]));
			}
		}
	}

	if (Mode == 1) {
		fprintf(Fp, "\n\n");
	}
}

/**************************************************************************
 *  5 numbers each line.
 *  Mode = 0, read; Mode = 1, write.
 ****/
void IOA_rzt(FILE* Fp, short Nr, short Nz, short Nt, OutStru* Out_Ptr, char Mode) {
	IOAb_zt(Fp, Nz, Nt, Out_Ptr, Mode);

	if (Mode == 1) {
		/* flag. */
		fprintf(Fp, "%s\n%s\n%s\n%s\n%s\n%s\n", "# A[r][z][t]. [1/(cm3 ps)]", "# A[0][0][0], [0][0][1],..[0][0][nt-1]",
				"# A[0][1][0], [0][1][1],..[0][1][nt-1]", "# ...",
				"# A[nr-1][nz-1][0], [nr-1][nz-1][1],..[nr-1][nz-1][nt-1]", "A_rzt");
	}
	else {
		FindDataLine(Fp);
	}

	short i = 0;
	for (short ir = 0; ir < Nr; ir++) {
		for (short iz = 0; iz < Nz; iz++) {
			for (short it = 0; it < Nt; it++) {
				if (Mode == 1) {
					fprintf(Fp, "%12.4E ", Out_Ptr->A_rzt[ir][iz][it]);
					if (++i % 5 == 0) {
						fprintf(Fp, "\n");
					}
				}
				else {
					fscanf_s(Fp, "%lf", &(Out_Ptr->A_rzt[ir][iz][it]));
				}
			}
		}
	}

	if (Mode == 1) {
		fprintf(Fp, "\n\n");
	}
}

/**************************************************************************
 *  1 number each line.
 *  Mode = 0, read; Mode = 1, write.
 ****/
void IOAb_z(FILE* Fp, short Nz, OutStru* Out_Ptr, char Mode) {
	if (Mode == 1) {
		fprintf(Fp, "Ab_z #Ab[0], [1],..Ab[nz-1]. [1/cm]\n"); /* flag. */
	}
	else {
		FindDataLine(Fp);
	}

	for (short iz = 0; iz < Nz; iz++) {
		if (Mode == 1) {
			fprintf(Fp, "%12.4E\n", Out_Ptr->Ab_z[iz]);
		}
		else {
			fscanf_s(Fp, "%lf", &(Out_Ptr->Ab_z[iz]));
		}
	}

	if (Mode == 1) {
		fprintf(Fp, "\n");
	}
}

/**************************************************************************
 *  5 numbers each line.
 *  Mode = 0, read; Mode = 1, write.
 ****/
void IOA_rz(FILE* Fp, short Nr, short Nz, OutStru* Out_Ptr, char Mode) {
	IOAb_z(Fp, Nz, Out_Ptr, Mode);

	if (Mode == 1) {
		/* flag. */
		fprintf(Fp, "%s\n%s\n%s\n%s\n%s\n", "# A[r][z]. [1/cm3]", "# A[0][0], [0][1],..[0][nz-1]", "# ...",
				"# A[nr-1][0], [nr-1][1],..[nr-1][nz-1]", "A_rz");
	}
	else {
		/* skip A_rz line. */
		FindDataLine(Fp);
	}

	short i = 0;
	for (short ir = 0; ir < Nr; ir++) {
		for (short iz = 0; iz < Nz; iz++) {
			if (Mode == 1) {
				fprintf(Fp, "%12.4E ", Out_Ptr->A_rz[ir][iz]);
				if (++i % 5 == 0) {
					fprintf(Fp, "\n");
				}
			}
			else {
				fscanf_s(Fp, "%lf", &(Out_Ptr->A_rz[ir][iz]));
			}
		}
	}

	if (Mode == 1) {
		fprintf(Fp, "\n\n");
	}
}

/**************************************************************************
 *  5 numbers each line.
 *  Mode = 0, read; Mode = 1, write.
 ****/
void IOA_zt(FILE* Fp, short Nz, short Nt, OutStru* Out_Ptr, char Mode) {
	if (Mode == 1) {
		/* flag. */
		fprintf(Fp, "%s\n%s\n%s\n%s\n%s\n%s\n", "# A[z][t]. [1/(cm ps)]", "# A[0][0], [0][1],..[0][nt-1]",
				"# A[1][0], [1][1],..[1][nt-1]", "# ...", "# A[nz-1][0], [nz-1][1],..[nz-1][nt-1]", "A_zt");
	}
	else {
		/* skip A_zt line. */
		FindDataLine(Fp);
	}

	short i = 0;
	for (short iz = 0; iz < Nz; iz++) {
		for (short it = 0; it < Nt; it++) {
			if (Mode == 1) {
				fprintf(Fp, "%12.4E ", Out_Ptr->A_zt[iz][it]);
				if (++i % 5 == 0) {
					fprintf(Fp, "\n");
				}
			}
			else {
				fscanf_s(Fp, "%lf", &(Out_Ptr->A_zt[iz][it]));
			}
		}
	}

	if (Mode == 1) {
		fprintf(Fp, "\n\n");
	}
}

/**************************************************************************
 *  1 number each line.
 *  Mode = 0, read; Mode = 1, write.
 ****/
void IOA_z(FILE* Fp, short Nz, OutStru* Out_Ptr, char Mode) {
	if (Mode == 1) {
		/* flag. */
		fprintf(Fp, "A_z #A[0], [1],..A[nz-1]. [1/cm]\n");
	}
	else {
		/* skip A_z line. */
		FindDataLine(Fp);
	}

	for (short iz = 0; iz < Nz; iz++) {
		if (Mode == 1) {
			fprintf(Fp, "%12.4E\n", Out_Ptr->A_z[iz]);
		}
		else {
			fscanf_s(Fp, "%lf", &(Out_Ptr->A_z[iz]));
		}
	}

	if (Mode == 1) {
		fprintf(Fp, "\n");
	}
}

/**************************************************************************
 *  1 number each line.
 *  Mode = 0, read; Mode = 1, write.
 ****/
void IOA_t(FILE* Fp, short Nt, OutStru* Out_Ptr, char Mode) {
	if (Mode == 1) {
		/* flag. */
		fprintf(Fp, "A_t #A[0], [1],..A[nt-1]. [1/ps]\n");
	}
	else {
		FindDataLine(Fp);
	}

	for (short it = 0; it < Nt; it++) {
		if (Mode == 1) {
			fprintf(Fp, "%12.4E\n", Out_Ptr->A_t[it]);
		}
		else {
			fscanf_s(Fp, "%lf", &(Out_Ptr->A_t[it]));
		}
	}

	if (Mode == 1) {
		fprintf(Fp, "\n");
	}
}

/**************************************************************************
 *  5 numbers each line.
 *  Mode = 0, read; Mode = 1, write.
 ****/
void IORd_rat(FILE* Fp, short Nr, short Na, short Nt, OutStru* Out_Ptr, char Mode) {
	if (Mode == 1) {
		/* flag. */
		fprintf(Fp, "%s\n%s\n%s\n%s\n%s\n%s\n", "# Rd[r][a][t]. [1/(cm2 sr ps)]",
				"# Rd[0][0][0], [0][0][1],..[0][0][nt-1]", "# Rd[0][1][0], [0][1][1],..[0][1][nt-1]", "# ...",
				"# Rd[nr-1][na-1][0], [nr-1][na-1][1],..[nr-1][na-1][nt-1]", "Rd_rat");
	}
	else {
		FindDataLine(Fp);
	}

	short i = 0;
	for (short ir = 0; ir < Nr; ir++) {
		for (short ia = 0; ia < Na; ia++) {
			for (short it = 0; it < Nt; it++) {
				if (Mode == 1) {
					fprintf(Fp, "%12.4E ", Out_Ptr->Rd_rat[ir][ia][it]);
					if (++i % 5 == 0) {
						fprintf(Fp, "\n");
					}
				}
				else {
					fscanf_s(Fp, "%lf", &(Out_Ptr->Rd_rat[ir][ia][it]));
				}
			}
		}
	}

	if (Mode == 1) {
		fprintf(Fp, "\n\n");
	}
}

/**************************************************************************
 *  5 numbers each line.
 *  Mode = 0, read; Mode = 1, write.
 ****/
void IORd_ra(FILE* Fp, short Nr, short Na, OutStru* Out_Ptr, char Mode) {
	if (Mode == 1) {
		/* flag. */
		fprintf(Fp, "%s\n%s\n%s\n%s\n%s\n%s\n", "# Rd[r][angle]. [1/(cm2 sr)].", "# Rd[0][0], [0][1],..[0][na-1]",
				"# Rd[1][0], [1][1],..[1][na-1]", "# ...", "# Rd[nr-1][0], [nr-1][1],..[nr-1][na-1]", "Rd_ra");
	}
	else {
		FindDataLine(Fp);
	}

	for (short ir = 0; ir < Nr; ir++) {
		for (short ia = 0; ia < Na; ia++) {
			if (Mode == 1) {
				fprintf(Fp, "%12.4E ", Out_Ptr->Rd_ra[ir][ia]);
				if ((ir * Na + ia + 1) % 5 == 0) {
					fprintf(Fp, "\n");
				}
			}
			else {
				fscanf_s(Fp, "%lf", &(Out_Ptr->Rd_ra[ir][ia]));
			}
		}
	}

	if (Mode == 1) {
		fprintf(Fp, "\n\n");
	}
}

/**************************************************************************
 *  5 numbers each line.
 *  Mode = 0, read; Mode = 1, write.
 ****/
void IORd_rt(FILE* Fp, short Nr, short Nt, OutStru* Out_Ptr, char Mode) {
	if (Mode == 1) {
		/* flag. */
		fprintf(Fp, "%s\n%s\n%s\n%s\n%s\n%s\n", "# Rd[r][t]. [1/(cm2 ps)]", "# Rd[0][0], [0][1],..[0][nt-1]",
				"# Rd[0][0], [0][1],..[0][nt-1]", "# ...", "# Rd[nr-1][0], [nr-1][1],..[nr-1][nt-1]", "Rd_rt");
	}
	else {
		FindDataLine(Fp);
	}

	short i = 0;
	for (short ir = 0; ir < Nr; ir++) {
		for (short it = 0; it < Nt; it++) {
			if (Mode == 1) {
				fprintf(Fp, "%12.4E ", Out_Ptr->Rd_rt[ir][it]);
				if (++i % 5 == 0) {
					fprintf(Fp, "\n");
				}
			}
			else {
				fscanf_s(Fp, "%lf", &(Out_Ptr->Rd_rt[ir][it]));
			}
		}
	}

	if (Mode == 1) {
		fprintf(Fp, "\n\n");
	}
}

/**************************************************************************
 *  5 numbers each line.
 *  Mode = 0, read; Mode = 1, write.
 ****/
void IORd_at(FILE* Fp, short Na, short Nt, OutStru* Out_Ptr, char Mode) {
	if (Mode == 1) {
		/* flag. */
		fprintf(Fp, "%s\n%s\n%s\n%s\n%s\n%s\n", "# Rd[a][t]. [1/(sr ps)]", "# Rd[0][0], [0][1],..[0][nt-1]",
				"# Rd[1][0], [1][1],..[1][nt-1]", "# ...", "# Rd[na-1][0], [na-1][1],..[na-1][nt-1]", "Rd_at");
	}
	else {
		FindDataLine(Fp);
	}

	short i = 0;
	for (short ia = 0; ia < Na; ia++) {
		for (short it = 0; it < Nt; it++) {
			if (Mode == 1) {
				fprintf(Fp, "%12.4E ", Out_Ptr->Rd_at[ia][it]);
				if (++i % 5 == 0) {
					fprintf(Fp, "\n");
				}
			}
			else {
				fscanf_s(Fp, "%lf", &(Out_Ptr->Rd_at[ia][it]));
			}
		}
	}

	if (Mode == 1) {
		fprintf(Fp, "\n\n");
	}
}

/**************************************************************************
 *  1 number each line.
 *  Mode = 0, read; Mode = 1, write.
 ****/
void IORd_r(FILE* Fp, short Nr, OutStru* Out_Ptr, char Mode) {
	if (Mode == 1) {
		/* flag. */
		fprintf(Fp, "Rd_r #Rd[0], [1],..Rd[nr-1]. [1/cm2]\n");
	}
	else {
		FindDataLine(Fp);
	}

	for (short ir = 0; ir < Nr; ir++) {
		if (Mode == 1) {
			fprintf(Fp, "%12.4E\n", Out_Ptr->Rd_r[ir]);
		}
		else {
			fscanf_s(Fp, "%lf", &(Out_Ptr->Rd_r[ir]));
		}
	}

	if (Mode == 1) {
		fprintf(Fp, "\n");
	}
}

/**************************************************************************
 *  1 number each line.
 *  Mode = 0, read; Mode = 1, write.
 ****/
void IORd_a(FILE* Fp, short Na, OutStru* Out_Ptr, char Mode) {
	if (Mode == 1) {
		/* flag. */
		fprintf(Fp, "Rd_a #Rd[0], [1],..Rd[na-1]. [1/sr]\n");
	}
	else {
		FindDataLine(Fp);
	}

	for (short ia = 0; ia < Na; ia++) {
		if (Mode == 1) {
			fprintf(Fp, "%12.4E\n", Out_Ptr->Rd_a[ia]);
		}
		else {
			fscanf_s(Fp, "%lf", &(Out_Ptr->Rd_a[ia]));
		}
	}

	if (Mode == 1) {
		fprintf(Fp, "\n");
	}
}

/**************************************************************************
 *  1 number each line.
 *  Mode = 0, read; Mode = 1, write.
 ****/
void IORd_t(FILE* Fp, short Nt, OutStru* Out_Ptr, char Mode) {
	if (Mode == 1) {
		/* flag. */
		fprintf(Fp, "Rd_t #Rd[0], [1],..Rd[nt-1]. [1/ps]\n");
	}
	else {
		FindDataLine(Fp);
	}

	for (short it = 0; it < Nt; it++) {
		if (Mode == 1) {
			fprintf(Fp, "%12.4E\n", Out_Ptr->Rd_t[it]);
		}
		else {
			fscanf_s(Fp, "%lf", &(Out_Ptr->Rd_t[it]));
		}
	}

	if (Mode == 1) {
		fprintf(Fp, "\n");
	}
}

/**************************************************************************
 *  5 numbers each line.
 *  Mode = 0, read; Mode = 1, write.
 ****/
void IOTd_rat(FILE* Fp, short Nr, short Na, short Nt, OutStru* Out_Ptr, char Mode) {
	if (Mode == 1) {
		/* flag. */
		fprintf(Fp, "%s\n%s\n%s\n%s\n%s\n%s\n", "# Td[r][a][t]. [1/(cm2 sr ps)]",
				"# Td[0][0][0], [0][0][1],..[0][0][nt-1]", "# Td[0][1][0], [0][1][1],..[0][1][nt-1]", "# ...",
				"# Td[nr-1][na-1][0], [nr-1][na-1][1],..[nr-1][na-1][nt-1]", "Td_rat");
	}
	else {
		FindDataLine(Fp);
	}

	short i = 0;
	for (short ir = 0; ir < Nr; ir++) {
		for (short ia = 0; ia < Na; ia++) {
			for (short it = 0; it < Nt; it++) {
				if (Mode == 1) {
					fprintf(Fp, "%12.4E ", Out_Ptr->Td_rat[ir][ia][it]);
					if (++i % 5 == 0) {
						fprintf(Fp, "\n");
					}
				}
				else {
					fscanf_s(Fp, "%lf", &(Out_Ptr->Td_rat[ir][ia][it]));
				}
			}
		}
	}

	if (Mode == 1) {
		fprintf(Fp, "\n\n");
	}
}

/**************************************************************************
 *  5 numbers each line.
 *  Mode = 0, read; Mode = 1, write.
 ****/
void IOTd_ra(FILE* Fp, short Nr, short Na, OutStru* Out_Ptr, char Mode) {
	if (Mode == 1) {
		/* flag. */
		fprintf(Fp, "%s\n%s\n%s\n%s\n%s\n%s\n", "# Td[r][angle]. [1/(cm2 sr)].", "# Td[0][0], [0][1],..[0][na-1]",
				"# Td[1][0], [1][1],..[1][na-1]", "# ...", "# Td[nr-1][0], [nr-1][1],..[nr-1][na-1]", "Td_ra");
	}
	else {
		FindDataLine(Fp);
	}

	for (short ir = 0; ir < Nr; ir++) {
		for (short ia = 0; ia < Na; ia++) {
			if (Mode == 1) {
				fprintf(Fp, "%12.4E ", Out_Ptr->Td_ra[ir][ia]);
				if ((ir * Na + ia + 1) % 5 == 0) {
					fprintf(Fp, "\n");
				}
			}
			else {
				fscanf_s(Fp, "%lf", &(Out_Ptr->Td_ra[ir][ia]));
			}
		}
	}

	if (Mode == 1) {
		fprintf(Fp, "\n\n");
	}
}

/**************************************************************************
 *  5 numbers each line.
 *  Mode = 0, read; Mode = 1, write.
 ****/
void IOTd_rt(FILE* Fp, short Nr, short Nt, OutStru* Out_Ptr, char Mode) {
	if (Mode == 1) {
		/* flag. */
		fprintf(Fp, "%s\n%s\n%s\n%s\n%s\n%s\n", "# Td[r][t]. [1/(cm2 ps)]", "# Td[0][0], [0][1],..[0][nt-1]",
				"# Td[0][0], [0][1],..[0][nt-1]", "# ...", "# Td[nr-1][0], [nr-1][1],..[nr-1][nt-1]", "Td_rt");
	}
	else {
		FindDataLine(Fp);
	}

	short i = 0;
	for (short ir = 0; ir < Nr; ir++) {
		for (short it = 0; it < Nt; it++) {
			if (Mode == 1) {
				fprintf(Fp, "%12.4E ", Out_Ptr->Td_rt[ir][it]);
				if (++i % 5 == 0) {
					fprintf(Fp, "\n");
				}
			}
			else {
				fscanf_s(Fp, "%lf", &(Out_Ptr->Td_rt[ir][it]));
			}
		}
	}

	if (Mode == 1) {
		fprintf(Fp, "\n\n");
	}
}

/**************************************************************************
 *  5 numbers each line.
 *  Mode = 0, read; Mode = 1, write.
 ****/
void IOTd_at(FILE* Fp, short Na, short Nt, OutStru* Out_Ptr, char Mode) {
	if (Mode == 1) {
		/* flag. */
		fprintf(Fp, "%s\n%s\n%s\n%s\n%s\n%s\n", "# Td[a][t]. [1/(sr ps)]", "# Td[0][0], [0][1],..[0][nt-1]",
				"# Td[1][0], [1][1],..[1][nt-1]", "# ...", "# Td[na-1][0], [na-1][1],..[na-1][nt-1]", "Td_at");
	}
	else {
		FindDataLine(Fp);
	}

	short i = 0;
	for (short ia = 0; ia < Na; ia++) {
		for (short it = 0; it < Nt; it++) {
			if (Mode == 1) {
				fprintf(Fp, "%12.4E ", Out_Ptr->Td_at[ia][it]);
				if (++i % 5 == 0) {
					fprintf(Fp, "\n");
				}
			}
			else {
				fscanf_s(Fp, "%lf", &(Out_Ptr->Td_at[ia][it]));
			}
		}
	}

	if (Mode == 1) {
		fprintf(Fp, "\n\n");
	}
}

/**************************************************************************
 *  1 number each line.
 *  Mode = 0, read; Mode = 1, write.
 ****/
void IOTd_r(FILE* Fp, short Nr, OutStru* Out_Ptr, char Mode) {
	if (Mode == 1) {
		/* flag. */
		fprintf(Fp, "Td_r #Td[0], [1],..Td[nr-1]. [1/cm2]\n");
	}
	else {
		FindDataLine(Fp);
	}

	for (short ir = 0; ir < Nr; ir++) {
		if (Mode == 1) {
			fprintf(Fp, "%12.4E\n", Out_Ptr->Td_r[ir]);
		}
		else {
			fscanf_s(Fp, "%lf", &(Out_Ptr->Td_r[ir]));
		}
	}

	if (Mode == 1) {
		fprintf(Fp, "\n");
	}
}

/**************************************************************************
 *  1 number each line.
 *  Mode = 0, read; Mode = 1, write.
 ****/
void IOTd_a(FILE* Fp, short Na, OutStru* Out_Ptr, char Mode) {
	if (Mode == 1) {
		/* flag. */
		fprintf(Fp, "Td_a #Td[0], [1],..Td[na-1]. [1/sr]\n");
	}
	else {
		FindDataLine(Fp);
	}

	for (short ia = 0; ia < Na; ia++) {
		if (Mode == 1) {
			fprintf(Fp, "%12.4E\n", Out_Ptr->Td_a[ia]);
		}
		else {
			fscanf_s(Fp, "%lf", &(Out_Ptr->Td_a[ia]));
		}
	}

	if (Mode == 1) {
		fprintf(Fp, "\n");
	}
}

/**************************************************************************
 *  1 number each line.
 *  Mode = 0, read; Mode = 1, write.
 ****/
void IOTd_t(FILE* Fp, short Nt, OutStru* Out_Ptr, char Mode) {
	if (Mode == 1) {
		/* flag. */
		fprintf(Fp, "Td_t #Rd[0], [1],..Td[nt-1]. [1/ps]\n");
	}
	else {
		FindDataLine(Fp);
	}

	for (short it = 0; it < Nt; it++) {
		if (Mode == 1) {
			fprintf(Fp, "%12.4E\n", Out_Ptr->Td_t[it]);
		}
		else {
			fscanf_s(Fp, "%lf", &(Out_Ptr->Td_t[it]));
		}
	}

	if (Mode == 1) {
		fprintf(Fp, "\n");
	}
}

/**************************************************************************
 *  Mode = 0, read result back from a output file.
 *  Mode = 1, write result to a output file;
 ****/
void IOResult(FILE* Fp, InStru* In_Ptr, OutStru* Out_Ptr, char Mode) {
	if (Mode == 1) {
		if (toupper(In_Ptr->out_fformat) == 'A') {
			WriteVersion(Fp, "mcmloA2.1");
		}
		else {
			WriteVersion(Fp, "mcmloB2.1");
		}

		PutInputToFile(Fp, In_Ptr);
		SaveRandomStatus(Fp);
		WriteRAT(Fp, Out_Ptr);
	}
	else {
		RestoreRandomStatus(Fp);
		ReadRAT(Fp, Out_Ptr);
	}

	/* reflectance, absorption, transmittance. */
	if (In_Ptr->record.A_rzt) {
		IOA_rzt(Fp, In_Ptr->nr, In_Ptr->nz, In_Ptr->nt, Out_Ptr, Mode);
	}
	if (In_Ptr->record.A_rz) {
		IOA_rz(Fp, In_Ptr->nr, In_Ptr->nz, Out_Ptr, Mode);
	}
	if (In_Ptr->record.A_zt) {
		IOA_zt(Fp, In_Ptr->nz, In_Ptr->nt, Out_Ptr, Mode);
	}
	if (In_Ptr->record.A_z) {
		IOA_z(Fp, In_Ptr->nz, Out_Ptr, Mode);
	}
	if (In_Ptr->record.A_t) {
		IOA_t(Fp, In_Ptr->nt, Out_Ptr, Mode);
	}

	if (In_Ptr->record.Rd_rat) {
		IORd_rat(Fp, In_Ptr->nr, In_Ptr->na, In_Ptr->nt, Out_Ptr, Mode);
	}
	if (In_Ptr->record.Rd_ra) {
		IORd_ra(Fp, In_Ptr->nr, In_Ptr->na, Out_Ptr, Mode);
	}
	if (In_Ptr->record.Rd_rt) {
		IORd_rt(Fp, In_Ptr->nr, In_Ptr->nt, Out_Ptr, Mode);
	}
	if (In_Ptr->record.Rd_at) {
		IORd_at(Fp, In_Ptr->na, In_Ptr->nt, Out_Ptr, Mode);
	}
	if (In_Ptr->record.Rd_r) {
		IORd_r(Fp, In_Ptr->nr, Out_Ptr, Mode);
	}
	if (In_Ptr->record.Rd_a) {
		IORd_a(Fp, In_Ptr->na, Out_Ptr, Mode);
	}
	if (In_Ptr->record.Rd_t) {
		IORd_t(Fp, In_Ptr->nt, Out_Ptr, Mode);
	}

	if (In_Ptr->record.Td_rat) {
		IOTd_rat(Fp, In_Ptr->nr, In_Ptr->na, In_Ptr->nt, Out_Ptr, Mode);
	}
	if (In_Ptr->record.Td_ra) {
		IOTd_ra(Fp, In_Ptr->nr, In_Ptr->na, Out_Ptr, Mode);
	}
	if (In_Ptr->record.Td_rt) {
		IOTd_rt(Fp, In_Ptr->nr, In_Ptr->nt, Out_Ptr, Mode);
	}
	if (In_Ptr->record.Td_at) {
		IOTd_at(Fp, In_Ptr->na, In_Ptr->nt, Out_Ptr, Mode);
	}
	if (In_Ptr->record.Td_r) {
		IOTd_r(Fp, In_Ptr->nr, Out_Ptr, Mode);
	}
	if (In_Ptr->record.Td_a) {
		IOTd_a(Fp, In_Ptr->na, Out_Ptr, Mode);
	}
	if (In_Ptr->record.Td_t) {
		IOTd_t(Fp, In_Ptr->nt, Out_Ptr, Mode);
	}

	fclose(Fp);
}
