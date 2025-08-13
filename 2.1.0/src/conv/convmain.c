/**************************************************************************
 *  Copyright Univ. of Texas M.D. Anderson Cancer Center, 1992.
 *
 *  Convolution program for Monte Carlo simulation of photon
 *  distribution in multilayered turbid media.
 ****/

#include "conv.h"

#define EPS      0.1 /* default relative error in convolution. */
#define COLWIDTH 80  /* column width for printing. */

// Function prototypes with descriptive parameter names
void ReadMcoFile(InStru *in_ptr, OutStru *out_ptr, ConvStru *conv_ptr);
void ExtractRAT(InStru *in_ptr, OutStru *out_ptr, ConvStru *conv_ptr);
void ExtractOrigData(InStru *in_ptr, OutStru *out_ptr, ConvStru *conv_ptr);
void ExtractConvData(InStru *in_ptr, OutStru *out_ptr, ConvStru *conv_ptr);
void LaserBeam(BeamStru *beam_ptr);
void ConvError(float *error_ptr);

/**************************************************************************
 * Show the main menu of commands.
 ****/
void ShowMainMenu(void) {
	puts("  a = About CONV.");
	puts("  i = Input a file of MCML output.");
	puts("  r = Reflectance, absorption, and transmittance.");
	puts("  o = Extract original data.\n");

	puts("  b = Specify laser beam.");
	puts("  e = Specify convolution error. ");
	puts("  c = Extract convolved data.");
	puts("  q = Quit from the program.");
	puts("  * Commands in conv are not case-sensitive");
}

/**************************************************************************
 * Quit the program after confirmation.
 ****/
void QuitProgram(void) {
	char cmd_str[STRLEN];
	char input_buffer[STRLEN];

	printf("Do you really want to exit CONV? (y/n): ");
	if (fgets(input_buffer, sizeof(input_buffer), stdin) != NULL) {
		// Remove trailing newline
		size_t len = strlen(input_buffer);
		if (len > 0 && input_buffer[len - 1] == '\n') {
			input_buffer[len - 1] = '\0';
		}
		// Copy input with length check to avoid truncation warning
		size_t input_len = strlen(input_buffer);
		size_t max_copy = sizeof(cmd_str) - 1;
		if (input_len < max_copy) {
			strcpy(cmd_str, input_buffer);
		}
		else {
			memcpy(cmd_str, input_buffer, max_copy);
			cmd_str[max_copy] = '\0';
		}

		if (toupper(cmd_str[0]) == 'Y') { /* really quit. */
			exit(0);
		}
	}
}

/**************************************************************************
 *  Center a string according to the column width.
 ****/
void CtrPuts(const char *InStr) {
	short nspaces;             /* number of spaces to be left-filled. */
	char outstr[STRLEN] = {0}; /* Initialize to zero */

	nspaces = (COLWIDTH - (short)strlen(InStr)) / 2;
	if (nspaces < 0) {
		nspaces = 0;
	}

	// Use snprintf for safer string formatting
	int pos = 0;
	for (int i = 0; i < nspaces && pos < STRLEN - 1; i++) {
		outstr[pos++] = ' ';
	}

	size_t remaining = STRLEN - pos - 1;
	if (remaining > 0) {
		strncpy(outstr + pos, InStr, remaining);
		outstr[STRLEN - 1] = '\0'; // Ensure null termination
	}

	puts(outstr);
}

/**************************************************************************
 *  Display the program information.
 ****/
void AboutConv(void) {
	CtrPuts(" ");
	CtrPuts("CONV 2.1, Copyright (c) 1992-1996, 2025\n");
	CtrPuts("Convolution of MCML Simulation Data\n");

	CtrPuts("Lihong Wang, Ph.D.");
	CtrPuts("Bioengineering Program, Texas A&M University");
	CtrPuts("College Station, Texas 77843-3120, USA");

	CtrPuts("Liqiong Zheng, B.S.");
	CtrPuts("Summer student from Dept. of Computer Science,");
	CtrPuts("University of Houston, Texas, USA.");

	CtrPuts("Steven L. Jacques, Ph.D.");
	CtrPuts("Oregon Medical Laser Center, Providence/St. Vincent Hospital");
	CtrPuts("9205 SW Barnes Rd., Portland, Oregon 97225, USA");
}

/**************************************************************************
 *  Main command dispatcher for the CONV program.
 *  This function interprets the command and calls the appropriate function.
 ****/
void BranchMainCmd(char *Cmd, InStru *In_Ptr, OutStru *Out_Ptr, ConvStru *Conv_Ptr) {
	switch (toupper(Cmd[0])) {
		case 'A': AboutConv(); break;
		case 'I': ReadMcoFile(In_Ptr, Out_Ptr, Conv_Ptr); break;
		case 'R': ExtractRAT(In_Ptr, Out_Ptr, Conv_Ptr); break;
		case 'O': ExtractOrigData(In_Ptr, Out_Ptr, Conv_Ptr); break;
		case 'B': LaserBeam(&Conv_Ptr->beam); break;
		case 'E': ConvError(&Conv_Ptr->eps); break;
		case 'C': ExtractConvData(In_Ptr, Out_Ptr, Conv_Ptr); break;
		case 'H': ShowMainMenu(); break;
		case 'Q': QuitProgram(); break;
		default: puts("...Wrong command");
	}
}

/**************************************************************************
 *  Read the MCML output file and initialize the structures.
 ****/
int main(void) {
	InStru in_parm = {0};
	OutStru out_parm = {0};
	ConvStru conv_parm;
	char cmd_str[STRLEN];

	puts(" ");
	CtrPuts("CONV Version 2.1, Copyright (c) 1992-1996, 2025\n");
	conv_parm.eps = EPS;
	conv_parm.beam.type = original;
	conv_parm.datain = 0; /* data is not read in yet. */

	do {
		printf("\n> Main menu (h for help) => ");
		do { /* get the command input. */
			if (fgets(cmd_str, sizeof(cmd_str), stdin) != NULL) {
				// Remove trailing newline
				size_t len = strlen(cmd_str);
				if (len > 0 && cmd_str[len - 1] == '\n') {
					cmd_str[len - 1] = '\0';
				}
			}
			else {
				cmd_str[0] = '\0'; // Handle EOF or error
			}
		}
		while (!strlen(cmd_str));
		BranchMainCmd(cmd_str, &in_parm, &out_parm, &conv_parm);
	}
	while (1);
}
