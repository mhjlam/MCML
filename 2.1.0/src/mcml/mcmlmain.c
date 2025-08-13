/**************************************************************************
 *	Copyright Univ. of Texas M.D. Anderson Cancer Center, 1992-1996.
 *	Program name: MCML.
 *	The main program for Monte Carlo simulation of light transport
 *	in multi-layered turbid media.
 ****/

#include "mcml.h"

/* Declare before they are used in main(). */
/* Modern C17 with restrict keywords and const correctness */
void InitOutputData(const InStru* restrict, OutStru* restrict);
void CtrPuts(const char* restrict);
char* FindDataLine(FILE* restrict);
void InterReadParam(InStru* restrict);
void LaunchPhoton(double, const InStru* restrict, OutStru* restrict, PhotonStru* restrict);
void AboutMCML(void);
void IOResult(FILE* restrict, const InStru* restrict, OutStru* restrict, char);
void CheckParamFromFile(FILE* restrict, const InStru* restrict);
void ReadRunParam(FILE* restrict, InStru* restrict);
void ScaleResult(const InStru* restrict, OutStru* restrict, char);
double Rspecular(const LayerStru* restrict);
bool ReadMediumListQ(FILE* restrict, InStru* restrict);
void TracePhoton(const InStru* restrict, PhotonStru* restrict, OutStru* restrict);
bool RunChangedInput(InStru* restrict);
FILE* GetFile(char* restrict, const char* restrict);
void FreeData(const InStru* restrict, OutStru* restrict);
bool ReadNumPhotonsQ(FILE* restrict, InStru* restrict, char);
bool CheckFileVersionQ(FILE* restrict, const char* restrict);

#define SWAP(x, y)  \
	{               \
		long temp;  \
		temp = x;   \
		(x) = y;    \
		(y) = temp; \
	}

/**************************************************************************
 *  If F = 0, reset the clock and return 0.
 *
 *  If F = 1, pass the real time to Msg, print Msg on the
 *  screen, return the real time elapsed since F=0.
 *
 *  If F = 2, same as F=1 except no printing.
 *
 *  Note that clock() and time() return user time and real
 *  time respectively.  User time is whatever the system allocates
 *  to the running of the program; real time is wall-clock time.
 *  In a time-shared system, they need not be the same.
 *
 *  clock() only hold 16 bit integer, which is about 32768
 *  clock ticks.  Because this fact can lead to wrap-around, it
 *  is dangerous to use clock() in Monte Carlo simulations.
 *  Therefore, we only keep track of the real time.
 *
 *  On UNIX machines, users can use time command to get the
 *  user time.  On personal computers, the user time is equal
 *  to the real time unless the system is multi-tasking during
 *  the Monte Carlo simulation, which usually is not true.
 ****/
time_t PunchTime(char F, char* restrict Msg, const InStru* restrict In_Ptr) {
	/* real time reference - thread-local for thread safety */
	static _Thread_local time_t rt0 = 0;

	if (F == 0) {
		rt0 = time(NULL);
		return 0;
	}

	const double real_time_secs = difftime(time(NULL), rt0);

	/* show & pass real time. */
	if (F == 1 && Msg && In_Ptr) {
		const double total_time = real_time_secs + In_Ptr->add_num_seconds;
		snprintf(Msg, STRLEN, "Real time for this simulation: %.1f sec = %.2f hr.", total_time, total_time / 3600.0);
	}
	return (time_t)real_time_secs;
}

/**************************************************************************
 *	Print the current time and the estimated finishing time.
 *  Modernized with variables declared at point of use.
 *
 *	P1 is the number of computed photon packets.
 *	Pt is the total number of photon packets.
 ****/
void PredictDoneTime(long P1, const InStru* restrict In_Ptr) {
	if (!In_Ptr || P1 <= 0) {
		return;
	}

	const time_t now = time(NULL);
	struct tm date;
	localtime_s(&date, &now);

	char time_str[SMALL_BUF];
	strftime(time_str, sizeof(time_str), "%H:%M %a %m/%d/%Y", &date);
	printf("%s\t", time_str);

	if (In_Ptr->control_bit == 1 || In_Ptr->control_bit == 3) {
		char msg[STRLEN];
		const time_t elapsed = PunchTime(2, msg, In_Ptr);
		const double time_per_photon = (double)elapsed / P1;
		const long remaining_photons = In_Ptr->num_photons - P1;

		time_t done_time = now + (time_t)(time_per_photon * remaining_photons);

		if (In_Ptr->control_bit == 3) {
			const time_t deadline = now + In_Ptr->num_seconds - elapsed;
			if (done_time > deadline) {
				done_time = deadline;
			}
		}

		localtime_s(&date, &done_time);
		char done_str[SMALL_BUF];
		strftime(done_str, sizeof(done_str), "%H:%M %a %m/%d/%Y", &date);
		printf("%s", done_str);
	}
	printf("\n");
}

/**************************************************************************
 *	Generate a string representing the user-specified done time.
 ****/
void FormDateString(char* String, InStru* In_Ptr) {
	char msg[STRLEN];
	time_t now = time(NULL);
	time_t done_time = now + In_Ptr->num_seconds - PunchTime(2, msg, In_Ptr);

	struct tm date;
	localtime_s(&date, &done_time);
	strftime(String, 80, "%H:%M on %Y/%m/%d", &date);
}

/**************************************************************************
 *	Report how and when the simultion will be terminated.
 ****/
void ReportControlInfo(short NumRunsLeft, InStru* In_Ptr) {
	char string[STRLEN];

	printf("%d runs left.", NumRunsLeft);
	if (In_Ptr->control_bit == 1) {
		printf("\nThe simulation will terminate after tracing %ld photons.\n\n", In_Ptr->num_photons);
		printf("\tPhotons Done\tCurrent Time\t\tEstimated Done Time\n");
		printf("\t------------\t--------------------\t--------------------\n");
	}
	else if (In_Ptr->control_bit == 2) {
		FormDateString(string, In_Ptr);
		printf("\nThe simulation will terminate at %s.\n\n", string);
		printf("\tPhotons Done\tCurrent Time\n");
		printf("\t------------\t--------------------\n");
	}
	else {
		FormDateString(string, In_Ptr);
		printf("\nThe simulation will terminate after tracing %ld photons ", In_Ptr->num_photons);
		printf("or at %s, whichever comes first.\n\n", string);
		printf("\tPhotons Done\tCurrent Time\t\tEstimated Done Time\n");
		printf("\t------------\t--------------------\t--------------------\n");
	}
}

/**************************************************************************
 *	Report the estimated time, number of photons and runs left
 *	after calculating 10 photons or every 1/10 of total
 *	number of photons.
 *
 *	Pi is the number of traced photons so far.
 ****/
void ReportStatus(long Pi, InStru* In_Ptr) {
	if (In_Ptr->control_bit == 1) {
		printf("%11ld(%6.2f%%)\t", Pi, (float)Pi * 100 / In_Ptr->num_photons);
	}
	else if (In_Ptr->control_bit == 2) {
		printf("\t%12ld\t", Pi);
	}
	else {
		printf("%11ld(%6.2f%%)\t", Pi, (float)Pi * 100 / In_Ptr->num_photons);
	}

	PredictDoneTime(Pi, In_Ptr);
}

/**************************************************************************
 *	Report time, photon number traced and write results.
 ****/
void ReportResult(InStru* In_Ptr, OutStru* Out_Ptr) {
	char time_report[STRLEN];
	PunchTime(1, time_report, In_Ptr);
	printf("\n%ld photons have been traced in this simulation.\n", In_Ptr->num_photons);

	ScaleResult(In_Ptr, Out_Ptr, 0);

	FILE* fp;
	if (fopen_s(&fp, In_Ptr->out_fname, "w")) {
		printf("Can not open output file to write.\n");
		exit(1);
	}
	IOResult(fp, In_Ptr, Out_Ptr, 1);
}

/**************************************************************************
 *	Get the file name of the input data file from the
 *	argument to the command line.
 ****/
void GetFnameFromArgv(int argc, char* argv[], char* input_filename) {
	/* filename in command line */
	if (argc >= 2) {
		strcpy_s(input_filename, STRLEN, argv[1]);
	}
	else {
		input_filename[0] = '\0';
	}
}

/**************************************************************************
 *	Execute Monte Carlo simulation for one independent run.
 *      Type = 0, start a new simulation;
 *      Type = 1, continue previous simulation.
 ****/
void DoOneRun(short NumRunsLeft, InStru* In_Ptr, OutStru* Out_Ptr, char Type) {
	int tens = 10;

	/* start a new simulation. */
	if (Type == 0) {
		if (In_Ptr->slayer == 0) {
			Out_Ptr->Rsp = Rspecular(In_Ptr->layerspecs);
		}
		/* initialize the generator. */
		RandomGen(0, 1, NULL);
	}

	char msg[STRLEN];
	PunchTime(0, msg, In_Ptr);
	ReportControlInfo(NumRunsLeft, In_Ptr);

	/* photon number traced.  */
	long i_photon = 1;

	/* switch to terminate simulation. */
	bool exit_switch = 0;

	do {
		PhotonStru photon;
		LaunchPhoton(Out_Ptr->Rsp, In_Ptr, Out_Ptr, &photon);
		TracePhoton(In_Ptr, &photon, Out_Ptr);

		/* report status every ten photons. */
		if (i_photon == tens) {
			tens *= 10;
			ReportStatus(i_photon, In_Ptr);
		}
		i_photon++;
		if (In_Ptr->control_bit == 1) {
			exit_switch = (i_photon > In_Ptr->num_photons);
		}
		else if (In_Ptr->control_bit == 2) {
			exit_switch = (PunchTime(2, msg, In_Ptr) >= In_Ptr->num_seconds);
		}
		else {
			exit_switch = (i_photon > In_Ptr->num_photons) || (PunchTime(2, msg, In_Ptr) >= In_Ptr->num_seconds);
		}
	}
	while (!exit_switch);

	In_Ptr->num_photons = In_Ptr->add_num_photons + i_photon - 1;
	In_Ptr->num_seconds = In_Ptr->add_num_seconds + (long)PunchTime(2, msg, In_Ptr);
	In_Ptr->control_bit = 3;

	ReportResult(In_Ptr, Out_Ptr);
	FreeData(In_Ptr, Out_Ptr);
}

/**************************************************************************
 *	In continuation runs, ask for additional number of photons or time.
 ****/
void AddNumPhotons(InStru* In_Ptr) {
	printf("\n%ld photons have been traced in the previous simulation.", In_Ptr->num_photons);
	printf("\nSpecify additional photons or compution time in hh:mm format,");
	printf("\nor both in one line (e.g. 10000 5:30): ");

	while (!ReadNumPhotonsQ(stdin, In_Ptr, 1)) {
		printf("Input agian: ");
	}

	printf("\n");
}

/**************************************************************************
 *	Start a new simulation non-interactively.
 *  Read input parameter from file input_filename.
 ****/
void NonInterSimu(FILE* Fp, InStru* In_Ptr, OutStru* Out_Ptr) {
	CheckParamFromFile(Fp, In_Ptr);
	short num_runs_left = In_Ptr->num_runs;
	while (num_runs_left--) {
		ReadRunParam(Fp, In_Ptr);
		InitOutputData(In_Ptr, Out_Ptr);
		DoOneRun(num_runs_left, In_Ptr, Out_Ptr, 0);
	}
	fclose(Fp);
	exit(0);
}

/**************************************************************************
 *	Read input parameters from a file with interactive change.
 ****/
void FileInterSimu(InStru* In_Ptr, OutStru* Out_Ptr) {
	char input_filename[STRLEN];
	FILE* input_file_ptr = GetFile(input_filename, "mcmli2.1");

	if (input_file_ptr != NULL) {
		if (ReadMediumListQ(input_file_ptr, In_Ptr)) {
			ReadRunParam(input_file_ptr, In_Ptr);
			printf("The parameters of the first run have been read in.\n");

			if (RunChangedInput(In_Ptr)) {
				InitOutputData(In_Ptr, Out_Ptr);
				DoOneRun(0, In_Ptr, Out_Ptr, 0);
				fclose(input_file_ptr);
				exit(0);
			}
			fclose(input_file_ptr);
		}
	}
}

/**************************************************************************
 *	Continue a previous simulation.
 ****/
void ContinueSimu(InStru* In_Ptr, OutStru* Out_Ptr) {
	printf("Specify the output file name of a previous simulation. \n");

	char input_filename[STRLEN];
	FILE* fp = GetFile(input_filename, "mcmloA2.1");
	if (fp == NULL) {
		return;
	}

	/* skip the line of file version. */
	FindDataLine(fp);
	if (!ReadMediumListQ(fp, In_Ptr)) {
		exit(1);
	}
	ReadRunParam(fp, In_Ptr);

	AddNumPhotons(In_Ptr);
	InitOutputData(In_Ptr, Out_Ptr);
	IOResult(fp, In_Ptr, Out_Ptr, 0);
	ScaleResult(In_Ptr, Out_Ptr, 1);

	SWAP(In_Ptr->num_photons, In_Ptr->add_num_photons);
	SWAP(In_Ptr->num_seconds, In_Ptr->add_num_seconds);
	DoOneRun(0, In_Ptr, Out_Ptr, 1);
	exit(0);
}

/**************************************************************************
****/
void QuitProgram(void) {
	printf("Do you really want to quit MCML? (y/n): ");

	char cmd_str[STRLEN];
	fgets(cmd_str, STRLEN, stdin);

	/* really quit. */
	if (toupper(cmd_str[0]) == 'Y') {
		exit(0);
	}
}

/**************************************************************************
 ****/
void ShowMainMenu(void) {
	puts("  a = About MCML.");
	puts("  r = Run an input file non-interactively.");
	puts("  m = Input and modify parameters of a file (the first run only).");
	puts("  i = Input parameters interactively.");
	puts("  c = Continue a previous simulation.");
	puts("  q = Quit from the program.");
	puts("  * Commands here are not case-sensitive.");
}

/**************************************************************************
 ****/
void BranchMainMenu(char* string, InStru* In_Ptr, OutStru* Out_Ptr) {
	switch (toupper(string[0])) {
		case 'A': {
			AboutMCML();
			break;
		}

		/* non-interactive. */
		case 'R': {
			char input_filename[STRLEN] = {0};
			FILE* input_file_ptr = GetFile(input_filename, "mcmli2.1");

			if (input_file_ptr != NULL) {
				NonInterSimu(input_file_ptr, In_Ptr, Out_Ptr);
			}
			break;
		}

		/* read a file with an interactive change. */
		case 'M': {
			FileInterSimu(In_Ptr, Out_Ptr);
			break;
		}

		/* interactive. */
		case 'I': {
			InterReadParam(In_Ptr);
			if (RunChangedInput(In_Ptr)) {
				InitOutputData(In_Ptr, Out_Ptr);
				DoOneRun(0, In_Ptr, Out_Ptr, 0);
				exit(0);
			}
			break;
		}

		case 'C': {
			ContinueSimu(In_Ptr, Out_Ptr);
			break;
		}

		case 'H': {
			ShowMainMenu();
			break;
		}

		case 'Q': {
			QuitProgram();
			break;
		}

		default: {
			puts("...Unknown command");
		}
	}
}

/**************************************************************************
 *	The argument to the command line is the input filename, if any.
 ****/
int main(int argc, char* argv[]) {
	printf("MCML Version 2.1.0, Copyright (c) 1992-1996, 2025\n\n");
	/* \251 is the formal copyright sign but does not show up on some terminals.*/

	/* non-interactive. */
	if (argc >= 2) {
		char input_filename[STRLEN];
		GetFnameFromArgv(argc, argv, input_filename);

		FILE* input_file_ptr;
		fopen_s(&input_file_ptr, input_filename, "r");

		if (input_file_ptr != NULL) {
			if (CheckFileVersionQ(input_file_ptr, "mcmli2.1")) {
				InStru in_param;
				OutStru out_param;
				NonInterSimu(input_file_ptr, &in_param, &out_param);
			}
		}
		exit(0);
	}
	/* accept commands from console. */
	else {
		while (1) {
			char str[STRLEN];
			do {
				printf("\n> Main menu (h for help) => ");
				fgets(str, STRLEN, stdin);
			}
			while (!strlen(str));

			InStru in_param;
			OutStru out_param;
			BranchMainMenu(str, &in_param, &out_param);
		}
	}
}
