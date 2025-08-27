/*******************************************************************************
 *	Copyright Univ. of Texas M.D. Anderson Cancer Center, 1992-1996.
 *  Copyright M.H.J. Lam, 2025.
 *	Convolution program for MCML data processing in C++20.
 ****/

#include "processor.hpp"

#include <exception>
#include <iostream>
#include <sstream>

/**
 * @brief Display program information and credits
 */
static void about() {
	std::cout << "CONV 3.0, Copyright (c) 1992-1996, 2025" << std::endl;
	std::cout << "Convolution Program for Monte Carlo Multi-Layer (MCML) Output Processing" << std::endl;

	std::cout << std::endl;

	std::cout << "Lihong Wang, Ph.D." << std::endl;
	std::cout << "Bioengineering Program, Texas A&M University" << std::endl;
	std::cout << "College Station, Texas, USA" << std::endl;

	std::cout << std::endl;

	std::cout << "Liqiong Zheng, B.S." << std::endl;
	std::cout << "Dept. of Computer Science," << std::endl;
	std::cout << "University of Houston, Texas, USA." << std::endl;

	std::cout << std::endl;

	std::cout << "Steven L. Jacques, Ph.D." << std::endl;
	std::cout << "Oregon Medical Laser Center, Providence/St. Vincent Hospital" << std::endl;
	std::cout << "Portland, Oregon, USA" << std::endl;

	std::cout << std::endl;

	std::cout << "M.H.J. Lam, MSc." << std::endl;
	std::cout << "Utrecht University" << std::endl;
	std::cout << "Utrecht, Netherlands" << std::endl;

	std::cout << std::endl;

	std::cout << "Obtain the original program from omlc.org/software/mc" << std::endl;
	std::cout << "This C++20 version modernizes the original C implementation" << std::endl;
	
	std::cout << std::endl;
}

/**
 * @brief Display usage information
 */
static void usage(const char* program_name) {
	std::cout << "Usage: " << program_name << " [options] [input_file]" << std::endl;
	std::cout << std::endl;
	std::cout << "Options:" << std::endl;
	std::cout << "  -h, --help              Show this help message" << std::endl;
	std::cout << "  -a, --about             Show program information" << std::endl;
	std::cout << "  -v, --verbose           Enable verbose output" << std::endl;
	std::cout << "  -i, --interactive       Force interactive mode" << std::endl;
	std::cout << "  -b, --batch             Force batch mode" << std::endl;
	std::cout << "  -o, --output FILE       Specify output filename" << std::endl;
	std::cout << "  -f, --force             Overwrite existing output files" << std::endl;
	std::cout << "  --beam-type TYPE        Beam type: original, flat, gaussian, arbitrary" << std::endl;
	std::cout << "  --beam-radius R         Beam radius in cm" << std::endl;
	std::cout << "  --beam-power P          Beam power in J" << std::endl;
	std::cout << "  --epsilon E             Convolution relative error (default: 0.1)" << std::endl;
	std::cout << std::endl;
	std::cout << "Examples:" << std::endl;
	std::cout << "  " << program_name << "                        # Interactive mode" << std::endl;
	std::cout << "  " << program_name << " input.mco              # Process with default settings" << std::endl;
	std::cout << "  " << program_name << " -v --beam-type flat --beam-radius 0.1 input.mco" << std::endl;
	std::cout << std::endl;
}

/**
 * @brief Main entry point
 */
int main(int argc, char* argv[]) {
	try {
		// Handle special command-line arguments
		for (int i = 1; i < argc; ++i) {
			std::string arg = argv[i];
			if (arg == "-h" || arg == "--help") {
				usage(argv[0]);
				return 0;
			}
			if (arg == "-a" || arg == "--about") {
				about();
				return 0;
			}
		}

		// Create and run processor
		conv::ConvProcessor processor;

		if (argc == 1) {
			// No arguments - run in interactive mode
			return processor.run_interactive();
		} else {
			// Arguments provided - run in batch mode
			return processor.run_batch(argc, argv);
		}

	} catch (const conv::ConvException& e) {
		std::cerr << "CONV Error: " << e.what() << std::endl;
		return 1;
	} catch (const std::exception& e) {
		std::cerr << "Error: " << e.what() << std::endl;
		return 1;
	} catch (...) {
		std::cerr << "Unknown error occurred" << std::endl;
		return 1;
	}

	return 0;
}
