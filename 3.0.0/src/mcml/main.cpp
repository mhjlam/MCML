/*******************************************************************************
 *	Copyright Univ. of Texas M.D. Anderson Cancer Center, 1992-1996.
 *  Copyright M.H.J. Lam, 2025.
 *	Monte Carlo simulation of light transport in multi-layered turbid mediums.
 ****/


#include "mcml.hpp"

#include <chrono>
#include <format>
#include <fstream>
#include <iostream>

#include "timer.hpp"
#include "random.hpp"
#include "simulator.hpp"


static void about()
{
    std::cout << "MCML 3.0, Copyright (c) 1992-1996, 2025" << std::endl;
    std::cout << "Monte Carlo Simulation of Light Transport in Multi-Layered Turbid Media" << std::endl;

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

    std::cout << std::endl;

    std::cout << "Please cite the following article in your publications:" << std::endl;
    std::cout << "  L.-H. Wang, S. L. Jacques, and L.-Q. Zheng," << std::endl;
    std::cout << "  MCML - Monte Carlo modeling of photon transport in multi-layered tissues," << std::endl;
    std::cout << "  Computer Methods and Programs in Biomedicine, 47, 131-146 (1995)" << std::endl;
}

static void help()
{
    std::cout << "  A = About MCML." << std::endl;
    std::cout << "  R = Run an input file non-interactively." << std::endl;
    std::cout << "  M = Input and modify parameters of a file (the first run only)." << std::endl;
    std::cout << "  I = Input parameters interactively." << std::endl;
    std::cout << "  C = Continue a previous simulation." << std::endl;
    std::cout << "  Q = Quit from the program." << std::endl;
    std::cout << "  * Commands here are not case-sensitive." << std::endl;
}

static void quit()
{
    std::cout << "Do you really want to quit MCML? (Y/N): ";

    char command;
    do {
        std::cin.get(command);
        command = std::toupper(command);
    } while (command != 'Y' && command != 'N');

    if (command == 'Y') {
        std::exit(0);
    }
}


int main(int argc, char* argv[])
{
    std::cout << "MCML Version 3.0, Copyright (c) 1992-1996, 2025" << std::endl << std::endl;

    try {
        // Read parameters from input file
        if (argc >= 2) {
            std::string input_filename = (argc >= 2) ? std::string(argv[1]) : std::string();
            std::shared_ptr<Simulator> simulator = std::make_shared<Simulator>(input_filename);
            simulator->Simulate();
        }
        // Accept commands from console
        else {
            std::shared_ptr<Simulator> simulator = std::make_shared<Simulator>();

            char command = '\0';
            while (true) {
                std::cout << std::endl << "> Main menu (H for help) => ";

                do {
                    // Read input
                    std::cin.get(command);
                    command = std::toupper(command);
                } while (command == '\0' || command == '\n');

                // Clear buffer
                std::cin.ignore(max_size, '\n');

                switch (command) {
                    case 'A': { about(); break; }
                    case 'R': { simulator->Simulate(); break; }
                    case 'M': { simulator->InteractiveEdit(); break; }
                    case 'I': { simulator->Interactive(); break; }
                    case 'C': { simulator->Resume(); break; }
                    case 'H': { help(); break; }
                    case 'Q': { quit(); break; }
                    default: { std::cerr << "Unknown command." << std::endl; }
                }
            }
        }
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
    }
}
