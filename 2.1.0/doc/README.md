# MCML 2.1

Monte Carlo simulation program for Multi-layered Turbid Media.

## Overview

**MCML** is a Monte Carlo simulation program for multi-layered turbid media with an infinitely narrow photon beam as the light source. The simulation is specified by an input text file (e.g., "sample.mci"), which can be edited with any simple text editor. The output is another text file (e.g., "sample.mco"). File names are arbitrary.

**CONV** is a convolution program that uses MCML output files to convolve photon beams of variable size or shape (Gaussian or flat field). CONV provides various output formats (reflectance, transmission, iso-fluence contours, etc.) compatible with standard graphics applications.

## Version History

| Version | Year | Description |
|---------|------|-------------|
| 1.2     | 1993 | Original MCML release by Lihong Wang and Steven Jacques |
| 1.2.2   | 2000 | Bug fixes and improvements by Lihong Wang |
| 2.0     | 2024 | Modernization by Lihong Wang and Scott Prahl |
| 2.1     | 2025 | C17 conformity, improved memory safety |

This version (2.1) is based on [Scott Prahl's MCML 2.0](https://github.com/scottprahl/MCML) modernization, with additional improvements including:

- **Registry-based memory management**: Centralized allocation tracking with automatic cleanup
- **Cross-platform compatibility**: Windows, Linux, macOS support with unified makefiles  
- **Modern C standards**: ANSI C compliance with enhanced safety features
- **Buffer overflow fixes**: Corrected array dimension specifications throughout
- **Enhanced build system**: Simplified makefiles with comprehensive testing
- **Interactive testing**: Automated test suites without persistent temporary files

## Building and Running

```bash
cd build/           # Navigate to build directory
make               # Build both applications
make test          # Test the build
cd ../bin/         # Go to executables
./mcml             # Run MCML simulation
./conv             # Run CONV convolution
```

Use `make help` for all available build options.

## Original Authors

**Lihong Wang, Ph.D.**  
Bren Professor of Medical Engineering and Electrical Engineering  
California Institute of Technology  
Pasadena, California  
Email: lvw@caltech.edu  
Web: http://coilab.caltech.edu

**Steven L. Jacques, Ph.D.**  
Department of Bioengineering  
University of Washington  
Seattle, Washington  
Email: stevjacq@uw.edu  
URL: https://spie.org/profile/Steve.Jacques-7427

**Scott Prahl, Ph.D.**  
Electrical Engineering & Renewable Energy  
Oregon Institute of Technology  
Portland, Oregon  
Email: scott.prahl@oit.edu  
URL: https://omlc.org/~prahl/

**Liqiong Zheng, B.S.**  
Department of Computer Science  
University of Houston  
Houston, Texas

## User Guide

### MCML Monte Carlo Simulation

MCML simulates light transport in multi-layered turbid media:

1. **Prepare Input**: Create or edit a `.mci` file specifying:
   - Number of photon packets to trace
   - Tissue layer properties (absorption, scattering, anisotropy, thickness)
   - Grid resolution for spatial sampling
   - Refractive indices for ambient media

2. **Run Simulation**: Execute `./mcml` and enter the input filename
3. **View Results**: Output `.mco` files contain absorption, reflectance, and transmittance data

**Key Features:**

- Multi-layer tissue modeling with arbitrary layer count
- Cylindrical coordinate system (r, z, α)
- Absorption probability, fluence, reflectance, and transmittance outputs
- Statistical accuracy through Monte Carlo photon tracking

### CONV Convolution Processing

CONV processes MCML output to simulate finite-size beam illumination:

1. **Load MCML Data**: Input `.mco` files from MCML simulations
2. **Configure Beam**: Specify Gaussian or flat circular beam profiles
3. **Generate Output**: Various formats for visualization and analysis

**Key Features:**

- Gaussian and flat beam profiles
- Configurable beam radius and total energy
- Multiple output formats: column data, contour lines, scanning profiles
- Compatible with standard plotting software

### Sample Files

The `sample/` directory provides example configurations:

- `sample1.mci` - Single-layer skin model
- `sample2.mci` - Multi-layer tissue with varying optical properties  
- `template.mci` - Base template for custom simulations

### Command Reference

Use interactive help commands within the programs by typing `h` in the main menu of MCML and CONV.

For detailed build options, run `make help` in the build directory.

## Project Structure

```text
mcml/2.1.0/
├── README.md           # This file
├── doc/                # Technical documentation and manuals
├── build/              # Build directory with makefiles
├── src/                # Source code (mcml/ and conv/ subdirectories)
├── sample/             # Sample input files (.mci)
└── bin/                # Compiled executables (created during build)
```

## Prerequisites

- C Compiler (GCC, Clang, or Microsoft Visual C++)
- Make utility
- Supported platforms: Windows, Linux, macOS, WSL

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "No rule to make target" | Ensure you're in the `build/` directory |
| "Compiler not found" | Install GCC, Clang, or try `make CC=cc` |
| "Permission denied" | Check write permissions to the `bin/` directory |

## Additional Resources

- Check `man.pdf` for comprehensive usage instructions  
- Visit sample files for configuration examples
