# CONV 3.0 - Modern C++20 Convolution Program for MCML

## Overview

CONV 3.0 is a complete modernization of the original CONV program used for processing Monte Carlo Multi-Layer (MCML) simulation output data. This version has been completely rewritten in modern C++20, following the same architectural patterns and code organization as MCML 3.0.

## Key Features

### Modern C++20 Design
- **Object-oriented architecture** with clear separation of concerns
- **RAII and smart pointers** for automatic memory management
- **Strongly typed enums** and modern containers
- **Exception-safe design** with custom exception hierarchy
- **Header-only templates** for generic algorithms
- **Concepts** for type constraints and better error messages

### Core Functionality
- **Beam Profile Processing**: Support for original, flat, Gaussian, and arbitrary beam profiles
- **Advanced Convolution Engine**: High-performance convolution with caching and adaptive integration
- **Data Extraction**: Comprehensive extraction of reflectance, transmittance, and absorption data
- **Interactive and Batch Modes**: Full command-line interface with interactive menu system

### Architecture Components

#### Core Classes
- `ConvProcessor`: Main processing engine and user interface
- `McmlDataReader`: Reads and validates MCML output (.mco) files
- `BeamProfile`: Manages photon beam profiles for convolution
- `ConvolutionEngine`: Performs high-performance convolution computations
- `BinaryTree<T>`: Template-based caching system for intermediate results

#### Data Structures
- `Matrix2D<T>` and `Matrix3D<T>`: Modern matrix containers with bounds checking
- `ExtractableQuantities`: Bitset-based quantity selection system
- `ConvolutionConfig`: Configuration management with validation
- `ProcessingStats`: Comprehensive performance monitoring

## Building

### Prerequisites
- C++20 compatible compiler (GCC 10+, Clang 12+, MSVC 2019+)
- CMake 3.20 or higher

### Build Instructions

```bash
cd 3.0.0/build
cmake -S . -B output
cmake --build output --config Release
```

The executables will be built in `3.0.0/bin/`:
- `mcml.exe` - Monte Carlo simulation engine
- `conv.exe` - Convolution processing tool

## Usage

### Interactive Mode
```bash
conv
```
Launches the interactive menu system for step-by-step processing.

### Batch Mode
```bash
conv input.mco                          # Process with default settings
conv -v --beam-type flat --beam-radius 0.1 input.mco  # Verbose with flat beam
conv --help                             # Show all options
```

### Command-Line Options
- `-h, --help`: Show help message
- `-a, --about`: Show program information
- `-v, --verbose`: Enable verbose output
- `-i, --interactive`: Force interactive mode
- `-b, --batch`: Force batch mode
- `-o, --output FILE`: Specify output filename
- `-f, --force`: Overwrite existing output files
- `--beam-type TYPE`: Beam type (original, flat, gaussian, arbitrary)
- `--beam-radius R`: Beam radius in cm
- `--beam-power P`: Beam power in J
- `--epsilon E`: Convolution relative error (default: 0.1)

## File Formats

### Input Files
- **MCML Output (.mco)**: Binary or ASCII output from MCML simulations
- **Beam Profile Files**: ASCII files defining arbitrary beam intensity profiles

### Output Files
- **Processed Data**: Convolved results in ASCII format
- **Statistics**: Performance and accuracy reports

## Technical Details

### Convolution Algorithm
The convolution engine implements advanced numerical integration techniques:
- **Adaptive Simpson's Rule** for high accuracy
- **Result Caching** using binary search trees
- **Bessel Function Optimization** with cached computations
- **Error Estimation** and convergence monitoring

### Memory Management
- **Zero-copy operations** where possible
- **Smart pointer usage** throughout the codebase
- **RAII pattern** for automatic resource cleanup
- **Memory pool allocation** for large data sets

### Performance Features
- **Multi-threaded processing** (planned for future versions)
- **SIMD optimizations** where applicable
- **Cache-friendly data layouts**
- **Minimal memory allocations** in hot paths

## Code Organization

The codebase follows modern C++ best practices:

```
src/conv/
├── conv.hpp              # Main header with types and constants
├── main.cpp              # Program entry point
├── processor.hpp/.cpp    # Main processing engine
├── mcml_reader.hpp       # MCML file reading
├── beam_profile.hpp/.cpp # Beam profile management
├── convolution_engine.hpp # Convolution computations
├── matrix.hpp            # Matrix container templates
├── binary_tree.hpp       # Caching system
└── stubs.cpp             # Implementation stubs
```

## Version History

### Version 3.0.0 (2025)
- Complete rewrite in modern C++20
- Object-oriented design following MCML 3.0 patterns
- Enhanced numerical algorithms
- Comprehensive error handling
- Modern build system with CMake

### Version 2.1.0 (1996)
- Original C implementation
- Basic convolution functionality
- Command-line interface

## License

Copyright (c) 1992-1996 Univ. of Texas M.D. Anderson Cancer Center  
Copyright (c) 2025 M.H.J. Lam

This software continues the tradition of the original MCML/CONV programs while bringing modern C++ design and performance optimizations.

## Contributors

**Original Authors:**
- Lihong Wang, Ph.D. - Texas A&M University
- Steven L. Jacques, Ph.D. - Oregon Medical Laser Center
- Liqiong Zheng, B.S. - University of Houston

**C++20 Modernization:**
- M.H.J. Lam, MSc. - Utrecht University

## References

1. Jacques, S.L., et al. "Monte Carlo modeling of light transport in tissues." SPIE Institute Series (1989)
2. Wang, L., Jacques, S.L. "Monte Carlo Multi-Layered (MCML) User Manual" (1996)
3. Original program available at: omlc.org/software/mc
