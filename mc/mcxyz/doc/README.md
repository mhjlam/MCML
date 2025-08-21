# MCXYZ: Monte Carlo Photon Transport in 3D Voxelized Media

Monte Carlo simulation of photon transport through 3D voxelized tissue models with arbitrary optical properties.

## Overview

MCXYZ simulates photon propagation through complex tissue geometries using Monte Carlo techniques. This version preserves the original physics and algorithms by Steven L. Jacques and Ting Li (Oregon Health & Science University, 2010-2017) while providing modern project organization and build system.

### Original Development (2010-2017)

- **Authors**: Steven L. Jacques, Ting Li  
- **Institution**: Oregon Health & Science University
- **Legacy**: Proven Monte Carlo algorithms with extensive validation
- **Usage**: `mcxyz myname` reads `myname_H.mci` and `myname_T.bin`, outputs `myname_F.bin` and `myname_props.m`

### Current Version Features

- **Professional Project Structure**: Organized directories (`src/`, `build/`, `bin/`, `res/`, `matlab/`)
- **Cross-Platform Build System**: Modern Makefile with multiple targets  
- **Full Backward Compatibility**: Works with existing input/output files
- **MATLAB Integration**: Compatible with `lookmcxyz.m`, `maketissue.m`, and analysis scripts
- **Validated Output**: Identical results to original implementation

## Directory Structure

```
mcxyz/
├── build/       # Makefile and build artifacts
├── bin/         # Final executables
├── src/         # Source code (modular files)
├── doc/         # Documentation and original reference
├── res/         # Test data and examples
├── matlab/      # MATLAB integration scripts
└── README.md    # This file
```

## Quick Start

### Building

Navigate to the build directory and use make:

```bash
# Navigate to build directory
cd build

# Release build (optimized)
make CC=gcc

# Debug build
make debug CC=gcc

# Clean build artifacts
make clean

# Run tests
make test

# Show build info
make info
```

### Basic Usage

The original simple usage pattern is preserved:

```bash
# Run with sample data (from res directory)
cd res
../bin/mcxyz skinvessel

# From any directory (specify full path to input files)
./bin/mcxyz res/skinvessel
```

## Input Files

MCXYZ requires two input files:

1. **`<basename>_H.mci`** - Header file with simulation parameters
2. **`<basename>_T.bin`** - Binary tissue structure file

### Header File Format (`_H.mci`)

```
10.0            # Simulation time [minutes]
200             # Grid size X [voxels]
200             # Grid size Y [voxels] 
200             # Grid size Z [voxels]
0.0005          # Voxel spacing X [cm]
0.0005          # Voxel spacing Y [cm]
0.0005          # Voxel spacing Z [cm]
0               # Source type (0=uniform, 1=Gaussian, 2=isotropic, 3=rectangular)
0               # Launch flag (0=auto, 1=manual direction)
2               # Boundary flag (0=infinite, 1=escape all, 2=surface only)
0.0             # Source position X [cm]
0.0             # Source position Y [cm]
0.01            # Source position Z [cm]
0.0             # Focus X [cm]
0.0             # Focus Y [cm]
1.0e12          # Focus Z [cm]
0.0             # Direction X (if manual)
0.0             # Direction Y (if manual)  
1.0             # Direction Z (if manual)
0.03            # Radius [cm]
0.03            # Waist [cm]
9               # Number of tissue types
0.0001 1.0000 1.0000    # Tissue 1: μa μs g
0.0004 10.0000 1.0000   # Tissue 2: μa μs g
# ... additional tissue types
```

### Tissue File Format (`_T.bin`)

Binary file containing tissue type indices (1 byte per voxel) in row-major order:

- Dimensions: Nx × Ny × Nz bytes
- Values: 1-N tissue type indices (0 = background/air)
- Order: `tissue[z][x][y]` for z=0 to Nz-1, x=0 to Nx-1, y=0 to Ny-1

## Output Files

MCXYZ generates the following output files:

1. **`<basename>_F.bin`** - Fluence rate distribution [W/cm²/W delivered]
   - Binary file: Nx × Ny × Nz float values
   - Units: [W/cm²] per [W] delivered to tissue

2. **`<basename>_props.m`** - Tissue optical properties (MATLAB format)
   - Text file with μa, μs, g values for each tissue type

## Command Line Options

The original simple interface is preserved. The program expects input files with specific naming:

| Input File | Description |
|------------|-------------|
| `<name>_H.mci` | Header file with simulation parameters |
| `<name>_T.bin` | Binary tissue structure file |

| Output File | Description |
|-------------|-------------|
| `<name>_F.bin` | Fluence rate distribution [W/cm²/W delivered] |
| `<name>_props.m` | Tissue optical properties (MATLAB format) |

## Project Status & Validation

This reorganized version has been validated to produce **identical output** to the original:

### **Validation Results**

- **Properties Files**: Byte-for-byte identical between original and reorganized versions
- **Binary Format**: Correct IEEE 754 float format (32MB for 200³ voxel grids)
- **MATLAB Compatibility**: Works with existing `lookmcxyz.m`, `maketissue.m` scripts
- **Physics Preservation**: Same Monte Carlo algorithms and statistical behavior

### **Improvements Added**

- **Professional Structure**: Organized directories with proper build system
- **Cross-Platform Build**: Works on Windows, Linux, and macOS
- **Build Validation**: Comprehensive test suite and format verification
- **Documentation**: Enhanced README and inline documentation

## Modernization History

This version represents a reorganization of the original mcxyz codebase:

**Original (2010-2017)**:

- Single 1000+ line `mcxyz.c` file
- Basic compilation with `gcc mcxyz.c -lm`
- Proven Monte Carlo physics and extensive validation

**Current (2025)**:

- Professional project structure with organized directories  
- Cross-platform Makefile with build targets and testing
- Preserved original algorithms and full backward compatibility
- Enhanced maintainability and documentation

## MATLAB Integration

Results can be visualized using the provided MATLAB functions:

```matlab
% Create input files
maketissue          % Create tissue geometry and input files

% Load and display results  
lookmcxyz           % Load and visualize fluence data

% Analysis functions
reportHmci          % Display input parameters
```

## Technical Details

### Source Types

| Type | Description | Parameters |
|------|-------------|------------|
| 0 | Uniform flat-field beam | radius, waist, focus |
| 1 | Gaussian beam profile | radius, waist, focus |
| 2 | Isotropic point source | position only |
| 3 | Rectangular source | radius (half-width) |

### Boundary Conditions

| Type | Description |
|------|-------------|
| 0 | Infinite medium - photons continue beyond boundaries |
| 1 | Escape at all boundaries - photons terminate at any edge |
| 2 | Escape at surface only - photons escape at top (z=0) only |

## License & Citation

MCXYZ is distributed under the GNU General Public License v3.0.

**Original Citation:**
> Jacques, S.L. and Li, T. "Monte Carlo simulation of photon transport in 3D voxelized media." Oregon Health & Science University (2010-2017).

**Authors:**

- **Steven L. Jacques** - Original algorithm and implementation
- **Ting Li** - Original C implementation
- **Oregon Health & Science University** - Research and development

## Support

For questions about usage or technical issues:

- Check the `res/` directory for input file examples
- Review source code comments for implementation details
- See original mcxyz documentation at <https://omlc.org/software/mc/mcxyz/index.html>
