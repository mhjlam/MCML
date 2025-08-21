# MCML

Monte Carlo simulation of light transport in multi-layered turbid media.

## Version History

This project contains multiple versions of MCML (Monte Carlo Multi-Layered) and CONV (convolution program).

The MCML and CONV programs have a complex version history spanning over 30 years. Due to poor version management practices, the lineage is fragmented across multiple repositories and maintainers. I have tried to reconstruct and accumulate the existing software as best to my knowledge, and made new contributions.

### 1.0 - 1.1

- **Date**: 1991-1993
- **Authors**: Likely Lihong Wang and Steven L. Jacques
- **Source**: Unknown/unavailable

### 1.2

- **Date**: 1993
- **Authors**: Lihong Wang and Steven L. Jacques
- **Source**: <https://omlc.org/software/mc/mcml>
- **Changes**: Conform to ANSI Standard C and add angularly resolved reflectance and transmittance

### 1.2.1

- **Date**: 1993-1996
- **Authors**: Lihong Wang and Steven L. Jacques
- **Source**: Unknown/unavailable
- **Changes**: Fix memory issues memory corruption bug

### 1.2.2

- **Date**: 1996-2000
- **Author**: Lihong Wang
- **Source**: <https://github.com/lhvwang/MCML>
- **Changes**: Fix bugs introduced in 1.2.1

### 2.0

- **Date**: 1996
- **Author**: Lihong Wang and Liqiong Zheng
- **Source**: <https://github.com/lhvwang/MCML>
- **Changes**: Add interactive parameter input, multiple scoring, time-resolved simulation, adjustable/isotropic sources, simulation time control, standard error computation, continuation runs

### 2.0(.1)

Updated 2.0 version released recently by Scott Prahl, but version was not bumped.

- **Date**: 2024
- **Author**: Scott Prahl
- **Source**: <https://github.com/scottprahl/MCML>
- **Changes**: Add Makefiles for modern systems, fix build issues, improve code readability

### 2.1.0

- **Date**: 2025
- **Author**: Maurits Lam
- **Source**: <https://github.com/mhjlam/MCML>
- **Changes**: Update to C17, modernize random number generator, improve memory management, update Makefiles for Windows systems

### 3.0.0

- **Date**: 2025
- **Author**: Maurits Lam
- **Source**: <https://github.com/mhjlam/MCML>
- **Changes**: Complete rewrite in C++20 with a modern C++ design, object-oriented architecture, improved performance and improved user interactivity and experience

### Recommendations

- **Avoid version 1.2 from the MCML website!**
- For general use, use version 2.0(.1) by Scott Prahl, which is stable, well-tested, and has a modern build system.
- For legacy compatibility you can use version 1.2.2 (last stable 1.x release).
- For modern systems, use new version 2.1.0 written in C17 with more readable source code, and better performance, but less tested.

## Building and Testing

### Prerequisites

#### A modern C compiler

- **GCC**: <https://gcc.gnu.org> or <https://www.mingw-w64.org>
- **Clang**: <https://llvm.org/>
- **MSVC** (cl.exe): <https://visualstudio.microsoft.com/visual-cpp-build-tools>

#### Tooling

- **GNU Make**: <https://www.gnu.org/software/make>

### Build Commands

All versions include cross-platform Makefiles with automated testing:

```bash
# Build specific components
cd <version>/build
make                            # Build MCML and CONV
make -f Makefile.mcml           # Build MCML only
make -f Makefile.conv           # Build CONV only

# Test components
make test                       # Test MCML, then CONV
make -f Makefile.mcml test      # Test MCML with sample simulation
make -f Makefile.conv test      # Test CONV with sample data

# Clean build artifacts
make clean                      # Remove all object files and outputs
make -f Makefile.mcml clean     # Remove MCML files and outputs
make -f Makefile.conv clean     # Remove CONV files and outputs
```

## Additional Programs

This repository also includes several standalone Monte Carlo projects that are referenced on the MCML website at <https://omlc.org/software/mc/>. These have been updated to work on modern systems and are provided with Makefiles as well.

### Simple Monte Carlo

- `tiny_mc` - Simulates light propagation from a point source in an infinite medium with isotropic scattering (by Scott Prahl).
- `small_mc` - Simulates light propagation from normal irradiation of a semi-infinite medium with anisotropic scattering. It calculates the volumetric heating as a function of depth. (by Scott Prahl).
- `time_mc` - Simulates the time resolved backscattering of a semi-infinite medium with anisotropic scattering (by Scott Prahl).
- `trmc` - Simple time-resolved Monte Carlo (by Steven L. Jacques, 1998).
- `mcfluor` - Simulation of fluorescence in scattering medium (such as biological tissues). Models both excitation and fluorescence emission processes with integration to MATLAB for analysis and visualization (by Steven L. Jacques, 2007).
- `mcsub` - A small Monte Carlo subroutine library that can be used in other programs, and has MATLAB integration (by Steven L. Jacques, 2010).
- `mc321` - Simple steady-state Monte Carlo in spherical, cylindrical and planar coordinates (by Steven L. Jacques, 2017).

### **mcxyz** - 3D Monte Carlo

Simulates light transport in 3D tissue structures using a cube of voxels, where a voxel can represent one of several different tissue types. It creates detailed maps showing how light spreads through the tissue. Useful for medical applications like therapy planning and optical imaging. Includes MATLAB tools for setting up simulations and supports various light sources.

- **Date**: 2019
- **Authors**: Steven Jacques, Ting Li, Scott Prahl
- **Website**: <https://omlc.org/software/mc/mcxyz/index.html>

### CUDA-based Monte Carlo

#### **CUDAMC** - CUDA Monte Carlo

CUDAMC specializes in time-resolved photon transport simulations in homogeneous media using GPU acceleration. It focuses on temporal analysis of photon migration patterns, making it valuable for time-domain spectroscopy and diffuse optical tomography where timing information helps distinguish tissue types and detect abnormalities in medical diagnostics.

- **Date**: 2007
- **Authors**: Erik Alerstam, Tomas Svensson, Stefan Andersson-Engels
- **Website**: <https://www.atomic.physics.lu.se/biophotonics/research/monte-carlo-simulations/gpu-monte-carlo>

#### **CUDAMCML** - CUDA Monte Carlo Multi-Layered

CUDAMCML accelerates multi-layered tissue simulations using GPU parallel processing while maintaining compatibility with the original MCML format. It leverages thousands of GPU cores to simulate photon trajectories simultaneously, making it ideal for medical imaging, optical diagnostics, and biomedical research requiring rapid analysis of light propagation through complex layered tissue structures.

- **Date**: 2009
- **Authors**: Erik Alerstam, Tomas Svensson, Stefan Andersson-Engels
- **Website**: <https://www.atomic.physics.lu.se/biophotonics/research/monte-carlo-simulations/gpu-monte-carlo>

#### Update

Both CUDAMC and CUDAMCML have been comprehensively modernized and refactored to improve code quality, maintainability, and documentation while preserving their original performance characteristics and simulation accuracy. The modernization effort focused on applying modern C++ practices, enhancing code organization, and providing extensive documentation of the underlying physics and algorithms.

**Key improvements include:**

- **Comprehensive documentation**: Every function now includes detailed headers explaining the physics, mathematics, and implementation approach
- **Modern C++ practices**: Enhanced type safety, const correctness, RAII memory management, and improved error handling
- **Modular architecture**: Clear separation of concerns with dedicated modules for I/O operations, memory management, random number generation, and physics transport
- **Enhanced physics documentation**: Complete mathematical derivations for Fresnel reflection, Henyey-Greenberg scattering, absorption calculations, and other optical phenomena
- **Improved code organization**: Logical grouping of functionality with consistent naming conventions and coding standards
- **Better error handling**: Detailed error reporting with specific failure contexts and validation mechanisms
- **Performance optimization**: Maintained original GPU performance while improving memory access patterns and atomic operations

The modernized CUDA implementations maintain backward compatibility with existing input formats while providing significantly improved code readability and maintainability for future development and scientific research applications.
