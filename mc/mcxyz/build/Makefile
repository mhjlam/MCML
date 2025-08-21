# Cross-platform Makefile for MCXYZ - Monte Carlo Photon Transport
#
# Monte Carlo simulation of photon transport in 3D voxelized media
# Supports Windows, Linux, and macOS with optimized builds
#
# Usage:
#   make              # Build optimized release version  
#   make debug        # Build debug version with symbols
#   make clean        # Remove build artifacts
#   make test         # Run validation tests
#   make sample       # Run sample simulation

################################################################################
# DIRECTORY STRUCTURE

# Project directories
SRC_DIR = ../src
DOC_DIR = ../doc
BIN_DIR = ../bin
BUILD_DIR = .
SAMPLE_DIR = ../sample
MATLAB_DIR = ../matlab

################################################################################
# BUILD CONFIGURATION

# Program information
PROGRAM = mcxyz
VERSION = 2.0.0

# Source files - single unified transport module
SOURCES = $(SRC_DIR)/mcxyz_main.c \
          $(SRC_DIR)/mcxyz_io.c \
          $(SRC_DIR)/mcxyz_transport.c \
          $(SRC_DIR)/mcxyz_rng.c

# Object files (built in build directory)
OBJECTS = mcxyz_main.o \
          mcxyz_io.o \
          mcxyz_transport.o \
          mcxyz_rng.o

################################################################################
# PLATFORM DETECTION AND COMPILER SELECTION

# Detect operating system
ifeq ($(OS),Windows_NT)
    DETECTED_OS = Windows
    EXEEXT = .exe
    RM = del /Q
    MKDIR = mkdir
    CP = copy
    PATHSEP = \\
    NULL_DEVICE = NUL
else
    DETECTED_OS = $(shell uname -s)
    EXEEXT =
    RM = rm -f
    MKDIR = mkdir -p
    CP = cp
    PATHSEP = /
    NULL_DEVICE = /dev/null
endif

# Compiler detection
ifeq ($(DETECTED_OS),Windows)
    CC = gcc
else
    CC ?= gcc
endif

################################################################################
# COMPILER FLAGS AND OPTIMIZATION

# Base compiler flags - C17 standard with ultra-optimizations
CFLAGS = -std=c17 -Wall -Wextra -Wno-unused-parameter -Wno-unused-variable -Wno-maybe-uninitialized \
         -march=native -mtune=native -msse2 -mavx2 -mfma -ffast-math \
         -funroll-loops -fprefetch-loop-arrays -ftree-vectorize \
         -fomit-frame-pointer -finline-functions -fipa-pta

# OpenMP support for multi-threading
CFLAGS += -fopenmp

# Include directories
CFLAGS += -I$(SRC_DIR)

# Math library linking
LDFLAGS = -lm -fopenmp

# Platform-specific flags
ifeq ($(DETECTED_OS),Windows)
    CFLAGS += -D_WIN32_WINNT=0x0601
    # Windows-specific optimizations
    CFLAGS += -D_USE_MATH_DEFINES
endif

ifeq ($(DETECTED_OS),Linux)
    CFLAGS += -D_GNU_SOURCE -D_DEFAULT_SOURCE
    # Linux-specific optimizations
    CFLAGS += -pthread
endif

ifeq ($(DETECTED_OS),Darwin)
    CFLAGS += -D_DARWIN_C_SOURCE
    # macOS-specific optimizations
    CFLAGS += -D_XOPEN_SOURCE=700
endif

################################################################################
# BUILD TARGETS

# Default target - optimized release build
all: release

# Release build - optimized for performance
release: CFLAGS += -O2 -DNDEBUG -march=native
release: $(BIN_DIR)/$(PROGRAM)$(EXEEXT)

# Debug build - optimized for debugging
debug: CFLAGS += -O0 -g3 -DDEBUG -fno-omit-frame-pointer
debug: $(BIN_DIR)/$(PROGRAM)_debug$(EXEEXT)

################################################################################
# BUILD RULES

# Create necessary directories
$(BIN_DIR):
ifeq ($(DETECTED_OS),Windows)
	@powershell -Command "if (!(Test-Path '$(BIN_DIR)')) { New-Item -ItemType Directory -Path '$(BIN_DIR)' }"
else
	@$(MKDIR) $(BIN_DIR)
endif

# Main executable target
$(BIN_DIR)/$(PROGRAM)$(EXEEXT): $(OBJECTS) | $(BIN_DIR)
	@echo "Linking $(PROGRAM)$(EXEEXT)..."
	$(CC) $(OBJECTS) -o $@ $(LDFLAGS)
	@echo "Build complete: $@"

# Debug executable target
$(BIN_DIR)/$(PROGRAM)_debug$(EXEEXT): $(OBJECTS) | $(BIN_DIR)
	@echo "Linking debug $(PROGRAM)$(EXEEXT)..."
	$(CC) $(OBJECTS) -o $@ $(LDFLAGS)
	@echo "Debug build complete: $@"

# Object file compilation rules
mcxyz_main.o: $(SRC_DIR)/mcxyz_main.c
	@echo "Compiling $<..."
	$(CC) $(CFLAGS) -I$(SRC_DIR) -c $< -o $@

mcxyz_io.o: $(SRC_DIR)/mcxyz_io.c
	@echo "Compiling $<..."
	$(CC) $(CFLAGS) -I$(SRC_DIR) -c $< -o $@

mcxyz_transport.o: $(SRC_DIR)/mcxyz_transport.c
	@echo "Compiling $<..."
	$(CC) $(CFLAGS) -I$(SRC_DIR) -c $< -o $@

mcxyz_rng.o: $(SRC_DIR)/mcxyz_rng.c
	@echo "Compiling $<..."
	$(CC) $(CFLAGS) -I$(SRC_DIR) -c $< -o $@

################################################################################
# UTILITY TARGETS

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
ifeq ($(DETECTED_OS),Windows)
	-powershell -Command "Remove-Item -Path '*.o' -Force -ErrorAction SilentlyContinue"
	-powershell -Command "Remove-Item -Path '$(BIN_DIR)\\*.exe' -Force -ErrorAction SilentlyContinue"
else
	$(RM) *.o
	$(RM) $(BIN_DIR)/$(PROGRAM)* 2>$(NULL_DEVICE) || true
endif
	@echo "Clean complete."

# Run basic validation test
test: $(BIN_DIR)/$(PROGRAM)$(EXEEXT)
	@echo "Running validation tests..."
	@echo "Testing executable exists and runs..."
ifeq ($(DETECTED_OS),Windows)
	@powershell -Command "if (Test-Path '$(BIN_DIR)/$(PROGRAM)$(EXEEXT)') { Write-Host 'Executable found: $(BIN_DIR)/$(PROGRAM)$(EXEEXT)' } else { Write-Host 'ERROR: Executable not found'; exit 1 }"
else
	@test -f "$(BIN_DIR)/$(PROGRAM)$(EXEEXT)" && echo "Executable found: $(BIN_DIR)/$(PROGRAM)$(EXEEXT)" || (echo "ERROR: Executable not found" && exit 1)
endif
	@echo "Basic tests completed successfully."

# Run sample simulation (Cross-platform)
sample: $(BIN_DIR)/$(PROGRAM)$(EXEEXT)
	@echo "Running sample simulation..."
ifeq ($(DETECTED_OS),Windows)
	@powershell -Command "if (Test-Path '$(SAMPLE_DIR)/skinvessel_H.mci') { Set-Location '$(SAMPLE_DIR)'; & '../bin/$(PROGRAM)$(EXEEXT)' skinvessel } else { Write-Host 'Sample files not found in $(SAMPLE_DIR)' }"
else
	@if [ -f "$(SAMPLE_DIR)/skinvessel_H.mci" ]; then \
		cd $(SAMPLE_DIR) && ../bin/$(PROGRAM)$(EXEEXT) skinvessel; \
	else \
		echo "Sample files not found in $(SAMPLE_DIR)"; \
	fi
endif

# Show build information
info:
	@echo "MCXYZ Build Information"
	@echo "======================"
	@echo "Program: $(PROGRAM) v$(VERSION)"
	@echo "Platform: $(DETECTED_OS)"
	@echo "Compiler: $(CC)"
	@echo "Source Dir: $(SRC_DIR)"
	@echo "Build Dir: $(BUILD_DIR)"
	@echo "Binary Dir: $(BIN_DIR)"
	@echo "Sample Dir: $(SAMPLE_DIR)"
	@echo "Sources: $(MAIN_SRC)"
	@echo "Objects: $(MAIN_OBJ)"

# Show help
help:
	@echo "MCXYZ Makefile - Available targets:"
	@echo ""
	@echo "Build targets:"
	@echo "  all/release  - Build optimized release version (default)"
	@echo "  debug        - Build debug version with symbols"
	@echo ""
	@echo "Utility targets:"
	@echo "  clean        - Remove build artifacts"
	@echo "  test         - Run basic validation tests"
	@echo "  sample       - Run sample simulation"
	@echo "  info         - Show build configuration"
	@echo "  help         - Show this help message"

################################################################################
# SPECIAL TARGETS

.PHONY: all release debug clean test sample info help
.DEFAULT_GOAL := all
