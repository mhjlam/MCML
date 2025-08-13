# MCML 2.1.0 Project - Master Makefile
# Unified Makefile for building MCML, CONV, or both

# Project info
VERSION = 2.1.0
BINDIR = ../bin

# Platform detection and cross-platform commands
ifeq ($(OS),Windows_NT)
	SHELL = cmd.exe
	EXEEXT = .exe
	MKDIR_P = if not exist "$(BINDIR)" mkdir "$(BINDIR)"
	RM = del /Q
	DEVNULL = NUL
	MAKE_CMD = $(MAKE)
else
	# Unix-like environment (Linux, macOS, WSL when not Windows_NT)
	EXEEXT =
	MKDIR_P = mkdir -p $(BINDIR)
	RM = rm -f
	DEVNULL = /dev/null
	MAKE_CMD = $(MAKE)
endif

# Default target - build both applications
all: mcml conv
	@echo Successfully built MCML and CONV $(VERSION)

# Build only MCML
mcml: $(BINDIR)
	@echo Building MCML $(VERSION)...
	$(MAKE_CMD) -f Makefile.mcml all

# Build only CONV
conv: $(BINDIR)
	@echo Building CONV $(VERSION)...
	$(MAKE_CMD) -f Makefile.conv all

# Test both applications
test: test-mcml test-conv

# Test MCML
test-mcml:
	@echo Testing MCML...
	$(MAKE_CMD) -f Makefile.mcml test

# Test CONV
test-conv:
	@echo Testing CONV...
	$(MAKE_CMD) -f Makefile.conv test

# Create bin directory if it doesn't exist
$(BINDIR):
	$(MKDIR_P)

# Clean object files for both projects
clean:
	@echo Cleaning MCML...
	$(MAKE_CMD) -f Makefile.mcml clean
	@echo Cleaning CONV...
	$(MAKE_CMD) -f Makefile.conv clean

# Clean everything including executables
realclean:
	@echo Real cleaning MCML...
	$(MAKE_CMD) -f Makefile.mcml realclean
	@echo Real cleaning CONV...
	$(MAKE_CMD) -f Makefile.conv realclean

# Install both applications (copy to system path or specified location)
install: all
	@echo Installing MCML and CONV to $(BINDIR)...
	@echo Installation complete. Add $(BINDIR) to your PATH if desired.

# Help target
help:
	@echo MCML Project $(VERSION) Master Build System
	@echo Available targets:
	@echo   all         - Build both MCML and CONV (default)
	@echo   mcml        - Build only MCML
	@echo   conv        - Build only CONV
	@echo   test        - Test both applications
	@echo   test-mcml   - Test only MCML
	@echo   test-conv   - Test only CONV
	@echo   clean       - Clean object files for both projects
	@echo   realclean   - Remove all generated files for both projects
	@echo   install     - Install both applications
	@echo   help        - Show this help message
	@echo
	@echo Individual project makefiles:
	@echo   make -f Makefile.mcml [target]  - Build MCML specifically
	@echo   make -f Makefile.conv [target]  - Build CONV specifically
	@echo
	@echo Compiler selection (pass to individual makefiles):
	@echo   CC=gcc      - Use GCC compiler (default on Windows and Unix)
	@echo   CC=clang    - Use Clang compiler
	@echo   CC=cl       - Use Microsoft Visual C++ compiler (Windows)

# Phony targets
.PHONY: all mcml conv test test-mcml test-conv clean realclean install help
