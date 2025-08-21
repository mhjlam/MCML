# Cross-platform Makefile for CUDAMCML (CUDA Monte Carlo Multi-Layered)

PROGRAM = cudamcml
SRCDIR = ../src
BINDIR = ../bin
SRCS = $(SRCDIR)/cudamcml.cu $(SRCDIR)/cudamcml_io.cu $(SRCDIR)/cudamcml_mem.cu $(SRCDIR)/cudamcml_rng.cu $(SRCDIR)/cudamcml_transport.cu
HEADERS = $(SRCDIR)/cudamcml.h

ifeq ($(OS),Windows_NT)
	EXEEXT = .exe
	RM = del /Q
	RMDIR = rmdir /S /Q
	NVCC = nvcc.exe
	SHELL = cmd.exe
	.SHELLFLAGS = /c
	MKDIR = if not exist "$(BINDIR)" mkdir "$(BINDIR)"
	# Auto-detect highest CUDA architecture
	ARCH ?= $(shell nvidia-smi.exe --query-gpu=compute_cap --format=csv,noheader 2>nul | head -1 | sed "s/\.//")
	ifeq ($(ARCH),)
		ARCH := 86
	endif
else
	EXEEXT =
	RM = rm -f
	RMDIR = rm -rf
	NVCC = nvcc
	MKDIR = mkdir -p $(BINDIR)
	# Auto-detect highest CUDA architecture
	ARCH ?= $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | sed "s/\.//")
	ifeq ($(ARCH),)
		ARCH := 75
	endif
endif

CFLAGS = -arch=sm_$(ARCH) -O3 -Xptxas -v
IFLAGS =
LFLAGS = -lcuda -lcudart

TARGET = $(BINDIR)/$(PROGRAM)$(EXEEXT)

all: $(TARGET)

$(BINDIR):
	$(MKDIR)

ifeq ($(OS),Windows_NT)
$(TARGET): $(SRCS) $(HEADERS) | $(BINDIR)
	@echo Building CUDAMCML with CUDA architecture sm_$(ARCH)
	@if exist "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" ( \
		call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64 >nul && $(NVCC) $(CFLAGS) $(SRCDIR)/cudamcml.cu -o $@ $(IFLAGS) $(LFLAGS) \
	) else if exist "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvarsall.bat" ( \
		call "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvarsall.bat" x64 >nul && $(NVCC) $(CFLAGS) $(SRCDIR)/cudamcml.cu -o $@ $(IFLAGS) $(LFLAGS) \
	) else if exist "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvarsall.bat" ( \
		call "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary Build\vcvarsall.bat" x64 >nul && $(NVCC) $(CFLAGS) $(SRCDIR)/cudamcml.cu -o $@ $(IFLAGS) $(LFLAGS) \
	) else ( \
		echo Visual Studio not found, trying without vcvarsall... && $(NVCC) $(CFLAGS) $(SRCDIR)/cudamcml.cu -o $@ $(IFLAGS) $(LFLAGS) \
	)
	@echo Copying sample files to bin directory...
	@if exist "../sample" ( \
		if not exist "$(BINDIR)\sample" mkdir "$(BINDIR)\sample" && \
		xcopy /E /I /Y "..\sample\*" "$(BINDIR)\sample\" >nul \
	) else ( \
		echo No sample directory found to copy \
	)
else
$(TARGET): $(SRCS) $(HEADERS) | $(BINDIR)
	$(NVCC) $(CFLAGS) $(SRCDIR)/cudamcml.cu -o $@ $(IFLAGS) $(LFLAGS)
	@echo Copying sample files to bin directory...
	@if [ -d "../sample" ]; then \
		mkdir -p $(BINDIR)/sample && \
		cp -r ../sample/* $(BINDIR)/sample/ ; \
	else \
		echo "No sample directory found to copy" ; \
	fi
endif

clean:
ifeq ($(OS),Windows_NT)
	@if exist "$(BINDIR)\$(PROGRAM)$(EXEEXT)" del /Q "$(BINDIR)\$(PROGRAM)$(EXEEXT)"
	@if exist "$(BINDIR)\sample" $(RMDIR) "$(BINDIR)\sample" 2>nul || echo Sample directory already removed
	@if exist "*~" del /Q "*~"
	@if exist "*.o" del /Q "*.o" 
	@if exist "*.linkinfo" del /Q "*.linkinfo"
else
	$(RM) $(BINDIR)/$(PROGRAM)$(EXEEXT) *~ *.o *.linkinfo
	$(RMDIR) $(BINDIR)/sample 2>/dev/null || true
endif

realclean: clean
ifeq ($(OS),Windows_NT)
	@if exist "$(BINDIR)" $(RMDIR) "$(BINDIR)" 2>nul || echo Directory already removed
	@if exist "../sample.mco" del /Q "../sample.mco"
else
	$(RMDIR) $(BINDIR) 2>/dev/null || true
	$(RM) ../sample.mco 2>/dev/null || true
endif

.PHONY: all clean realclean
