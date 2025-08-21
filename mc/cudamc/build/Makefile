# Cross-platform Makefile for CUDAMC (CUDA)

PROGRAM = cudamc
SRCDIR = ../src
BINDIR = ../bin
SRCS = $(SRCDIR)/cudamc.cu $(SRCDIR)/cudamc_transport.cu

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

# CUDA architecture (auto-detected from GPU, overridable with make ARCH=XX)
CFLAGS = -arch=sm_$(ARCH) -O3 -Xptxas -v
LFLAGS = -lcuda -lcudart

TARGET = $(BINDIR)/$(PROGRAM)$(EXEEXT)

all: $(TARGET)

$(BINDIR):
	$(MKDIR)

ifeq ($(OS),Windows_NT)
$(TARGET): $(SRCS) | $(BINDIR)
	@echo Building CUDAMC with CUDA architecture sm_$(ARCH)
	@if exist "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" ( \
		call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64 >nul && $(NVCC) $(CFLAGS) $(SRCDIR)/cudamc.cu -o $@ $(LFLAGS) \
	) else if exist "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvarsall.bat" ( \
		call "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvarsall.bat" x64 >nul && $(NVCC) $(CFLAGS) $(SRCDIR)/cudamc.cu -o $@ $(LFLAGS) \
	) else if exist "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvarsall.bat" ( \
		call "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvarsall.bat" x64 >nul && $(NVCC) $(CFLAGS) $(SRCDIR)/cudamc.cu -o $@ $(LFLAGS) \
	) else ( \
		echo Visual Studio not found, trying without vcvarsall... && $(NVCC) $(CFLAGS) $(SRCDIR)/cudamc.cu -o $@ $(LFLAGS) \
	)
else
$(TARGET): $(SRCS) | $(BINDIR)
	$(NVCC) $(CFLAGS) $(SRCDIR)/cudamc.cu -o $@ $(LFLAGS)
endif

clean:
ifeq ($(OS),Windows_NT)
	@if exist "$(BINDIR)\$(PROGRAM)$(EXEEXT)" del /Q "$(BINDIR)\$(PROGRAM)$(EXEEXT)"
	@if exist "*~" del /Q "*~"
	@if exist "*.o" del /Q "*.o" 
	@if exist "*.linkinfo" del /Q "*.linkinfo"
	@if exist "outp.txt" del /Q "outp.txt"
	@if exist "outph.txt" del /Q "outph.txt"
else
	$(RM) $(BINDIR)/$(PROGRAM)$(EXEEXT) *~ *.o *.linkinfo outp.txt outph.txt
endif

realclean: clean
ifeq ($(OS),Windows_NT)
	@if exist "$(BINDIR)" $(RMDIR) "$(BINDIR)" 2>nul || echo Directory already removed
	@if exist "../outp.txt" del /Q "../outp.txt"
	@if exist "../outph.txt" del /Q "../outph.txt"
else
	$(RMDIR) $(BINDIR) 2>/dev/null || true
	$(RM) ../outp.txt ../outph.txt 2>/dev/null || true
endif

.PHONY: all clean realclean
