# Cross-platform Makefile for CUDAMC (CUDA)

PROGRAM = CUDAMC
SRCS = CUDAMC.cu CUDAMCtransport.cu

ifeq ($(OS),Windows_NT)
	EXEEXT = .exe
	RM = del /Q
	NVCC = nvcc.exe
	SHELL = cmd.exe
	.SHELLFLAGS = /c
	# Auto-detect highest CUDA architecture
	ARCH ?= $(shell nvidia-smi.exe --query-gpu=compute_cap --format=csv,noheader 2>nul | head -1 | sed "s/\.//")
	ifeq ($(ARCH),)
		ARCH := 86
	endif
else
	EXEEXT =
	RM = rm -f
	NVCC = nvcc
	# Auto-detect highest CUDA architecture
	ARCH ?= $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | sed "s/\.//")
	ifeq ($(ARCH),)
		ARCH := 75
	endif
endif

# CUDA architecture (auto-detected from GPU, overridable with make ARCH=XX)
CFLAGS = -arch=sm_$(ARCH) -O3 -Xptxas -v
LFLAGS = -lcuda -lcudart

TARGET = $(PROGRAM)$(EXEEXT)

all: $(TARGET)

ifeq ($(OS),Windows_NT)
$(TARGET): $(SRCS)
	@echo Building CUDAMC with CUDA architecture sm_$(ARCH)
	@if exist "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" ( \
		call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64 >nul && $(NVCC) $(CFLAGS) CUDAMC.cu -o $@ $(LFLAGS) \
	) else if exist "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvarsall.bat" ( \
		call "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvarsall.bat" x64 >nul && $(NVCC) $(CFLAGS) CUDAMC.cu -o $@ $(LFLAGS) \
	) else if exist "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvarsall.bat" ( \
		call "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvarsall.bat" x64 >nul && $(NVCC) $(CFLAGS) CUDAMC.cu -o $@ $(LFLAGS) \
	) else ( \
		echo Visual Studio not found, trying without vcvarsall... && $(NVCC) $(CFLAGS) CUDAMC.cu -o $@ $(LFLAGS) \
	)
else
$(TARGET): $(SRCS)
	$(NVCC) $(CFLAGS) CUDAMC.cu -o $@ $(LFLAGS)
endif

clean:
	$(RM) *~ *.o *.linkinfo $(PROGRAM)$(EXEEXT)

.PHONY: all clean
