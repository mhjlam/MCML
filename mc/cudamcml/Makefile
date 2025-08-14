
# Cross-platform Makefile for CUDAMCML (CUDA Monte Carlo Multi-Layered)

PROGRAM = CUDAMCML
SRCDIR = ./src
SRCS = $(SRCDIR)/CUDAMCMLmain.cu $(SRCDIR)/CUDAMCMLio.cu $(SRCDIR)/CUDAMCMLmem.cu $(SRCDIR)/CUDAMCMLrng.cu $(SRCDIR)/CUDAMCMLtransport.cu
HEADERS = $(SRCDIR)/CUDAMCML.h

ifeq ($(OS),Windows_NT)
	EXEEXT = .exe
	RM = del /Q
	NVCC = nvcc.exe
else
	EXEEXT =
	RM = rm -f
	NVCC = nvcc
endif

CFLAGS = -arch=sm_11 -O3 -Xptxas -v
IFLAGS =
LFLAGS = -lcuda -lcudart

TARGET = $(PROGRAM)$(EXEEXT)

all: $(TARGET)

$(TARGET): $(SRCS) $(HEADERS)
	$(NVCC) $(CFLAGS) $(SRCDIR)/CUDAMCMLmain.cu -o $@ $(IFLAGS) $(LFLAGS)

clean:
	$(RM) *~ *.o *.linkinfo $(PROGRAM)$(EXEEXT)

.PHONY: all clean
