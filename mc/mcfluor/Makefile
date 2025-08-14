# Cross-platform Makefile for MCFLUOR

PROGRAM = mcfluor
SRCS = mcfluor.c mcsubLIB.c

ifeq ($(OS),Windows_NT)
	EXEEXT = .exe
	CC = gcc
	RM = del /Q
else
	EXEEXT =
	CC = cc
	RM = rm -f
endif

TARGET = $(PROGRAM)$(EXEEXT)

all: $(TARGET)

$(TARGET): $(SRCS)
	$(CC) -o $@ $(SRCS) -lm

clean:
	$(RM) $(TARGET)

.PHONY: all clean
