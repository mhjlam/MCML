# Makefile for all standalone programs in small_mc (cross-platform)

SOURCES := mc321.c small_mc.c time_mc.c tiny_mc.c trmc.c test_mcsub.c mcfluor.c
EXES := $(SOURCES:.c=)

ifeq ($(OS),Windows_NT)
	EXEEXT = .exe
	CC = gcc
	RM = cmd /c del /Q
else
	EXEEXT =
	CC = cc
	RM = rm -f
endif


all: $(EXES)

%: %.c
	$(CC) -o $@ $(CFLAGS) $< -lm

clean:
	$(RM) $(addsuffix $(EXEEXT),$(EXES))

.PHONY: all clean
