
# MCML 2.0.0 - Combined Makefile for build directory (cross-platform)

all: mcml conv

mcml:
	$(MAKE) -f Makefile.mcml

conv:
	$(MAKE) -f Makefile.conv

clean:
	$(MAKE) -f Makefile.mcml clean
	$(MAKE) -f Makefile.conv clean

.PHONY: all mcml conv clean
