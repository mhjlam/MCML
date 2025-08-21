SHELL = cmd.exe

all:
	gcc -Wall -Wextra -std=c17 -O2 -I.. -c -o mc321.o ../mc321.c
	gcc -Wall -Wextra -std=c17 -O2 -I.. -c -o small_mc.o ../small_mc.c
	gcc -Wall -Wextra -std=c17 -O2 -I.. -c -o time_mc.o ../time_mc.c
	gcc -Wall -Wextra -std=c17 -O2 -I.. -c -o tiny_mc.o ../tiny_mc.c
	gcc -Wall -Wextra -std=c17 -O2 -I.. -c -o trmc.o ../trmc.c
	gcc -Wall -Wextra -std=c17 -O2 -I.. -c -o test_mcsub.o ../test_mcsub.c
	gcc -Wall -Wextra -std=c17 -O2 -I.. -c -o mcfluor.o ../mcfluor.c
	gcc -o ../bin/mc321.exe mc321.o -lm
	gcc -o ../bin/small_mc.exe small_mc.o -lm
	gcc -o ../bin/time_mc.exe time_mc.o -lm
	gcc -o ../bin/tiny_mc.exe tiny_mc.o -lm
	gcc -o ../bin/trmc.exe trmc.o -lm
	gcc -o ../bin/test_mcsub.exe test_mcsub.o -lm
	gcc -o ../bin/mcfluor.exe mcfluor.o -lm

clean:
	del *.o

realclean: clean
	del ..\bin\*.exe

help:
	@echo Available targets: all, clean, realclean, help
	@echo   clean     - Remove object files
	@echo   realclean - Remove object files and executables

.PHONY: all clean realclean help

.PHONY: all clean help
