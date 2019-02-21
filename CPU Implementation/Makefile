######################################################################
 #
 # Makefile -- The makefile for the CPU Implementation
 #
 # Michail Iason Pavlidis <michailpg@ece.auth.gr>
 # John Flionis <iflionis@auth.gr>
 #
######################################################################

GCC = gcc

CFLAGS = -O3

PROJECT_NAME = trianglesCPU

all: trianglesCPU.o validation.o readCSV.o
	$(GCC) $(CFLAGS) $^ -o $(PROJECT_NAME)

%.o: %.c
	$(GCC) -c $(CFLAGS) $^

clean:
	rm -f *.o *~ $(PROJECT_NAME)
