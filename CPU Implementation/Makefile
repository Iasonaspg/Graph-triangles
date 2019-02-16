######################################################################
 #
 # Makefile -- The makefile for the CPU Implementation
 #
 # Michail Iason Pavlidis <michailpg@ece.auth.gr>
 # John Flionis <iflionis@auth.gr>
 #
######################################################################

CC = gcc
#HEADER = /home/johnfli/MATLABextern/include/mat.h
CFLAGS = -O3 #-Iinclude

PROJECT_NAME = triangles

all: build # $(HEADER)
	$(CC) $(CFLAGS) $^ -o $(PROJECT_NAME)

build:
	$(CC) $(CFLAGS) *.c

clean:
	rm *.o

