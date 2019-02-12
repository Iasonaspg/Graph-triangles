######################################################################
 #
 # Makefile -- The makefile for the CPU Implementation
 #
 # Michail Iason Pavlidis <michailpg@ece.auth.gr>
 # John Flionis <iflionis@auth.gr>
 #
######################################################################

CC = gcc
CFLAGS = -O3

PROJECT_NAME = triangles

all: build
	$(CC) $(CFLAGS) $^ -o $(PROJECT_NAME)

build:
	$(CC) $(CFLAGS) *.c

clean:
	rm *.o

