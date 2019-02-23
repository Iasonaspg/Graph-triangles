######################################################################
 #
 # Makefile -- The makefile for our CUDA app
 #
 # Inspiration from Patrick:
 # https://github.com/hellopatrick/cuda-samples/
 # blob/master/game_of_life_graphics
 #
 # Thanks to Frank Blanning <frankgou@auth.gr>
 #
 # Michail Iason Pavlidis <michailpg@ece.auth.gr>
 # John Flionis <iflionis@auth.gr>
 #
######################################################################

PROJECT_NAME = triangles

# NVCC is path to nvcc. Here it is assumed /usr/local/cuda is on one's PATH.
# CC is the compiler for C host code.

NVCC = nvcc
CC = gcc

CUDAPATH = /usr/local/cuda

CFLAGS = -c -I $(CUDAPATH)/include
NVCCFLAGS = -c -I $(CUDAPATH)/include -Wno-deprecated-gpu-targets

LFLAGS = -L$(CUDAPATH)/lib -lcuda -lcusparse -Wno-deprecated-gpu-targets -ftz=true #here of above?

all: build

build: gpu #cpu
	$(NVCC) $(LFLAGS) -o $(PROJECT_NAME) *.o

gpu:
	$(NVCC) $(NVCCFLAGS) *.cu

# Try and fix this later, so that .c files don't need to be named and compiled as .cu
#cpu:			
#	$(CC) $(CFLAGS) *.c

clean:
	rm *.o

#run:
#	./$(PROJECT_NAME) -v 5 5 2

