#!/bin/bash

for N in {15..25}
do
    for M in {15..23}
    do
		# add for loop for the different input .mat file names
	    echo "N:$N M:$M"
	    echo "N:$N M:$M" >> batch.txt
	    ./triangles $N $M >> batch.txt
    done
done
echo All done
