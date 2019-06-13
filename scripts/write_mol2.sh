#!/bin/bash

i=1
for file in ~/champs-scalar-coupling/structures/*; do
	outfile=$(basename $file .xyz)
	obabel -i xyz -o mol2 $file -O ~/champs-scalar-coupling/mol2_structures/$outfile.mol2
	echo "Files converted: $i"
	((i++))
done;
