#!/usr/bin/env bash

cij_total_file='poly_cij.out'

out_cij='avg_cij'

cwd=$(pwd)

for dir in _L_*_001_*
do
    echo $dir

    cd $dir


    for poly in poly_cij*
    do
        time=$(echo $poly | awk -F "_" '{print $3}' | awk -F "." '{print $1}')
        out_cij="avg_cij_$time"
        if [ -f "$poly" ]; 
        then
            
            awk 'NR>=11&&NR<=16{print}' $poly | sed 's/[][]//g' > $out_cij

        fi

    done
    cd $cwd

done








