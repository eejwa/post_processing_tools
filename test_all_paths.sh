#!/usr/bin/env bash 

while getopts s:t: flag
do
    case "${flag}" in
    t) test_time=${OPTARG};;
    s) time_step=${OPTARG};;
    esac
done

awk 'NR>2{print}' path_summary.txt >temp.txt

while read line 
do

  rad=$(echo $line | awk '{print $1}') 
  lat=$(echo $line | awk '{print $2}')
  lon=$(echo $line | awk '{print $3}')
  lfile="_L_${rad}.0_${lat}_${lon}"

  /nfs/a300/eejwa/Anisotropy_Flow/post_processing_tools/test_pathline_memory.py -s 010 -l $lfile -ts $time_step -t $test_time > test_010.out &
  /nfs/a300/eejwa/Anisotropy_Flow/post_processing_tools/test_pathline_memory.py -s 100 -l $lfile -ts $time_step -t $test_time > test_100.out &
  /nfs/a300/eejwa/Anisotropy_Flow/post_processing_tools/test_pathline_memory.py -s 001 -l $lfile -ts $time_step -t $test_time > test_001.out &

done < temp.txt

