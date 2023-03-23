#!/bin/bash -xe

mkdir -p /Users/joanatirana/Desktop/SplitPipe/experiments/find_saturation/dataowners_$1_

declare -i y=0
y=$(( 3 + $1 ))

for i in $(seq 4 1 $y)
do
    ../../build/profile_data_owner -i $i -d $i > /Users/joanatirana/Desktop/SplitPipe/experiments/find_saturation/dataowners_$1_/d$i.data &
done