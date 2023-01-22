#!/bin/bash -xe

mkdir -p /root/experiments/find_saturation/dataowners_$1_

declare -i y=0
y=$(( 5 + $1 ))

for i in $(seq 6 1 $y)
do
    ../../build/profile_data_owner $i > /root/experiments/find_saturation/dataowners_$1_/d$i.data &
done