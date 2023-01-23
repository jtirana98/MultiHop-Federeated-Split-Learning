#!/bin/bash -xe

mkdir -p /root/experiments/simulations/dataowners_$1_

declare -i y=0
y=$(( 5 + $1 ))

for i in $(seq 1 1 5)
do
    ../../build/simulated_data_owner $i > /root/experiments/simulations/dataowners_$1_/d$i.data &
done

../../build/simulated_data_owner 0 > /root/experiments/simulations/dataowners_$1_/d0.data &