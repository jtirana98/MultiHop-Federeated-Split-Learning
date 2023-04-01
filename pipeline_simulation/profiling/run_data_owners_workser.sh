#!/bin/bash -xe

# 1: num of data owners
# 2: num of compute nodes

export LD_LIBRARY_PATH=/usr/local/lib:/usr/lib:/usr/local/lib64:/usr/lib64

mkdir -p /root/experiments/simulations_check/compute_nodes_$2
mkdir -p /root/experiments/simulations_check/compute_nodes_$2/dataowners_$1_

declare -i start=0
declare -i end=0
declare -i port=0
declare -i port_start=0
start=$((  18 ))
end=$(( $1 + $start -1))

for i in $(seq $start 1 $end)
do
    port=$(( 8081 + $port_start ))
    sudo iptables -I INPUT -p tcp -m tcp --dport $port -j ACCEPT
    ../../build/simulated_data_owner $i > "/root/experiments/simulations_check/compute_nodes_$2/dataowners_$1_/d$i.data" &
    port_start=$(( $port_start + 1 ))
done