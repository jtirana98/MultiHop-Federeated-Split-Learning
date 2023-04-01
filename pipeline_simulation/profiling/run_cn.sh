#!/bin/bash -xe

# 1: id
# 2: num of compute nodes --> we need that to find the directory

mkdir -p /root/experiments/simulations_cn
mkdir -p /root/experiments/simulations_cn/compute_nodes_$2_
mkdir -p /root/experiments/simulations_check/compute_nodes_$2/

export LD_LIBRARY_PATH=/usr/local/lib:/usr/lib:/usr/local/lib64:/usr/lib64

declare -i port=0
port=$(( 8081 + $1 ))

sudo iptables -I INPUT -p tcp -m tcp --dport $port -j ACCEPT

../../build/compute_node $1 >/root/experiments/simulations_check/compute_nodes_$2/cn_$1_.data &
