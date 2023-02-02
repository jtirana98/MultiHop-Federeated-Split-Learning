#!/bin/bash -xe

# 1: num of data owners
# 2: configuration
# 3: num of compute nodes --> we need that to find the directory

export LD_LIBRARY_PATH=/usr/local/lib:/usr/lib:/usr/local/lib64:/usr/lib64
mkdir -p /root/experiments/aggregator

declare -i port=0
port=$(( 8081 + $2 ))

sudo iptables -I INPUT -p tcp -m tcp --dport $port -j ACCEPT

../../build/aggregator $1 $2 > /root/experiments/aggregator/out_a_c$3_d$1.data &
