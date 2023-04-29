#!/bin/bash

pmap -x $1 > out_mem.txt
sleep  1s

for i in  $(seq 0 1 1000)
do
	pmap -x $1 >> out_mem.txt
	sleep 1s
done
