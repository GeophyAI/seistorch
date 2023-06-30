#!/bin/bash

while true; do
    mem_usage=$(free -m | awk 'NR==2{printf "%.2f", $3*100/$2 }')
    gpu_mem_usage=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{ sum += $1 } END { printf "%.2f", sum }')
    echo "$(date +"%T") $mem_usage $gpu_mem_usage" >> memory_usage.log
    sleep 1
done

