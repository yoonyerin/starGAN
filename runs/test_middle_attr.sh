#!/bin/bash


mode="train"
num_iters=300000

gpu_id=0
 
for group_idx in 0 1 2 3 4 5 

do
    python stargan.py --mode $mode\       
        --num_iters $num_iters \
        --gpu_id $gpu_id 
done
