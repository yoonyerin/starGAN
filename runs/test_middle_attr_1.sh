#!/bin/bash

index=1

mode="train"
num_iters=300000

gpu_id=$index
 

#for group_idx in 0 1 2 3 4 5 

for group_idx in 1

do
    python stargan.py --mode $mode \
        --num_iters $num_iters \
        --gpu_id $gpu_id \
		--group_index $group_idx
done
