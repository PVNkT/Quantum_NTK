#!/bin/sh

function sparsity_block {
    #

    echo "input sparsity : sparsity must be factor of kernel's row or column shape"
    echo "ex) If kernel shape is 128*128, sparsity must be factor of 128, 2 4 8 16 32 64" 
    read -a sparsity
    echo "choose number of sampling"
    read SAMPLING

    ITER=${#sparsity[*]}
    TOTAL_ITER=$(echo "$ITER * $SAMPLING" |bc)

    echo "Sparsity Iteration number :" $ITER
    echo "Sampling number per Iteration :" $SAMPLING
    echo "Total Iteration number :" $TOTAL_ITER

    #for multiple sampling
    progress=0
    for spars in ${sparsity[*]}
    do
        for (( samp=1; samp<=$SAMPLING; samp++ ))
        do
            progress=$(echo "$progress+1" |bc) 
            echo "Progress Status : [$progress / $TOTAL_ITER]"
            seed_num=$RANDOM
            python3 main.py sparse.method=block sparse.sparsity=$spars seed=$seed_num
        
        done
    done    
}

#original code
# FIRST=$1

# while [ "$2" -gt "$FIRST" ]
# do
#     python3 main.py sparse.method=block sparse.sparsity=$FIRST seed=7777
#     FIRST=$(echo "$FIRST*2" |bc -l)
# done

