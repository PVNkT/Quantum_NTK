#!/bin/sh

function sparsity_random {

    echo "set first sparsity"
    read FIRST
    echo "set last sparsity"
    read LAST
    echo "set term"
    read TERM
    echo "choose number of sampling"
    read SAMPLING


    ITER=$(echo "$LAST-$FIRST" |bc)
    ITER=$(echo "$ITER/$TERM" |bc)
    TOTAL_ITER=$(echo "$ITER * $SAMPLING" |bc)
    # ITER=$(($LAST - $FIRST)) 
    # ITER=$(($ITER/$TERM))

    echo "Sparsity Iteration number :" $ITER
    echo "Sampling number per Iteration :" $SAMPLING
    echo "Total Iteration number :" $TOTAL_ITER

    #for multiple sampling
    progress=0
    while :
    do
        for (( samp=1; samp<=$SAMPLING; samp++ ))
        do
            progress=$(echo "$progress+1" |bc) 
            echo "Progress Status : [$progress / $TOTAL_ITER]"
            seed_num=$RANDOM
            python3 main.py sparse.method=random sparse.sparsity=$FIRST seed=$seed_num
        
        done
        
        FIRST=$(echo "$FIRST+$TERM" |bc -l)
        ITER=$(echo "$LAST-$FIRST" |bc)
        ITER=$(echo "$ITER/$TERM" |bc)
        if [ $ITER -le 0 ]; then
            break
        fi
    done    
}
# #for single sampling
# while :
# do
#     python3 main.py sparse.method=random sparse.sparsity=$FIRST seed=1234
#     FIRST=$(echo "$FIRST+$TERM" |bc -l)
#     ITER=$(echo "$LAST-$FIRST" |bc)
#     ITER=$(echo "$ITER/$TERM" |bc)
#     if [ $ITER -le 0 ]; then
#         break
#     fi
# done