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

    progress=0
    for (( samp=1; samp<=$SAMPLING; samp++ ))
    do
        seed_num=$RANDOM
        python3 main.py sparse.method=random seed=$seed_num trial=$progress sparse.first=$FIRST sparse.last=$LAST sparse.term=$TERM
        progress=$(echo "$progress+1" |bc) 
        echo "Progress Status [$progress / $SAMPLING]"        
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