#!/bin/sh


echo "choose mode"
read MODE
echo "set first sparsity"
read FIRST
echo "set last sparsity"
read LAST
echo "set term"
read TERM
echo "choose number of sampling"
read SAMPLING
echo "set options"
read -a options

progress=0
for (( samp=1; samp<=$SAMPLING; samp++ ))
do
    seed_num=$RANDOM
    python3 main.py sparse.method=$MODE seed=$seed_num trial=$progress sparse.first=$FIRST sparse.last=$LAST sparse.term=$TERM $options
    progress=$(echo "$progress+1" |bc) 
    echo "Progress Status [$progress / $SAMPLING]"        
done      
        
        
