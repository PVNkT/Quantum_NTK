#!/bin/sh

FIRST=$1

while [ "$2" -gt "$FIRST" ]
do
    python3 main.py sparse.method=block sparse.sparsity=$FIRST seed=7777
    FIRST=$(echo "$FIRST*2" |bc -l)
done
