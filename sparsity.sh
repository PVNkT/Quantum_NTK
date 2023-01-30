#!/bin/sh

echo "set first sparsity"
read FIRST
echo "set last sparsity"
read LAST
echo "set term"
read TERM

ITER=$(echo "$LAST-$FIRST" |bc)
ITER=$(echo "$ITER/$TERM" |bc)
echo $ITER

while [ echo `expr $ITER > 0` ]
do
    python3 main.py sparsity=$FIRST
    FIRST=$(echo "$FIRST+$TERM" |bc -l)
    ITER=$(echo "$LAST-$FIRST" |bc)
    ITER=$(echo "$ITER/$TERM" |bc)
done