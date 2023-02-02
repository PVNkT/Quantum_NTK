#!/bin/sh

echo "set first sparsity"
read FIRST
echo "set last sparsity"
read LAST
echo "set term"
read TERM

ITER=$(echo "$LAST-$FIRST" |bc)
ITER=$(echo "$ITER/$TERM" |bc)


while :
do
    python3 main.py sparse.sparsity=$FIRST seed=1234
    FIRST=$(echo "$FIRST+$TERM" |bc -l)
    ITER=$(echo "$LAST-$FIRST" |bc)
    ITER=$(echo "$ITER/$TERM" |bc)
    if [ $ITER -le 0 ]; then
        break
    fi
done
