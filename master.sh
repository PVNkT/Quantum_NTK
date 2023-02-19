#!/bin/sh

# sparsity mode 입력
echo "choose sparsity mode : [random, origin, block]"
read mode


if [ $mode == 'random' ]
then
    source ./bash/sparsity_random.sh
    sparsity_random

elif [ $mode == 'origin' ]
then
    source ./bash/sparsity_origin.sh
    sparsity_origin

elif [ $mode == 'block' ]
then
    source ./bash/sparsity_block.sh
    sparsity_block

fi