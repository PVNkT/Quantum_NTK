#!/bin/sh

function sparsity_origin {

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
            python3 main.py sparse.sparsity=$FIRST seed=$seed_num
            #kill -9 `pgrep -f "python3 main.py"`
            #http://taewan.kim/tip/ps_grep_kill/ : cache 삭제에 대한 참고자료.
            #killall -9 "python3 main.py"
            
        done

        FIRST=$(echo "$FIRST+$TERM" |bc -l)
        ITER=$(echo "$LAST-$FIRST" |bc)
        ITER=$(echo "$ITER/$TERM" |bc)
        
        if [ $ITER -le 0 ]; then          #-le = '<='
            break
        fi
    done
}

# #for single sampling
# while :
# do
#     seed_num=$Random
  
#     python3 main.py sparse.sparsity=$FIRST seed=$seed_num
#     FIRST=$(echo "$FIRST+$TERM" |bc -l)
#     ITER=$(echo "$LAST-$FIRST" |bc)
#     ITER=$(echo "$ITER/$TERM" |bc)
    
#     if [ $ITER -le 0 ]; then          #-le = '<='
#         break
#     fi
# done

# echo : python의 print문
# read 인자값 입력받아오기.
# |bc는 사칙 연산을 위한 파이프라인. 