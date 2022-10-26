#!/usr/bin/env bash

CONFIG=$1
#   sh ./tools/mul_test.sh  /home/tju531/hwr/work_dirs/
dir=$(ls -l $1 |awk '/^d/ {print $NF}')
for i in $dir
do
#    echo $i
    echo $1$i'/'$i'.py'
    python tools/test.py \
     --config $1$i'/'$i'.py' \
     --checkpoint $1$i'/'best.pth \
     --work-dir $1$i'/'best/  \
     --out $1$i'/'result.pkl  \
     --eval bbox
done