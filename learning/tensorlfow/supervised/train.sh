#!/bin/bash
for f in `find data/*.h5 | shuf` 
do
    python train.py -f $f
done 