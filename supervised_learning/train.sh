#!/bin/bash
for f in `find data/*.h5` 
do
    python train.py -f $f
done 