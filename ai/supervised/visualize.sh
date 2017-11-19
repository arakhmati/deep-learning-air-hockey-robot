#!/bin/bash
cd "$(dirname "$0")"
for f in `find data/*.h5 | shuf` 
do
    echo $f
    python visualize.py -f $f
done 