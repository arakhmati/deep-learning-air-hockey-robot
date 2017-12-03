#!/bin/bash
cd "$(dirname "$0")"
for f in `find mixed_data/*.h5 | shuf` 
do
    echo $f
    python visualize_data.py -f $f
done 