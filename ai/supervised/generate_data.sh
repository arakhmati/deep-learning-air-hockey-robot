#!/bin/bash
for i in `seq 1 50`;
do
    python generate_data.py -n 5000 -dt $(( ( RANDOM % 3 )  + 1 ))
done 