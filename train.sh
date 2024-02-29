#!/bin/bash

counter=0

while true; do
    if [ $counter -eq 0 ]; then
        python3 lstm.py output.csv
    else
        python3 lstm.py data.csv
    fi
    python3 gan.py
    ((counter++))
done
