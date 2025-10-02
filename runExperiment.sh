#!/bin/bash

log=$(mktemp XXX.log)

python code/concatenate.py
script -c "python code/APIcall+evaluate.py" "$log"

echo "SAVED LOG TO $log"
