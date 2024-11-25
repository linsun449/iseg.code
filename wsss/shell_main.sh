#!/bin/bash

current_path=$(pwd)
script_dir=$(dirname "$current_path")
echo "root dir: $script_dir"

export PYTHONPATH=$PYTHONPATH:$script_dir

python main.py --save_file '1125_voc_train.txt'
