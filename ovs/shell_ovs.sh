#!/bin/bash

current_path=$(pwd)
script_dir=$(dirname "$current_path")
echo "root dir: $script_dir"

export PYTHONPATH=$PYTHONPATH:$script_dir

#python predict_labels.py
python diff_for_openvocabulary.py --save_file "1125.txt"