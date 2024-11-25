#!/bin/bash

current_path=$(pwd)
script_dir=$(dirname "$current_path")
echo "root dir: $script_dir"

export PYTHONPATH=$PYTHONPATH:$script_dir

# select ENT and ITER
for ents in 0.02
do
  for iters in 5
  do
    echo "ents: $ents, iters: $iters"
    python diff_for_openvocabulary.py --save_file 'hyper.txt' --iter $iters --enhanced 1.6 --ent $ents
  done
done

# select REC
#for rec in 1.4 1.5 1.6 1.7 1.8
#do
#  echo "recs: $rec"
#  python diff_for_openvocabulary.py --save_file 'hyper_rec.txt' --enhanced $rec
#done
