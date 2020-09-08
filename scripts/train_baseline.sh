#!/usr/bin/env bash

# This is a script that will train all of the models for scene graph classification and then evaluate them.
export CUDA_VISIBLE_DEVICES=$1

echo "TRAINING THE BASELINE"
python models/train_rels.py -m sgcls -model motifnet -nl_obj 0 -val_size 0 -nl_edge 0 -b 6 \
-clip 5 -p 100 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -save_dir /home/xujing/tmp/checkpoints/baseline2 \
-nepoch 50 -use_bias -test