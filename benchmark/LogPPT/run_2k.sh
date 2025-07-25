#!/bin/bash

echo 'Starting LogPPT run'
echo $(date)

# conda activate LogPPT
# export CUDA_VISIBLE_DEVICES=3

python fewshot_sampling.py
./train_2k.sh

# cd ../evaluation/
# # conda activate logevaluate
# python LogPPT_eval.py -otc

# cd ../LogPPT

echo 'Ending LogPPT run'
echo $(date)
