#!/usr/bin/env bash
# source ~/.zshrc

param="-orig"
param="-full"

export CUDA_VISIBLE_DEVICES=3
# conda activate UniParser
# unset LD_LIBRARY_PATH
# python process_log_parsing_input_to_ner.py ${param}
# python TrainNERLogAll.py -epoch 1000 ${param}
python InferNERLogAll.py -epoch 1000 ${param}
# conda deactivate

# cd ../evaluation/
# # conda activate logevaluate
# python UniParser_eval.py -otc

# cd ../UniParser
# # conda deactivate
