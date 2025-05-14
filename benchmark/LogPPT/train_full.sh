#!/bin/bash

shot=32
max_train_steps=400 #1000
task_output_dir="../../result/result_LogPPT_full/"

# max_train_steps=0
# task_output_dir="../../result/result_LogPPT-untrained_full/"

dataset=OpenSSH


for dataset in Apache Hadoop HealthApp HPC Linux Mac OpenSSH OpenStack Proxifier Zookeeper
# for dataset in OpenSSH
do
  trf="datasets/${dataset}/${shot}shot/2.json"
  tef="datasets/${dataset}/test.json"
  python train.py --seed 42 --mode prompt-tuning --train_file ${trf} \
    --validation_file ${tef} \
    --model_name_or_path "./pretrained_models/roberta-base" \
    --per_device_train_batch_size 8 \
    --learning_rate 5e-5 \
    --lr_scheduler_type polynomial \
    --task_name log-parsing \
    --num_warmup_steps 20 \
    --log_file ../../full_dataset/${dataset}/${dataset}_full.log_structured.csv \
    --shot $shot \
    --dataset_name ${dataset} \
    --max_train_steps ${max_train_steps} \
    --task_output_dir ${task_output_dir}
    end=$(date +%s.%N)
    elapsed=$(echo "$end - $start" | bc)
  done

# for split in 1 2 3 4 5 6
# do
#   echo "Split: ${split}"
#   trf="datasets/${dataset}/${shot}shot/2.json"
#   tef="datasets/${dataset}/test.json"
#   python train.py --seed 42 --mode prompt-tuning --train_file ${trf} \
#     --validation_file ${tef} \
#     --model_name_or_path "./pretrained_models/roberta-base" \
#     --per_device_train_batch_size 8 \
#     --learning_rate 5e-5 \
#     --lr_scheduler_type polynomial \
#     --task_name log-parsing \
#     --num_warmup_steps 20 \
#     --log_file ../../full_dataset/${dataset}/${dataset}_full.log_structured.csv \
#     --shot $shot \
#     --dataset_name ${dataset} \
#     --max_train_steps ${max_train_steps} \
#     --task_output_dir ${task_output_dir}
#     end=$(date +%s.%N)
#     elapsed=$(echo "$end - $start" | bc)
#   done
