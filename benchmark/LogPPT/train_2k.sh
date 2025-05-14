#!/bin/bash

shot=32
max_train_steps=1000
task_output_dir="../../result/result_LogPPT_2k/"

for dataset in Apache BGL Hadoop HDFS HealthApp HPC Linux Mac OpenSSH OpenStack Proxifier Spark Thunderbird Zookeeper Android Windows
# for dataset in Apache
    do
        trf="datasets/${dataset}/${shot}shot/1.json"
        tef="datasets/${dataset}/test.json"
        python train.py --seed 42 --mode prompt-tuning --train_file ${trf} \
            --validation_file ${tef} \
            --model_name_or_path "./pretrained_models/roberta-base" \
            --per_device_train_batch_size 8 \
            --learning_rate 5e-5 \
            --lr_scheduler_type polynomial \
            --task_name log-parsing \
            --num_warmup_steps 20 \
            --log_file ../../2k_dataset/${dataset}/${dataset}_2k.log_structured_corrected.csv \
            --shot $shot \
            --dataset_name ${dataset} \
            --max_train_steps ${max_train_steps} \
            --task_output_dir ${task_output_dir}
    done
