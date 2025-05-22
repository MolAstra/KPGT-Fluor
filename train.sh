#!/bin/bash

set -e

split_method="random"
fold=1
dataset_name="consolidation" # consolidation
task_name="quantum_yield"    # absorption, emission, quantum_yield, log_molar_absorptivity
cuda_id=2

split_file="datasets/${split_method}/${dataset_name}_fold${fold}/${task_name}/splits.npy"
if [ ! -f "$split_file" ]; then
    echo "Splits file ($split_file) not found, creating splits..."
    python preprocess_downstream_dataset.py --data_path datasets/${split_method}/${dataset_name}_fold${fold} --dataset ${task_name}
fi

CUDA_VISIBLE_DEVICES=${cuda_id} python finetune.py --config base --model_path models/pretrained/base.pth --data_path datasets/${split_method}/${dataset_name}_fold${fold} --dataset ${task_name} --dataset_type regression --metric r2 --split splits --weight_decay 0 --dropout 0.1 --lr 3e-5 --save_dir models/downstream/${split_method}/${dataset_name}_fold${fold}/${task_name}
