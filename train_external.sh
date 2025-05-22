#!/bin/bash

set -e

fold=1
split_method="random"
dataset_name="xanthene"            # cyanine, xanthene
task_name="log_molar_absorptivity" # absorption, emission, quantum_yield, log_molar_absorptivity
cuda_id=3

base_data_path="datasets/${split_method}/${dataset_name}_fold${fold}"
split_file="${base_data_path}/${task_name}/splits.npy"

# Check if the splits file exists, if not, create it
if [ ! -f "$split_file" ]; then
    echo "Splits file ($split_file) not found, creating splits..."
    python preprocess_downstream_dataset.py --data_path "$base_data_path" --dataset "$task_name"
else
    echo "Splits file ($split_file) already exists, skipping split creation."
fi

CUDA_VISIBLE_DEVICES=${cuda_id} python finetune_external.py \
    --config base \
    --model_path "models/downstream/scaffold/consolidation_fold1/${task_name}.pth" \
    --data_path "$base_data_path" \
    --dataset ${task_name} \
    --dataset_type regression \
    --metric r2 \
    --split splits \
    --weight_decay 0 \
    --dropout 0.1 \
    --lr 3e-5 \
    --save_dir "models/downstream/${split_method}/${dataset_name}_fold${fold}/${task_name}"

