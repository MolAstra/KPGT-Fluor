#!/bin/bash

set -e

split_method="scaffold"
fold=1
dataset_name="consolidation" # consolidation
task_name="absorption"    # absorption, emission, quantum_yield, log_molar_absorptivity
cuda_id=3

python ../preprocess_downstream_dataset.py --data_path "datasets/${task_name}/input.csv" --dataset ${task_name}


CUDA_VISIBLE_DEVICES=${cuda_id} python finetune.py --config base --model_path models/pretrained/base.pth --data_path datasets/${split_method}/${dataset_name}_fold${fold} --dataset ${task_name} --dataset_type regression --metric r2 --split splits --weight_decay 0 --dropout 0.1 --lr 3e-5 --save_dir models/downstream/${split_method}/${dataset_name}_fold${fold}/${task_name}


split_method="scaffold"
fold=1
dataset_name="consolidation" # consolidation
task_name="absorption"       # absorption, emission, quantum_yield, log_molar_absorptivity
cuda_id=3

CUDA_VISIBLE_DEVICES=${cuda_id} python predict.py \
  --config base \
  --model_path "models/downstream/${split_method}/${dataset_name}_fold${fold}/${task_name}.pth" \
  --dataset "${task_name}" \
  --data_path "datasets/${split_method}/${dataset_name}_fold${fold}" \
  --dataset_type regression \
  --metric r2 \
  --split splits \
  --results_dir "case_study/${task_name}"
