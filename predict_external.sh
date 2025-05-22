#!/bin/bash

set -e

fold=1
split_method="random"
dataset_name="xanthene" # cyanine, xanthene
task_name="log_molar_absorptivity" # absorption, emission, quantum_yield, log_molar_absorptivity
cuda_id=3

CUDA_VISIBLE_DEVICES=${cuda_id} python predict.py --config base --model_path "models/downstream/${split_method}/${dataset_name}_fold${fold}/${task_name}.pth" --dataset "${task_name}" --data_path "datasets/${split_method}/${dataset_name}_fold${fold}" --dataset_type regression --metric r2 --split splits --results_dir "results/${split_method}/${dataset_name}_fold${fold}/${task_name}"

