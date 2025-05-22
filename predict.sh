#!/bin/bash

set -e

split_method="random"
fold=$1
dataset_name="consolidation"  # consolidation
task_name="absorption"  # absorption, emission, quantum_yield, log_molar_absorptivity
cuda_id=$2

CUDA_VISIBLE_DEVICES=${cuda_id} python predict.py --config base --model_path "models/downstream/${split_method}/${dataset_name}_fold${fold}/${task_name}.pth" --dataset "${task_name}" --data_path "datasets/${split_method}/${dataset_name}_fold${fold}" --dataset_type regression --metric r2 --split splits --results_dir "results/${split_method}/${dataset_name}_fold${fold}/${task_name}"

