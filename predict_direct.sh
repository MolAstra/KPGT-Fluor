#!/bin/bash

set -e

split_method="random"
for fold in 0 1 2 3 4; do
    dataset_name="consolidation"
    external_dataset_name="xanthene"  # cyanine, xanthene
    task_name="log_molar_absorptivity"  # absorption, emission, quantum_yield, log_molar_absorptivity
    cuda_id=1

    CUDA_VISIBLE_DEVICES=${cuda_id} python predict_direct.py --config base --model_path "models/downstream/${split_method}/${dataset_name}_fold${fold}/${task_name}.pth" --dataset "${task_name}" --data_path "datasets/${split_method}/${external_dataset_name}_fold${fold}" --dataset_type regression --metric r2 --split splits --results_dir "results_direct/${split_method}/${external_dataset_name}_fold${fold}/${task_name}"
done
