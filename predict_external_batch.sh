#!/bin/bash

set -e
cuda_id=2
for fold in 0 1 2 3 4; do
    for dataset_name in cyanine xanthene; do
        for task_name in absorption emission quantum_yield log_molar_absorptivity; do
            for split_method in scaffold random; do
                CUDA_VISIBLE_DEVICES=${cuda_id} python predict.py --config base --model_path "models/downstream/${split_method}/${dataset_name}_fold${fold}/${task_name}.pth" --dataset "${task_name}" --data_path "datasets/${split_method}/${dataset_name}_fold${fold}" --dataset_type regression --metric r2 --split splits --results_dir "results/${split_method}/${dataset_name}_fold${fold}/${task_name}"
            done
        done
    done
done
