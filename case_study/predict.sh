#!/bin/bash

set -e

cuda_id=3

for task_name in "absorption" "emission" "quantum_yield" "log_molar_absorptivity"; do
    CUDA_VISIBLE_DEVICES=${cuda_id} python ../predict.py --config base --model_path "../models/downstream/scaffold/consolidation_fold1/${task_name}.pth" --dataset "${task_name}" --data_path "./datasets" --dataset_type regression --metric r2 --split splits --results_dir "results/${task_name}"
done
