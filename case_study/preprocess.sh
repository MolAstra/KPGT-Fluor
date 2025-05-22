for task_name in "absorption" "emission" "quantum_yield" "log_molar_absorptivity"; do
    python ../preprocess_downstream_dataset.py --data_path datasets --dataset ${task_name}
done
