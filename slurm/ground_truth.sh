#!/usr/bin/bash
rm slurm/*.txt
# Run wandb sweep command and capture output

sweep_id=""
if [[ -z "$sweep_id" ]]; then
    echo "Running wandb sweep..."
    sweep_output=$(wandb sweep --project policy_eval config/ground_truth.yaml 2>&1)
    echo "$sweep_output"
    # Extract sweep ID from output using grep
    sweep_id=$(echo "$sweep_output" | grep -oP '(?<=ID: )\w+')
    echo "Extracted sweep ID: $sweep_id"
fi

for i in $(seq 1 10); do
    sbatch slurm/ground_truth.slurm $sweep_id &
done

echo "All processes started."
