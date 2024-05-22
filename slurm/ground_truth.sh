#!/bin/bash
rm slurm/*.txt
# Run wandb sweep command and capture output
echo "Running wandb sweep..."
sweep_output=$(wandb sweep --project policy_eval config/ground_truth.yaml 2>&1)


# Extract sweep ID from output using grep
sweep_id=$(echo "$sweep_output" | grep -oP '(?<=ID: )\w+')
echo "Extracted sweep ID: $sweep_id"

for i in $(seq 1 10); do
    sbatch slurm/ground_truth.slurm $sweep_id &
done

echo "All processes started."
