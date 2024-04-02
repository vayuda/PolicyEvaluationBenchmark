#!/bin/bash
rm slurm/*.txt
# Run wandb sweep command and capture output
echo "Running wandb sweep..."
sweep_output=$(wandb sweep --project ground_truth config/ground_truth.yaml 2>&1)
echo "Sweep output: $sweep_output"

# Extract sweep ID from output using grep
echo "$sweep_output" | grep -oP '\w+'
sweep_id=$(echo "$sweep_output" | grep -oP '(?<=ID: )\w+')
echo "Extracted sweep ID: $sweep_id"

for i in $(seq 1 10); do
    sbatch slurm/p2.slurm $sweep_id &
done

echo "All processes started."
