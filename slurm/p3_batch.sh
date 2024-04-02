#!/bin/bash
rm slurm/*.txt
algorithm="mc"
# Run wandb sweep command and capture output
echo "Running wandb sweep..."
sweep_output=$(wandb sweep --project "${algorithm}_eval" config/$algorithm.yaml 2>&1)
echo "Sweep output: $sweep_output"

# Extract sweep ID from output using grep
echo "$sweep_output" | grep -oP '\w+'
sweep_id=$(echo "$sweep_output" | grep -oP '(?<=ID: )\w+')
echo "Extracted sweep ID: $sweep_id"


for i in $(seq 1 10); do
    sbatch slurm/p3.slurm "${algorithm}_eval" $sweep_id &
done

echo "All processes started."
