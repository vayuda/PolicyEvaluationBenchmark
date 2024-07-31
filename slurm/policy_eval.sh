#!/bin/bash
rm *.txt
sweep_id=""

if [[ -z "$sweep_id" ]]; then
    # Run wandb sweep command and capture output
    echo "Creating a new sweep as one was not provided..."
    sweep_output=$(wandb sweep --project policy_eval config/policy_eval.yaml 2>&1)
    echo "$sweep_output"
    # Extract sweep ID from output using grep
    sweep_id=$(echo "$sweep_output" | grep -oP '(?<=ID: )\w+')
fi

slurm_jobs=()
for i in $(seq 1 $1); do
    job_id=$(sbatch slurm/policy_eval.slurm $sweep_id | awk '{print $4}')
    echo "Started job $job_id"
    slurm_jobs+=("$job_id")
done