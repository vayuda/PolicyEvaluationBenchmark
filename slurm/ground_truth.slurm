#!/bin/bash

# Example of running python script with a job array

#SBATCH --partition=cpu
#SBATCH --gpus-per-node=0
#SBATCH --cpus-per-task=4                # one CPU core per task
#SBATCH --mem=10G
#SBATCH --time=4-00:00:00
#SBATCH --output slurm/policyGT-%A.txt            
#SBATCH --ntasks 7                    # how many repetition time for srun
# wandb sweep --project ground_truth ../ground_truth.yaml
srun wandb agent mengomango/policy_eval/$1