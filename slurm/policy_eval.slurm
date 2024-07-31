#!/bin/bash

# Example of running python script with a job array

#SBATCH --partition=cpu
#SBATCH --gpus-per-node=0
#SBATCH --cpus-per-task=4                # one CPU core per task
#SBATCH --mem=10G
#SBATCH --time=4-00:00:00
#SBATCH --output slurm/policy_eval-%A.txt            
#SBATCH --ntasks 8                   # how many repetition time for srun
srun wandb agent mengomango/policy_eval/$1