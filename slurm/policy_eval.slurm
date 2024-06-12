#!/bin/bash

# Example of running python script with a job array

#SBATCH --partition=cpu
#SBATCH --gpus-per-node=0
#SBATCH --cpus-per-task=4                # one CPU core per task
#SBATCH --mem=10G
#SBATCH --time=4-00:00:00
#SBATCH --output slurm/$1-%A.txt            
#SBATCH --ntasks 4                   # how many repetition time for srun
srun wandb agent mengomango/policy_eval/$1