#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --gpus-per-node=0
#SBATCH --cpus-per-task=3                # one CPU core per task
#SBATCH --mem=3G
#SBATCH --time=4-00:00:00
#SBATCH --output slurm/policygen-%A.txt            
#SBATCH --ntasks 1                    # how many repetition time for srun

srun wandb agent mengomango/policy_eval/$1