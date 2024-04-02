#!/bin/bash

# Example of running python script with a job array

#SBATCH --exclude=adriatic[01-04],cheetah01,cheetah02,cheetah03,jaguar01,jaguar02,jaguar04,lotus,lynx[01-02]
#SBATCH --partition=main
#SBATCH --gpus-per-node=0
#SBATCH --cpus-per-task=4                # one CPU core per task
#SBATCH --mem=3G
#SBATCH --time=4-00:00:00
#SBATCH --output slurm/$1-%A.txt            
#SBATCH --ntasks 16                    # how many repetition time for srun
srun wandb agent mengomango/$1/$2