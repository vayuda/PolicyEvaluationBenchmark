# wandb sweep --project policy_gen ../config.yaml
rm slurm/*.txt
for i in $(seq 1 2); do
    sbatch slurm/p1.slurm &
done
