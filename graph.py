import pandas as pd
import wandb
import numpy as np
import matplotlib.pyplot as plt
import yaml

env_id = 'CartPole-v1'
algo = "mc"
# Import data from CSV file
mse = pd.read_csv(f'results/{env_id}/{algo}_mse.csv')
norm_err = pd.read_csv(f'results/{env_id}/{algo}_norm_err.csv')


# Calculate column-wise averages
mse_avg = mse.mean(axis=0)
norm_avg = norm_err.mean(axis=0)

# Calculate column-wise standard errors
mse_se = mse.sem(axis=0)
norm_se = norm_err.sem(axis=0)

#save average mse into a yaml file
with open(f'results/{env_id}/{algo}/summary.yaml', 'w') as f:
    yaml.dump({
        "mse_1": mse_avg.iloc[0],
        "mse_10k": mse_avg.iloc[-1],
        "mse": mse_avg,
        "norm_err": norm_avg
    }, f)
# Graph mse with standard error in a lighter color +- 1 standard error
plt.plot(mse_avg, label='MSE', color='blue')
plt.fill_between(range(len(mse_avg)), mse_avg - mse_se, mse_avg + mse_se, color='lightblue')

plt.plot(mse_avg, label='NormErr', color='red')
plt.fill_between(range(len(norm_avg)), norm_avg - norm_se, norm_avg + norm_se, color='lightred')

plt.xlabel('Iterations')
plt.ylabel('Error')
plt.title(f'{algo} Error vs Iterations')
plt.legend()
plt.savefig(f'results/{env_id}/{algo}/error-test.png')