import pandas as pd
import wandb
import numpy as np
import matplotlib.pyplot as plt
import yaml
import csv
env_id = 'CartPole-v1'
algo = "mc"
# Import data from CSV file
mc_mse = pd.read_csv(f'results/{env_id}/{algo}/mse.csv')
mc_norm_err = pd.read_csv(f'results/{env_id}/{algo}/norm_err.csv')
print("read csv files")
algo = "ros"

# Import ros data from CSV file
ros_mse = pd.read_csv(f'results/{env_id}/{algo}/mse.csv')
ros_norm_err = pd.read_csv(f'results/{env_id}/{algo}/norm_err.csv')
print("read csv files")

# Calculate column-wise averages
mc_mse_avg = mc_mse.mean(axis=0)
mc_norm_avg = mc_norm_err.mean(axis=0)
ros_mse_avg = mc_mse.mean(axis=0)
ros_norm_avg = mc_norm_err.mean(axis=0)
print("calculated averages")

# Calculate column-wise standard errors
mc_mse_se = mc_mse.sem(axis=0)
mc_norm_se = mc_norm_err.sem(axis=0)
ros_mse_se = mc_mse.sem(axis=0)
ros_norm_se = mc_norm_err.sem(axis=0)
print("calculated standard errors")

# #save average mse into a yaml file
# with open(f'results/{env_id}/{algo}/summary.csv', 'w') as f:
#     writer = csv.writer(f)
#     writer.writerow(mc_mse_avg.iloc[0])
#     print("saved summary")

# Graph mse with standard error in a lighter color +- 1 standard error
# plt.plot(mse_avg, label='MSE', color='blue')
# plt.fill_between(range(len(mse_avg)), mse_avg - mse_se, mse_avg + mse_se, color='lightblue', alpha = 0.5)
# print("plotted mse")

plt.plot(mc_norm_avg, label='NormErr', color='lightblue')
lower_bound = mc_norm_avg - mc_norm_se
upper_bound = mc_norm_avg + mc_norm_se
plt.fill_between(range(len(mc_norm_avg)), lower_bound, upper_bound, color='blue', alpha = 0.3)
print("plotted mc norm_err")

plt.plot(ros_norm_avg, label='NormErr', color='lightblue')
lower_bound = ros_norm_avg - ros_norm_se
upper_bound = ros_norm_avg + ros_norm_se
plt.fill_between(range(len(ros_norm_avg)), lower_bound, upper_bound, color='blue', alpha = 0.3)
print("plotted ros norm_err")

plt.yscale('log')
plt.xlabel('Iterations')
plt.ylabel('Error')
plt.title(f'{algo} Error vs Iterations')
plt.legend()
plt.savefig(f'results/{env_id}/{algo}/norm_error.png')
print("saved graph")