import pandas as pd
import wandb
import numpy as np
import matplotlib.pyplot as plt
import yaml
import pickle
import os

name = 'policy_ground_truth'
steps = 4000
seeds = [0,1,2,3,4,5,6,7,8,9,10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
models = [500,1000,1500,2000,2500,3000,3500,4000,4500,5000]
samplers = ["MonteCarlo", "ROS_1e4"]
graph_name = 'ros_1000k' 


# aggregate error data into the dataframe
for sampler in samplers:
    results = []
    print(f'Loading data for {sampler}')
    for model in models:
        for seed in seeds:
            results_file = f'results/{name}/{sampler}/{model}_{seed}.pkl'
            with open(results_file, 'rb') as f:
                errs = pickle.load(f)
                results.append(errs**2)
                
    print(f'creating graph for {sampler}')
    results = pd.DataFrame(results)
    # calculate the mean squared error
    mse = results.mean(axis=0)
    std_err = results.sem(axis=0)
    plt.plot(mse, label=sampler)
    lower = mse - std_err
    upper = mse + std_err
    plt.fill_between(range(steps), lower, upper, alpha=0.2)
    


os.makedirs(f"plots/{name}", exist_ok=True)
# mc1 =mc_mse[0]

# norm_mc_mse = mc_mse/ mc1
# norm_ros_mse = ros_mse / mc1

# norm_mc_error = np.abs(mc_estimate - truth_value) / mc1
# norm_ros_error = np.abs(ros_estimate - truth_value) / mc1
plt.xlabel('Steps')
plt.ylabel('Mean Squared Error')
plt.legend()
# plt.xscale('log')
plt.yscale('log')
plt.savefig(f'plots/{name}/{graph_name}.png')
