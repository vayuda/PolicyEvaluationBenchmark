import pandas as pd
import wandb
import numpy as np
import matplotlib.pyplot as plt
import yaml
import pickle
import os

def bounds_to_list(min, max):
    return list(range(min, max + 1))


with open("config/policy_eval.yaml", 'r') as file:
    config = yaml.safe_load(file)
    
result_dir = "results/MultiBandit"
parameters = config.get('parameters', {})
steps = parameters["num_episodes"]["value"]
seeds = bounds_to_list(parameters["seed"]["min"], parameters["seed"]["max"])
models = parameters["policy"]["values"]
samplers = parameters["sampler"]["values"]
plot_dir = f"{result_dir}/plots"
os.makedirs(plot_dir, exist_ok=True)

'''
["BPS_100_1e-2",
"BPS_100_1e-1", 
"BPS_100_1",
"BPS_100_1e2",
"BPS_100_1e3", 
"BPS_100_1e4", 
"BPS_100_1e5", 
"BPS_100_1e6",
"ROS_1e-2",
"ROS_1",
"ROS_1e3",
"ROS_1e4",
"ROS_1e5"]
'''


# aggregate error data into the dataframe
def graph_mse():
    name = 'mse-1k'
    samplers = ["BPS_1000_1e-4", "MonteCarlo"]
    for sampler in samplers:
        first = True
        results = []
        print(f'Loading data for {sampler}')
        with open(f"{result_dir}/{sampler}_results.pkl", "rb") as pickle_file:
            results = pickle.load(pickle_file)
                    
        print(f'creating graph for {sampler}')
        results = pd.DataFrame(results)
        # calculate the mean squared error
        mse = results.mean(axis=0)
        std_err = results.sem(axis=0)
        plt.plot(mse, label=sampler)
        lower = mse - std_err
        upper = mse + std_err
        plt.fill_between(range(steps), lower, upper, alpha=0.2)
    

    
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
    plt.savefig(f'{plot_dir}/{name}.png')


def create_histogram():
    with open(f'{result_dir}/final_means.pkl', 'rb') as f:
        data = pickle.load(f)
        for sampler in samplers:
            if sampler == "MonteCarlo":
                continue
            print("sampler", sampler)
            sampler_result = data[sampler]
            plt.hist(sampler_result, bins=20, alpha=0.5, label=sampler)
            plt.legend()
            plt.xlabel('MC - sampler MSE difference')
            # plt.xlim(-0.01,0.01)
            plt.ylabel('Frequency')
            plt.savefig(f'{plot_dir}/{sampler}_hist.png')
            plt.clf()
            sampler_result.sort()
            print(f"worst 10: {[int(i*1000)/1000. for i in sampler_result[:10]]}")
            sampler_result.sort(reverse=True)
            print(f"best 10: {[int(i*1000)/1000. for i in sampler_result[:10]]}")
            print(f'mse < MC %: {100*len([i for i in sampler_result if i > 0]) / len(sampler_result):.3f}')
        
if __name__ == '__main__':
    import sys
    if sys.argv[1] == 'mse':
        graph_mse()
    elif sys.argv[1] == 'hist':
        create_histogram()