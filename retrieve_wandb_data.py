import wandb
import yaml
import numpy as np
import csv
def calculate_step_errors(cumulative_steps, err):
    step_mse = []
    for i in range(len(cumulative_steps)-1):
       step_mse += list(np.linspace(err[i],err[i+1], cumulative_steps[i+1]-cumulative_steps[i]))
    return step_mse

def retrieve_ground_truth_data(sweepid, env_id):
    #retrieve data with wandb api
    api = wandb.Api()
    sweep = api.sweep(sweepid)
    runs = sweep.runs

    #aggregate data by policy_id
    policies = { i+1 : {'truth_reward': [], "truth_steps": [], "variance": []} for i in range(100) }
    for run in runs:
        policy_id = run.config['policy_id']
        policies[policy_id]["truth_reward"].append(run.summary['truth_reward'])
        policies[policy_id]["truth_steps"].append(run.summary['truth_steps'])
        policies[policy_id]["variance"].append(run.summary['variance'])

    # aggregate statistics
    for i in range(1,101):
        cumulative_steps = sum(policies[i]["truth_steps"])
        policies[i]["variance"] = sum(
            [policies[i]["truth_steps"][j] * policies[i]["variance"][j] for j in range(len(policies[i]["variance"]))]
        ) / cumulative_steps
        policies[i]["truth_reward"] = sum(policies[i]["truth_reward"]) / len(policies[i]["truth_reward"])
        policies[i]["truth_steps"] = cumulative_steps / len(policies[i]["truth_steps"])

    # store results in a file
    with open(f'results/{env_id}/ground_truth_results.yaml', 'w') as f:
        yaml.dump(policies, f)


def retrieve_mc_data(sweepid, env_id):
    api = wandb.Api()
    sweep = api.sweep(sweepid)
    runs = sweep.runs

    #aggregate across runs
    mc_data = {
            'mse': np.zeros(10000), 
            'norm_err': np.zeros(10000)
    }

    for run in runs:
        mc_data["mse"]+=run.summary['mse']
        mc_data["norm_err"]+=run.summary['norm_err']

    mc_data["mse"]/=len(runs)
    mc_data["norm_err"]/=len(runs)
    
    with open(f'results/{env_id}/mc_results.yaml', 'w') as f:
        yaml.dump(mc_data, f)

def download_wandb_data(sweepid):
    api = wandb.Api()
    sweep = api.sweep(sweepid)
    runs = sweep.runs
    with open(f"results/{run.config['env_id']}/wandb_data/mc.csv", 'w') as f:
        writer = csv.writer(f)
        writer.writerow(["cumulative_steps", "return_estimate", "mse", "norm_err", "standard_error"])
        for run in runs:
            hist = run.scan_history(keys=["cumulative_steps", "return_estimate", "mse", "norm_err", "standard_error"])
            writer.writerows(hist.values())
if __name__ == "__main__":
    # retrieve_ground_truth_data("mengomango/ground_truth/4rgm4uz4", "CartPole-v1")
    retrieve_mc_data("mengomango/mc_eval/67advpqh", "CartPole-v1")