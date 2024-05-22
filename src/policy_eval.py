import time
import numpy as np
import torch
import gymnasium as gym
import os
import pickle
import yaml
import wandb

from models import Policy, REINFORCE
from environments import get_exp_config
from sampling import MonteCarlo, ROS, DataCollector

N_ITER = 30

def pickle_loadall(filename):
    with open(filename, "rb") as f:
        while True:
            try:
                yield pickle.load(f)
            except EOFError:
                break

def policy_evaluation(
        env: gym.Env,
        max_time_step: int,
        collector: DataCollector, # wrapper for evaluation method and policy
        gamma: float = 1.,
        n_trajectory: int = 1e3,
        show: bool = True,
):
    n = 0
    estimation = 0.
    estimations = []
    returns = []
    start_time = time.time()

    s = env.reset()
    g = 0
    time_step = 0
    step = 0
    avg_pie = 0
    while True:
        if type(s) == np.ndarray:
            s = torch.tensor(s).unsqueeze(0)
        elif type(s) == tuple:
            s = torch.tensor(s[0]).unsqueeze(0)
            
        a = collector.act(s)
        s_, r, truncated, terminated, _ = env.step(a)
        d = truncated or terminated or time_step == max_time_step - 1
        g += r * (gamma ** time_step)
        step += 1
        
        if d:
            collector.update(s, a, False)
            returns.append(g)
            estimation = n / (n + 1) * estimation + g / (n + 1)
            estimations.append(estimation)
            n += 1

            if show and n % int(wandb.config.num_episodes /100) == 0:
                import sys
                sys.stdout.write('\b' * 1000 + f'tr {n}, time: {int(time.time() - start_time)}s, avg_step: {step / n}, step {time_step}, est: {estimation:.3f} var: {np.var(returns):.3f}')
                sys.stdout.flush()

            if n == n_trajectory:
                return estimation, np.var(returns), step / n, returns, estimations

            s = env.reset()
            g = 0
            time_step = 0
            
        else:
            collector.update(s, a)
            s = s_
            time_step += 1
            
with open('config/policy_eval.yaml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        wandb.init(config=config)

np.random.seed(wandb.config.seed)
torch.random.manual_seed(wandb.config.seed)
env_config = get_exp_config(wandb.config.env_id)

pi_e = REINFORCE(env_config, 
                 file=f'policies/{wandb.config.env_id}/REINFORCE/model_{wandb.config.policy}_{wandb.config.seed}.pt',
                 device='cpu')

if wandb.config.sampler == 'MonteCarlo':
    collector = MonteCarlo(pi_e)
    ground_truth = False
    
elif wandb.config.sampler.startswith('ROS'):
    collector = ROS(pi_e, 
                    env_config['env'],
                    env_config['max_time_step'],
                    **{'ros_lr': float(wandb.config.sampler.split('_')[1]),
                       'ros_dynamic_lr': False,
                       'ros_first_layer': False,
                       'ros_only_weight': False})
    ground_truth = False
    
elif wandb.config.sampler == 'GroundTruth':
    collector = MonteCarlo(pi_e)
    ground_truth = True
    
else:
    raise ValueError(f'Unknown collector: {wandb.config.collector}')
err_data = []
for i in range(N_ITER):
    policy_value, policy_variance, avg_steps, returns, estimations = policy_evaluation(
        env_config['env'], 
        env_config['max_time_step'],
        collector, env_config['gamma'], 
        n_trajectory=wandb.config.num_episodes,
        show=True)

    print(f'\n{wandb.config.env_id} {i}: avg_step: {avg_steps}, policy value and variance: {policy_value}, {policy_variance}')
    ground_truth_file = f"results/{wandb.config.env_id}/GroundTruth/{wandb.config.policy}_{wandb.config.seed}.pkl"
    if ground_truth:  
        with open(ground_truth_file,'wb') as f:
            pickle.dump({'truth_steps': avg_steps,
                    'truth_value': policy_value,
                    }, f)
        break # ground truth doesn't need repeats
        
    else:
        #calculate errors
        ground_truth_file = f"results/{wandb.config.env_id}/GroundTruth/truth_map.pkl"
        with open(ground_truth_file,'rb') as f:
            info = pickle.load(f)
            truth_steps, truth_value = info[f"{wandb.config.policy}_{wandb.config.seed}"]
            
        estimations = np.array(estimations)
        error =  np.power(estimations - truth_value,2)
        err_data.append(error)

err_data = np.array(err_data)
#create folders if they dont exist and then save run data, appending if it already exists
save_dir = f"results/{wandb.config.env_id}/{wandb.config.sampler}"
save_file = f"{save_dir}/{wandb.config.policy}_{wandb.config.seed}.pkl"
os.makedirs(save_dir, exist_ok=True)
        
with open(save_file,'wb') as f:    
    pickle.dump(err_data, f)