import time
import numpy as np
import torch
import gymnasium as gym
import matplotlib.pyplot as plt
import argparse
import pickle

from models import Policy, REINFORCE
from policy_gen import get_exp_config
from sampling import MonteCarlo, ROS, DataCollector

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default = 0)
parser.add_argument('--policy_name', type=str, default='CartPole_5000')
parser.add_argument('--sampler', type=str, default='MonteCarlo')
parser.add_argument('--n', type=int, default=1e6)
args = parser.parse_args()

 

def mc_estimation_true_value(
        env: gym.Env,
        max_time_step: int,
        collector: DataCollector,
        gamma: float = 1.,
        n_trajectory: int = 1e3,
        show: bool = True,
):
    n = 0
    estimation = 0.
    estimations = []
    returns = []
    start_time = time.time()

    s = env.reset(seed=args.seed)
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
            collector.update(s, a, True)
            returns.append(g)
            estimation = n / (n + 1) * estimation + g / (n + 1)
            estimations.append(estimation)
            n += 1

            if show and n % 100 == 0:
                import sys
                sys.stdout.write('\b' * 1000 + f'tr {n}, time: {int(time.time() - start_time)}s, avg_step: {step / n}, step {time_step}, returns: {g}, estimation and variance: {estimation}, {np.var(returns)}')
                sys.stdout.flush()

            if n == n_trajectory:
                return estimation, np.var(returns), step / n, returns

            s = env.reset(seed=args.seed)
            g = 0
            time_step = 0
        else:
            collector.update(s, a)
            s = s_
            time_step += 1
            

np.random.seed(args.seed)
torch.random.manual_seed(args.seed)
exp_name = args.policy_name.split('_')[0]
env_config = get_exp_config(exp_name)

pi_e = REINFORCE(env_config, file=f'policies/model_{args.policy_name}.pt', device='cpu')
if args.sampler == 'MonteCarlo':
    collector = MonteCarlo(pi_e)
elif args.sampler == 'ROS':
    collector = ROS(pi_e, env_config['env'], env_config['max_time_step'], **{'ros_lr': 0.1, 'ros_dynamic_lr': False, 'ros_first_layer': False, 'ros_only_weight': False})
else:
    raise ValueError(f'Unknown collector: {args.collector}')
true_value, true_variance, avg_steps, returns = mc_estimation_true_value(env_config['env'], env_config['max_time_step'], collector, env_config['gamma'], n_trajectory=args.n, show=True)
print(f'\n{exp_name}: avg_step: {avg_steps}, true policy value and variance: {true_value}, {true_variance}')

with open("policies/CartPole.txt","a") as f:
    f.write(f'\n{args.policy_name}: avg_step: {avg_steps}, true policy value and variance: {true_value}, {true_variance}')

with open("results/CartPole-v1/ros/CartPole_5000.pkl",'wb') as f:
    pickle.dump({'truth_steps': avg_steps,
                 'truth_value': true_value,
                 'truth_variance': true_variance,
                 'returns': returns}, f)