import time
import numpy as np
import torch
import gymnasium as gym
import os
import pickle
import yaml
import wandb
from collections import namedtuple

from policies import REINFORCE
from environments import get_exp_config
from evaluators import MonteCarlo, ROS, PolicyEvaluator

N_ITER = 30
USE_WANDB = True
def policy_evaluation(
        env: gym.Env,
        max_time_step: int,
        evaluator: PolicyEvaluator,
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
            s = torch.tensor(s,dtype=torch.float32).unsqueeze(0)
        elif type(s) == tuple:
            s = torch.tensor(s[0], dtype=torch.float32).unsqueeze(0)
            
        a = evaluator.act(s)
        s_, r, truncated, terminated, _ = env.step(a)
        d = truncated or terminated or time_step == max_time_step - 1
        g += r * (gamma ** time_step)
        step += 1
        
        if d:
            evaluator.update(s, a)
            returns.append(g)
            estimation = n / (n + 1) * estimation + g / (n + 1)
            estimations.append(estimation)
            n += 1

            if show and n % int(n_trajectory /100) == 0:
                import sys
                sys.stdout.write('\b' * 1000 + f'tr {n}, time: {int(time.time() - start_time)}s, avg_step: {step / n}, step {time_step}, est: {estimation:.3f} var: {np.var(returns):.3f}')
                sys.stdout.flush()

            if n == n_trajectory:
                return estimation, np.var(returns), step / n, returns, estimations

            s = env.reset()
            g = 0
            time_step = 0
            
        else:
            evaluator.update(s, a)
            s = s_
            time_step += 1

RunConfig = namedtuple('RunConfig', ['env_id', 'policy', 'seed', 'num_episodes', 'sampler', "show"])
if __name__ == '__main__':
    if USE_WANDB: # uses wandb and slurm to run experiments with different configs in parallel            
        with open('config/policy_eval.yaml') as f:
                config = yaml.load(f, Loader=yaml.FullLoader)
                wandb.init(config=config)
                
        cfg = RunConfig(
            wandb.config.env_id, wandb.config.policy, 
            wandb.config.seed, wandb.config.num_episodes, 
            wandb.config.sampler, False
        )
        
    else:  # for local debugging
        cfg = RunConfig('GridWorld',  0, 0, 1000, 'GroundTruth', True)
    

    np.random.seed(cfg.seed)
    torch.random.manual_seed(cfg.seed)
    env_config = get_exp_config(cfg.env_id)

    pi_e = REINFORCE(env_config, 
                    file=f'policies/{cfg.env_id}/REINFORCE/model_{cfg.policy}_{cfg.seed}.pt',
                    device='cpu')
    
    os.makedirs(f"results/{cfg.env_id}/{cfg.sampler}", exist_ok=True)
    
    if cfg.sampler == 'MonteCarlo':
        collector = MonteCarlo(pi_e)
        batch_size = 1
        ground_truth = False
        
    elif cfg.sampler.startswith('ROS'):
        collector = ROS(pi_e, 
                        env_config['env'],
                        env_config['max_time_step'],
                        **{'ros_lr': float(cfg.sampler.split('_')[1]),
                        'ros_dynamic_lr': False,
                        'ros_first_layer': False,
                        'ros_only_weight': False})
        ground_truth = False
        batch_size = 1
    elif cfg.sampler == 'BehaviorPolicySearch':
        pass
    elif cfg.sampler == 'GroundTruth':
        collector = MonteCarlo(pi_e)
        batch_size = 1
        ground_truth = True
        
    else:
        raise ValueError(f'Unknown collector: {cfg.collector}')

    err_data = []
    for i in range(N_ITER):
        policy_value, policy_variance, avg_steps, returns, estimations = policy_evaluation(
            env_config['env'], 
            env_config['max_time_step'],
            collector, env_config['gamma'], 
            n_trajectory=cfg.num_episodes,
            show=cfg.show)

        # print(f'\n{cfg.env_id} {i}: avg_step: {avg_steps}, policy value and variance: {policy_value}, {policy_variance}')
        ground_truth_file = f"results/{cfg.env_id}/GroundTruth/{cfg.policy}_{cfg.seed}.pkl"
        if ground_truth:  
            with open(ground_truth_file,'wb') as f:
                pickle.dump({'truth_steps': avg_steps,
                        'truth_value': policy_value,
                        }, f)
            break # only need to calculate ground truth once and then were done
            
        else:
            #calculate errors
            
            ground_truth_file = f"results/{cfg.env_id}/GroundTruth/truth_map.pkl"
            with open(ground_truth_file,'rb') as f:
                info = pickle.load(f)
                truth_steps, truth_value = info[f"{cfg.policy}_{cfg.seed}"]
                
            estimations = np.array(estimations)
            error =  np.power(estimations - truth_value,2)
            err_data.append(error)
            
    if not ground_truth: 
        err_data = np.array(err_data)
        #create folders if they dont exist and then save run data, appending if it already exists
        save_dir = f"results/{cfg.env_id}/{cfg.sampler}"
        save_file = f"{save_dir}/{cfg.policy}_{cfg.seed}.pkl"
        os.makedirs(save_dir, exist_ok=True)
        wandb.log({"error": err_data[-1].mean()})
        with open(save_file,'wb') as f:    
            pickle.dump(err_data, f)
            