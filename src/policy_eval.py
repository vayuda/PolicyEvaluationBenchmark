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
from evaluators import MonteCarlo, ROS, PolicyEvaluator, BehaviorPolicySearch

# config
MODE = "online"
USE_WANDB = True


def policy_evaluation(
    env: gym.Env,
    max_time_step: int,
    evaluator: PolicyEvaluator,
    gamma: float = 1.,
    n_trajectory: int = 1e4,
    show: bool = True,
):
    n = 0
    estimation = 0.
    estimations = []
    returns = []
    trajectory = []
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
            
        a, pi_b = evaluator.act(s)
        s_, r, truncated, terminated, _ = env.step(a)
        a= torch.tensor(a)
        d = truncated or terminated or time_step == max_time_step - 1
        g += r * (gamma ** time_step)
        step += 1

        trajectory.append((s.squeeze(dim=0),a, pi_b))
        if d:
            evaluator.update(s, a, (g,trajectory))
            trajectory = []
            returns.append(g)
            estimation = n / (n + 1) * estimation + g / (n + 1)
            estimations.append(estimation)
            n += 1

            if show and n % int(n_trajectory /100) == 0:
                print(f'tr {n}, time: {int(time.time() - start_time)}s, est: {estimation:.5f} var: {np.var(returns):.3f}')

            if n == n_trajectory:
                return estimation, np.var(returns), step / n, returns, estimations

            s = env.reset()
            g = 0
            time_step = 0
            
        else:
            evaluator.update(s, a)
            s = s_
            time_step += 1


if __name__ == '__main__':
    
    RunConfig = namedtuple('RunConfig', ['env_id', 'policy', 'seed','repeats', 'num_episodes', 'sampler', "show"])
    
    if USE_WANDB: # uses wandb and slurm to run experiments with different configs in parallel            
        with open('config/policy_eval.yaml') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            wandb.init(config=config, mode=MODE)
                
        cfgs = [RunConfig(
            wandb.config.env_id, wandb.config.policy, 
            wandb.config.seed, wandb.config.repeats,
            wandb.config.num_episodes, 
            wandb.config.sampler, False
        )]
        
    else:  # for local debugging
        cfgs = [RunConfig('MultiBandit',  1000, 2, 30, 10000, 'BPS_100_1e-2', False),
                RunConfig('MultiBandit',  1000, 2, 30, 10000, 'ROS_1e4', False),
                RunConfig('MultiBandit',  1000, 2, 30, 10000, 'MonteCarlo', False)]
    for cfg in cfgs:
        # env initialization
        np.random.seed(cfg.seed)
        torch.random.manual_seed(cfg.seed)
        env_config = get_exp_config(cfg.env_id)
        ground_truth = False
        
        # save info
        results_folder = f"results/{cfg.env_id}/{cfg.sampler}"
        results_file = f"{cfg.policy}_{cfg.seed}.pkl"
        if os.path.exists(os.path.join(results_folder, results_file)):
            err_data = pickle.load(open(os.path.join(results_folder, results_file), 'rb'))
            if USE_WANDB:
                wandb.log({"error": err_data[-1].mean()})
        os.makedirs(results_folder, exist_ok=True)
        
        
        
        pi_e = REINFORCE(
            env_config, 
            file=f'policies/{cfg.env_id}/REINFORCE/model_{cfg.policy}_{cfg.seed}.pt',
            device='cpu'
        )
        
        # policy evaluation selector and setup
        if cfg.sampler == 'MonteCarlo':
            collector = MonteCarlo(pi_e)
            
        elif cfg.sampler.startswith('ROS'):
            collector = ROS(
                pi_e, 
                env_config['env'],
                env_config['max_time_step'],
                lr=float(cfg.sampler.split('_')[1])
            )
            
        elif cfg.sampler.startswith('BPS'):
            collector = BehaviorPolicySearch(
                pi_e, 
                env_config['env'],
                env_config['max_time_step'],
                k=int(cfg.sampler.split('_')[1]),
                lr=float(cfg.sampler.split('_')[2])
            )
            
        elif cfg.sampler == 'GroundTruth':
            collector = MonteCarlo(pi_e)
            ground_truth = True
            
        else:
            raise ValueError(f'Unknown collector: {cfg.collector}')

        # policy evaluation main loop
        err_data = []
        for i in range(cfg.repeats):
            policy_value, policy_variance, avg_steps, returns, estimations = policy_evaluation(
                env_config['env'], 
                env_config['max_time_step'],
                collector, env_config['gamma'], 
                n_trajectory=cfg.num_episodes,
                show=cfg.show
            )
            # write back results
            
            if ground_truth:  
                with open(os.path.join(results_folder, results_file),'wb') as f:
                    pickle.dump({'truth_steps': avg_steps,
                            'truth_value': policy_value,
                            }, f)
                break # only need to calculate ground truth once and then we're done
                
            else:
                # save mse
                ground_truth_file = f"results/{cfg.env_id}/truth_map.pkl"
                with open(ground_truth_file,'rb') as f:
                    info = pickle.load(f)
                    truth_steps, truth_value = info[f"{cfg.policy}_{cfg.seed}"]
                    
                estimations = np.array(estimations)
                error =  np.power(estimations - truth_value,2)
                err_data.append(error)
                
        if not ground_truth: 
            err_data = np.array(err_data)
            if USE_WANDB:
                wandb.log({"error": err_data[-1].mean()}) # report final mse average over 30 trials to wandb
            else:
                print(f"{cfg.sampler} Final MSE: {err_data[-1].mean()}")
            with open(os.path.join(results_folder, results_file),'wb') as f:    
                pickle.dump(err_data, f)
                