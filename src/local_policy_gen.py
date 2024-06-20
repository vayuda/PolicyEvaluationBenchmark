import gymnasium as gym
from gym.spaces import Box
import numpy as np
import torch
import os

from models import REINFORCE
from environments import get_exp_config


def play_episode(env: gym.Env, max_time_step: int, actor: REINFORCE):
    t = 0
    d = False
    returns = 0
    s = env.reset()
    trajectory = []
    while not d:
        if type(s) == np.ndarray:
            s = torch.tensor(s,dtype=torch.float32).unsqueeze(0)
        elif type(s) == tuple:
            s = torch.tensor(s[0], dtype=torch.float32).unsqueeze(0)
        a, _ = actor.sample(s, min_sigma=0.00)
        a = [np.clip(a[0], env.action_space.low[0], env.action_space.high[0])] if isinstance(env.action_space, Box) else a

        s_, r, truncated, terminated , info = env.step(a)
        d = truncated or terminated
        d = d or t == max_time_step - 1
        returns += r
        trajectory.append((t, s, a, r, d, s_))
        s = s_
        t += 1

    return trajectory, returns


def train(env_name, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    config = get_exp_config(env_name)
    print(f'training on: {env_name}, config: {config}')
    
    
    env = config['env']
    max_time_step = config['max_time_step']
    gamma = config['gamma']
    save_episodes = config['save_episodes']
    
    training_episodes = max(save_episodes) + 1
    actor = REINFORCE(config, device='cpu')
    batch = []
    batch_size = 10
    save_dir = f'policies/{env_name}/REINFORCE'
    os.makedirs(save_dir, exist_ok=True)
    for episode in range(training_episodes):
        if not episode % 2500:
            actor.save(os.path.join(save_dir, f'model_{episode}_{seed}.pt'))

        tr, returns = play_episode(env, max_time_step, actor)
        batch.append(tr)

        if len(batch) == batch_size:
            actor.update(batch)
            batch.clear()

        if episode % 500 == 0:
            eval_trs, eval_returns = zip(*[play_episode(env, max_time_step, actor) for i in range(100)])
            print( f'ep {episode}: train_returns: {returns:.2f} eval_returns={np.mean(eval_returns):.2f}({np.min(eval_returns):.2f}, {np.max(eval_returns):.2f})') 

if __name__ == '__main__': 
    env = "GridWorld"
    train(env)