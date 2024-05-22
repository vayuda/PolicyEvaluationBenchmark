import gymnasium as gym
from gym.spaces import Box
import numpy as np
import torch
import yaml
import wandb
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
            s = torch.tensor(s).unsqueeze(0)
        elif type(s) == tuple:
            s = torch.tensor(s[0]).unsqueeze(0)
        a, _ = actor.sample(s, min_sigma=0.00)
        a = [np.clip(a[0], env.action_space.low[0], env.action_space.high[0])] if isinstance(env.action_space, Box) else a

        s_, r, truncated, terminated , info = env.step(a)
        # print(a, r)
        d = truncated or terminated
        d = d or t == max_time_step - 1
        returns += r

        # learning trick for MountainCar
        if wandb.config.env_id == 'MountainCar' or wandb.config.env_id == 'MountainCarContinuous':
            d = d or (t == 200)  # max_time_step
            r = (max([t[1][0] for t in trajectory]) + 0.5) * 100 if d else r  # higher reward if closer to goal

        trajectory.append((t, s, a, r, d, s_))
        s = s_
        t += 1

    return trajectory, returns


def train():
    torch.manual_seed(wandb.config.seed)
    np.random.seed(wandb.config.seed)
    config = get_exp_config(wandb.config.env_id)
    print(wandb.config.policy)
    print(f'training: {wandb.config.env_id}, config: {config}')
    
    
    env = config['env']
    max_time_step = config['max_time_step']
    gamma = config['gamma']
    save_episodes = config['save_episodes']
    
    training_episodes = max(save_episodes) + 1
    actor = REINFORCE(config, device='cpu')
    batch = []
    batch_size = 17
    for episode in range(training_episodes):
        if episode in save_episodes:
            actor.save(f'policies/{wandb.config.env_id}/{wandb.config.policy}/model_{episode}_{wandb.config.seed}.pt')

        tr, returns = play_episode(env, max_time_step, actor)
        batch.append(tr)

        if len(batch) == batch_size:
            actor.update(batch)
            batch.clear()

        if episode % 500 == 0:
            eval_trs, eval_returns = zip(*[play_episode(env, max_time_step, actor) for i in range(100)])
            print(
                f'episode {episode}\t'
                f'training_tot_returns={returns:.2f}, '
                f'training_dis_returns={sum([t[3] * gamma ** i for i, t in enumerate(tr)]):.2f}\t'
                f'eval_tot_returns={np.mean(eval_returns):.2f}({np.min(eval_returns):.2f}, {np.max(eval_returns):.2f})\t'
                f'eval_dis_returns={np.mean([sum([t[3] * gamma ** i for i, t in enumerate(tr)]) for tr in eval_trs]):.2f}\t'
            )   

if __name__ == '__main__': 
    with open('config/policy_gen.yaml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        wandb.init(config=config)
    train()